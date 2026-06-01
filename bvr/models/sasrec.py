"""
SASRec: Self-Attentive Sequential Recommendation
Kang & McAuley, ICDM 2018  —  https://arxiv.org/abs/1808.09781

Tests whether the TEMPORAL ORDER of venue visits adds signal beyond
aggregate behavioral features (BiRank) or graph structure (LightGCN).

Architecture:
  - Item embedding table
  - L transformer blocks: multi-head self-attention + position-wise FFN
  - Causal (left-to-right) masking — predicts next item from prefix
  - Trained with BCE loss on last item held-out per sequence

Usage:
    ranking = build_sasrec_ranking(train_df)
    # Returns {venue_id: float_score} compatible with evaluate_per_user()
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional


# ── Defaults ─────────────────────────────────────────────────────────────────
N_LAYERS    = 2       # Kang & McAuley use 2 for most datasets
N_HEADS     = 1       # 1 head is optimal for small datasets
EMB_DIM     = 64      # matches LightGCN / MF for fair comparison
MAX_SEQ_LEN = 50      # truncate sequences to last 50 visits
DROPOUT     = 0.2
N_EPOCHS    = 50
BATCH_SIZE  = 256
LR          = 0.001


def _get_device() -> torch.device:
    # Sparse ops not needed here — MPS works for dense transformer
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class PointWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.conv1  = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.conv2  = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop   = nn.Dropout(dropout)
        self.act    = nn.GELU()

    def forward(self, x):                 # x: (B, T, D)
        out = self.drop(self.act(self.conv1(x.transpose(-1, -2))))
        out = self.drop(self.conv2(out)).transpose(-1, -2)
        return out


class SASRec(nn.Module):
    """
    SASRec model.

    Input:  padded item sequences, shape (B, T)
    Output: next-item logits over vocabulary, shape (B, T, |V|)
    """

    def __init__(self, n_items: int, emb_dim: int = EMB_DIM,
                 n_layers: int = N_LAYERS, n_heads: int = N_HEADS,
                 max_seq_len: int = MAX_SEQ_LEN, dropout: float = DROPOUT):
        super().__init__()
        self.n_items = n_items
        self.emb_dim = emb_dim

        # Item embedding (index 0 = padding)
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_seq_len + 1, emb_dim)
        self.emb_drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn_norm": nn.LayerNorm(emb_dim),
                "attn":      nn.MultiheadAttention(emb_dim, n_heads,
                                                    dropout=dropout,
                                                    batch_first=True),
                "ffn_norm":  nn.LayerNorm(emb_dim),
                "ffn":       PointWiseFeedForward(emb_dim, dropout),
            })
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(emb_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.pos_emb.weight)

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """seqs: (B, T) int64. Returns logit embeddings (B, T, D)."""
        B, T = seqs.shape
        positions = torch.arange(1, T + 1, device=seqs.device).unsqueeze(0).expand(B, T)
        x = self.emb_drop(self.item_emb(seqs) + self.pos_emb(positions))

        # Additive causal mask: -1e9 for future positions (safer than -inf for softmax)
        causal_mask = torch.triu(
            torch.full((T, T), -1e9, device=seqs.device), diagonal=1
        )

        for layer in self.layers:
            # Pre-norm attention
            normed = layer["attn_norm"](x)
            attn_out, _ = layer["attn"](
                normed, normed, normed,
                attn_mask=causal_mask,
            )
            x = x + attn_out
            # Pre-norm FFN
            x = x + layer["ffn"](layer["ffn_norm"](x))

        return self.out_norm(x)      # (B, T, D)


# ── Dataset helpers ───────────────────────────────────────────────────────────

def build_sequences(train: pd.DataFrame, v2i: dict, max_len: int = MAX_SEQ_LEN):
    """
    Build per-user chronological item sequences, truncated to max_len.
    Returns: list of (sequence: list[int], target: int) pairs.
    """
    seqs = []
    for uid, grp in train.sort_values("timestamp").groupby("user_id"):
        items = [v2i[v] for v in grp["business_id"] if v in v2i]
        if len(items) < 2:
            continue
        # Sliding window: every prefix → next item
        for end in range(1, len(items)):
            prefix = items[max(0, end - max_len): end]
            target = items[end]
            seqs.append((prefix, target))
    return seqs


def pad_sequence(seq: list, max_len: int) -> list:
    """Left-pad with zeros to max_len."""
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq


# ── Public API ────────────────────────────────────────────────────────────────

def build_sasrec_ranking(
    train: pd.DataFrame,
    n_layers:    int   = N_LAYERS,
    emb_dim:     int   = EMB_DIM,
    n_epochs:    int   = N_EPOCHS,
    batch_size:  int   = BATCH_SIZE,
    lr:          float = LR,
    max_seq_len: int   = MAX_SEQ_LEN,
    device:      Optional[torch.device] = None,
    verbose:     bool  = True,
) -> dict:
    """
    Train SASRec and return global venue scores.

    Global score: item_emb[v] · mean_user_representation
    (mean of final-position hidden states across all users' sequences)
    Compatible with evaluate_per_user().
    """
    if device is None:
        device = _get_device()

    if verbose:
        print(f"  [SASRec] device={device}  layers={n_layers}  "
              f"dim={emb_dim}  epochs={n_epochs}  max_seq={max_seq_len}")

    venues = train["business_id"].unique()
    v2i    = {v: i + 1 for i, v in enumerate(venues)}   # 0 reserved for padding
    i2v    = {i: v for v, i in v2i.items()}
    n_items = len(venues)

    if verbose:
        print(f"  [SASRec] {train['user_id'].nunique():,} users · "
              f"{n_items:,} venues · building sequences...")

    seqs = build_sequences(train, v2i, max_seq_len)
    if len(seqs) < 10:
        print("  [SASRec] too few sequences — returning empty ranking")
        return {}

    if verbose:
        print(f"  [SASRec] {len(seqs):,} (prefix, target) training pairs")

    model = SASRec(n_items, emb_dim, n_layers, 1, max_seq_len, DROPOUT).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Pre-build padded arrays
    X = np.array([pad_sequence(s[0], max_seq_len) for s in seqs], dtype=np.int64)
    Y = np.array([s[1] for s in seqs], dtype=np.int64)
    n_total = len(X)

    for epoch in range(n_epochs):
        model.train()
        idx       = np.random.permutation(n_total)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n_total, batch_size):
            batch = idx[start: start + batch_size]
            x_b = torch.from_numpy(X[batch]).to(device)    # (B, T)
            y_b = torch.from_numpy(Y[batch]).to(device)    # (B,)

            opt.zero_grad()
            hidden = model(x_b)                             # (B, T, D)
            # Use the last (rightmost non-pad) hidden state
            last_pos  = (x_b != 0).sum(dim=1) - 1          # index of last real item
            last_pos  = last_pos.clamp(min=0)
            h_last    = hidden[torch.arange(len(batch), device=device), last_pos]  # (B, D)

            # Positive scores
            pos_emb   = model.item_emb(y_b)                # (B, D)
            pos_score = (h_last * pos_emb).sum(dim=1)      # (B,)

            # Negative sampling (random items)
            neg_ids   = torch.randint(1, n_items + 1, y_b.shape, device=device)
            neg_emb   = model.item_emb(neg_ids)
            neg_score = (h_last * neg_emb).sum(dim=1)

            loss = F.binary_cross_entropy_with_logits(
                torch.cat([pos_score, neg_score]),
                torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches  += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  [SASRec] epoch {epoch+1:>3}/{n_epochs}  "
                  f"loss={epoch_loss/n_batches:.4f}")

    # Global venue score: item_emb · mean_user_final_hidden
    model.eval()
    with torch.no_grad():
        # Collect final hidden states for all users (one per user, last sequence)
        user_hiddens = []
        user_seqs_per_user: dict = {}
        for uid, grp in train.sort_values("timestamp").groupby("user_id"):
            items = [v2i[v] for v in grp["business_id"] if v in v2i]
            if len(items) >= 1:
                user_seqs_per_user[uid] = pad_sequence(items[-max_seq_len:], max_seq_len)

        if user_seqs_per_user:
            all_seqs = np.array(list(user_seqs_per_user.values()), dtype=np.int64)
            # Process in batches to avoid OOM
            hiddens = []
            for start in range(0, len(all_seqs), 512):
                chunk = torch.from_numpy(all_seqs[start: start + 512]).to(device)
                h = model(chunk)
                last = (chunk != 0).sum(dim=1) - 1
                last = last.clamp(min=0)
                h_l = h[torch.arange(len(chunk), device=device), last]
                hiddens.append(h_l.cpu())
            mean_h = torch.cat(hiddens, dim=0).mean(dim=0)   # (D,)
        else:
            mean_h = torch.zeros(emb_dim)

        # Score = item_emb[i] · mean_h  (cosine after normalising)
        item_embs = model.item_emb.weight[1:].cpu()           # skip padding idx 0
        mean_h_n  = F.normalize(mean_h.unsqueeze(0), dim=1)
        item_n    = F.normalize(item_embs, dim=1)
        scores    = (item_n @ mean_h_n.T).squeeze().numpy()

    result = {i2v[i + 1]: float(scores[i]) for i in range(n_items)}
    if verbose:
        s = list(result.values())
        print(f"  [SASRec] done — score range [{min(s):.4f}, {max(s):.4f}]")
    return result
