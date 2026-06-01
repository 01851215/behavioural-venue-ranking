"""
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
He et al., SIGIR 2020  —  https://arxiv.org/abs/2002.02126

Key idea: remove feature transformation matrices and non-linear activations from NGCF.
Keep only neighbourhood aggregation. Final embedding = mean of all layer embeddings
(including E^0). Trained with BPR loss on implicit feedback.

Usage:
    ranking = build_lightgcn_ranking(train_df, n_layers=3, emb_dim=64, n_epochs=50)
    # Returns {venue_id: global_score} compatible with evaluate_per_user()
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from typing import Optional


# ── Defaults (tuned for RecSys literature standard) ──────────────────────────
N_LAYERS   = 3      # He et al. find L=3 optimal across most datasets
EMB_DIM    = 64     # matches MF baseline for fair comparison
N_EPOCHS   = 50
BATCH_SIZE = 2048
LR         = 0.001
REG        = 1e-4   # L2 regularisation coefficient on initial embeddings


def _get_device() -> torch.device:
    # MPS does not support sparse tensor ops (torch.sparse.mm on MPS raises
    # NotImplementedError). The adjacency propagation requires sparse matmul,
    # so we keep everything on CPU. For Apple Silicon the CPU BLAS is fast enough.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_norm_adj(user_idx: np.ndarray, item_idx: np.ndarray,
                    n_users: int, n_items: int,
                    device: torch.device) -> torch.Tensor:
    """
    Build D^(-1/2) A D^(-1/2) for the augmented bipartite adjacency:

        A = [ 0    R  ]
            [ R^T  0  ]

    where R[u, i] = 1 if user u interacted with item i.

    Returns a (n_users+n_items) × (n_users+n_items) sparse torch tensor.
    """
    n = n_users + n_items
    # Both directions: user→item and item→user
    row = np.concatenate([user_idx, item_idx + n_users])
    col = np.concatenate([item_idx + n_users, user_idx])

    A = sp.csr_matrix(
        (np.ones(len(row), dtype=np.float32), (row, col)),
        shape=(n, n)
    )

    deg = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0).astype(np.float32)
    D = sp.diags(d_inv_sqrt)
    A_norm = (D @ A @ D).tocoo()

    indices = torch.from_numpy(
        np.vstack([A_norm.row, A_norm.col]).astype(np.int64)
    )
    values = torch.from_numpy(A_norm.data)
    return torch.sparse_coo_tensor(indices, values, (n, n)).to(device)


class LightGCN(nn.Module):
    """
    LightGCN model.

    Forward pass: K rounds of normalised graph propagation, then mean-pool
    embeddings from all K+1 layers (E^0 through E^K).
    """

    def __init__(self, n_users: int, n_items: int,
                 emb_dim: int = EMB_DIM, n_layers: int = N_LAYERS):
        super().__init__()
        self.n_users  = n_users
        self.n_items  = n_items
        self.n_layers = n_layers

        self.embedding = nn.Embedding(n_users + n_items, emb_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, norm_adj: torch.Tensor):
        e = self.embedding.weight          # (n_users+n_items, emb_dim)
        all_embs = [e]
        for _ in range(self.n_layers):
            e = torch.sparse.mm(norm_adj, e)
            all_embs.append(e)
        final = torch.stack(all_embs, dim=1).mean(dim=1)
        return final[:self.n_users], final[self.n_users:]

    def bpr_loss(self, u_idx, pos_idx, neg_idx,
                 u_emb: torch.Tensor, i_emb: torch.Tensor) -> torch.Tensor:
        """BPR loss + L2 regularisation on initial (not propagated) embeddings."""
        u   = u_emb[u_idx]
        pos = i_emb[pos_idx]
        neg = i_emb[neg_idx]
        loss = -torch.log(
            torch.sigmoid((u * pos).sum(1) - (u * neg).sum(1)) + 1e-10
        ).mean()
        # Regularise initial embeddings only (standard in LightGCN paper)
        e0 = self.embedding.weight
        reg = (e0[u_idx] ** 2 +
               e0[self.n_users + pos_idx] ** 2 +
               e0[self.n_users + neg_idx] ** 2).mean()
        return loss + REG * reg


# ── Public API ────────────────────────────────────────────────────────────────

def build_lightgcn_ranking(
    train: pd.DataFrame,
    n_layers:   int   = N_LAYERS,
    emb_dim:    int   = EMB_DIM,
    n_epochs:   int   = N_EPOCHS,
    batch_size: int   = BATCH_SIZE,
    lr:         float = LR,
    device:     Optional[torch.device] = None,
    verbose:    bool  = True,
) -> dict:
    """
    Train LightGCN on implicit feedback (visit counts → binary) and return
    a global venue score dict compatible with evaluate_per_user().

    Global score: item_emb · mean_user_emb
    (same approach as build_mf_ranking in run_london_pipeline.py)

    Args:
        train: DataFrame with columns [user_id, business_id]
    Returns:
        {business_id: float_score}
    """
    if device is None:
        device = _get_device()

    if verbose:
        print(f"  [LightGCN] device={device}  layers={n_layers}  "
              f"dim={emb_dim}  epochs={n_epochs}")

    # Index users and items
    users  = train["user_id"].unique()
    venues = train["business_id"].unique()
    u2i = {u: i for i, u in enumerate(users)}
    v2i = {v: i for i, v in enumerate(venues)}
    i2v = {i: v for v, i in v2i.items()}
    n_users, n_items = len(users), len(venues)

    # Deduplicate interactions (count-weighted → binary for graph structure)
    pairs = (
        train[["user_id", "business_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    u_arr = np.array([u2i[u] for u in pairs["user_id"]])
    v_arr = np.array([v2i[v] for v in pairs["business_id"]])
    n_pairs = len(u_arr)

    if verbose:
        print(f"  [LightGCN] {n_users:,} users · {n_items:,} venues · "
              f"{n_pairs:,} unique interactions")

    # Build normalised adjacency once (stays on device for full training)
    norm_adj = _build_norm_adj(u_arr, v_arr, n_users, n_items, device)

    model = LightGCN(n_users, n_items, emb_dim, n_layers).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    # Pre-convert to tensors for fast sampling
    u_tensor = torch.from_numpy(u_arr).long()
    v_tensor = torch.from_numpy(v_arr).long()

    # Sample size per epoch: all pairs if small, else cap at 8K for speed
    sample_n = min(n_pairs, 8192)

    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad()

        # KEY OPTIMISATION: propagate ONCE per epoch, not per minibatch.
        # One forward pass → one BPR loss on a large random sample → one backward.
        # Correct gradient flow: the full computation graph is alive for the
        # single backward call, then freed. No retain_graph needed.
        u_emb, i_emb = model(norm_adj)

        idx   = np.random.choice(n_pairs, sample_n, replace=False)
        b_u   = u_tensor[idx]
        b_pos = v_tensor[idx]
        b_neg = torch.randint(0, n_items, (sample_n,))

        loss = model.bpr_loss(b_u, b_pos, b_neg, u_emb, i_emb)
        loss.backward()
        opt.step()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  [LightGCN] epoch {epoch+1:>3}/{n_epochs}  "
                  f"loss={loss.item():.4f}")

    # Final embeddings
    model.eval()
    with torch.no_grad():
        u_emb, i_emb = model(norm_adj)
        u_np = u_emb.cpu().float().numpy()
        i_np = i_emb.cpu().float().numpy()

    # Global venue score: mean cosine similarity between each item and all users.
    # Raw dot product fails because user embeddings point in diverse directions
    # and their mean collapses to ~0. L2-normalising before aggregating preserves
    # directional signal and gives a well-scaled score in [-1, +1].
    u_norm       = u_np / (np.linalg.norm(u_np, axis=1, keepdims=True) + 1e-10)
    i_norm       = i_np / (np.linalg.norm(i_np, axis=1, keepdims=True) + 1e-10)
    mean_u_norm  = u_norm.mean(axis=0)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        venue_scores = i_norm @ mean_u_norm                # cosine similarity to mean direction
    venue_scores = np.nan_to_num(venue_scores, nan=0.0, posinf=0.0, neginf=0.0)

    if verbose:
        print(f"  [LightGCN] done — score range "
              f"[{venue_scores.min():.4f}, {venue_scores.max():.4f}]")

    return {i2v[i]: float(venue_scores[i]) for i in range(n_items)}
