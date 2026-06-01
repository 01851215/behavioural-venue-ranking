"""
Temporal Graph Network (TGN) — Base Implementation

Based on: Rossi et al. (2020) "Temporal Graph Networks for Deep Learning
on Dynamic Graphs." ICML Workshop on Graph Representation Learning.
https://arxiv.org/abs/2006.10637

Architecture:
  1. Memory module: per-node state vector summarising past interactions
  2. Message function: computes messages from interactions
  3. Memory updater: GRU-based update of node memory
  4. Embedding module: computes final node embeddings

Usage:
    from temporal.models.tgn import TGN
    model = TGN(n_nodes=10000, node_dim=64)
    model.fit(events)  # events = list of (src, dst, timestamp, features)
    embeddings = model.get_embeddings()
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from pathlib import Path


class MemoryModule(nn.Module):
    """Per-node memory — summarises a node's interaction history."""

    def __init__(self, n_nodes: int, memory_dim: int):
        super().__init__()
        self.n_nodes    = n_nodes
        self.memory_dim = memory_dim
        self.register_buffer("memory",      torch.zeros(n_nodes, memory_dim))
        self.register_buffer("last_update", torch.zeros(n_nodes))

    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self.memory[node_ids]

    def set_memory(self, node_ids: torch.Tensor, new_memory: torch.Tensor):
        self.memory[node_ids] = new_memory

    def reset(self):
        self.memory.zero_()
        self.last_update.zero_()


class MessageFunction(nn.Module):
    """Computes messages from interaction events."""

    def __init__(self, memory_dim: int, time_dim: int, msg_dim: int):
        super().__init__()
        # Message = concat(src_memory, dst_memory, time_encoding)
        self.linear = nn.Linear(2 * memory_dim + time_dim, msg_dim)

    def forward(self, src_mem: torch.Tensor, dst_mem: torch.Tensor,
                t_enc: torch.Tensor) -> torch.Tensor:
        msg = torch.cat([src_mem, dst_mem, t_enc], dim=-1)
        return torch.relu(self.linear(msg))


class TimeEncoding(nn.Module):
    """Time2Vec positional encoding for timestamps."""

    def __init__(self, time_dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, time_dim))
        self.b = nn.Parameter(torch.randn(time_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) float timestamps
        t = t.unsqueeze(-1)                              # (B, 1)
        raw = t * self.w + self.b                        # (B, time_dim)
        enc = torch.cat([t, torch.sin(raw[:, 1:])], dim=-1)
        return enc


class TGN(nn.Module):
    """
    Temporal Graph Network.

    Processes a stream of timestamped events and maintains a memory
    per node that is updated after each interaction.
    """

    def __init__(self, n_nodes: int, memory_dim: int = 64,
                 time_dim: int = 16, msg_dim: int = 64,
                 emb_dim: int = 64):
        super().__init__()
        self.n_nodes    = n_nodes
        self.memory_dim = memory_dim
        self.emb_dim    = emb_dim

        self.memory    = MemoryModule(n_nodes, memory_dim)
        self.time_enc  = TimeEncoding(time_dim)
        self.msg_fn    = MessageFunction(memory_dim, time_dim, msg_dim)
        self.updater   = nn.GRUCell(msg_dim, memory_dim)
        self.embedder  = nn.Sequential(
            nn.Linear(memory_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def process_events(self, src_ids: torch.Tensor, dst_ids: torch.Tensor,
                       timestamps: torch.Tensor):
        """Update memory for a batch of events."""
        src_mem = self.memory.get_memory(src_ids)
        dst_mem = self.memory.get_memory(dst_ids)
        t_enc   = self.time_enc(timestamps.float())

        # Messages for both src and dst
        msg = self.msg_fn(src_mem, dst_mem, t_enc)

        # Update memories with GRU
        new_src_mem = self.updater(msg, src_mem)
        new_dst_mem = self.updater(msg, dst_mem)

        self.memory.set_memory(src_ids, new_src_mem.detach())
        self.memory.set_memory(dst_ids, new_dst_mem.detach())

    def get_embeddings(self, node_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return final embeddings from current memory state."""
        if node_ids is None:
            mem = self.memory.memory
        else:
            mem = self.memory.get_memory(node_ids)
        return self.embedder(mem)

    def reset_memory(self):
        self.memory.reset()


def build_tgn_ranking(
    train: "pd.DataFrame",
    n_epochs: int = 20,
    memory_dim: int = 64,
    emb_dim: int = 64,
    lr: float = 0.0005,
    batch_size: int = 512,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> dict:
    """
    Train TGN on temporal interactions and return global venue scores.

    Compatible with evaluate_per_user() — returns {venue_id: score}.
    """
    import pandas as pd
    from optional import Optional

    if device is None:
        device = torch.device("cpu")   # TGN uses dense ops, CPU is fine

    users  = train["user_id"].unique()
    venues = train["business_id"].unique()
    all_nodes = np.concatenate([users, venues])
    u2i = {u: i for i, u in enumerate(users)}
    v2i = {v: i + len(users) for i, v in enumerate(venues)}   # offset venues
    i2v = {i + len(users): v for i, v in enumerate(venues)}
    n_nodes = len(users) + len(venues)
    n_items = len(venues)

    # Sort events chronologically
    events = train.sort_values("timestamp")[["user_id", "business_id", "timestamp"]].values

    model = TGN(n_nodes=n_nodes, memory_dim=memory_dim, emb_dim=emb_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    if verbose:
        print(f"  [TGN] device={device}  memory={memory_dim}  dim={emb_dim}  "
              f"epochs={n_epochs}  {len(users):,} users  {len(venues):,} venues")

    # Encode timestamps as seconds since epoch
    t_min = train["timestamp"].min().timestamp()

    for epoch in range(n_epochs):
        model.train()
        model.reset_memory()

        idx = np.random.permutation(len(events))
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, len(events), batch_size):
            batch = idx[start: start + batch_size]
            b_src = torch.tensor([u2i[e[0]] for e in events[batch]], device=device)
            b_dst = torch.tensor([v2i[e[1]] for e in events[batch]], device=device)
            b_ts  = torch.tensor(
                [(pd.Timestamp(e[2]).timestamp() - t_min) / 86400 for e in events[batch]],
                device=device
            )
            b_neg = torch.randint(len(users), len(users) + n_items, (len(batch),), device=device)

            # Update memory
            model.process_events(b_src, b_dst, b_ts)

            # BPR loss on current embeddings
            opt.zero_grad()
            all_emb  = model.get_embeddings()
            u_emb    = all_emb[b_src]
            pos_emb  = all_emb[b_dst]
            neg_emb  = all_emb[b_neg]

            pos_score = (u_emb * pos_emb).sum(dim=1)
            neg_score = (u_emb * neg_emb).sum(dim=1)
            loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches  += 1

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  [TGN] epoch {epoch+1:>3}/{n_epochs}  loss={epoch_loss/n_batches:.4f}")

    # Global venue score: item_emb · mean_user_emb (cosine, normalised)
    model.eval()
    with torch.no_grad():
        all_emb   = model.get_embeddings()
        u_embs    = all_emb[:len(users)].cpu().numpy()
        i_embs    = all_emb[len(users):].cpu().numpy()

    u_norm  = u_embs / (np.linalg.norm(u_embs, axis=1, keepdims=True) + 1e-10)
    i_norm  = i_embs / (np.linalg.norm(i_embs, axis=1, keepdims=True) + 1e-10)
    mean_u  = u_norm.mean(axis=0)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = i_norm @ mean_u
    scores = np.nan_to_num(scores, 0.0)

    if verbose:
        print(f"  [TGN] done — score range [{scores.min():.4f}, {scores.max():.4f}]")

    return {i2v[len(users) + i]: float(scores[i]) for i in range(n_items)}
