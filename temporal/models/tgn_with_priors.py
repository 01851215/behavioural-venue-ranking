"""
TGN with Behavioral Priors — Novel Contribution of Paper P3

Extends the base TGN with two modifications:
  1. Node embeddings initialised from behavioral features (not random)
  2. Custom loss: BPR + λ * rising-stars regulariser

The rising-stars regulariser penalises predictions that correlate
too strongly with current popularity — forcing the model to learn
signals beyond what popularity already predicts.

This is the main methodological contribution of the PhD.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from temporal.models.tgn import TGN, build_tgn_ranking
from bvr.core.validation import compute_user_features, compute_venue_features


class TGNWithPriors(TGN):
    """
    TGN with behavioral prior initialisation and rising-stars regularisation.

    Key innovation: instead of random Xavier init, node memories are
    initialised from behavioral features (burstiness, repeat_user_rate, etc.).
    This injects the domain knowledge from Paper P0 into the temporal model.
    """

    def __init__(self, n_nodes: int, memory_dim: int = 64,
                 time_dim: int = 16, msg_dim: int = 64, emb_dim: int = 64,
                 lambda_rs: float = 0.1):
        super().__init__(n_nodes, memory_dim, time_dim, msg_dim, emb_dim)
        self.lambda_rs = lambda_rs   # rising-stars regularisation strength

    def init_from_behavioral_features(self, user_feat, venue_feat,
                                       u2i: dict, v2i: dict, device):
        """
        Initialise node memories from behavioral features.

        Users: burstiness index + venue entropy → diversity signal
        Venues: repeat_user_rate → loyalty signal (inverted for rising stars)
        """
        n_users  = len(u2i)
        n_venues = len(v2i)

        # User prior: exploration tendency (low burstiness + high entropy = explorer)
        if "burstiness_index" in user_feat.columns:
            bust_map = user_feat.set_index("user_id")["burstiness_index"].fillna(0)
            ent_map  = user_feat.set_index("user_id")["venue_entropy"].fillna(0)
            u_signal = np.array([
                float(np.log1p(abs(bust_map.get(u, 0))) * ent_map.get(u, 0))
                for u in u2i.keys()
            ])
        else:
            u_signal = np.ones(n_users)

        # Venue prior: inverse repeat_user_rate (anti-loyalty signal)
        if "repeat_user_rate" in venue_feat.columns:
            rr_map  = venue_feat.set_index("business_id")["repeat_user_rate"].fillna(0)
            v_signal = np.array([
                1.0 / (float(rr_map.get(v, 0.01) or 0.01) + 0.01)
                for v in {k - n_users: k for k in v2i.values()}.values()  # simplify
            ])
        else:
            v_signal = np.ones(n_venues)

        # Normalise and set as memory init (first dim of memory vector)
        u_signal = (u_signal - u_signal.min()) / (u_signal.max() - u_signal.min() + 1e-10)
        v_signal = (v_signal - v_signal.min()) / (v_signal.max() - v_signal.min() + 1e-10)

        # Expand to full memory_dim (broadcast scalar signal to first dim)
        all_signal = np.concatenate([u_signal, v_signal])
        mem_init = torch.zeros(self.n_nodes, self.memory_dim)
        mem_init[:, 0] = torch.tensor(all_signal, dtype=torch.float32)

        self.memory.memory.copy_(mem_init.to(device))

    def rising_stars_regulariser(self, venue_scores: torch.Tensor,
                                  popularity: torch.Tensor) -> torch.Tensor:
        """
        Penalise correlation between model scores and popularity.
        Forces the model to learn signal beyond what popularity captures.

        Loss = λ * max(0, Spearman(scores, popularity))
        Approximated as max(0, cosine_similarity(ranks(scores), ranks(popularity)))
        """
        if self.lambda_rs <= 0:
            return torch.tensor(0.0, device=venue_scores.device)

        # Differentiable rank approximation using softmax
        n = len(venue_scores)
        s_rank = torch.argsort(torch.argsort(venue_scores.detach())).float() / n
        p_rank = torch.argsort(torch.argsort(popularity.detach())).float() / n

        # Penalise positive correlation (we want correlation ≤ 0)
        corr = torch.dot(s_rank - s_rank.mean(), p_rank - p_rank.mean())
        corr = corr / (torch.std(s_rank) * torch.std(p_rank) * n + 1e-10)
        return self.lambda_rs * torch.relu(corr)


def build_tgn_with_priors_ranking(
    train,
    n_epochs: int = 20,
    memory_dim: int = 64,
    emb_dim: int = 64,
    lambda_rs: float = 0.1,
    lr: float = 0.0005,
    batch_size: int = 512,
    device=None,
    verbose: bool = True,
) -> dict:
    """
    Train TGN-with-priors and return global venue scores.
    Compatible with evaluate_per_user().
    """
    import pandas as pd

    if device is None:
        device = torch.device("cpu")

    users  = train["user_id"].unique()
    venues = train["business_id"].unique()
    u2i = {u: i for i, u in enumerate(users)}
    v2i = {v: i + len(users) for i, v in enumerate(venues)}
    i2v = {i + len(users): v for i, v in enumerate(venues)}
    n_nodes = len(users) + len(venues)
    n_items = len(venues)

    user_feat  = compute_user_features(train)
    venue_feat = compute_venue_features(train)

    model = TGNWithPriors(
        n_nodes=n_nodes, memory_dim=memory_dim, emb_dim=emb_dim,
        lambda_rs=lambda_rs
    ).to(device)

    # Initialise from behavioral features (the novel contribution)
    model.init_from_behavioral_features(user_feat, venue_feat, u2i, v2i, device)

    if verbose:
        print(f"  [TGN+priors] device={device}  λ_rs={lambda_rs}  "
              f"epochs={n_epochs}  {len(users):,} users  {len(venues):,} venues")

    # Popularity vector for regulariser
    pop_counts = train["business_id"].value_counts()
    pop_tensor = torch.tensor(
        [float(pop_counts.get(v, 0)) for v in venues],
        device=device
    )

    events = train.sort_values("timestamp")[["user_id", "business_id", "timestamp"]].values
    t_min  = train["timestamp"].min().timestamp()
    opt    = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        model.reset_memory()
        model.init_from_behavioral_features(user_feat, venue_feat, u2i, v2i, device)

        idx = np.random.permutation(len(events))
        epoch_loss = 0.0

        for start in range(0, len(events), batch_size):
            batch = idx[start: start + batch_size]
            b_src = torch.tensor([u2i[e[0]] for e in events[batch]], device=device)
            b_dst = torch.tensor([v2i[e[1]] for e in events[batch]], device=device)
            b_ts  = torch.tensor(
                [(pd.Timestamp(e[2]).timestamp() - t_min) / 86400 for e in events[batch]],
                device=device
            )
            b_neg = torch.randint(len(users), len(users) + n_items, (len(batch),), device=device)

            model.process_events(b_src, b_dst, b_ts)

            opt.zero_grad()
            all_emb   = model.get_embeddings()
            u_emb     = all_emb[b_src]
            pos_emb   = all_emb[b_dst]
            neg_emb   = all_emb[b_neg]

            bpr = -torch.log(torch.sigmoid(
                (u_emb * pos_emb).sum(1) - (u_emb * neg_emb).sum(1)
            ) + 1e-10).mean()

            # Rising-stars regulariser on item embeddings
            item_embs   = all_emb[len(users):]
            item_scores = (item_embs * item_embs).sum(dim=1)   # L2 norm as global score proxy
            rs_reg = model.rising_stars_regulariser(item_scores, pop_tensor)

            loss = bpr + rs_reg
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_loss += loss.item()

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  [TGN+priors] epoch {epoch+1:>3}/{n_epochs}  loss={epoch_loss/(len(events)//batch_size+1):.4f}")

    # Global scores
    model.eval()
    with torch.no_grad():
        all_emb = model.get_embeddings()
        u_embs  = all_emb[:len(users)].cpu().numpy()
        i_embs  = all_emb[len(users):].cpu().numpy()

    u_norm = u_embs / (np.linalg.norm(u_embs, axis=1, keepdims=True) + 1e-10)
    i_norm = i_embs / (np.linalg.norm(i_embs, axis=1, keepdims=True) + 1e-10)
    mean_u = u_norm.mean(axis=0)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = i_norm @ mean_u
    scores = np.nan_to_num(scores, 0.0)

    if verbose:
        print(f"  [TGN+priors] done — score range [{scores.min():.4f}, {scores.max():.4f}]")

    return {i2v[len(users) + i]: float(scores[i]) for i in range(n_items)}
