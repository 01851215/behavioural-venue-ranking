"""
Task B: Implement BiRank Algorithm

BiRank is a co-ranking algorithm for bipartite graphs that iteratively
updates scores for both user and venue nodes.

Algorithm:
  p_{t+1} = α · S_u · q_t + (1-α) · p_0
  q_{t+1} = β · S_v^T · p_{t+1} + (1-β) · q_0

Where:
  p = user scores, q = venue scores
  S_u, S_v = normalized transition matrices
  α, β = damping factors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import sparse
import time

# Configuration
DATA_DIR = Path(__file__).parent
EDGES_FILE = DATA_DIR / "coffee_bipartite_edges.csv"
VENUE_FEATURES_FILE = DATA_DIR / "coffee_venue_features.csv"

VENUE_SCORES_FILE = DATA_DIR / "coffee_birank_venue_scores.csv"
USER_SCORES_FILE = DATA_DIR / "coffee_birank_user_scores.csv"

# BiRank parameters
DEFAULT_ALPHA = 0.85  # User damping factor
DEFAULT_BETA = 0.85   # Venue damping factor
CONVERGENCE_THRESHOLD = 1e-8
MAX_ITERATIONS = 200

# Set plotting style
sns.set_style("whitegrid")

def load_edges():
    """Load bipartite graph edges."""
    print("Loading bipartite graph edges...")
    edges_df = pd.read_csv(EDGES_FILE)
    print(f"Loaded {len(edges_df):,} edges")
    return edges_df

def build_adjacency_matrix(edges_df):
    """
    Build weighted adjacency matrix W (users × businesses).
    
    Returns:
        W: sparse matrix (users × businesses)
        user_to_idx: dict mapping user_id to matrix row index
        business_to_idx: dict mapping business_id to matrix column index
        idx_to_user: reverse mapping
        idx_to_business: reverse mapping
    """
    print("\nBuilding adjacency matrix...")
    
    # Create index mappings
    unique_users = edges_df['user_id'].unique()
    unique_businesses = edges_df['business_id'].unique()
    
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    business_to_idx = {biz: idx for idx, biz in enumerate(unique_businesses)}
    
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    idx_to_business = {idx: biz for biz, idx in business_to_idx.items()}
    
    n_users = len(unique_users)
    n_businesses = len(unique_businesses)
    
    print(f"  Matrix dimensions: {n_users:,} users × {n_businesses:,} businesses")
    
    # Build sparse matrix
    row_indices = [user_to_idx[user] for user in edges_df['user_id']]
    col_indices = [business_to_idx[biz] for biz in edges_df['business_id']]
    weights = edges_df['weight'].values
    
    W = sparse.csr_matrix(
        (weights, (row_indices, col_indices)),
        shape=(n_users, n_businesses),
        dtype=np.float64
    )
    
    print(f"  Non-zero entries: {W.nnz:,}")
    print(f"  Sparsity: {(1 - W.nnz / (n_users * n_businesses)) * 100:.4f}%")
    
    return W, user_to_idx, business_to_idx, idx_to_user, idx_to_business

def normalize_matrix_rowwise(W):
    """
    Row-normalize matrix W to get transition probabilities.
    S[i,j] = W[i,j] / sum_j(W[i,j])
    
    Handles rows with zero sum (isolated nodes).
    """
    row_sums = np.array(W.sum(axis=1)).flatten()
    
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    
    # Create diagonal matrix of 1/row_sums
    row_sums_inv = sparse.diags(1.0 / row_sums)
    
    # Multiply to get normalized matrix
    S = row_sums_inv @ W
    
    return S

def normalize_matrix_colwise(W):
    """
    Column-normalize matrix W to get transition probabilities.
    S[i,j] = W[i,j] / sum_i(W[i,j])
    
    Handles columns with zero sum (isolated nodes).
    """
    col_sums = np.array(W.sum(axis=0)).flatten()
    
    # Avoid division by zero
    col_sums[col_sums == 0] = 1.0
    
    # Create diagonal matrix of 1/col_sums
    col_sums_inv = sparse.diags(1.0 / col_sums)
    
    # Multiply to get normalized matrix
    S = W @ col_sums_inv
    
    return S

def birank_iterate(W, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, 
                   p0=None, q0=None, max_iters=MAX_ITERATIONS,
                   convergence_threshold=CONVERGENCE_THRESHOLD,
                   verbose=True):
    """
    Run BiRank algorithm with iterative updates.
    
    Args:
        W: weighted adjacency matrix (users × businesses)
        alpha: user damping factor
        beta: venue damping factor
        p0: prior user scores (default: uniform)
        q0: prior venue scores (default: uniform)
        max_iters: maximum iterations
        convergence_threshold: L1 convergence threshold
        verbose: print progress
        
    Returns:
        p: final user scores
        q: final venue scores
        iterations: number of iterations to convergence
        convergence_log: list of (iter, p_change, q_change) tuples
    """
    n_users, n_businesses = W.shape
    
    # Initialize priors
    if p0 is None:
        p0 = np.ones(n_users) / n_users
    if q0 is None:
        q0 = np.ones(n_businesses) / n_businesses
    
    # Normalize matrices
    if verbose:
        print("\n  Normalizing transition matrices...")
    S_u = normalize_matrix_rowwise(W)  # User → Business transitions
    S_v = normalize_matrix_colwise(W)   # Business → User transitions
    
    # Initialize scores
    p = p0.copy()
    q = q0.copy()
    
    convergence_log = []
    
    if verbose:
        print(f"  Starting BiRank iteration (α={alpha}, β={beta})...")
        print(f"  Convergence threshold: {convergence_threshold}")
    
    start_time = time.time()
    
    for iteration in range(1, max_iters + 1):
        # Store previous scores
        p_prev = p.copy()
        q_prev = q.copy()
        
        # Update user scores: p_{t+1} = α * S_u * q_t + (1-α) * p_0
        p = alpha * (S_u @ q) + (1 - alpha) * p0
        
        # Update venue scores: q_{t+1} = β * S_v^T * p_{t+1} + (1-β) * q_0
        q = beta * (S_v.T @ p) + (1 - beta) * q0
        
        # Normalize to prevent numerical drift
        p = p / p.sum()
        q = q / q.sum()
        
        # Check convergence
        p_change = np.abs(p - p_prev).sum()
        q_change = np.abs(q - q_prev).sum()
        
        convergence_log.append((iteration, p_change, q_change))
        
        if verbose and iteration % 10 == 0:
            print(f"    Iter {iteration:3d}: p_change={p_change:.2e}, q_change={q_change:.2e}")
        
        # Check if converged
        if p_change < convergence_threshold and q_change < convergence_threshold:
            elapsed = time.time() - start_time
            if verbose:
                print(f"\n  ✓ Converged at iteration {iteration}")
                print(f"    Final p_change: {p_change:.2e}")
                print(f"    Final q_change: {q_change:.2e}")
                print(f"    Time: {elapsed:.2f}s")
            break
    else:
        elapsed = time.time() - start_time
        if verbose:
            print(f"\n  ! Reached max iterations ({max_iters})")
            print(f"    Final p_change: {p_change:.2e}")
            print(f"    Final q_change: {q_change:.2e}")
            print(f"    Time: {elapsed:.2f}s")
    
    return p, q, iteration, convergence_log

def sensitivity_analysis(W, alphas=[0.7, 0.85, 0.9], betas=[0.7, 0.85, 0.9]):
    """
    Test BiRank sensitivity to different damping factor combinations.
    """
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS")
    print("="*60)
    
    results = []
    
    for alpha in alphas:
        for beta in betas:
            print(f"\nTesting α={alpha}, β={beta}...")
            p, q, iterations, log = birank_iterate(W, alpha=alpha, beta=beta, verbose=False)
            
            print(f"  Converged in {iterations} iterations")
            print(f"  Top venue score: {q.max():.6e}")
            print(f"  Mean venue score: {q.mean():.6e}")
            
            results.append({
                'alpha': alpha,
                'beta': beta,
                'iterations': iterations,
                'max_venue_score': q.max(),
                'mean_venue_score': q.mean(),
                'scores': q.copy()
            })
    
    # Check top-10 stability
    print("\nTop-10 venue rank stability:")
    reference_top10 = set(np.argsort(results[4]['scores'])[-10:])  # α=0.85, β=0.85
    
    for i, result in enumerate(results):
        current_top10 = set(np.argsort(result['scores'])[-10:])
        overlap = len(reference_top10 & current_top10)
        print(f"  α={result['alpha']}, β={result['beta']}: {overlap}/10 overlap with reference")
    
    print("="*60)
    
    return results

def plot_convergence(convergence_log):
    """Plot convergence curves."""
    print("\nPlotting convergence curves...")
    
    iterations = [log[0] for log in convergence_log]
    p_changes = [log[1] for log in convergence_log]
    q_changes = [log[2] for log in convergence_log]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(iterations, p_changes, 'o-', label='User score change', linewidth=2, markersize=4)
    ax.plot(iterations, q_changes, 's-', label='Venue score change', linewidth=2, markersize=4)
    ax.axhline(y=CONVERGENCE_THRESHOLD, color='red', linestyle='--', 
               label=f'Threshold ({CONVERGENCE_THRESHOLD:.0e})', linewidth=2)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('L1 Change', fontsize=12)
    ax.set_title('BiRank Convergence', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = DATA_DIR / "birank_convergence.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Convergence plot saved to {plot_file}")
    plt.close()

def main():
    """
    Main execution function for Task B.
    """
    print("\n" + "="*60)
    print("TASK B: IMPLEMENT BIRANK ALGORITHM")
    print("="*60)
    
    # Load edges
    edges_df = load_edges()
    
    # Build adjacency matrix
    W, user_to_idx, business_to_idx, idx_to_user, idx_to_business = build_adjacency_matrix(edges_df)
    
    # Run BiRank with default parameters
    print("\n" + "="*60)
    print("RUNNING BIRANK (α=0.85, β=0.85)")
    print("="*60)
    
    p, q, iterations, convergence_log = birank_iterate(
        W, 
        alpha=DEFAULT_ALPHA, 
        beta=DEFAULT_BETA,
        verbose=True
    )
    
    # Create venue scores dataframe
    print("\nCreating venue scores dataframe...")
    venue_scores = []
    for idx, score in enumerate(q):
        business_id = idx_to_business[idx]
        venue_scores.append({
            'business_id': business_id,
            'birank_score': score
        })
    
    venue_scores_df = pd.DataFrame(venue_scores)
    venue_scores_df = venue_scores_df.sort_values('birank_score', ascending=False)
    venue_scores_df['rank'] = range(1, len(venue_scores_df) + 1)
    
    print(f"\nTop 10 venues by BiRank score:")
    print(venue_scores_df.head(10).to_string(index=False))
    
    # Save venue scores
    print(f"\nSaving venue scores to {VENUE_SCORES_FILE}...")
    venue_scores_df.to_csv(VENUE_SCORES_FILE, index=False)
    print(f"Saved {len(venue_scores_df)} venue scores")
    
    # Create user scores dataframe
    print("\nCreating user scores dataframe...")
    user_scores = []
    for idx, score in enumerate(p):
        user_id = idx_to_user[idx]
        user_scores.append({
            'user_id': user_id,
            'birank_score': score
        })
    
    user_scores_df = pd.DataFrame(user_scores)
    user_scores_df = user_scores_df.sort_values('birank_score', ascending=False)
    
    # Save user scores
    print(f"\nSaving user scores to {USER_SCORES_FILE}...")
    user_scores_df.to_csv(USER_SCORES_FILE, index=False)
    print(f"Saved {len(user_scores_df)} user scores")
    
    # Plot convergence
    plot_convergence(convergence_log)
    
    # Sensitivity analysis
    sensitivity_results = sensitivity_analysis(W)
    
    # Summary statistics
    print("\n" + "="*60)
    print("BIRANK SCORE SUMMARY")
    print("="*60)
    print(f"\nVenue scores:")
    print(f"  Mean: {q.mean():.6e}")
    print(f"  Median: {np.median(q):.6e}")
    print(f"  Max: {q.max():.6e}")
    print(f"  Min: {q.min():.6e}")
    print(f"  Std: {q.std():.6e}")
    
    print(f"\nUser scores:")
    print(f"  Mean: {p.mean():.6e}")
    print(f"  Median: {np.median(p):.6e}")
    print(f"  Max: {p.max():.6e}")
    print(f"  Min: {p.min():.6e}")
    print(f"  Std: {p.std():.6e}")
    print("="*60)
    
    print(f"\n✓ Task B complete.")
    print(f"  Venue scores: {VENUE_SCORES_FILE}")
    print(f"  User scores: {USER_SCORES_FILE}")

if __name__ == "__main__":
    main()
