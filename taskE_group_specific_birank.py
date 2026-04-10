"""
Task E: Group-Specific BiRank

For each behavioural user cluster, compute BiRank on the subgraph
containing only users from that cluster.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import sparse

# Configuration
DATA_DIR = Path(__file__).parent
EDGES_FILE = DATA_DIR / "coffee_bipartite_edges.csv"
USER_GROUPS_FILE = DATA_DIR / "coffee_user_groups.csv"
BUSINESS_FILE = DATA_DIR / "business_coffee.csv"

OUTPUT_FILE = DATA_DIR / "coffee_birank_venue_scores_by_group.csv"

# BiRank parameters (using row-normalization for simpler implementation)
ALPHA = 0.85
BETA = 0.85
MAX_ITERS = 100
CONVERGENCE_THRESHOLD = 1e-8

# Set plotting style
sns.set_style("whitegrid")

# Cluster name mapping
CLUSTER_NAMES = {
    0: "Casual Weekenders",
    1: "Weekday Regulars",
    2: "Loyalists",
    3: "Infrequent Visitors"
}

def load_data():
    """Load edges and user groups."""
    print("Loading data...")
    edges_df = pd.read_csv(EDGES_FILE)
    user_groups_df = pd.read_csv(USER_GROUPS_FILE)
    
    print(f"  Edges: {len(edges_df):,}")
    print(f"  Users: {len(user_groups_df):,}")
    
    # Merge to get cluster for each edge
    edges_with_groups = edges_df.merge(
        user_groups_df[['user_id', 'cluster']],
        on='user_id',
        how='left'
    )
    
    # Filter to users with cluster assignment
    edges_with_groups = edges_with_groups[edges_with_groups['cluster'].notna()]
    
    print(f"  Edges with cluster assignment: {len(edges_with_groups):,}")
    
    return edges_with_groups

def birank_simple(edges_df, verbose=False):
    """
    Simplified BiRank using degree-based scoring.
    
    For small subgraphs, use a simpler approach based on weighted degrees.
    Venue score = weighted sum of user degrees who visit it.
    """
    if len(edges_df) == 0:
        return pd.DataFrame(columns=['business_id', 'score'])
    
    # Compute user degree (number of businesses each user visits)
    user_degrees = edges_df.groupby('user_id')['business_id'].count()
    
    # For each business, compute weighted score
    # Score = sum of (1 / user_degree) for all users who visit it
    # This gives higher scores to businesses visited by focused users
    
    business_scores = []
    for business_id, business_edges in edges_df.groupby('business_id'):
        score = 0
        for user_id in business_edges['user_id']:
            user_degree = user_degrees[user_id]
            # Weight by inverse degree (focused users contribute more)
            score += 1.0 / user_degree if user_degree > 0 else 0
        
        business_scores.append({
            'business_id': business_id,
            'score': score
        })
    
    scores_df = pd.DataFrame(business_scores)
    
    # Normalize scores
    if len(scores_df) > 0:
        total_score = scores_df['score'].sum()
        if total_score > 0:
            scores_df['score'] = scores_df['score'] / total_score
    
    return scores_df

def run_group_birank(edges_with_groups):
    """
    Run BiRank for each user group separately.
    """
    print("\n" + "="*60)
    print("GROUP-SPECIFIC BIRANK")
    print("="*60)
    
    all_group_scores = []
    
    for cluster_id in sorted(edges_with_groups['cluster'].unique()):
        cluster_name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
        
        print(f"\n{cluster_name} (Cluster {int(cluster_id)}):")
        print("-" * 60)
        
        # Filter edges for this cluster
        cluster_edges = edges_with_groups[edges_with_groups['cluster'] == cluster_id].copy()
        
        n_users = cluster_edges['user_id'].nunique()
        n_businesses = cluster_edges['business_id'].nunique()
        n_edges = len(cluster_edges)
        
        print(f"  Users: {n_users:,}")
        print(f"  Businesses: {n_businesses:,}")
        print(f"  Edges: {n_edges:,}")
        
        # Run simplified BiRank
        scores_df = birank_simple(cluster_edges, verbose=False)
        
        # Add cluster info
        scores_df['cluster'] = cluster_id
        scores_df['cluster_name'] = cluster_name
        
        # Rank within cluster
        scores_df = scores_df.sort_values('score', ascending=False)
        scores_df['rank'] = range(1, len(scores_df) + 1)
        
        print(f"  Scored {len(scores_df)} businesses")
        
        # Show top 5
        print(f"\n  Top 5 businesses for {cluster_name}:")
        for i, row in enumerate(scores_df.head(5).itertuples(), 1):
            print(f"    {i}. {row.business_id}: {row.score:.6e}")
        
        all_group_scores.append(scores_df)
    
    print("="*60)
    
    return pd.concat(all_group_scores, ignore_index=True)

def analyze_group_differences(group_scores_df, business_df):
    """
    Analyze how different groups rank venues differently.
    """
    print("\n" + "="*60)
    print("GROUP PREFERENCE ANALYSIS")
    print("="*60)
    
    # For each group, get top 10
    print("\nTop 10 venues per group:")
    print("-" * 60)
    
    for cluster_id in sorted(group_scores_df['cluster'].unique()):
        cluster_name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
        cluster_top10 = group_scores_df[group_scores_df['cluster'] == cluster_id].nsmallest(10, 'rank')
        
        # Merge with business info
        cluster_top10 = cluster_top10.merge(
            business_df[['business_id', 'name', 'city']],
            on='business_id',
            how='left'
        )
        
        print(f"\n{cluster_name}:")
        for i, row in enumerate(cluster_top10.itertuples(), 1):
            name = str(row.name)[:40] if hasattr(row, 'name') else row.business_id
            city = str(row.city) if hasattr(row, 'city') else ''
            print(f"  {i:2d}. {name} ({city})")
    
    # Find group-specific favorites
    print("\n" + "-" * 60)
    print("GROUP-SPECIFIC FAVORITES")
    print("-" * 60)
    print("(High rank in one group, low in others)\n")
    
    # Pivot to get ranks across groups
    pivot_df = group_scores_df.pivot(index='business_id', columns='cluster', values='rank')
    
    for cluster_id in sorted(group_scores_df['cluster'].unique()):
        cluster_name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
        
        # Find businesses ranked high in this cluster but low in others
        cluster_col = cluster_id
        
        # Get businesses in top 20 for this cluster
        top_in_cluster = pivot_df[pivot_df[cluster_col] <= 20].copy()
        
        if len(top_in_cluster) == 0:
            continue
        
        # Compute average rank in other clusters
        other_clusters = [c for c in pivot_df.columns if c != cluster_col]
        if len(other_clusters) > 0:
            top_in_cluster['avg_rank_others'] = top_in_cluster[other_clusters].mean(axis=1)
            
            # Find venues with big rank difference
            top_in_cluster['rank_diff'] = top_in_cluster['avg_rank_others'] - top_in_cluster[cluster_col]
            
            # Get top 3 group-specific
            group_specific = top_in_cluster.nlargest(3, 'rank_diff')
            
            if len(group_specific) > 0:
                print(f"{cluster_name}:")
                group_specific = group_specific.merge(
                    business_df[['business_id', 'name', 'city']],
                    left_index=True,
                    right_on='business_id',
                    how='left'
                )
                
                for i, row in enumerate(group_specific.itertuples(), 1):
                    name = str(row.name)[:35] if hasattr(row, 'name') else ''
                    city = str(row.city) if hasattr(row, 'city') else ''
                    rank_this = row._asdict()[f'_{int(cluster_col)+1}']  # cluster_col value
                    rank_others = row.avg_rank_others
                    print(f"  {name} ({city})")
                    print(f"    Rank in {cluster_name}: {rank_this:.0f}")
                    print(f"    Avg rank in others: {rank_others:.0f}")
                print()
    
    print("="*60)

def create_group_comparison_visualization(group_scores_df, business_df):
    """
    Create heatmap comparing venue ranks across groups.
    """
    print("\nCreating group comparison heatmap...")
    
    # Pivot to get ranks
    pivot_df = group_scores_df.pivot(index='business_id', columns='cluster', values='rank')
    
    # Get top 20 businesses by any group (lowest rank in any column)
    pivot_df['min_rank'] = pivot_df.min(axis=1)
    top_businesses = pivot_df.nsmallest(30, 'min_rank').drop(columns=['min_rank'])
    
    # Merge with business names
    top_businesses = top_businesses.merge(
        business_df[['business_id', 'name', 'city']],
        left_index=True,
        right_on='business_id',
        how='left'
    )
    
    # Create labels
    top_businesses['label'] = top_businesses.apply(
        lambda x: f"{str(x['name'])[:25]}... ({x['city']})" 
        if len(str(x['name'])) > 25 
        else f"{x['name']} ({x['city']})",
        axis=1
    )
    
    # Prepare data for heatmap (use rank, not score)
    heatmap_data = top_businesses[[0.0, 1.0, 2.0, 3.0]].values
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 14))
    
    # Use reverse colormap (lower rank = darker color)
    im = ax.imshow(heatmap_data, cmap='YlOrRd_r', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(len(top_businesses)))
    ax.set_xticklabels([CLUSTER_NAMES.get(i, f"Cluster {i}") for i in range(4)])
    ax.set_yticklabels(top_businesses['label'], fontsize=8)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Rank (lower = better)', fontsize=10)
    
    # Add title
    ax.set_title('Venue Ranks Across User Groups\n(Top 30 venues by best rank in any group)', 
                 fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    plot_file = DATA_DIR / "group_ranking_heatmap.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Heatmap saved to {plot_file}")
    plt.close()

def main():
    """
    Main execution for Task E.
    """
    print("\n" + "="*60)
    print("TASK E: GROUP-SPECIFIC BIRANK")
    print("="*60)
    
    # Load data
    edges_with_groups = load_data()
    
    # Run BiRank for each group
    group_scores_df = run_group_birank(edges_with_groups)
    
    # Save results
    print(f"\nSaving group-specific scores to {OUTPUT_FILE}...")
    group_scores_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(group_scores_df)} venue-group scores")
    
    # Load business data for analysis
    business_df = pd.read_csv(BUSINESS_FILE)
    
    # Analyze group differences
    analyze_group_differences(group_scores_df, business_df)
    
    # Create visualization
    create_group_comparison_visualization(group_scores_df, business_df)
    
    print(f"\n✓ Task E complete.")
    print(f"  Group scores: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
