"""
Task A: Build Bipartite Graph (Users ↔ Coffee Shops)

Create a bipartite graph from user-venue interactions.
Since check-ins lack user_id, we use review-based edges only.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Configuration
DATA_DIR = Path(__file__).parent
INTERACTIONS_FILE = DATA_DIR / "coffee_interactions.csv"
OUTPUT_EDGES_FILE = DATA_DIR / "coffee_bipartite_edges.csv"

# Edge weighting configuration
REVIEW_WEIGHT = 1.0
CHECKIN_WEIGHT = 1.0  # Not used since check-ins lack user_id

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_interactions():
    """Load interaction data."""
    print("Loading interaction data...")
    interactions_df = pd.read_csv(INTERACTIONS_FILE)
    print(f"Loaded {len(interactions_df):,} interactions")
    print(f"Interaction types: {interactions_df['interaction_type'].value_counts().to_dict()}")
    return interactions_df

def filter_user_interactions(interactions_df):
    """
    Filter to interactions with user_id (reviews only).
    
    IMPORTANT: Check-ins in the dataset lack user_id.
    We can only build user-venue edges from review data.
    """
    print("\nFiltering to user-linked interactions...")
    user_interactions = interactions_df[interactions_df['user_id'].notna()].copy()
    
    print(f"Total interactions: {len(interactions_df):,}")
    print(f"User-linked interactions (reviews): {len(user_interactions):,}")
    print(f"Percentage with user_id: {len(user_interactions)/len(interactions_df)*100:.1f}%")
    
    return user_interactions

def create_weighted_edges(user_interactions):
    """
    Create weighted edges between users and businesses.
    Weight = number of interactions (reviews) per (user, business) pair.
    """
    print("\nCreating weighted edges...")
    
    # Group by user and business to count interactions
    edge_weights = user_interactions.groupby(['user_id', 'business_id']).size().reset_index(name='weight')
    
    print(f"Created {len(edge_weights):,} unique edges (user-business pairs)")
    
    # Weight distribution
    print(f"\nEdge weight statistics:")
    print(f"  Mean weight: {edge_weights['weight'].mean():.2f}")
    print(f"  Median weight: {edge_weights['weight'].median():.0f}")
    print(f"  Max weight: {edge_weights['weight'].max()}")
    print(f"  Edges with weight > 1: {(edge_weights['weight'] > 1).sum():,}")
    
    return edge_weights

def compute_graph_statistics(edges_df):
    """
    Compute and report graph statistics.
    """
    print("\n" + "="*60)
    print("BIPARTITE GRAPH STATISTICS")
    print("="*60)
    
    # Node counts
    n_users = edges_df['user_id'].nunique()
    n_businesses = edges_df['business_id'].nunique()
    n_edges = len(edges_df)
    
    print(f"\nNode counts:")
    print(f"  User nodes: {n_users:,}")
    print(f"  Business nodes: {n_businesses:,}")
    print(f"  Total nodes: {n_users + n_businesses:,}")
    
    print(f"\nEdge counts:")
    print(f"  Total edges: {n_edges:,}")
    print(f"  Total weight: {edges_df['weight'].sum():,}")
    
    # Degree statistics
    user_degrees = edges_df.groupby('user_id')['business_id'].count()
    business_degrees = edges_df.groupby('business_id')['user_id'].count()
    
    print(f"\nUser degree (businesses per user):")
    print(f"  Mean: {user_degrees.mean():.2f}")
    print(f"  Median: {user_degrees.median():.0f}")
    print(f"  Max: {user_degrees.max()}")
    print(f"  Min: {user_degrees.min()}")
    
    print(f"\nBusiness degree (users per business):")
    print(f"  Mean: {business_degrees.mean():.2f}")
    print(f"  Median: {business_degrees.median():.0f}")
    print(f"  Max: {business_degrees.max()}")
    print(f"  Min: {business_degrees.min()}")
    
    # Graph density
    max_possible_edges = n_users * n_businesses
    density = n_edges / max_possible_edges
    print(f"\nGraph density: {density:.6f} ({density*100:.4f}%)")
    
    print("="*60)
    
    return {
        'n_users': n_users,
        'n_businesses': n_businesses,
        'n_edges': n_edges,
        'density': density,
        'user_degrees': user_degrees,
        'business_degrees': business_degrees
    }

def plot_degree_distributions(stats):
    """
    Visualize degree distributions for users and businesses.
    """
    print("\nCreating degree distribution plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # User degree distribution
    ax1 = axes[0]
    user_degrees = stats['user_degrees']
    ax1.hist(user_degrees, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Number of Businesses Visited', fontsize=12)
    ax1.set_ylabel('Number of Users', fontsize=12)
    ax1.set_title('User Degree Distribution', fontsize=13, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'Mean: {user_degrees.mean():.1f}\nMedian: {user_degrees.median():.0f}\nMax: {user_degrees.max()}'
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    # Business degree distribution
    ax2 = axes[1]
    business_degrees = stats['business_degrees']
    ax2.hist(business_degrees, bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax2.set_xlabel('Number of Users (Reviewers)', fontsize=12)
    ax2.set_ylabel('Number of Businesses', fontsize=12)
    ax2.set_title('Business Degree Distribution', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'Mean: {business_degrees.mean():.1f}\nMedian: {business_degrees.median():.0f}\nMax: {business_degrees.max()}'
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    plt.tight_layout()
    
    plot_file = DATA_DIR / "bipartite_graph_degree_distributions.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Plot saved to {plot_file}")
    plt.close()

def main():
    """
    Main execution function for Task A.
    """
    print("\n" + "="*60)
    print("TASK A: BUILD BIPARTITE GRAPH")
    print("="*60 + "\n")
    
    # Load interactions
    interactions_df = load_interactions()
    
    # Filter to user-linked interactions
    user_interactions = filter_user_interactions(interactions_df)
    
    # Create weighted edges
    edges_df = create_weighted_edges(user_interactions)
    
    # Save edge list
    print(f"\nSaving edge list to {OUTPUT_EDGES_FILE}...")
    edges_df.to_csv(OUTPUT_EDGES_FILE, index=False)
    print(f"Saved {len(edges_df):,} edges")
    
    # Compute and display graph statistics
    stats = compute_graph_statistics(edges_df)
    
    # Plot degree distributions
    plot_degree_distributions(stats)
    
    print(f"\n✓ Task A complete. Output saved to: {OUTPUT_EDGES_FILE}")
    
    print("\n" + "!"*60)
    print("DATA STRUCTURE NOTE")
    print("!"*60)
    print("Check-ins in the dataset lack user_id information.")
    print("Therefore, the bipartite graph uses REVIEW-BASED edges only.")
    print(f"This gives us {stats['n_edges']:,} edges from {stats['n_users']:,} users")
    print(f"to {stats['n_businesses']:,} businesses.")
    print("This is sufficient for BiRank and validates actual user engagement.")
    print("!"*60)

if __name__ == "__main__":
    main()
