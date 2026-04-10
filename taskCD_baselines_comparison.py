"""
Tasks C & D: Baseline Rankings and Comparison

Compute baseline rankings (rating, popularity, revisit) and compare with BiRank.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr

# Configuration
DATA_DIR = Path(__file__).parent
REVIEWS_FILE = DATA_DIR / "coffee_reviews.csv"
INTERACTIONS_FILE = DATA_DIR / "coffee_interactions.csv"
VENUE_FEATURES_FILE = DATA_DIR / "coffee_venue_features.csv"
BIRANK_SCORES_FILE = DATA_DIR / "coffee_birank_venue_scores.csv"
BUSINESS_FILE = DATA_DIR / "business_coffee.csv"

BASELINES_FILE = DATA_DIR / "coffee_baselines.csv"
COMPARISON_FILE = DATA_DIR / "coffee_ranking_comparison.csv"

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def compute_rating_baseline():
    """Compute mean star rating per business."""
    print("Computing rating baseline...")
    reviews_df = pd.read_csv(REVIEWS_FILE)
    
    rating_baseline = reviews_df.groupby('business_id')['stars'].agg([
        ('rating_mean', 'mean'),
        ('rating_count', 'count')
    ]).reset_index()
    
    print(f"  {len(rating_baseline)} businesses with ratings")
    print(f"  Mean rating: {rating_baseline['rating_mean'].mean():.2f}")
    
    return rating_baseline

def compute_popularity_baseline():
    """Compute total interaction count per business."""
    print("\nComputing popularity baseline...")
    interactions_df = pd.read_csv(INTERACTIONS_FILE)
    
    popularity_baseline = interactions_df.groupby('business_id').size().reset_index(name='popularity_visits')
    
    print(f"  {len(popularity_baseline)} businesses with interactions")
    print(f"  Mean visits: {popularity_baseline['popularity_visits'].mean():.1f}")
    print(f"  Median visits: {popularity_baseline['popularity_visits'].median():.0f}")
    
    return popularity_baseline

def compute_revisit_baseline():
    """Load revisit rate from venue features."""
    print("\nComputing revisit baseline...")
    venue_features_df = pd.read_csv(VENUE_FEATURES_FILE)
    
    revisit_baseline = venue_features_df[['business_id', 'revisit_rate', 'unique_users', 'repeat_users']].copy()
    
    print(f"  {len(revisit_baseline)} businesses with revisit rates")
    print(f"  Mean revisit rate: {revisit_baseline['revisit_rate'].mean():.3f}")
    print(f"  Median revisit rate: {revisit_baseline['revisit_rate'].median():.3f}")
    
    return revisit_baseline

def merge_baselines(rating_baseline, popularity_baseline, revisit_baseline):
    """Merge all baseline metrics."""
    print("\nMerging baselines...")
    
    # Start with rating baseline
    baselines_df = rating_baseline.copy()
    
    # Merge popularity
    baselines_df = baselines_df.merge(popularity_baseline, on='business_id', how='outer')
    
    # Merge revisit
    baselines_df = baselines_df.merge(revisit_baseline, on='business_id', how='outer')
    
    # Fill missing values
    baselines_df['rating_mean'] = baselines_df['rating_mean'].fillna(baselines_df['rating_mean'].median())
    baselines_df['popularity_visits'] = baselines_df['popularity_visits'].fillna(0)
    baselines_df['revisit_rate'] = baselines_df['revisit_rate'].fillna(0)
    
    print(f"  Combined {len(baselines_df)} businesses")
    
    return baselines_df

def compute_ranks(df, score_col, rank_col):
    """Compute ranks from scores (higher score = better rank = lower rank number)."""
    df[rank_col] = df[score_col].rank(ascending=False, method='min').astype(int)
    return df

def merge_with_birank(baselines_df):
    """Merge baselines with BiRank scores."""
    print("\nMerging with BiRank scores...")
    birank_df = pd.read_csv(BIRANK_SCORES_FILE)
    
    comparison_df = baselines_df.merge(birank_df, on='business_id', how='outer')
    
    # Fill missing BiRank scores
    comparison_df['birank_score'] = comparison_df['birank_score'].fillna(0)
    comparison_df['rank'] = comparison_df['rank'].fillna(len(comparison_df)).astype(int)
    
    # Compute ranks for all metrics
    comparison_df = compute_ranks(comparison_df, 'rating_mean', 'rating_rank')
    comparison_df = compute_ranks(comparison_df, 'popularity_visits', 'popularity_rank')
    comparison_df = compute_ranks(comparison_df, 'revisit_rate', 'revisit_rank')
    
    # BiRank rank is already computed, but rename for clarity
    comparison_df.rename(columns={'rank': 'birank_rank'}, inplace=True)
    
    print(f"  Final comparison table: {len(comparison_df)} businesses")
    
    return comparison_df

def compute_correlations(comparison_df):
    """Compute Spearman rank correlations."""
    print("\n" + "="*60)
    print("RANK CORRELATIONS (Spearman)")
    print("="*60)
    
    # Filter to businesses with all metrics
    complete_df = comparison_df.dropna(subset=['birank_rank', 'rating_rank', 'popularity_rank', 'revisit_rank'])
    
    print(f"\nBusinesses with complete data: {len(complete_df)}")
    
    # BiRank vs Rating
    corr_rating, p_rating = spearmanr(complete_df['birank_rank'], complete_df['rating_rank'])
    print(f"\nBiRank ↔ Rating rank:")
    print(f"  Spearman ρ = {corr_rating:.4f} (p={p_rating:.2e})")
    
    # BiRank vs Popularity
    corr_pop, p_pop = spearmanr(complete_df['birank_rank'], complete_df['popularity_rank'])
    print(f"\nBiRank ↔ Popularity rank:")
    print(f"  Spearman ρ = {corr_pop:.4f} (p={p_pop:.2e})")
    
    # BiRank vs Revisit
    corr_revisit, p_revisit = spearmanr(complete_df['birank_rank'], complete_df['revisit_rank'])
    print(f"\nBiRank ↔ Revisit rank:")
    print(f"  Spearman ρ = {corr_revisit:.4f} (p={p_revisit:.2e})")
    
    print("="*60)
    
    return {
        'rating': corr_rating,
        'popularity': corr_pop,
        'revisit': corr_revisit
    }

def compute_topk_overlap(comparison_df, k_values=[10, 20, 50]):
    """Compute top-k overlap between BiRank and baselines."""
    print("\n" + "="*60)
    print("TOP-K OVERLAP (Jaccard)")
    print("="*60)
    
    for k in k_values:
        print(f"\nTop-{k} overlap:")
        
        birank_topk = set(comparison_df.nsmallest(k, 'birank_rank')['business_id'])
        rating_topk = set(comparison_df.nsmallest(k, 'rating_rank')['business_id'])
        popularity_topk = set(comparison_df.nsmallest(k, 'popularity_rank')['business_id'])
        revisit_topk = set(comparison_df.nsmallest(k, 'revisit_rank')['business_id'])
        
        # Jaccard overlap
        overlap_rating = len(birank_topk & rating_topk) / k
        overlap_pop = len(birank_topk & popularity_topk) / k
        overlap_revisit = len(birank_topk & revisit_topk) / k
        
        print(f"  BiRank ∩ Rating: {len(birank_topk & rating_topk)}/{k} ({overlap_rating:.1%})")
        print(f"  BiRank ∩ Popularity: {len(birank_topk & popularity_topk)}/{k} ({overlap_pop:.1%})")
        print(f"  BiRank ∩ Revisit: {len(birank_topk & revisit_topk)}/{k} ({overlap_revisit:.1%})")
    
    print("="*60)

def create_visualizations(comparison_df):
    """Create comprehensive comparison visualizations."""
    print("\nCreating comparison visualizations...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: BiRank vs Rating scatter
    ax1 = fig.add_subplot(gs[0, 0])
    scatter_df = comparison_df.dropna(subset=['birank_score', 'rating_mean'])
    ax1.scatter(scatter_df['rating_mean'], scatter_df['birank_score'], 
                alpha=0.4, s=30, color='steelblue')
    
    # Highlight top-20 BiRank
    top20_birank = scatter_df.nsmallest(20, 'birank_rank')
    ax1.scatter(top20_birank['rating_mean'], top20_birank['birank_score'],
                alpha=0.8, s=100, color='red', marker='*', 
                edgecolor='black', linewidth=0.5, label='Top 20 BiRank')
    
    ax1.set_xlabel('Mean Rating (stars)', fontsize=11)
    ax1.set_ylabel('BiRank Score', fontsize=11)
    ax1.set_title('BiRank Score vs. Rating', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: BiRank vs Popularity scatter
    ax2 = fig.add_subplot(gs[0, 1])
    scatter_df = comparison_df.dropna(subset=['birank_score', 'popularity_visits'])
    ax2.scatter(scatter_df['popularity_visits'], scatter_df['birank_score'],
                alpha=0.4, s=30, color='coral')
    
    # Highlight top-20 BiRank
    top20_birank = scatter_df.nsmallest(20, 'birank_rank')
    ax2.scatter(top20_birank['popularity_visits'], top20_birank['birank_score'],
                alpha=0.8, s=100, color='red', marker='*',
                edgecolor='black', linewidth=0.5, label='Top 20 BiRank')
    
    ax2.set_xlabel('Total Visits (log scale)', fontsize=11)
    ax2.set_ylabel('BiRank Score', fontsize=11)
    ax2.set_title('BiRank Score vs. Popularity', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: BiRank score histogram
    ax3 = fig.add_subplot(gs[1, 0])
    scores = comparison_df['birank_score'].dropna()
    ax3.hist(scores, bins=50, edgecolor='black', alpha=0.7, color='teal')
    ax3.axvline(scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {scores.mean():.2e}')
    ax3.axvline(scores.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {scores.median():.2e}')
    ax3.set_xlabel('BiRank Score', fontsize=11)
    ax3.set_ylabel('Number of Venues', fontsize=11)
    ax3.set_title('BiRank Score Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: BiRank vs Revisit scatter
    ax4 = fig.add_subplot(gs[1, 1])
    scatter_df = comparison_df.dropna(subset=['birank_score', 'revisit_rate'])
    ax4.scatter(scatter_df['revisit_rate'], scatter_df['birank_score'],
                alpha=0.4, s=30, color='purple')
    
    # Highlight top-20 BiRank
    top20_birank = scatter_df.nsmallest(20, 'birank_rank')
    ax4.scatter(top20_birank['revisit_rate'], top20_birank['birank_score'],
                alpha=0.8, s=100, color='red', marker='*',
                edgecolor='black', linewidth=0.5, label='Top 20 BiRank')
    
    ax4.set_xlabel('Revisit Rate', fontsize=11)
    ax4.set_ylabel('BiRank Score', fontsize=11)
    ax4.set_title('BiRank Score vs. Revisit Rate', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5 & 6: Top 20 venues bar chart (combined)
    ax5 = fig.add_subplot(gs[2, :])
    
    # Get top 20 by BiRank
    top20 = comparison_df.nsmallest(20, 'birank_rank').copy()
    
    # Merge with business names
    business_df = pd.read_csv(BUSINESS_FILE)
    top20 = top20.merge(business_df[['business_id', 'name', 'city']], on='business_id', how='left')
    
    # Create labels (truncate long names)
    top20['label'] = top20.apply(lambda x: f"{x['name'][:25]}... ({x['city']})" 
                                  if len(str(x['name'])) > 25 
                                  else f"{x['name']} ({x['city']})", axis=1)
    
    x = np.arange(len(top20))
    width = 0.25
    
    # Normalize metrics for display (0-1 scale)
    max_rating = 5.0
    max_visits = top20['popularity_visits'].max()
    max_revisit = 1.0
    
    bars1 = ax5.bar(x - width, top20['rating_mean'] / max_rating, width, 
                    label='Rating (norm.)', color='steelblue', alpha=0.8)
    bars2 = ax5.bar(x, top20['popularity_visits'] / max_visits, width,
                    label='Popularity (norm.)', color='coral', alpha=0.8)
    bars3 = ax5.bar(x + width, top20['revisit_rate'] / max_revisit, width,
                    label='Revisit Rate (norm.)', color='purple', alpha=0.8)
    
    ax5.set_ylabel('Normalized Score', fontsize=11)
    ax5.set_title('Top 20 Venues by BiRank: Multi-Metric Comparison', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(top20['label'], rotation=45, ha='right', fontsize=8)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    plot_file = DATA_DIR / "ranking_comparison_plots.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Plots saved to {plot_file}")
    plt.close()

def main():
    """
    Main execution for Tasks C & D.
    """
    print("\n" + "="*60)
    print("TASKS C & D: BASELINES AND COMPARISON")
    print("="*60)
    
    # Task C: Compute baselines
    print("\n" + "-"*60)
    print("TASK C: COMPUTING BASELINES")
    print("-"*60)
    
    rating_baseline = compute_rating_baseline()
    popularity_baseline = compute_popularity_baseline()
    revisit_baseline = compute_revisit_baseline()
    
    baselines_df = merge_baselines(rating_baseline, popularity_baseline, revisit_baseline)
    
    print(f"\nSaving baselines to {BASELINES_FILE}...")
    baselines_df.to_csv(BASELINES_FILE, index=False)
    print(f"Saved {len(baselines_df)} business baselines")
    
    # Task D: Comparison
    print("\n" + "-"*60)
    print("TASK D: RANKING COMPARISON")
    print("-"*60)
    
    comparison_df = merge_with_birank(baselines_df)
    
    print(f"\nSaving comparison table to {COMPARISON_FILE}...")
    comparison_df.to_csv(COMPARISON_FILE, index=False)
    print(f"Saved {len(comparison_df)} business comparisons")
    
    # Correlation analysis
    correlations = compute_correlations(comparison_df)
    
    # Top-k overlap
    compute_topk_overlap(comparison_df)
    
    # Visualizations
    create_visualizations(comparison_df)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nTop 10 venues by BiRank:")
    top10 = comparison_df.nsmallest(10, 'birank_rank')
    business_df = pd.read_csv(BUSINESS_FILE)
    top10 = top10.merge(business_df[['business_id', 'name', 'city']], on='business_id', how='left')
    
    for i, row in enumerate(top10.itertuples(), 1):
        print(f"{i:2d}. {row.name} ({row.city})")
        print(f"    BiRank: {row.birank_score:.6e} | Rating: {row.rating_mean:.2f} | Visits: {row.popularity_visits:.0f} | Revisit: {row.revisit_rate:.3f}")
    
    print("="*60)
    print(f"\n✓ Tasks C & D complete.")
    print(f"  Baselines: {BASELINES_FILE}")
    print(f"  Comparison: {COMPARISON_FILE}")

if __name__ == "__main__":
    main()
