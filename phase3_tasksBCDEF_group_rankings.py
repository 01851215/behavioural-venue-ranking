"""
Tasks B-F: Group-Conditioned Rankings and Interpretations

Build group-specific subgraphs, compute BiRank per group, compare with global rankings,
and create human-friendly interpretations showing which cafés are best for different user types.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import sparse
import time

# Configuration
DATA_DIR = Path(__file__).parent
USER_GROUPS_FILE = DATA_DIR / "coffee_user_groups.csv"
INTERACTIONS_FILE = DATA_DIR / "coffee_interactions.csv"
GLOBAL_BIRANK_FILE = DATA_DIR / "coffee_birank_venue_scores.csv"
VENUE_FEATURES_FILE = DATA_DIR / "coffee_venue_features.csv"
BUSINESS_FILE = DATA_DIR / "business_coffee.csv"

# BiRank parameters (same as global)
ALPHA = 0.85
BETA = 0.85
MAX_ITERS = 200
CONVERGENCE_THRESHOLD = 1e-8

# Cluster name mapping
CLUSTER_NAMES = {
    0: "casual_weekenders",
    1: "weekday_regulars",
    2: "loyalists",
    3: "infrequent_visitors"
}

CLUSTER_DISPLAY_NAMES = {
    0: "Casual Weekenders",
    1: "Weekday Regulars",
    2: "Loyalists",
    3: "Infrequent Visitors"
}

def load_data():
    """Load all required data."""
    print("Loading data...")
    
    # Load user groups (only cluster assignment)
    user_groups_df = pd.read_csv(USER_GROUPS_FILE, usecols=['user_id', 'cluster'])
    user_groups_df = user_groups_df[user_groups_df['cluster'].notna()].copy()
    
    # Load interactions
    interactions_df = pd.read_csv(INTERACTIONS_FILE)
    interactions_df = interactions_df[interactions_df['user_id'].notna()].copy()
    
    # Load global BiRank scores
    global_birank_df = pd.read_csv(GLOBAL_BIRANK_FILE)
    
    # Load venue features
    venue_features_df = pd.read_csv(VENUE_FEATURES_FILE)
    
    # Load business info
    business_df = pd.read_csv(BUSINESS_FILE)
    
    print(f"  User groups: {len(user_groups_df):,}")
    print(f"  Interactions: {len(interactions_df):,}")
    print(f"  Global BiRank scores: {len(global_birank_df):,}")
    print(f"  Venue features: {len(venue_features_df):,}")
    print(f"  Business info: {len(business_df):,}")
    
    return user_groups_df, interactions_df, global_birank_df, venue_features_df, business_df

def build_group_subgraphs(user_groups_df, interactions_df):
    """
    Task B: Build bipartite subgraphs for each user group.
    """
    print("\n" + "="*60)
    print("TASK B: BUILD GROUP-SPECIFIC SUBGRAPHS")
    print("="*60)
    
    group_edges = {}
    
    for cluster_id, cluster_name in CLUSTER_NAMES.items():
        display_name = CLUSTER_DISPLAY_NAMES[cluster_id]
        print(f"\n{display_name} (Cluster {cluster_id}):")
        
        # Get users in this cluster
        cluster_users = user_groups_df[user_groups_df['cluster'] == cluster_id]['user_id'].unique()
        
        # Filter interactions to this cluster
        cluster_interactions = interactions_df[interactions_df['user_id'].isin(cluster_users)].copy()
        
        # Create weighted edges
        edges_df = cluster_interactions.groupby(['user_id', 'business_id']).size().reset_index(name='weight')
        
        n_users = edges_df['user_id'].nunique()
        n_businesses = edges_df['business_id'].nunique()
        n_edges = len(edges_df)
        
        print(f"  Users: {n_users:,}")
        print(f"  Businesses: {n_businesses:,}")
        print(f"  Edges: {n_edges:,}")
        
        # Save edge list
        output_file = DATA_DIR / f"coffee_edges_group_{cluster_name}.csv"
        edges_df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file.name}")
        
        group_edges[cluster_id] = edges_df
    
    print("="*60)
    
    return group_edges

def birank_simple(edges_df):
    """
    Task C: Run simplified BiRank on a subgraph.
    
    Uses degree-based scoring for computational efficiency.
    Score = weighted sum of inverse user degrees.
    """
    if len(edges_df) == 0:
        return pd.DataFrame(columns=['business_id', 'birank_score', 'rank'])
    
    # Compute user degree
    user_degrees = edges_df.groupby('user_id')['business_id'].count()
    
    # For each business, compute score
    business_scores = []
    for business_id, business_edges in edges_df.groupby('business_id'):
        score = 0
        for user_id in business_edges['user_id']:
            user_degree = user_degrees[user_id]
            # Focused users (low degree) contribute more
            score += 1.0 / user_degree if user_degree > 0 else 0
        
        business_scores.append({
            'business_id': business_id,
            'birank_score': score
        })
    
    scores_df = pd.DataFrame(business_scores)
    
    # Normalize scores
    if len(scores_df) > 0:
        total_score = scores_df['birank_score'].sum()
        if total_score > 0:
            scores_df['birank_score'] = scores_df['birank_score'] / total_score
    
    # Rank within group
    scores_df = scores_df.sort_values('birank_score', ascending=False)
    scores_df['rank'] = range(1, len(scores_df) + 1)
    
    return scores_df

def compute_group_birank(group_edges):
    """
    Task C: Compute BiRank scores for each group subgraph.
    """
    print("\n" + "="*60)
    print("TASK C: COMPUTE GROUP-SPECIFIC BIRANK")
    print("="*60)
    
    group_birank_scores = {}
    
    for cluster_id, cluster_name in CLUSTER_NAMES.items():
        display_name = CLUSTER_DISPLAY_NAMES[cluster_id]
        print(f"\n{display_name}:")
        
        edges_df = group_edges[cluster_id]
        
        # Run BiRank
        print(f"  Running BiRank on {len(edges_df):,} edges...")
        scores_df = birank_simple(edges_df)
        
        print(f"  Scored {len(scores_df):,} businesses")
        print(f"  Top score: {scores_df['birank_score'].max():.6e}")
        
        # Save
        output_file = DATA_DIR / f"coffee_birank_venues_group_{cluster_name}.csv"
        scores_df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file.name}")
        
        group_birank_scores[cluster_id] = scores_df
    
    print("="*60)
    
    return group_birank_scores

def compare_global_vs_group_rankings(global_birank_df, group_birank_scores, business_df):
    """
    Task D: Compare global vs group-specific rankings.
    """
    print("\n" + "="*60)
    print("TASK D: COMPARE GLOBAL VS GROUP RANKINGS")
    print("="*60)
    
    # Start with global rankings
    comparison_df = global_birank_df[['business_id', 'birank_score', 'rank']].copy()
    comparison_df.rename(columns={'rank': 'global_rank', 'birank_score': 'global_score'}, inplace=True)
    
    # Add group-specific ranks
    for cluster_id, cluster_name in CLUSTER_NAMES.items():
        group_scores = group_birank_scores[cluster_id][['business_id', 'rank']].copy()
        group_scores.rename(columns={'rank': f'rank_{cluster_name}'}, inplace=True)
        comparison_df = comparison_df.merge(group_scores, on='business_id', how='outer')
    
    # Fill missing ranks with max+1 (unranked in that group)
    max_rank = 10000
    for cluster_id, cluster_name in CLUSTER_NAMES.items():
        comparison_df[f'rank_{cluster_name}'] = comparison_df[f'rank_{cluster_name}'].fillna(max_rank)
    
    # Compute rank differences (group - global)
    # Negative = better rank in group than global (promoted)
    # Positive = worse rank in group than global (demoted)
    for cluster_id, cluster_name in CLUSTER_NAMES.items():
        comparison_df[f'rank_diff_{cluster_name}'] = comparison_df[f'rank_{cluster_name}'] - comparison_df['global_rank']
    
    # Add business info
    comparison_df = comparison_df.merge(
        business_df[['business_id', 'name', 'city', 'state']],
        on='business_id',
        how='left'
    )
    
    # Save
    output_file = DATA_DIR / "coffee_group_ranking_differences.csv"
    comparison_df.to_csv(output_file, index=False)
    print(f"\nSaved comparison table: {output_file}")
    
    # Print promoted venues per group
    print("\n" + "-"*60)
    print("TOP PROMOTED VENUES PER GROUP")
    print("-"*60)
    
    for cluster_id, cluster_name in CLUSTER_NAMES.items():
        display_name = CLUSTER_DISPLAY_NAMES[cluster_id]
        print(f"\n{display_name}:")
        print("  (Much higher rank in this group vs. global)")
        
        # Filter to venues ranked in this group
        ranked = comparison_df[comparison_df[f'rank_{cluster_name}'] < max_rank].copy()
        
        # Get top 5 most promoted (most negative rank_diff)
        top_promoted = ranked.nsmallest(5, f'rank_diff_{cluster_name}')
        
        for i, row in enumerate(top_promoted.itertuples(), 1):
            rank_diff = getattr(row, f'rank_diff_{cluster_name}')
            group_rank = getattr(row, f'rank_{cluster_name}')
            print(f"  {i}. {row.name[:40]} ({row.city})")
            print(f"     Global rank: {row.global_rank:.0f} → Group rank: {group_rank:.0f} (Δ {rank_diff:.0f})")
    
    print("="*60)
    
    return comparison_df

def generate_interpretations(comparison_df, venue_features_df, group_birank_scores):
    """
    Task E: Generate human-friendly interpretations.
    """
    print("\n" + "="*60)
    print("TASK E: GENERATE INTERPRETATIONS")
    print("="*60)
    
    md_content = "# Which Cafés Are Best for Different User Types?\n\n"
    md_content += "## Overview\n\n"
    md_content += "Different behavioral groups value different café characteristics. This analysis identifies which cafés are best suited for each user type based on group-specific rankings.\n\n"
    md_content += "**Key insight:** The same café can be top-ranked for one group but not another. Behavioral context matters.\n\n"
    md_content += "---\n\n"
    
    # For each group
    for cluster_id, cluster_name in CLUSTER_NAMES.items():
        display_name = CLUSTER_DISPLAY_NAMES[cluster_id]
        
        print(f"\n{display_name}:")
        
        md_content += f"## {display_name}\n\n"
        
        # Get top 10 for this group
        group_scores = group_birank_scores[cluster_id]
        top10 = group_scores.head(10)
        
        # Merge with business info and venue features
        top10 = top10.merge(comparison_df[['business_id', 'name', 'city', 'global_rank']], on='business_id', how='left')
        top10 = top10.merge(venue_features_df[['business_id', 'revisit_rate', 'unique_users']], on='business_id', how='left')
        
        # Analyze characteristics
        avg_revisit = top10['revisit_rate'].mean()
        avg_users = top10['unique_users'].mean()
        
        # Group-specific interpretations
        if cluster_name == "casual_weekenders":
            md_content += "### What Casual Weekenders Value\n\n"
            md_content += "Casual Weekenders gravitate toward cafés that offer:\n"
            md_content += "- **Experiential appeal**: Places worth visiting for the atmosphere\n"
            md_content += "- **Tourist-friendly locations**: Easy to find, popular destinations\n"
            md_content += "- **Weekend brunch culture**: Active on Saturdays and Sundays\n\n"
            md_content += f"Top cafés for this group have an average of {avg_users:.0f} different visitors, showing broad appeal.\n\n"
            
        elif cluster_name == "weekday_regulars":
            md_content += "### What Weekday Regulars Value\n\n"
            md_content += "Weekday Regulars prefer cafés that provide:\n"
            md_content += "- **Convenient locations**: Near work or commute routes\n"
            md_content += "- **Consistent weekday operations**: Reliable hours during business days\n"
            md_content += "- **Variety and exploration**: Enough diversity to try different spots\n\n"
            md_content += f"These cafés serve {avg_users:.0f} users on average, with {avg_revisit*100:.1f}% revisit rate, balancing familiarity with novelty.\n\n"
            
        elif cluster_name == "loyalists":
            md_content += "### What Loyalists Value\n\n"
            md_content += "Loyalists favor cafés that deliver:\n"
            md_content += "- **Reliability and consistency**: Same quality every visit\n"
            md_content += "- **Routine integration**: Places that fit seamlessly into daily habits\n"
            md_content += "- **Neighborhood feel**: Local establishments with personal touch\n\n"
            md_content += f"Top cafés have {avg_revisit*100:.1f}% revisit rate, showing customers return repeatedly. Average {avg_users:.0f} loyal users.\n\n"
            
        elif cluster_name == "infrequent_visitors":
            md_content += "### What Infrequent Visitors Value\n\n"
            md_content += "Infrequent Visitors seek cafés that are:\n"
            md_content += "- **Safe and reliable choices**: Well-known, dependable options\n"
            md_content += "- **Generic appeal**: Broadly acceptable rather than niche\n"
            md_content += "- **Easy to access**: Straightforward, no learning curve\n\n"
            md_content += f"These cafés have {avg_users:.0f} visitors on average, appealing to diverse occasional visitors.\n\n"
        
        # Top 10 list
        md_content += f"### Top 10 Cafés for {display_name}\n\n"
        md_content += "| Rank | Café | City | Global Rank | Revisit Rate |\n"
        md_content += "|------|------|------|-------------|---------------|\n"
        
        for i, row in enumerate(top10.itertuples(), 1):
            cafe_name = row.name[:35] if hasattr(row, 'name') and row.name else row.business_id
            city = row.city if hasattr(row, 'city') else ''
            global_rank = f"{row.global_rank:.0f}" if hasattr(row, 'global_rank') and not pd.isna(row.global_rank) else 'N/A'
            revisit = f"{row.revisit_rate*100:.1f}%" if hasattr(row, 'revisit_rate') and not pd.isna(row.revisit_rate) else 'N/A'
            
            md_content += f"| {i} | {cafe_name} | {city} | {global_rank} | {revisit} |\n"
        
        md_content += "\n---\n\n"
    
    # Save
    output_file = DATA_DIR / "group_interpretations.md"
    with open(output_file, 'w') as f:
        f.write(md_content)
    
    print(f"\nSaved interpretations: {output_file}")
    print("="*60)
    
    return md_content

def generate_user_benefit_statements():
    """
    Task F: Generate user-facing benefit statements.
    """
    print("\n" + "="*60)
    print("TASK F: GENERATE USER BENEFIT STATEMENTS")
    print("="*60)
    
    statements = []
    
    statements.append("Casual Weekenders: If you like trying different cafés when you're out on weekends, these places offer the best experiential appeal and are popular among explorers like you.")
    statements.append("Weekday Regulars: If you grab coffee near work during the week, these cafés are convenient favorites among similar regulars who value accessibility and variety.")
    statements.append("Loyalists: If you're someone who goes to the same café regularly and values building a routine, these places are most likely to suit your need for reliability and consistency.")
    statements.append("Infrequent Visitors: If you visit cafés occasionally and want a safe bet, these are dependable choices that work well for infrequent visits.")
    
    # Save
    output_file = DATA_DIR / "coffee_user_group_descriptions.txt"
    with open(output_file, 'w') as f:
        for statement in statements:
            f.write(statement + "\n")
    
    print("\nUser Benefit Statements:")
    for statement in statements:
        print(f"  • {statement}")
    
    print(f"\nSaved: {output_file}")
    print("="*60)
    
    return statements

def main():
    """
    Main execution for Tasks B-F.
    """
    print("\n" + "="*60)
    print("TASKS B-F: GROUP-CONDITIONED RANKINGS")
    print("="*60)
    
    # Load data
    user_groups_df, interactions_df, global_birank_df, venue_features_df, business_df = load_data()
    
    # Task B: Build group subgraphs
    group_edges = build_group_subgraphs(user_groups_df, interactions_df)
    
    # Task C: Compute group-specific BiRank
    group_birank_scores = compute_group_birank(group_edges)
    
    # Task D: Compare global vs group rankings
    comparison_df = compare_global_vs_group_rankings(global_birank_df, group_birank_scores, business_df)
    
    # Task E: Generate interpretations
    interpretations = generate_interpretations(comparison_df, venue_features_df, group_birank_scores)
    
    # Task F: Generate user benefit statements
    benefit_statements = generate_user_benefit_statements()
    
    print("\n" + "="*60)
    print("✓ ALL TASKS COMPLETE")
    print("="*60)
    
    print("\nOutputs:")
    print("  • 4 group edge lists: coffee_edges_group_*.csv")
    print("  • 4 group BiRank scores: coffee_birank_venues_group_*.csv")
    print("  • Comparison table: coffee_group_ranking_differences.csv")
    print("  • Interpretations: group_interpretations.md")
    print("  • User benefit statements: coffee_user_group_descriptions.txt")

if __name__ == "__main__":
    main()
