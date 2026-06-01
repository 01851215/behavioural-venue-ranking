"""
Task A: Define User Behavioral Groups Explicitly

Load user groups and document their behavioral characteristics.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).parent
USER_GROUPS_FILE = DATA_DIR / "coffee_user_groups.csv"
USER_FEATURES_FILE = DATA_DIR / "coffee_user_features.csv"
INTERACTIONS_FILE = DATA_DIR / "coffee_interactions.csv"

OUTPUT_FILE = DATA_DIR / "group_profiles.md"

# Cluster name mapping
CLUSTER_NAMES = {
    0: "Casual Weekenders",
    1: "Weekday Regulars", 
    2: "Loyalists",
    3: "Infrequent Visitors"
}

def load_data():
    """Load user groups and features."""
    print("Loading user data...")
    
    # Load only cluster assignment from user_groups
    user_groups_df = pd.read_csv(USER_GROUPS_FILE, usecols=['user_id', 'cluster'])
    user_features_df = pd.read_csv(USER_FEATURES_FILE)
    
    print(f"  User groups: {len(user_groups_df):,} users")
    print(f"  User features: {len(user_features_df):,} users")
    
    # Merge to get features per group
    merged_df = user_groups_df.merge(user_features_df, on='user_id', how='inner')
    
    print(f"  Merged data: {len(merged_df):,} users with both cluster and features")
    
    return merged_df

def compute_group_statistics(merged_df):
    """
    Compute comprehensive statistics for each behavioral group.
    """
    print("\n" + "="*60)
    print("BEHAVIORAL GROUP STATISTICS")
    print("="*60)
    
    # Filter to users with cluster assignment
    clustered_df = merged_df[merged_df['cluster'].notna()].copy()
    print(f"\nUsers with cluster assignment: {len(clustered_df):,} ({len(clustered_df)/len(merged_df)*100:.1f}%)")
    
    group_stats = []
    
    for cluster_id in sorted(clustered_df['cluster'].unique()):
        cluster_name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
        cluster_data = clustered_df[clustered_df['cluster'] == cluster_id]
        
        print(f"\n{cluster_name} (Cluster {int(cluster_id)}):")
        print("-" * 60)
        
        # Group size
        n_users = len(cluster_data)
        pct_of_total = n_users / len(clustered_df) * 100
        
        print(f"  Group size: {n_users:,} users ({pct_of_total:.1f}% of total)")
        
        # Average visits per user (total_visits)
        avg_visits = cluster_data['total_visits'].mean()
        median_visits = cluster_data['total_visits'].median()
        
        print(f"  Average visits per user: {avg_visits:.2f} (median: {median_visits:.0f})")
        
        # Average revisit ratio
        avg_revisit = cluster_data['revisit_ratio'].mean()
        median_revisit = cluster_data['revisit_ratio'].median()
        
        print(f"  Average revisit ratio: {avg_revisit:.3f} ({avg_revisit*100:.1f}%)")
        print(f"  Median revisit ratio: {median_revisit:.3f} ({median_revisit*100:.1f}%)")
        
        # Average venue diversity (unique_shops)
        avg_diversity = cluster_data['unique_shops'].mean()
        median_diversity = cluster_data['unique_shops'].median()
        
        print(f"  Average venue diversity: {avg_diversity:.2f} unique cafés")
        print(f"  Median venue diversity: {median_diversity:.0f} unique cafés")
        
        # Weekday ratio
        avg_weekday = cluster_data['weekday_ratio'].mean()
        
        print(f"  Average weekday ratio: {avg_weekday:.3f} ({avg_weekday*100:.1f}% weekday visits)")
        
        # Time gap (avg_time_gap_days)
        avg_time_gap = cluster_data['avg_time_gap_days'].mean()
        median_time_gap = cluster_data['avg_time_gap_days'].median()
        
        print(f"  Average time between visits: {avg_time_gap:.0f} days")
        print(f"  Median time between visits: {median_time_gap:.0f} days")
        
        # Store for markdown
        group_stats.append({
            'cluster_id': cluster_id,
            'cluster_name': cluster_name,
            'n_users': n_users,
            'pct_of_total': pct_of_total,
            'avg_visits': avg_visits,
            'median_visits': median_visits,
            'avg_revisit_ratio': avg_revisit,
            'median_revisit_ratio': median_revisit,
            'avg_diversity': avg_diversity,
            'median_diversity': median_diversity,
            'avg_weekday_ratio': avg_weekday,
            'avg_time_gap_days': avg_time_gap,
            'median_time_gap_days': median_time_gap
        })
    
    print("="*60)
    
    return pd.DataFrame(group_stats)

def generate_markdown(group_stats_df):
    """
    Generate comprehensive markdown documentation of group profiles.
    """
    print("\nGenerating group profiles markdown...")
    
    md_content = """# User Behavioral Group Profiles

## Overview

This document defines the four behavioral user groups identified from coffee shop interaction patterns. These groups exhibit distinct visiting behaviors and café preferences.

**Total users analyzed:** {:,}

---

## Group Summary Table

| Group | Users | % of Total | Avg Visits | Avg Revisit Ratio | Avg Diversity | Weekday % |
|-------|-------|------------|------------|-------------------|---------------|-----------|
""".format(int(group_stats_df['n_users'].sum()))
    
    # Add summary table rows
    for _, row in group_stats_df.iterrows():
        md_content += f"| {row['cluster_name']} | {int(row['n_users']):,} | {row['pct_of_total']:.1f}% | {row['avg_visits']:.1f} | {row['avg_revisit_ratio']*100:.1f}% | {row['avg_diversity']:.1f} | {row['avg_weekday_ratio']*100:.1f}% |\n"
    
    md_content += "\n---\n\n## Detailed Group Profiles\n\n"
    
    # Detailed profiles for each group
    for _, row in group_stats_df.iterrows():
        md_content += f"""### {row['cluster_name']}

**Group Size:** {int(row['n_users']):,} users ({row['pct_of_total']:.1f}% of total)

#### Behavioral Characteristics

**Visit Frequency:**
- Average visits: {row['avg_visits']:.1f} (median: {row['median_visits']:.0f})
- Time between visits: {row['avg_time_gap_days']:.0f} days average, {row['median_time_gap_days']:.0f} days median

**Loyalty & Exploration:**
- Revisit ratio: {row['avg_revisit_ratio']*100:.1f}% average ({row['median_revisit_ratio']*100:.1f}% median)
- Venue diversity: {row['avg_diversity']:.1f} unique cafés average ({row['median_diversity']:.0f} median)

**Temporal Patterns:**
- Weekday visits: {row['avg_weekday_ratio']*100:.1f}% of visits occur on weekdays
- Weekend preference: {(1-row['avg_weekday_ratio'])*100:.1f}% of visits on weekends

"""
        
        # Add behavioral interpretation
        if row['cluster_name'] == "Casual Weekenders":
            md_content += """#### Behavioral Summary

Casual Weekenders are **exploratory weekend users** who visit cafés infrequently and prefer trying new places. They have:
- **Low loyalty**: Only {:.1f}% of visits are revisits
- **Weekend preference**: {:.1f}% weekend visits
- **Moderate exploration**: Visit {:.1f} different cafés on average
- **Infrequent visits**: ~{} days between café visits

**Typical behavior:** Occasional weekend brunch at different trendy spots.

""".format(row['avg_revisit_ratio']*100, (1-row['avg_weekday_ratio'])*100, row['avg_diversity'], int(row['avg_time_gap_days']))
        
        elif row['cluster_name'] == "Weekday Regulars":
            md_content += """#### Behavioral Summary

Weekday Regulars are **work-routine oriented users** who frequently visit cafés during the week. They have:
- **Low-moderate loyalty**: {:.1f}% revisit ratio
- **Strong weekday preference**: {:.1f}% weekday visits
- **High exploration**: Visit {:.1f} different cafés (most exploratory group)
- **Regular visits**: ~{} days between visits

**Typical behavior:** Daily/weekly coffee runs near work, trying different convenient locations.

""".format(row['avg_revisit_ratio']*100, row['avg_weekday_ratio']*100, row['avg_diversity'], int(row['avg_time_gap_days']))
        
        elif row['cluster_name'] == "Loyalists":
            md_content += """#### Behavioral Summary

Loyalists are **high-commitment users** who return to the same cafés repeatedly. They have:
- **High loyalty**: {:.1f}% revisit ratio (41× higher than Casuals!)
- **Focused visits**: Only {:.1f} unique cafés (most focused group)
- **Moderate weekday preference**: {:.1f}% weekday visits
- **Longer gaps**: ~{} days between visits (but to same places)

**Typical behavior:** Regular customers with "their café" where staff know their order.

""".format(row['avg_revisit_ratio']*100, row['avg_diversity'], row['avg_weekday_ratio']*100, int(row['avg_time_gap_days']))
        
        elif row['cluster_name'] == "Infrequent Visitors":
            md_content += """#### Behavioral Summary

Infrequent Visitors are **sporadic users** with very long gaps between café visits. They have:
- **Minimal loyalty**: {:.1f}% revisit ratio
- **Very infrequent visits**: ~{:.0f} days ({:.1f} years) between visits
- **Limited exploration**: Visit {:.1f} cafés on average
- **Mixed timing**: {:.1f}% weekday, {:.1f}% weekend

**Typical behavior:** Occasional café visits, mostly first-time experiences with long gaps.

""".format(row['avg_revisit_ratio']*100, row['avg_time_gap_days'], row['avg_time_gap_days']/365, row['avg_diversity'], row['avg_weekday_ratio']*100, (1-row['avg_weekday_ratio'])*100)
        
        md_content += "---\n\n"
    
    # Add key distinctions section
    md_content += """## Key Group Distinctions

### Loyalty Spectrum

**Highest to Lowest Revisit Ratio:**
1. **Loyalists**: {:.1f}% - Return to same cafés repeatedly
2. **Weekday Regulars**: {:.1f}% - Some routine, but explore more
3. **Casual Weekenders**: {:.1f}% - Rarely revisit
4. **Infrequent Visitors**: {:.1f}% - Almost never revisit

### Exploration Behavior

**Most to Least Diverse:**
1. **Weekday Regulars**: {:.1f} unique cafés - Try many locations
2. **Casual Weekenders**: {:.1f} unique cafés - Moderate exploration
3. **Loyalists**: {:.1f} unique cafés - Focus on favorites
4. **Infrequent Visitors**: {:.1f} unique cafés - Limited exposure

### Temporal Patterns

**Weekday Preference (Highest to Lowest):**
1. **Weekday Regulars**: {:.1f}% - Work routine integration
2. **Loyalists**: {:.1f}% - Mixed but slightly weekday
3. **Infrequent Visitors**: {:.1f}% - No clear pattern
4. **Casual Weekenders**: {:.1f}% - Weekend preference

---

## Implications for Café Ranking

Different groups express value differently:

- **Loyalists** value **reliability and routine integration** → High revisit rate cafés
- **Weekday Regulars** value **convenience and variety** → Accessible locations with options
- **Casual Weekenders** value **experience and novelty** → Trendy, weekend brunch spots
- **Infrequent Visitors** value **safety and genericness** → Well-known, reliable choices

**These groups will rank the same cafés differently.**

---

*Generated from Yelp Open Dataset coffee shop user behavioral analysis*
""".format(
        # Loyalty spectrum
        group_stats_df[group_stats_df['cluster_name']=='Loyalists']['avg_revisit_ratio'].iloc[0]*100,
        group_stats_df[group_stats_df['cluster_name']=='Weekday Regulars']['avg_revisit_ratio'].iloc[0]*100,
        group_stats_df[group_stats_df['cluster_name']=='Casual Weekenders']['avg_revisit_ratio'].iloc[0]*100,
        group_stats_df[group_stats_df['cluster_name']=='Infrequent Visitors']['avg_revisit_ratio'].iloc[0]*100,
        # Diversity
        group_stats_df[group_stats_df['cluster_name']=='Weekday Regulars']['avg_diversity'].iloc[0],
        group_stats_df[group_stats_df['cluster_name']=='Casual Weekenders']['avg_diversity'].iloc[0],
        group_stats_df[group_stats_df['cluster_name']=='Loyalists']['avg_diversity'].iloc[0],
        group_stats_df[group_stats_df['cluster_name']=='Infrequent Visitors']['avg_diversity'].iloc[0],
        # Weekday
        group_stats_df[group_stats_df['cluster_name']=='Weekday Regulars']['avg_weekday_ratio'].iloc[0]*100,
        group_stats_df[group_stats_df['cluster_name']=='Loyalists']['avg_weekday_ratio'].iloc[0]*100,
        group_stats_df[group_stats_df['cluster_name']=='Infrequent Visitors']['avg_weekday_ratio'].iloc[0]*100,
        group_stats_df[group_stats_df['cluster_name']=='Casual Weekenders']['avg_weekday_ratio'].iloc[0]*100,
    )
    
    return md_content

def main():
    """
    Main execution for Task A.
    """
    print("\n" + "="*60)
    print("TASK A: DEFINE USER BEHAVIORAL GROUPS")
    print("="*60)
    
    # Load data
    merged_df = load_data()
    
    # Compute group statistics
    group_stats_df = compute_group_statistics(merged_df)
    
    # Generate markdown
    md_content = generate_markdown(group_stats_df)
    
    # Save markdown
    print(f"\nSaving group profiles to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        f.write(md_content)
    
    print(f"Saved group profiles documentation")
    
    print(f"\n✓ Task A complete.")
    print(f"  Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
