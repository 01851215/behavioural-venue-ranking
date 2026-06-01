"""
Compute Group Summary v2

Recompute group summaries with new behavioral features.

Input: coffee_user_features_v2.csv, coffee_user_groups.csv
Output: coffee_user_group_summary_v2.csv
"""

import pandas as pd
import numpy as np

print("Loading data...")

# Load user features v2
user_features = pd.read_csv('coffee_user_features_v2.csv')
print(f"Loaded {len(user_features):,} user features")

# Load user groups
user_groups = pd.read_csv('coffee_user_groups.csv')
print(f"Loaded {len(user_groups):,} user group assignments")

# Merge
data = user_features.merge(user_groups[['user_id', 'cluster', 'cluster_description']], 
                            on='user_id', how='inner')
print(f"Merged: {len(data):,} users with both features and group assignments")

# ============================================================================
# Compute Group Summaries
# ============================================================================

print("\nComputing group summaries...")

# Group by cluster
grouped = data.groupby('cluster_description')

# Aggregate statistics for new features
summary_stats = grouped.agg({
    # Existing features
    'total_visits': ['count', 'mean', 'std'],
    'unique_shops': ['mean', 'std'],
    'revisit_ratio': ['mean', 'std'],
    
    # New commitment features
    'top1_venue_share': ['mean', 'std', 'median'],
    'top3_venue_share': ['mean', 'std', 'median'],
    'max_visits_single_venue': ['mean', 'median', 'max'],
    
    # New burstiness features  
    'burstiness_index': ['mean', 'std', lambda x: x.dropna().median()],
    'active_span_days': ['mean', 'median'],
    'visits_per_active_day': ['mean', lambda x: x.replace([np.inf], np.nan).median()],
    
    # New temporal loyalty features
    'months_active': ['mean', 'median'],
    'repeat_month_ratio': ['mean', 'median'],
    
    # New diversity features
    'venue_entropy': ['mean', 'std', 'median'],
    'unique_venue_ratio': ['mean', 'std', 'median']
})

# Flatten multi-index columns
summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
summary_stats = summary_stats.reset_index()

print(f"Computed summaries for {len(summary_stats)} groups")

# ============================================================================
# Add Percentile Distributions
# ============================================================================

print("\nComputing percentile distributions...")

# For key features, compute percentiles
percentile_features = ['top1_venue_share', 'burstiness_index', 'venue_entropy']

for feature in percentile_features:
    if feature in data.columns:
        percentiles = grouped[feature].describe(percentiles=[0.25, 0.5, 0.75])
        for pct in ['25%', '50%', '75%']:
            if pct in percentiles.columns:
                col_name = f"{feature}_p{pct.replace('%', '')}"
                summary_stats[col_name] = percentiles[pct].values

# ============================================================================
# Save Output
# ============================================================================

output_file = 'coffee_user_group_summary_v2.csv'
summary_stats.to_csv(output_file, index=False)

print(f"\n✓ Saved: {output_file}")
print(f"  Groups: {len(summary_stats)}")
print(f"  Metrics: {len(summary_stats.columns)}")

# Display summary
print("\n--- Group Summary (Key Metrics) ---")
display_cols = [
    'cluster_description',
    'total_visits_count',
    'top1_venue_share_mean',
    'burstiness_index_mean', 
    'venue_entropy_mean',
    'repeat_month_ratio_mean'
]
available_cols = [c for c in display_cols if c in summary_stats.columns]
print(summary_stats[available_cols].to_string(index=False))

print("\nDone!")
