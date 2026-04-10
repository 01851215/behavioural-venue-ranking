"""
Compute Venue Features v2

Extended behavioral metrics for coffee shop venues:
- Loyalty concentration (Gini coefficient)
- Temporal stability (weekly variance, seasonality)
- Loyalty depth (repeat user metrics)

Input: coffee_interactions.csv, coffee_venue_features.csv
Output: coffee_venue_features_v2.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

print("Loading data...")

# Load interactions
interactions = pd.read_csv('coffee_interactions.csv')
print(f"Loaded {len(interactions):,} interactions")

# Load existing venue features
existing_features = pd.read_csv('coffee_venue_features.csv')
print(f"Loaded {len(existing_features):,} existing venue features")

# Parse timestamps
interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])

# ============================================================================
# Helper Functions
# ============================================================================

def compute_gini_coefficient(values):
    """
    Compute Gini coefficient for a list of values.
    
    Gini = 0 means perfect equality (all values the same)
    Gini = 1 means maximum inequality (one value has everything)
    
    Formula: G = (2 * Σ(i * x_i)) / (n * Σ(x_i)) - (n+1)/n
    where x_i are sorted values in ascending order
    """
    if len(values) == 0:
        return 0
    
    # Sort values
    sorted_values = np.sort(values)
    n = len(sorted_values)
    
    # Compute Gini
    index = np.arange(1, n + 1)
    sum_values = np.sum(sorted_values)
    
    if sum_values == 0:
        return 0
    
    gini = (2 * np.sum(index * sorted_values)) / (n * sum_values) - (n + 1) / n
    
    return gini

def compute_venue_metrics(venue_df):
    """
    Compute all new metrics for a single venue.
    
    Args:
        venue_df: DataFrame of interactions for one venue
    
    Returns:
        dict of computed metrics
    """
    metrics = {}
    
    total_visits = len(venue_df)
    
    # ---- LOYALTY CONCENTRATION ----
    
    # User visit counts for this venue
    user_counts = venue_df['user_id'].value_counts()
    
    # Gini coefficient of user contribution
    metrics['gini_user_contribution'] = compute_gini_coefficient(user_counts.values)
    
    # Top user share
    if total_visits > 0:
        metrics['top_user_share'] = user_counts.iloc[0] / total_visits if len(user_counts) > 0 else 0
    else:
        metrics['top_user_share'] = 0
    
    # ---- TEMPORAL STABILITY ----
    
    if total_visits >= 7:  # Need at least 1 week of data
        # Aggregate by week
        venue_df['week'] = venue_df['timestamp'].dt.to_period('W')
        weekly_visits = venue_df.groupby('week').size()
        
        metrics['weekly_visit_mean'] = weekly_visits.mean()
        metrics['weekly_visit_std'] = weekly_visits.std()
        
        # Coefficient of variation
        if metrics['weekly_visit_mean'] > 0:
            metrics['stability_cv'] = metrics['weekly_visit_std'] / metrics['weekly_visit_mean']
        else:
            metrics['stability_cv'] = np.nan
    else:
        metrics['weekly_visit_mean'] = total_visits  # Too few visits for weekly analysis
        metrics['weekly_visit_std'] = 0
        metrics['stability_cv'] = 0 if total_visits > 0 else np.nan
    
    # Seasonal consistency (quarterly variance)
    if total_visits >= 4:
        venue_df['quarter'] = venue_df['timestamp'].dt.to_period('Q')
        quarterly_visits = venue_df.groupby('quarter').size()
        
        if len(quarterly_visits) >= 2:
            metrics['seasonal_variance'] = quarterly_visits.var()
        else:
            metrics['seasonal_variance'] = 0
    else:
        metrics['seasonal_variance'] = 0
    
    # ---- LOYALTY DEPTH ----
    
    # Unique users
    unique_users = user_counts.count()
    
    # Repeat users (visited 2+ times)
    repeat_users = (user_counts >= 2).sum()
    
    metrics['repeat_user_count'] = repeat_users
    
    # Repeat user rate
    if unique_users > 0:
        metrics['repeat_user_rate'] = repeat_users / unique_users
    else:
        metrics['repeat_user_rate'] = 0
    
    # Average repeat visits among repeat users
    if repeat_users > 0:
        repeat_user_visits = user_counts[user_counts >= 2]
        metrics['avg_user_repeat_visits'] = repeat_user_visits.mean()
    else:
        metrics['avg_user_repeat_visits'] = 0
    
    return metrics

# ============================================================================
# Compute Metrics for All Venues
# ============================================================================

print("\nComputing venue metrics...")

# Group by venue
venue_groups = interactions.groupby('business_id')

# Compute metrics for each venue
venue_metrics_list = []

for business_id, venue_df in venue_groups:
    metrics = compute_venue_metrics(venue_df)
    metrics['business_id'] = business_id
    venue_metrics_list.append(metrics)

# Convert to DataFrame
new_features = pd.DataFrame(venue_metrics_list)

print(f"Computed metrics for {len(new_features):,} venues")

# ============================================================================
# Merge with Existing Features
# ============================================================================

print("\nMerging with existing features...")

# Merge on business_id
features_v2 = existing_features.merge(new_features, on='business_id', how='left')

print(f"Final feature set: {len(features_v2):,} venues, {len(features_v2.columns)} features")

# ============================================================================
# Data Quality Checks
# ============================================================================

print("\n--- Data Quality Summary ---")

# Check for NaN values in new features
new_feature_cols = [
    'gini_user_contribution', 'top_user_share',
    'weekly_visit_mean', 'weekly_visit_std', 'stability_cv',
    'seasonal_variance',
    'repeat_user_count', 'repeat_user_rate', 'avg_user_repeat_visits'
]

for col in new_feature_cols:
    if col in features_v2.columns:
        nan_count = features_v2[col].isna().sum()
        nan_pct = (nan_count / len(features_v2)) * 100
        print(f"{col}: {nan_count:,} NaN ({nan_pct:.2f}%)")

# Display sample statistics
print("\n--- Sample Statistics ---")
print(features_v2[new_feature_cols].describe())

# ============================================================================
# Save Output
# ============================================================================

output_file = 'coffee_venue_features_v2.csv'
features_v2.to_csv(output_file, index=False)

print(f"\n✓ Saved: {output_file}")
print(f"  Venues: {len(features_v2):,}")
print(f"  Features: {len(features_v2.columns)}")

# Display sample rows
print("\n--- Sample Rows (new features only) ---")
sample_cols = ['business_id', 'name'] + new_feature_cols
available_cols = [c for c in sample_cols if c in features_v2.columns]
print(features_v2[available_cols].head(10).to_string())

print("\nDone!")
