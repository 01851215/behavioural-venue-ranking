"""
Compute User Features v2

Extended behavioral metrics for coffee shop users:
- Commitment strength (venue concentration)
- Behavioral burstiness (tourist vs. routine)
- Temporal loyalty (consistency over time)
- Improved diversity metrics (entropy)

Input: coffee_interactions.csv, coffee_user_features.csv
Output: coffee_user_features_v2.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

print("Loading data...")

# Load interactions (prefer check-ins over reviews)
interactions = pd.read_csv('coffee_interactions.csv')
print(f"Loaded {len(interactions):,} interactions")

# Load existing user features to merge
existing_features = pd.read_csv('coffee_user_features.csv')
print(f"Loaded {len(existing_features):,} existing user features")

# Timestamp is already in the interactions file, just parse it
interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])

# ============================================================================
# Helper Functions
# ============================================================================

def compute_user_metrics(user_df):
    """
    Compute all new metrics for a single user.
    
    Args:
        user_df: DataFrame of interactions for one user
    
    Returns:
        dict of computed metrics
    """
    metrics = {}
    
    # Total visits
    total_visits = len(user_df)
    
    # ---- COMMITMENT STRENGTH ----
    
    # Venue visit counts
    venue_counts = user_df['business_id'].value_counts()
    
    # Top 1 venue share
    if total_visits > 0:
        metrics['top1_venue_share'] = venue_counts.iloc[0] / total_visits if len(venue_counts) > 0 else 0
    else:
        metrics['top1_venue_share'] = 0
    
    # Top 3 venue share
    if len(venue_counts) >= 3:
        metrics['top3_venue_share'] = venue_counts.iloc[:3].sum() / total_visits
    elif len(venue_counts) > 0:
        metrics['top3_venue_share'] = venue_counts.sum() / total_visits
    else:
        metrics['top3_venue_share'] = 0
    
    # Max visits to single venue
    metrics['max_visits_single_venue'] = venue_counts.iloc[0] if len(venue_counts) > 0 else 0
    
    # ---- BEHAVIORAL BURSTINESS ----
    
    if total_visits >= 2:
        # Sort by timestamp
        timestamps = user_df['timestamp'].sort_values()
        
        # Compute inter-visit intervals in days
        intervals = timestamps.diff().dt.total_seconds() / 86400  # Convert to days
        intervals = intervals.dropna()
        
        if len(intervals) > 0:
            mean_interval = intervals.mean()
            std_interval = intervals.std()
            
            # Burstiness index
            if mean_interval > 0:
                metrics['burstiness_index'] = std_interval / mean_interval
            else:
                metrics['burstiness_index'] = np.nan
            
            # Active span
            metrics['active_span_days'] = (timestamps.max() - timestamps.min()).total_seconds() / 86400
            
            # Visits per active day
            if metrics['active_span_days'] > 0:
                metrics['visits_per_active_day'] = total_visits / metrics['active_span_days']
            else:
                metrics['visits_per_active_day'] = 0
        else:
            metrics['burstiness_index'] = np.nan
            metrics['active_span_days'] = 0
            metrics['visits_per_active_day'] = 0
    else:
        # Single visit or no visits
        metrics['burstiness_index'] = np.nan
        metrics['active_span_days'] = 0
        metrics['visits_per_active_day'] = 0 if total_visits == 0 else np.inf
    
    # ---- TEMPORAL LOYALTY ----
    
    if total_visits > 0:
        # Extract year-month
        user_df['year_month'] = user_df['timestamp'].dt.to_period('M')
        
        # Months active
        metrics['months_active'] = user_df['year_month'].nunique()
        
        # Study window: 2005-01 to 2022-12 (from Yelp dataset)
        study_start = pd.Period('2005-01', freq='M')
        study_end = pd.Period('2022-12', freq='M')
        total_months_in_window = (study_end - study_start).n + 1
        
        # Repeat month ratio
        metrics['repeat_month_ratio'] = metrics['months_active'] / total_months_in_window
    else:
        metrics['months_active'] = 0
        metrics['repeat_month_ratio'] = 0
    
    # ---- DIVERSITY / EXPLORATION ----
    
    if total_visits > 1:
        # Venue visit distribution
        venue_probs = venue_counts / total_visits
        
        # Shannon entropy
        # H = -Σ(p_i * log2(p_i))
        entropy = -np.sum(venue_probs * np.log2(venue_probs))
        metrics['venue_entropy'] = entropy
        
        # Unique venue ratio
        metrics['unique_venue_ratio'] = len(venue_counts) / total_visits
    elif total_visits == 1:
        metrics['venue_entropy'] = 0  # Only one venue, no diversity
        metrics['unique_venue_ratio'] = 1.0
    else:
        metrics['venue_entropy'] = 0
        metrics['unique_venue_ratio'] = 0
    
    return metrics

# ============================================================================
# Compute Metrics for All Users
# ============================================================================

print("\nComputing user metrics...")

# Group by user
user_groups = interactions.groupby('user_id')

# Compute metrics for each user
user_metrics_list = []

for user_id, user_df in user_groups:
    metrics = compute_user_metrics(user_df)
    metrics['user_id'] = user_id
    user_metrics_list.append(metrics)

# Convert to DataFrame
new_features = pd.DataFrame(user_metrics_list)

print(f"Computed metrics for {len(new_features):,} users")

# ============================================================================
# Merge with Existing Features
# ============================================================================

print("\nMerging with existing features...")

# Merge on user_id
features_v2 = existing_features.merge(new_features, on='user_id', how='left')

print(f"Final feature set: {len(features_v2):,} users, {len(features_v2.columns)} features")

# ============================================================================
# Data Quality Checks
# ============================================================================

print("\n--- Data Quality Summary ---")

# Check for NaN values in new features
new_feature_cols = [
    'top1_venue_share', 'top3_venue_share', 'max_visits_single_venue',
    'burstiness_index', 'active_span_days', 'visits_per_active_day',
    'months_active', 'repeat_month_ratio',
    'venue_entropy', 'unique_venue_ratio'
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

output_file = 'coffee_user_features_v2.csv'
features_v2.to_csv(output_file, index=False)

print(f"\n✓ Saved: {output_file}")
print(f"  Users: {len(features_v2):,}")
print(f"  Features: {len(features_v2.columns)}")

# Display sample rows
print("\n--- Sample Rows (new features only) ---")
sample_cols = ['user_id'] + new_feature_cols
print(features_v2[sample_cols].head(10).to_string())

print("\nDone!")
