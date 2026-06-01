"""
Task 5: Multi-Scopic Behaviour Feature Extraction
Extract user-level, venue-level, and group-level behavioural features.

This is the core of behaviour-based ranking analysis.
We extract features that capture how users interact with venues over time,
NOT demographic attributes or assumed identities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Configuration
DATA_DIR = Path(__file__).parent
INTERACTIONS_FILE = DATA_DIR / "coffee_interactions.csv"
REVIEWS_FILE = DATA_DIR / "coffee_reviews.csv"  # Needed for stars field

USER_FEATURES_FILE = DATA_DIR / "coffee_user_features.csv"
VENUE_FEATURES_FILE = DATA_DIR / "coffee_venue_features.csv"
USER_GROUPS_FILE = DATA_DIR / "coffee_user_groups.csv"

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

def load_data():
    """
    Load interaction data.
    Remember: only review interactions have user_id.
    """
    print("Loading interaction data...")
    interactions_df = pd.read_csv(INTERACTIONS_FILE)
    
    # Parse timestamps
    interactions_df['datetime'] = pd.to_datetime(interactions_df['timestamp'], errors='coerce')
    
    print(f"Loaded {len(interactions_df):,} interactions")
    print(f"User-linked interactions: {interactions_df['user_id'].notna().sum():,}")
    
    return interactions_df

def extract_user_features(interactions_df):
    """
    Extract user-level behavioural features.
    Use only review interactions (those with user_id).
    
    Features:
    - total_visits: total coffee shop reviews by this user
    - unique_shops: number of unique coffee shops visited
    - revisit_ratio: proportion of visits that are revisits
    - avg_time_gap_days: average time between consecutive visits
    - time_of_day_mode: most common hour of day for visits
    - weekday_ratio: proportion of visits on weekdays (Mon-Fri)
    """
    print("\n" + "="*60)
    print("EXTRACTING USER-LEVEL FEATURES")
    print("="*60)
    
    # Filter to user-linked interactions only
    user_interactions = interactions_df[interactions_df['user_id'].notna()].copy()
    print(f"Processing {len(user_interactions):,} user-linked interactions")
    print(f"Unique users: {user_interactions['user_id'].nunique():,}")
    
    user_features_list = []
    
    # Group by user
    grouped = user_interactions.groupby('user_id')
    total_users = len(grouped)
    
    print("Computing features for each user...")
    for i, (user_id, user_data) in enumerate(grouped, 1):
        # Sort by time
        user_data = user_data.sort_values('datetime')
        
        # Basic counts
        total_visits = len(user_data)
        unique_shops = user_data['business_id'].nunique()
        
        # Revisit ratio
        # Revisit = when total_visits > unique_shops
        revisits = total_visits - unique_shops
        revisit_ratio = revisits / total_visits if total_visits > 0 else 0
        
        # Time gap between visits
        if len(user_data) > 1:
            time_diffs = user_data['datetime'].diff().dropna()
            avg_time_gap_days = time_diffs.dt.total_seconds().mean() / (24 * 3600)
        else:
            avg_time_gap_days = np.nan
        
        # Time of day distribution (hour)
        valid_times = user_data['datetime'].notna()
        if valid_times.sum() > 0:
            hours = user_data.loc[valid_times, 'datetime'].dt.hour
            # Mode (most common hour)
            if len(hours) > 0:
                time_of_day_mode = hours.mode().iloc[0] if len(hours.mode()) > 0 else np.nan
            else:
                time_of_day_mode = np.nan
        else:
            time_of_day_mode = np.nan
        
        # Weekday vs weekend
        if valid_times.sum() > 0:
            weekdays = user_data.loc[valid_times, 'datetime'].dt.dayofweek < 5  # Mon=0, Sun=6
            weekday_ratio = weekdays.sum() / valid_times.sum() if valid_times.sum() > 0 else 0
        else:
            weekday_ratio = np.nan
        
        user_features_list.append({
            'user_id': user_id,
            'total_visits': total_visits,
            'unique_shops': unique_shops,
            'revisit_ratio': revisit_ratio,
            'avg_time_gap_days': avg_time_gap_days,
            'time_of_day_mode': time_of_day_mode,
            'weekday_ratio': weekday_ratio
        })
        
        # Progress indicator
        if i % 10000 == 0:
            print(f"  Processed {i:,} / {total_users:,} users...")
    
    user_features_df = pd.DataFrame(user_features_list)
    print(f"\n✓ Extracted features for {len(user_features_df):,} users")
    
    return user_features_df

def extract_venue_features(interactions_df):
    """
    Extract venue-level behavioural features.
    Can use ALL interactions (check-ins + reviews).
    
    Features:
    - total_visits: total interactions at this venue
    - unique_users: number of unique users (from reviews only)
    - repeat_users: number of users who visited more than once
    - revisit_rate: proportion of users who are repeat visitors
    - temporal_stability: coefficient of variation of visits over time
    """
    print("\n" + "="*60)
    print("EXTRACTING VENUE-LEVEL FEATURES")
    print("="*60)
    
    print(f"Processing {len(interactions_df):,} total interactions")
    print(f"Unique venues: {interactions_df['business_id'].nunique():,}")
    
    venue_features_list = []
    
    # Group by business
    grouped = interactions_df.groupby('business_id')
    total_venues = len(grouped)
    
    print("Computing features for each venue...")
    for i, (business_id, venue_data) in enumerate(grouped, 1):
        total_visits = len(venue_data)
        
        # User metrics (from review interactions only)
        user_interactions = venue_data[venue_data['user_id'].notna()]
        unique_users = user_interactions['user_id'].nunique()
        
        if unique_users > 0:
            # Count users who visited more than once
            user_visit_counts = user_interactions.groupby('user_id').size()
            repeat_users = (user_visit_counts > 1).sum()
            revisit_rate = repeat_users / unique_users
        else:
            repeat_users = 0
            revisit_rate = 0
        
        # Temporal stability
        # Measure consistency of visits over time using coefficient of variation
        venue_data_sorted = venue_data.sort_values('datetime')
        valid_times = venue_data_sorted['datetime'].notna()
        
        if valid_times.sum() > 10:  # Need sufficient data
            # Group by month and count visits
            venue_data_sorted['year_month'] = venue_data_sorted.loc[valid_times, 'datetime'].dt.to_period('M')
            monthly_visits = venue_data_sorted.groupby('year_month').size()
            
            if len(monthly_visits) > 1:
                # Coefficient of variation = std / mean
                cv = monthly_visits.std() / monthly_visits.mean() if monthly_visits.mean() > 0 else np.nan
                temporal_stability = 1 / (1 + cv) if not np.isnan(cv) else np.nan  # Higher = more stable
            else:
                temporal_stability = np.nan
        else:
            temporal_stability = np.nan
        
        venue_features_list.append({
            'business_id': business_id,
            'total_visits': total_visits,
            'unique_users': unique_users,
            'repeat_users': repeat_users,
            'revisit_rate': revisit_rate,
            'temporal_stability': temporal_stability
        })
        
        # Progress indicator
        if i % 1000 == 0:
            print(f"  Processed {i:,} / {total_venues:,} venues...")
    
    venue_features_df = pd.DataFrame(venue_features_list)
    print(f"\n✓ Extracted features for {len(venue_features_df):,} venues")
    
    return venue_features_df

def create_user_groups(user_features_df):
    """
    Create behavioural user groups using clustering.
    
    IMPORTANT:
    - We do NOT claim demographic identity
    - We describe groups BEHAVIOURALLY
    - Groups are based solely on interaction patterns
    
    Clustering dimensions:
    - Persistence: revisit_ratio, total_visits
    - Exploration: unique_shops vs total_visits
    - Temporal patterns: avg_time_gap, weekday_ratio
    """
    print("\n" + "="*60)
    print("CREATING BEHAVIOURAL USER GROUPS")
    print("="*60)
    
    print("\nClustering approach:")
    print("  Using K-Means clustering on behavioural features")
    print("  Features: revisit_ratio, unique_shops, avg_time_gap_days, weekday_ratio")
    print("  Goal: Identify behavioural archetypes (e.g., routines vs explorers)")
    
    # Select features for clustering
    clustering_features = ['revisit_ratio', 'unique_shops', 'avg_time_gap_days', 'weekday_ratio']
    
    # Filter users with complete data
    complete_data = user_features_df[clustering_features].notna().all(axis=1)
    users_for_clustering = user_features_df[complete_data].copy()
    
    print(f"\nUsers with complete feature data: {len(users_for_clustering):,} / {len(user_features_df):,}")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(users_for_clustering[clustering_features])
    
    # Determine optimal number of clusters using elbow method
    print("\nDetermining optimal number of clusters...")
    inertias = []
    K_range = range(2, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features_scaled)
        inertias.append(kmeans.inertia_)
    
    # Use 4 clusters as a reasonable default for behavioural archetypes
    n_clusters = 4
    print(f"Using {n_clusters} clusters (behavioural archetypes)")
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    users_for_clustering['cluster'] = kmeans.fit_predict(features_scaled)
    
    # Analyze cluster characteristics
    print("\nCluster characteristics:")
    print("-" * 60)
    
    cluster_descriptions = {}
    for cluster_id in range(n_clusters):
        cluster_data = users_for_clustering[users_for_clustering['cluster'] == cluster_id]
        
        print(f"\nCluster {cluster_id}: n={len(cluster_data):,}")
        print(f"  Revisit ratio: {cluster_data['revisit_ratio'].mean():.3f} (±{cluster_data['revisit_ratio'].std():.3f})")
        print(f"  Unique shops: {cluster_data['unique_shops'].mean():.1f} (±{cluster_data['unique_shops'].std():.1f})")
        print(f"  Avg time gap (days): {cluster_data['avg_time_gap_days'].mean():.1f} (±{cluster_data['avg_time_gap_days'].std():.1f})")
        print(f"  Weekday ratio: {cluster_data['weekday_ratio'].mean():.3f} (±{cluster_data['weekday_ratio'].std():.3f})")
        
        # Generate behavioural description
        revisit = cluster_data['revisit_ratio'].mean()
        exploration = cluster_data['unique_shops'].mean()
        time_gap = cluster_data['avg_time_gap_days'].mean()
        
        # Classify behaviour
        if revisit > 0.3 and exploration < 3:
            desc = "High-persistence, low-exploration (Loyalist)"
        elif revisit < 0.15 and exploration > 5:
            desc = "Low-persistence, high-exploration (Explorer)"
        elif time_gap < 30:
            desc = "Frequent, regular visitor (Routine)"
        else:
            desc = "Moderate engagement (Casual)"
        
        cluster_descriptions[cluster_id] = desc
        print(f"  → Behavioural archetype: {desc}")
    
    # Add cluster labels to all users (NaN for those without complete data)
    user_groups_df = user_features_df.copy()
    user_groups_df['cluster'] = np.nan
    user_groups_df.loc[complete_data, 'cluster'] = users_for_clustering['cluster'].values
    
    # Add descriptive labels
    user_groups_df['cluster_description'] = user_groups_df['cluster'].apply(
        lambda x: cluster_descriptions.get(x, 'Unknown') if pd.notna(x) else 'Insufficient data'
    )
    
    print(f"\n✓ Created {n_clusters} behavioural user groups")
    
    # Create visualization
    create_cluster_visualization(users_for_clustering, n_clusters)
    
    return user_groups_df

def create_cluster_visualization(users_for_clustering, n_clusters):
    """
    Create PCA visualization of user clusters.
    """
    print("\nCreating cluster visualization...")
    
    clustering_features = ['revisit_ratio', 'unique_shops', 'avg_time_gap_days', 'weekday_ratio']
    
    # Standardize and apply PCA
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(users_for_clustering[clustering_features])
    
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(features_scaled)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    for cluster_id in range(n_clusters):
        cluster_mask = users_for_clustering['cluster'] == cluster_id
        ax.scatter(pca_coords[cluster_mask, 0], 
                  pca_coords[cluster_mask, 1],
                  c=colors[cluster_id],
                  label=f'Cluster {cluster_id}',
                  alpha=0.6, 
                  s=30)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title('User Behavioural Clusters (PCA Visualization)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = DATA_DIR / "user_clusters_pca.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Visualization saved to {plot_file}")
    plt.close()

def main():
    """
    Main execution function for Task 5.
    """
    print("\n" + "="*60)
    print("TASK 5: MULTI-SCOPIC BEHAVIOUR FEATURE EXTRACTION")
    print("="*60 + "\n")
    
    # Load data
    interactions_df = load_data()
    
    # Part A: User-level features
    print("\n" + "#"*60)
    print("# PART A: USER-LEVEL FEATURES")
    print("#"*60)
    user_features_df = extract_user_features(interactions_df)
    print(f"\nSaving user features to {USER_FEATURES_FILE}...")
    user_features_df.to_csv(USER_FEATURES_FILE, index=False)
    print(f"✓ Saved {len(user_features_df):,} user feature records")
    
    # Part B: Venue-level features
    print("\n" + "#"*60)
    print("# PART B: VENUE-LEVEL FEATURES")
    print("#"*60)
    venue_features_df = extract_venue_features(interactions_df)
    print(f"\nSaving venue features to {VENUE_FEATURES_FILE}...")
    venue_features_df.to_csv(VENUE_FEATURES_FILE, index=False)
    print(f"✓ Saved {len(venue_features_df):,} venue feature records")
    
    # Part C: Group-level behaviour (clustering)
    print("\n" + "#"*60)
    print("# PART C: GROUP-LEVEL BEHAVIOUR")
    print("#"*60)
    user_groups_df = create_user_groups(user_features_df)
    print(f"\nSaving user groups to {USER_GROUPS_FILE}...")
    user_groups_df.to_csv(USER_GROUPS_FILE, index=False)
    print(f"✓ Saved {len(user_groups_df):,} user group records")
    
    print("\n" + "="*60)
    print("✓ TASK 5 COMPLETE")
    print("="*60)
    print("\nOutputs:")
    print(f"  - User features: {USER_FEATURES_FILE}")
    print(f"  - Venue features: {VENUE_FEATURES_FILE}")
    print(f"  - User groups: {USER_GROUPS_FILE}")

if __name__ == "__main__":
    main()
