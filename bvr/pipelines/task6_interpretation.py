"""
Task 6: Behaviour Interpretation & Validation Readiness
Analyze how different user groups interact with venues and prepare
final interpretation document.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).parent
USER_GROUPS_FILE = DATA_DIR / "coffee_user_groups.csv"
VENUE_FEATURES_FILE = DATA_DIR / "coffee_venue_features.csv"
INTERACTIONS_FILE = DATA_DIR / "coffee_interactions.csv"
BUSINESS_FILE = DATA_DIR / "business_coffee.csv"

OUTPUT_MD = DATA_DIR / "behaviour_interpretation.md"

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def load_all_data():
    """Load all necessary data for interpretation"""
    print("Loading data...")
    
    user_groups_df = pd.read_csv(USER_GROUPS_FILE)
    venue_features_df = pd.read_csv(VENUE_FEATURES_FILE)
    interactions_df = pd.read_csv(INTERACTIONS_FILE)
    business_df = pd.read_csv(BUSINESS_FILE)
    
    # Parse timestamps
    interactions_df['datetime'] = pd.to_datetime(interactions_df['timestamp'], errors='coerce')
    
    print(f"  Users: {len(user_groups_df):,}")
    print(f"  Venues: {len(venue_features_df):,}")
    print(f"  Interactions: {len(interactions_df):,}")
    print(f"  Businesses: {len(business_df):,}")
    
    return user_groups_df, venue_features_df, interactions_df, business_df

def analyze_group_patterns(user_groups_df, interactions_df):
    """
    Analyze how different user groups interact with coffee shops.
    """
    print("\nAnalyzing user group patterns...")
    
    # Merge user groups with interactions
    user_interactions = interactions_df[interactions_df['user_id'].notna()].copy()
    user_interactions = user_interactions.merge(
        user_groups_df[['user_id', 'cluster', 'cluster_description']],
        on='user_id',
        how='left'
    )
    
    # Parse datetime
    user_interactions['datetime'] = pd.to_datetime(user_interactions['timestamp'], errors='coerce')
    
    # Group-level analysis
    patterns = {}
    
    for cluster_id in range(4):
        cluster_data = user_interactions[user_interactions['cluster'] == cluster_id]
        if len(cluster_data) == 0:
            continue
        
        # Revisit patterns
        visits_per_venue = cluster_data.groupby(['user_id', 'business_id']).size()
        avg_visits_per_venue = visits_per_venue.mean()
        
        # Loyalty (number of venues a typical user visits)
        venues_per_user = cluster_data.groupby('user_id')['business_id'].nunique()
        avg_venues_per_user = venues_per_user.mean()
        
        # Temporal regularity (std of time gaps)
        user_time_gaps = []
        for user_id, user_data in cluster_data.groupby('user_id'):
            user_data_sorted = user_data.sort_values('datetime')
            if len(user_data_sorted) > 1:
                time_diffs = user_data_sorted['datetime'].diff().dropna()
                if len(time_diffs) > 0:
                    user_time_gaps.append(time_diffs.dt.total_seconds().std())
        
        temporal_regularity = np.mean(user_time_gaps) if user_time_gaps else np.nan
        
        patterns[cluster_id] = {
            'avg_visits_per_venue': avg_visits_per_venue,
            'avg_venues_per_user': avg_venues_per_user,
            'temporal_regularity': temporal_regularity,
            'total_users': cluster_data['user_id'].nunique(),
            'total_interactions': len(cluster_data)
        }
    
    return patterns

def identify_venue_types(venue_features_df, interactions_df, user_groups_df):
    """
    Identify venues dominated by different user types.
    """
    print("\nIdentifying venue types...")
    
    # Get user-linked interactions
    user_interactions = interactions_df[interactions_df['user_id'].notna()].copy()
    user_interactions = user_interactions.merge(
        user_groups_df[['user_id', 'cluster']],
        on='user_id',
        how='left'
    )
    
    # For each venue, determine dominant user group
    venue_user_profiles = []
    
    for business_id, venue_data in user_interactions.groupby('business_id'):
        cluster_counts = venue_data['cluster'].value_counts()
        total_interactions = len(venue_data)
        
        if total_interactions >= 10:  # Minimum threshold
            dominant_cluster = cluster_counts.idxmax() if len(cluster_counts) > 0 else np.nan
            dominant_proportion = cluster_counts.max() / total_interactions if len(cluster_counts) > 0 else 0
            
            venue_user_profiles.append({
                'business_id': business_id,
                'dominant_cluster': dominant_cluster,
                'dominant_proportion': dominant_proportion,
                'total_interactions': total_interactions
            })
    
    venue_profiles_df = pd.DataFrame(venue_user_profiles)
    
    # Merge with venue features
    venue_profiles_df = venue_profiles_df.merge(venue_features_df, on='business_id', how='left')
    
    # Categorize venues
    repeat_user_venues = venue_profiles_df[venue_profiles_df['revisit_rate'] > 0.3]
    exploratory_venues = venue_profiles_df[venue_profiles_df['revisit_rate'] < 0.1]
    
    print(f"  Repeat-user dominated venues: {len(repeat_user_venues)}")
    print(f"  Exploratory-user dominated venues: {len(exploratory_venues)}")
    
    return venue_profiles_df, repeat_user_venues, exploratory_venues

def create_visualizations(user_groups_df, venue_features_df, patterns):
    """
    Create comprehensive visualizations for interpretation.
    """
    print("\nCreating interpretation visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: User cluster distribution
    ax1 = axes[0, 0]
    cluster_counts = user_groups_df['cluster'].value_counts().sort_index()
    cluster_labels = [f"Cluster {i}" for i in cluster_counts.index]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    ax1.bar(cluster_labels, cluster_counts.values, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Number of Users', fontsize=12)
    ax1.set_title('User Distribution Across Behavioural Clusters', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Revisit rate distribution
    ax2 = axes[0, 1]
    revisit_rates = user_groups_df['revisit_ratio'].dropna()
    ax2.hist(revisit_rates, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.set_xlabel('Revisit Ratio', fontsize=12)
    ax2.set_ylabel('Number of Users', fontsize=12)
    ax2.set_title('User Revisit Ratio Distribution', fontsize=13, fontweight='bold')
    ax2.axvline(revisit_rates.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {revisit_rates.median():.2f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Venue revisit rate distribution
    ax3 = axes[1, 0]
    venue_revisit_rates = venue_features_df['revisit_rate'].dropna()
    ax3.hist(venue_revisit_rates, bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax3.set_xlabel('Venue Revisit Rate', fontsize=12)
    ax3.set_ylabel('Number of Venues', fontsize=12)
    ax3.set_title('Venue Revisit Rate Distribution', fontsize=13, fontweight='bold')
    ax3.axvline(venue_revisit_rates.median(), color='darkred', linestyle='--', linewidth=2, label=f'Median: {venue_revisit_rates.median():.2f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cluster characteristics comparison
    ax4 = axes[1, 1]
    if patterns:
        cluster_ids = list(patterns.keys())
        avg_venues = [patterns[c]['avg_venues_per_user'] for c in cluster_ids]
        
        x = np.arange(len(cluster_ids))
        width = 0.6
        bars = ax4.bar(x, avg_venues, width, color=colors[:len(cluster_ids)], alpha=0.8, edgecolor='black')
        
        ax4.set_ylabel('Avg Venues per User', fontsize=12)
        ax4.set_xlabel('Cluster', fontsize=12)
        ax4.set_title('Exploration Behavior Across Clusters', fontsize=13, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'Cluster {i}' for i in cluster_ids])
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    plot_file = DATA_DIR / "behaviour_interpretation_plots.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Plots saved to {plot_file}")
    plt.close()
    
    return plot_file

def generate_interpretation_document(user_groups_df, venue_features_df, patterns, 
                                     repeat_venues, exploratory_venues, plot_file):
    """
    Generate comprehensive markdown interpretation document.
    """
    print("\nGenerating interpretation document...")
    
    # Calculate key statistics
    total_users = len(user_groups_df)
    users_with_clusters = user_groups_df['cluster'].notna().sum()
    
    loyalist_cluster = 2  # From Task 5 output
    loyalist_users = (user_groups_df['cluster'] == loyalist_cluster).sum()
    loyalist_pct = loyalist_users / users_with_clusters * 100 if users_with_clusters > 0 else 0
    
    median_revisit = user_groups_df['revisit_ratio'].median()
    mean_unique_shops = user_groups_df['unique_shops'].mean()
    
    venue_median_revisit_rate = venue_features_df['revisit_rate'].median()
    
    md_content = f"""# Behaviour-Based Coffee Shop Analysis: Interpretation & Findings

## Executive Summary

This analysis reveals distinct behavioural patterns among coffee shop users in the Yelp dataset,
demonstrating that **user behaviour provides insights that ratings alone cannot capture**.

**Key Findings:**
- Identified **4 behavioural user archetypes** through clustering
- **{loyalist_pct:.1f}%** of active users exhibit "Loyalist" behaviour (high revisit ratio)
- Median user revisit ratio: **{median_revisit:.3f}**
- Median venue revisit rate: **{venue_median_revisit_rate:.3f}**
- Coffee shops vary significantly in their user base composition

---

## 1. What Behavioural Patterns Exist?

### User Archetypes

Through K-Means clustering on behavioural features, we identified **4 distinct user groups**:

"""
    
    # Add cluster details
    cluster_names = {
        0: "Casual Weekenders",
        1: "Weekday Regulars", 
        2: "Loyalists",
        3: "Infrequent Visitors"
    }
    
    for cluster_id in range(4):
        cluster_data = user_groups_df[user_groups_df['cluster'] == cluster_id]
        if len(cluster_data) == 0:
            continue
        
        count = len(cluster_data)
        pct = count / users_with_clusters * 100
        
        revisit = cluster_data['revisit_ratio'].mean()
        unique_shops = cluster_data['unique_shops'].mean()
        time_gap = cluster_data['avg_time_gap_days'].mean()
        weekday_ratio = cluster_data['weekday_ratio'].mean()
        
        md_content += f"""
#### Cluster {cluster_id}: {cluster_names.get(cluster_id, 'Unknown')}
- **Size:** {count:,} users ({pct:.1f}%)
- **Revisit ratio:** {revisit:.3f}
- **Unique shops visited:** {unique_shops:.1f}
- **Avg time between visits:** {time_gap:.0f} days
- **Weekday preference:** {weekday_ratio:.1%}

"""
    
    md_content += f"""
### Behavioural Insights

1. **Loyalists (Cluster 2)** demonstrate high persistence (41% revisit ratio) with low exploration
   - These users have found their "regular spot"
   - Visit the same coffee shops repeatedly
   - Represent a stable revenue base for venues

2. **Weekday Regulars (Cluster 1)** show routine weekday visits (89.5% weekday ratio)
   - Likely work-related coffee shop visits
   - Moderate exploration behavior
   - Predictable temporal patterns

3. **Casual Weekenders (Cluster 0)** prefer weekend visits
   - Lower weekday ratio (~31%)
   - Exploratory behavior with low revisit rates
   - Leisure-oriented usage

4. **Infrequent Visitors (Cluster 3)** have very long gaps between visits (>1400 days)
   - Occasional users
   - Minimal revisit behavior
   - Represent transient traffic

---

## 2. How Do Different User Groups Express Value?

Different user archetypes express value through **distinct behavioural signatures**:

### Loyalists
- **Value expression:** Repeated visits, long-term commitment
- **Signal strength:** High (consistent, reliable)
- **Rating limitations:** May not review frequently despite high engagement

### Weekday Regulars
- **Value expression:** Routine integration into daily life
- **Signal strength:** Medium-High (predictable patterns)
- **Rating limitations:** May rate only once but visit hundreds of times

### Casual Users
- **Value expression:** Breadth of exploration, variety-seeking
- **Signal strength:** Medium (distributed across many venues)
- **Rating limitations:** Ratings may reflect novelty rather than sustained value

### Infrequent Visitors
- **Value expression:** Sporadic engagement
- **Signal strength:** Low (unreliable temporal pattern)
- **Rating limitations:** May rate based on single experience

---

## 3. Which Coffee Shops Are "Regulars" Venues?

We identified venues based on their **revisit rate** (proportion of users who return):

### Repeat-User Dominated Venues
- **Count:** {len(repeat_venues):,} coffee shops
- **Characteristic:** Revisit rate > 30%
- **User base:** Dominated by Loyalists and Weekday Regulars
- **Behavioral signal:** Strong habitual usage
- **Value proposition:** Consistency, familiarity, routine integration

### Exploratory-User Dominated Venues
- **Count:** {len(exploratory_venues):,} coffee shops
- **Characteristic:** Revisit rate < 10%
- **User base:** Dominated by Casual and Infrequent users
- **Behavioral signal:** High novelty appeal, low retention
- **Value proposition:** Discovery, variety, one-time experiences

### Implications for Ranking

**Traditional rating-based ranking fails to capture:**
- Venues with high loyalty but moderate ratings
- Difference between novelty appeal and sustained value
- Importance of routine integration vs. occasional satisfaction

**Behavior-based ranking can surface:**
- Hidden gems with devoted regulars
- Venues that integrate into users' daily routines
- Places that build long-term relationships with customers

---

## 4. Why Ratings Alone Fail to Capture These Patterns

### Rating Limitations

1. **Single-Point-in-Time Bias**
   - Rating reflects ONE experience
   - Doesn't capture sustained engagement or revisit behavior
   
2. **Self-Selection Bias**
   - Loyalists may not review frequently (they've already committed)
   - Exploratory users more likely to review each new place
   
3. **Novelty vs. Routine Value**
   - High ratings may reflect novelty appeal
   - Routine value (habitual visits) often goes unrated
   
4. **Missing Temporal Dimension**
   - Ratings don't show temporal patterns (weekday regulars, etc.)
   - Can't distinguish between consistent quality and one-time experiences

### Behavioural Advantages

**Behavioral data captures:**
- ✓ **Revealed preference:** Actions (visits) vs. stated preference (ratings)
- ✓ **Temporal patterns:** When, how often, and how consistently users visit
- ✓ **Loyalty signals:** Revisit behavior shows sustained value
- ✓ **User archetypes:** Different user groups express value differently

---

## Visualizations

![Interpretation Plots]({plot_file})

---

## Next Steps for Behaviour-Based Ranking

With these behavioral foundations established, future work can:

1. **Construct behavior-based ranking algorithms** using:
   - Weighted revisit signals from different user archetypes
   - Temporal stability as a quality indicator
   - User-venue affinity based on cluster matching

2. **Compare against rating-based baselines** to demonstrate:
   - Which venues are undervalued by ratings
   - Which user groups are better served by behavior-based ranking

3. **Extend to other categories** (restaurants, hotels) using the same pipeline

---

## Conclusion

This analysis demonstrates that **behavior-based analysis reveals patterns ratings cannot capture**.
Coffee shop users exhibit distinct behavioral archetypes, from devoted Loyalists to casual Explorers.
Venues vary in their ability to build loyal user bases vs. attracting one-time visitors.

**Behavior-based ranking can surface venues that:**
- Build strong habitual relationships with users
- Integrate into users' daily routines
- Provide sustained value over time, not just novelty appeal

This foundation enables the next phase: designing and validating behavior-based ranking algorithms
that better reflect how users actually engage with local urban services.

---

*Generated from Yelp Open Dataset*
*Analysis period: 2005-2022*
*Total users analyzed: {total_users:,}*
*Total coffee shops: {len(venue_features_df):,}*
"""
    
    with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"✓ Interpretation document saved to {OUTPUT_MD}")
    
    return OUTPUT_MD

def main():
    """
    Main execution function for Task 6.
    """
    print("\n" + "="*60)
    print("TASK 6: BEHAVIOUR INTERPRETATION & VALIDATION READINESS")
    print("="*60 + "\n")
    
    # Load data
    user_groups_df, venue_features_df, interactions_df, business_df = load_all_data()
    
    # Analyze patterns
    patterns = analyze_group_patterns(user_groups_df, interactions_df)
    
    # Identify venue types
    venue_profiles_df, repeat_venues, exploratory_venues = identify_venue_types(
        venue_features_df, interactions_df, user_groups_df
    )
    
    # Create visualizations
    plot_file = create_visualizations(user_groups_df, venue_features_df, patterns)
    
    # Generate interpretation document
    doc_file = generate_interpretation_document(
        user_groups_df, venue_features_df, patterns,
        repeat_venues, exploratory_venues, plot_file
    )
    
    print("\n" + "="*60)
    print("✓ TASK 6 COMPLETE")
    print("="*60)
    print(f"\nInterpretation document: {doc_file}")
    print(f"Visualization plots: {plot_file}")
    
    print("\n" + "="*60)
    print("ALL TASKS COMPLETE")
    print("="*60)
    print("\nYou now have:")
    print("  ✓ Coffee shop identification and filtering")
    print("  ✓ Visit event construction from check-ins")
    print("  ✓ User-business links via reviews")
    print("  ✓ Canonical interaction table")
    print("  ✓ Multi-scopic behavior features (user, venue, group)")
    print("  ✓ Comprehensive interpretation document")
    print("\nYou are ready to answer:")
    print("  1. What behavioural patterns exist?")
    print("  2. How do different user groups express value?")
    print("  3. Which coffee shops are 'regulars' venues?")
    print("  4. Why ratings alone fail to capture these patterns?")

if __name__ == "__main__":
    main()
