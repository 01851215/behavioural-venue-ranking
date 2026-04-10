"""
Task 4: Build the Canonical Interaction Table
Combine check-in and review data into a single unified interaction table.
This table is the foundation for all behaviour analysis.

Schema: user_id | business_id | timestamp | interaction_type
Where interaction_type ∈ {checkin, review}
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).parent
VISITS_FILE = DATA_DIR / "coffee_visits_raw.csv"
REVIEWS_FILE = DATA_DIR / "coffee_reviews.csv"
OUTPUT_FILE = DATA_DIR / "coffee_interactions.csv"

def load_visits(file_path):
    """
    Load visit (check-in) data.
    Note: Check-ins don't have user_id, only business_id and timestamp.
    """
    print(f"Loading visit data from {file_path}...")
    visits_df = pd.read_csv(file_path)
    print(f"Loaded {len(visits_df):,} visit events")
    print(f"Columns: {list(visits_df.columns)}")
    return visits_df

def load_reviews(file_path):
    """
    Load review data.
    Reviews have user_id, business_id, review_timestamp, stars.
    """
    print(f"\nLoading review data from {file_path}...")
    reviews_df = pd.read_csv(file_path)
    print(f"Loaded {len(reviews_df):,} reviews")
    print(f"Columns: {list(reviews_df.columns)}")
    return reviews_df

def create_canonical_interactions(visits_df, reviews_df):
    """
    Create canonical interaction table combining both data sources.
    
    Important Note:
    - Check-ins have business_id + timestamp, but NO user_id
    - Reviews have user_id + business_id + timestamp
    
    This creates an asymmetry:
    - Review interactions can link users to businesses
    - Check-in interactions show business activity but not specific users
    
    For user-level analysis, we can only use review data.
    For venue-level analysis, we can use both.
    
    We'll create the canonical table but clearly mark the interaction type.
    """
    print("\nCreating canonical interaction table...")
    
    # Prepare review interactions
    print("  Processing review interactions...")
    review_interactions = reviews_df[['user_id', 'business_id', 'review_timestamp']].copy()
    review_interactions.rename(columns={'review_timestamp': 'timestamp'}, inplace=True)
    review_interactions['interaction_type'] = 'review'
    print(f"    Created {len(review_interactions):,} review interactions")
    
    # Prepare check-in interactions
    # Note: Check-ins don't have user_id, so we set it to NaN
    print("  Processing check-in interactions...")
    checkin_interactions = visits_df[['business_id', 'visit_timestamp']].copy()
    checkin_interactions.rename(columns={'visit_timestamp': 'timestamp'}, inplace=True)
    checkin_interactions['user_id'] = np.nan  # No user data for check-ins
    checkin_interactions['interaction_type'] = 'checkin'
    # Reorder columns to match review_interactions
    checkin_interactions = checkin_interactions[['user_id', 'business_id', 'timestamp', 'interaction_type']]
    print(f"    Created {len(checkin_interactions):,} check-in interactions (without user_id)")
    
    # Combine both interaction types
    print("  Combining both interaction types...")
    all_interactions = pd.concat([review_interactions, checkin_interactions], ignore_index=True)
    
    # Sort by timestamp
    print("  Sorting by timestamp...")
    all_interactions['datetime'] = pd.to_datetime(all_interactions['timestamp'], errors='coerce')
    all_interactions.sort_values('datetime', inplace=True)
    all_interactions.drop(columns=['datetime'], inplace=True)
    
    print(f"\nCanonical interaction table created:")
    print(f"  Total interactions: {len(all_interactions):,}")
    print(f"  Review interactions (with user_id): {(all_interactions['interaction_type'] == 'review').sum():,}")
    print(f"  Check-in interactions (no user_id): {(all_interactions['interaction_type'] == 'checkin').sum():,}")
    print(f"  Interactions with user_id: {all_interactions['user_id'].notna().sum():,}")
    
    return all_interactions

def print_statistics(interactions_df):
    """
    Print statistics about the canonical interaction table.
    """
    print("\n" + "="*60)
    print("CANONICAL INTERACTION TABLE STATISTICS")
    print("="*60)
    
    print(f"\nTotal interactions: {len(interactions_df):,}")
    
    # Interaction type breakdown
    print(f"\nInteraction type breakdown:")
    type_counts = interactions_df['interaction_type'].value_counts()
    for itype, count in type_counts.items():
        pct = count / len(interactions_df) * 100
        print(f"  {itype}: {count:,} ({pct:.1f}%)")
    
    # User coverage (only for reviews)
    user_interactions = interactions_df[interactions_df['user_id'].notna()]
    print(f"\nUser-linked interactions:")
    print(f"  Total: {len(user_interactions):,}")
    print(f"  Unique users: {user_interactions['user_id'].nunique():,}")
    print(f"  Unique businesses: {user_interactions['business_id'].nunique():,}")
    
    # Overall business coverage
    print(f"\nOverall business coverage:")
    print(f"  Unique businesses: {interactions_df['business_id'].nunique():,}")
    
    # Temporal distribution
    interactions_df['datetime'] = pd.to_datetime(interactions_df['timestamp'], errors='coerce')
    valid_times = interactions_df['datetime'].notna()
    if valid_times.sum() > 0:
        print(f"\nTemporal range:")
        print(f"  Earliest: {interactions_df.loc[valid_times, 'datetime'].min()}")
        print(f"  Latest: {interactions_df.loc[valid_times, 'datetime'].max()}")
        print(f"  Span: {(interactions_df.loc[valid_times, 'datetime'].max() - interactions_df.loc[valid_times, 'datetime'].min()).days} days")
    
    print("="*60)
    
    print("\n" + "!"*60)
    print("IMPORTANT NOTE ON DATA STRUCTURE")
    print("!"*60)
    print("Check-ins lack user_id information.")
    print()
    print("Implications for analysis:")
    print("  ✓ User-level features: Use REVIEW data only")
    print("  ✓ Venue-level features: Can use BOTH check-ins + reviews")
    print("  ✓ User-business connections: Review data only")
    print()
    print("This is NOT a limitation for behaviour-based ranking.")
    print("Reviews provide rich user-venue interaction data.")
    print("!"*60)

def main():
    """
    Main execution function for Task 4.
    """
    print("\n" + "="*60)
    print("TASK 4: BUILD CANONICAL INTERACTION TABLE")
    print("="*60 + "\n")
    
    # Load data
    visits_df = load_visits(VISITS_FILE)
    reviews_df = load_reviews(REVIEWS_FILE)
    
    # Create canonical interaction table
    interactions_df = create_canonical_interactions(visits_df, reviews_df)
    
    # Save results
    print(f"\nSaving results to {OUTPUT_FILE}...")
    # Drop the temporary datetime column if it exists
    save_df = interactions_df[['user_id', 'business_id', 'timestamp', 'interaction_type']].copy()
    save_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(save_df):,} interactions")
    
    # Print statistics
    print_statistics(interactions_df)
    
    print(f"\n✓ Task 4 complete. Output saved to: {OUTPUT_FILE}")
    print("\nThis table is the foundation for all behaviour analysis.")

if __name__ == "__main__":
    main()
