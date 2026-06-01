"""
Task 3: Link Users via Reviews (Identity & Baseline Only)
Load review data, filter for coffee shops, extract user-business connections.
Reviews will be used ONLY for user association and rating baselines, NOT for ranking.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).parent
REVIEW_FILE = DATA_DIR / "yelp_academic_dataset_review.json"
COFFEE_BUSINESS_FILE = DATA_DIR / "business_coffee.csv"
OUTPUT_FILE = DATA_DIR / "coffee_reviews.csv"

def load_coffee_businesses(file_path):
    """
    Load the set of coffee shop business IDs for filtering.
    """
    print(f"Loading coffee shop businesses from {file_path}...")
    coffee_df = pd.read_csv(file_path)
    coffee_ids = set(coffee_df['business_id'])
    print(f"Loaded {len(coffee_ids)} coffee shop business IDs")
    return coffee_ids

def load_and_filter_reviews(review_file, coffee_business_ids):
    """
    Load review data and filter for coffee shop reviews only.
    Extract: user_id, business_id, review_timestamp (from 'date'), stars.
    
    Note: Large file processing - will iterate line by line.
    """
    print(f"\nLoading and filtering reviews from {review_file}...")
    print("This may take several minutes due to the large file size...")
    
    coffee_reviews = []
    total_reviews = 0
    
    with open(review_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                review = json.loads(line)
                total_reviews += 1
                
                # Check if this review is for a coffee shop
                business_id = review.get('business_id')
                if business_id in coffee_business_ids:
                    # Extract relevant fields
                    coffee_reviews.append({
                        'user_id': review.get('user_id'),
                        'business_id': business_id,
                        'review_timestamp': review.get('date'),
                        'stars': review.get('stars')
                    })
                
                # Progress indicator for large files
                if line_num % 500000 == 0:
                    print(f"  Processed {line_num:,} reviews → {len(coffee_reviews):,} coffee shop reviews...")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    print(f"\nTotal reviews processed: {total_reviews:,}")
    print(f"Coffee shop reviews found: {len(coffee_reviews):,}")
    print(f"Filter rate: {len(coffee_reviews)/total_reviews*100:.2f}%")
    
    return pd.DataFrame(coffee_reviews)

def print_statistics(reviews_df):
    """
    Print detailed statistics about coffee shop reviews.
    """
    print("\n" + "="*60)
    print("COFFEE SHOP REVIEW STATISTICS")
    print("="*60)
    
    print(f"\nTotal coffee shop reviews: {len(reviews_df):,}")
    print(f"Unique users: {reviews_df['user_id'].nunique():,}")
    print(f"Unique coffee shops: {reviews_df['business_id'].nunique():,}")
    
    # Reviews per user
    reviews_per_user = reviews_df.groupby('user_id').size()
    print(f"\nReviews per user:")
    print(f"  Mean: {reviews_per_user.mean():.2f}")
    print(f"  Median: {reviews_per_user.median():.0f}")
    print(f"  Max: {reviews_per_user.max()}")
    
    # Reviews per coffee shop
    reviews_per_shop = reviews_df.groupby('business_id').size()
    print(f"\nReviews per coffee shop:")
    print(f"  Mean: {reviews_per_shop.mean():.2f}")
    print(f"  Median: {reviews_per_shop.median():.0f}")
    print(f"  Max: {reviews_per_shop.max()}")
    
    # Star rating distribution
    print(f"\nStar rating distribution:")
    star_counts = reviews_df['stars'].value_counts().sort_index()
    for stars, count in star_counts.items():
        pct = count / len(reviews_df) * 100
        print(f"  {stars} stars: {count:,} ({pct:.1f}%)")
    
    print(f"\nRating statistics:")
    print(f"  Mean rating: {reviews_df['stars'].mean():.2f}")
    print(f"  Median rating: {reviews_df['stars'].median():.1f}")
    print(f"  Std dev: {reviews_df['stars'].std():.2f}")
    
    # Temporal distribution
    if 'review_timestamp' in reviews_df.columns:
        reviews_df['datetime'] = pd.to_datetime(reviews_df['review_timestamp'], errors='coerce')
        valid_dates = reviews_df['datetime'].notna()
        if valid_dates.sum() > 0:
            print(f"\nTemporal range:")
            print(f"  Earliest review: {reviews_df.loc[valid_dates, 'datetime'].min()}")
            print(f"  Latest review: {reviews_df.loc[valid_dates, 'datetime'].max()}")
            print(f"  Time span: {(reviews_df.loc[valid_dates, 'datetime'].max() - reviews_df.loc[valid_dates, 'datetime'].min()).days} days")
    
    # Top reviewers (users with most coffee shop reviews)
    print(f"\nTop 10 users by number of coffee shop reviews:")
    top_users = reviews_per_user.nlargest(10)
    for i, (user_id, count) in enumerate(top_users.items(), 1):
        print(f"  {i}. {user_id}: {count} reviews")
    
    print("="*60)

def main():
    """
    Main execution function for Task 3.
    """
    print("\n" + "="*60)
    print("TASK 3: LINK USERS VIA REVIEWS")
    print("="*60 + "\n")
    
    # Load coffee shop business IDs
    coffee_business_ids = load_coffee_businesses(COFFEE_BUSINESS_FILE)
    
    # Load and filter reviews
    reviews_df = load_and_filter_reviews(REVIEW_FILE, coffee_business_ids)
    
    # Save results
    print(f"\nSaving results to {OUTPUT_FILE}...")
    reviews_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(reviews_df):,} coffee shop reviews")
    
    # Print statistics
    print_statistics(reviews_df)
    
    print(f"\n✓ Task 3 complete. Output saved to: {OUTPUT_FILE}")
    print("\nNote: Reviews are used ONLY for:")
    print("  1. Associating users with coffee shops")
    print("  2. Providing rating-based baselines for comparison")
    print("  3. They will NOT be used for behaviour-based ranking")

if __name__ == "__main__":
    main()
