"""
Task 1: Identify and Scope Coffee Shops
Load business data, filter for coffee shops, and save results.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).parent
BUSINESS_FILE = DATA_DIR / "yelp_academic_dataset_business.json"
OUTPUT_FILE = DATA_DIR / "business_coffee.csv"

def load_business_data(file_path):
    """
    Load business data from JSON file.
    Each line is a separate JSON object.
    """
    print(f"Loading business data from {file_path}...")
    businesses = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                business = json.loads(line)
                businesses.append(business)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(businesses)} businesses")
    return pd.DataFrame(businesses)

def is_coffee_shop(categories_str):
    """
    Determine if a business is a coffee shop based on categories.
    
    Category Matching Logic:
    - Check if categories contain keywords: "Coffee", "Café", "Cafe", 
      "Coffeehouse", "Coffee & Tea", "Espresso"
    - Case-insensitive matching
    - Categories field may be None or a semicolon/comma-separated string
    
    Returns:
        bool: True if business is a coffee shop
    """
    if pd.isna(categories_str) or categories_str is None:
        return False
    
    # Convert to lowercase for case-insensitive matching
    categories_lower = str(categories_str).lower()
    
    # Define coffee-related keywords
    coffee_keywords = [
        'coffee',
        'café',
        'cafe',
        'coffeehouse',
        'espresso',
        'coffee & tea',
        'coffee roasteries'
    ]
    
    # Check if any keyword appears in categories
    return any(keyword in categories_lower for keyword in coffee_keywords)

def filter_coffee_shops(df):
    """
    Filter businesses to include only coffee shops.
    Extract relevant fields.
    """
    print("\nFiltering for coffee shops...")
    
    # Apply coffee shop filter
    df['is_coffee'] = df['categories'].apply(is_coffee_shop)
    coffee_df = df[df['is_coffee']].copy()
    
    # Extract relevant columns
    relevant_columns = ['business_id', 'name', 'categories', 'city', 'latitude', 'longitude']
    # Also include state and stars for additional context
    if 'state' in coffee_df.columns:
        relevant_columns.append('state')
    if 'stars' in coffee_df.columns:
        relevant_columns.append('stars')
    if 'review_count' in coffee_df.columns:
        relevant_columns.append('review_count')
    
    # Select available columns
    available_columns = [col for col in relevant_columns if col in coffee_df.columns]
    coffee_df = coffee_df[available_columns]
    
    print(f"Found {len(coffee_df)} coffee shops")
    return coffee_df

def print_statistics(df):
    """
    Print basic statistics about coffee shops.
    """
    print("\n" + "="*60)
    print("COFFEE SHOP STATISTICS")
    print("="*60)
    
    print(f"\nTotal number of coffee shops: {len(df)}")
    
    # Cities covered
    if 'city' in df.columns:
        unique_cities = df['city'].nunique()
        print(f"Number of unique cities: {unique_cities}")
        
        # Top 10 cities by coffee shop count
        print("\nTop 10 cities by coffee shop count:")
        top_cities = df['city'].value_counts().head(10)
        for city, count in top_cities.items():
            print(f"  {city}: {count}")
    
    # States covered (if available)
    if 'state' in df.columns:
        unique_states = df['state'].nunique()
        print(f"\nNumber of unique states: {unique_states}")
        print(f"States: {sorted(df['state'].unique())}")
    
    # Geographic distribution
    if 'latitude' in df.columns and 'longitude' in df.columns:
        print(f"\nGeographic range:")
        print(f"  Latitude: {df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
        print(f"  Longitude: {df['longitude'].min():.4f} to {df['longitude'].max():.4f}")
    
    # Rating statistics (if available)
    if 'stars' in df.columns:
        print(f"\nRating statistics:")
        print(f"  Mean stars: {df['stars'].mean():.2f}")
        print(f"  Median stars: {df['stars'].median():.2f}")
        print(f"  Std dev: {df['stars'].std():.2f}")
    
    # Review count statistics (if available)
    if 'review_count' in df.columns:
        print(f"\nReview count statistics:")
        print(f"  Mean reviews: {df['review_count'].mean():.1f}")
        print(f"  Median reviews: {df['review_count'].median():.1f}")
        print(f"  Total reviews: {df['review_count'].sum()}")
    
    # Sample categories
    print(f"\nSample category strings (first 5):")
    for i, cats in enumerate(df['categories'].head(), 1):
        print(f"  {i}. {cats}")
    
    print("="*60)

def main():
    """
    Main execution function for Task 1.
    """
    print("\n" + "="*60)
    print("TASK 1: IDENTIFY AND SCOPE COFFEE SHOPS")
    print("="*60 + "\n")
    
    # Load business data
    business_df = load_business_data(BUSINESS_FILE)
    
    # Filter for coffee shops
    coffee_df = filter_coffee_shops(business_df)
    
    # Save results
    print(f"\nSaving results to {OUTPUT_FILE}...")
    coffee_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(coffee_df)} coffee shops")
    
    # Print statistics
    print_statistics(coffee_df)
    
    print(f"\n✓ Task 1 complete. Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
