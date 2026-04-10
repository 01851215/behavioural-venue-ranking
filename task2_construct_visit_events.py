"""
Task 2: Construct Raw Visit Events (Behaviour Grounding)
Load check-in data, expand timestamps into individual visit events,
join with coffee shops, and create visualizations.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Configuration
DATA_DIR = Path(__file__).parent
CHECKIN_FILE = DATA_DIR / "yelp_academic_dataset_checkin.json"
COFFEE_BUSINESS_FILE = DATA_DIR / "business_coffee.csv"
OUTPUT_FILE = DATA_DIR / "coffee_visits_raw.csv"

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_checkin_data(file_path):
    """
    Load check-in data from JSON file.
    Each line contains business_id and a 'date' field with comma-separated timestamps.
    """
    print(f"Loading check-in data from {file_path}...")
    checkins = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                checkin = json.loads(line)
                checkins.append(checkin)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            
            # Progress indicator for large files
            if line_num % 100000 == 0:
                print(f"  Processed {line_num:,} lines...")
    
    print(f"Loaded {len(checkins)} check-in records")
    return checkins

def expand_checkin_timestamps(checkins):
    """
    Expand check-in records so each timestamp becomes a separate visit event.
    
    Input format:
        {business_id: "xyz", date: "2015-04-03 12:34:56, 2015-05-10 14:23:45, ..."}
    
    Output format:
        [{business_id: "xyz", visit_timestamp: "2015-04-03 12:34:56"},
         {business_id: "xyz", visit_timestamp: "2015-05-10 14:23:45"}, ...]
    """
    print("\nExpanding check-in timestamps into individual visit events...")
    visits = []
    
    for i, checkin in enumerate(checkins, 1):
        business_id = checkin.get('business_id')
        date_str = checkin.get('date', '')
        
        if not business_id or not date_str:
            continue
        
        # Split by comma to get individual timestamps
        timestamps = [ts.strip() for ts in date_str.split(',') if ts.strip()]
        
        # Create a visit record for each timestamp
        for timestamp in timestamps:
            visits.append({
                'business_id': business_id,
                'visit_timestamp': timestamp
            })
        
        # Progress indicator
        if i % 50000 == 0:
            print(f"  Expanded {i:,} records → {len(visits):,} visits so far...")
    
    print(f"Total visit events created: {len(visits):,}")
    return pd.DataFrame(visits)

def filter_coffee_visits(visits_df, coffee_businesses_file):
    """
    Join visits with coffee shop businesses to keep only coffee shop visits.
    """
    print(f"\nLoading coffee shop businesses from {coffee_businesses_file}...")
    coffee_df = pd.read_csv(coffee_businesses_file)
    print(f"Loaded {len(coffee_df)} coffee shops")
    
    print("\nFiltering visits to coffee shops only...")
    initial_visits = len(visits_df)
    
    # Join to keep only coffee shop visits
    # Use inner join on business_id
    coffee_visits_df = visits_df.merge(
        coffee_df[['business_id']],
        on='business_id',
        how='inner'
    )
    
    filtered_visits = len(coffee_visits_df)
    print(f"Visits before filtering: {initial_visits:,}")
    print(f"Coffee shop visits: {filtered_visits:,}")
    print(f"Filtered out: {initial_visits - filtered_visits:,} visits")
    
    return coffee_visits_df

def create_sanity_check_plots(visits_df):
    """
    Create visualization plots for sanity checking the visit data.
    """
    print("\nCreating sanity-check plots...")
    
    # Convert timestamps to datetime for temporal analysis
    print("  Parsing timestamps...")
    visits_df['datetime'] = pd.to_datetime(visits_df['visit_timestamp'], errors='coerce')
    
    # Remove any rows with invalid timestamps
    valid_timestamps = visits_df['datetime'].notna()
    visits_df = visits_df[valid_timestamps].copy()
    print(f"  Valid timestamps: {len(visits_df):,}")
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Distribution of visits per coffee shop
    print("  Plot 1: Distribution of visits per coffee shop...")
    visits_per_shop = visits_df.groupby('business_id').size()
    
    ax1 = axes[0]
    ax1.hist(visits_per_shop, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Number of Visits', fontsize=12)
    ax1.set_ylabel('Number of Coffee Shops', fontsize=12)
    ax1.set_title('Distribution of Visits per Coffee Shop', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')  # Log scale due to likely heavy tail
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: {visits_per_shop.mean():.1f}\nMedian: {visits_per_shop.median():.0f}\nMax: {visits_per_shop.max()}'
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    # Plot 2: Visits over time
    print("  Plot 2: Visits over time...")
    visits_df['year_month'] = visits_df['datetime'].dt.to_period('M')
    visits_over_time = visits_df.groupby('year_month').size()
    
    ax2 = axes[1]
    # Convert Period to timestamp for plotting
    time_index = visits_over_time.index.to_timestamp()
    ax2.plot(time_index, visits_over_time.values, linewidth=2, color='steelblue')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Number of Visits', fontsize=12)
    ax2.set_title('Coffee Shop Visits Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add temporal range text
    time_range = f"Range: {visits_df['datetime'].min().strftime('%Y-%m-%d')} to {visits_df['datetime'].max().strftime('%Y-%m-%d')}"
    ax2.text(0.05, 0.95, time_range, transform=ax2.transAxes,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
             fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    plot_file = DATA_DIR / "coffee_visits_plots.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Plots saved to {plot_file}")
    plt.close()
    
    return visits_per_shop

def print_statistics(visits_df, visits_per_shop):
    """
    Print detailed statistics about coffee shop visits.
    """
    print("\n" + "="*60)
    print("COFFEE SHOP VISIT STATISTICS")
    print("="*60)
    
    print(f"\nTotal visit events: {len(visits_df):,}")
    print(f"Unique coffee shops with visits: {visits_df['business_id'].nunique():,}")
    
    print(f"\nVisits per coffee shop:")
    print(f"  Mean: {visits_per_shop.mean():.2f}")
    print(f"  Median: {visits_per_shop.median():.0f}")
    print(f"  Std dev: {visits_per_shop.std():.2f}")
    print(f"  Min: {visits_per_shop.min()}")
    print(f"  Max: {visits_per_shop.max()}")
    
    # Top 10 most visited coffee shops
    if len(visits_per_shop) > 0:
        print(f"\nTop 10 most visited coffee shops:")
        top_shops = visits_per_shop.nlargest(10)
        for i, (business_id, count) in enumerate(top_shops.items(), 1):
            print(f"  {i}. {business_id}: {count:,} visits")
    
    # Temporal statistics
    if 'datetime' in visits_df.columns:
        print(f"\nTemporal range:")
        print(f"  Earliest visit: {visits_df['datetime'].min()}")
        print(f"  Latest visit: {visits_df['datetime'].max()}")
        print(f"  Time span: {(visits_df['datetime'].max() - visits_df['datetime'].min()).days} days")
    
    print("="*60)

def main():
    """
    Main execution function for Task 2.
    """
    print("\n" + "="*60)
    print("TASK 2: CONSTRUCT RAW VISIT EVENTS")
    print("="*60 + "\n")
    
    # Load check-in data
    checkins = load_checkin_data(CHECKIN_FILE)
    
    # Expand timestamps into individual visit events
    visits_df = expand_checkin_timestamps(checkins)
    
    # Filter to coffee shop visits only
    coffee_visits_df = filter_coffee_visits(visits_df, COFFEE_BUSINESS_FILE)
    
    # Save results
    print(f"\nSaving results to {OUTPUT_FILE}...")
    coffee_visits_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(coffee_visits_df):,} coffee shop visits")
    
    # Create sanity-check plots
    visits_per_shop = create_sanity_check_plots(coffee_visits_df)
    
    # Print statistics
    print_statistics(coffee_visits_df, visits_per_shop)
    
    print(f"\n✓ Task 2 complete. Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
