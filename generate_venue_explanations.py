"""
Generate Venue Explanations

Create plain-language explanations for each venue based on their metrics.
NO algorithm jargon - use observable characteristics only.

Input: coffee_venue_features_v2.csv, coffee_baselines.csv
Output: venue_explanations.csv
"""

import pandas as pd
import numpy as np

print("Loading data...")

# Load venue features v2
venue_features = pd.read_csv('coffee_venue_features_v2.csv')
print(f"Loaded {len(venue_features):,} venue features")

# Load baselines for ratings
baselines = pd.read_csv('coffee_baselines.csv')
print(f"Loaded {len(baselines):,} baselines")

# Merge
data = venue_features.merge(baselines[['business_id', 'rating_mean']], 
                             on='business_id', how='left')

print(f"Merged: {len(data):,} venues")

# ============================================================================
# Generate Explanations
# ============================================================================

print("\nGenerating explanations...")

def generate_venue_explanation(row):
    """
    Generate up to 5 plain-language explanations for a venue.
    
    Rules (no algorithm jargon):
    1. High repeat_user_rate → "Many people come back repeatedly"
    2. Low stability_cv → "Visit activity is steady over time"
    3. Low gini → "Broad loyalty, not dominated by few users"
    4. High rating + high repeat → "Highly rated AND consistently chosen"
    5. Moderate rating + high repeat → "Not highest rated, but consistently chosen"
    6. Low seasonal_variance → "Popular year-round"
    7. Low top_user_share → "Healthy diverse patronage"
    """
    explanations = []
    
    repeat_rate = row.get('repeat_user_rate', 0)
    stability = row.get('stability_cv', 0)
    gini = row.get('gini_user_contribution', 0)
    rating = row.get('rating_mean', 0)
    seasonal_var = row.get('seasonal_variance', 0)
    top_user = row.get('top_user_share', 0)
    
    # Rule 1: High repeat user rate
    if repeat_rate > 0.10:
        explanations.append(f"Many people come back repeatedly ({repeat_rate*100:.0f}% return rate)")
    
    # Rule 2: Low stability CV (steady visits)
    if stability < 0.5 and not pd.isna(stability):
        explanations.append("Visit activity is steady over time")
    
    # Rule 3: Low Gini (broad loyalty)
    if gini < 0.7:
        explanations.append("Loyalty is broad across many users (not dominated by super-users)")
    
    # Rule 4 & 5: Rating + repeat combination
    if rating >= 4.0 and repeat_rate > 0.10:
        explanations.append("Highly rated AND consistently chosen by regulars")
    elif rating < 4.0 and repeat_rate > 0.15:
        explanations.append("Not the highest rated, but consistently chosen by regulars")
    
    # Rule 6: Low seasonal variance (year-round)
    if seasonal_var < 50 and not pd.isna(seasonal_var):
        explanations.append("Popular year-round, not just seasonal")
    
    # Rule 7: Low top user share (diverse)
    if top_user < 0.05:
        explanations.append("No single user dominates visits (healthy diverse patronage)")
    
    # Additional: High rating alone
    if rating >= 4.5 and "Highly rated" not in ' '.join(explanations):
        explanations.append(f"Excellent customer ratings ({rating:.1f}★)")
    
    # Pad to 5 explanations
    while len(explanations) < 5:
        explanations.append("")
    
    # Return exactly 5
    return explanations[:5]

# Generate explanations for all venues
explanations_list = []

for idx, row in data.iterrows():
    explanations = generate_venue_explanation(row)
    explanations_list.append({
        'business_id': row['business_id'],
        'explain_1': explanations[0],
        'explain_2': explanations[1],
        'explain_3': explanations[2],
        'explain_4': explanations[3],
        'explain_5': explanations[4]
    })

explanations_df = pd.DataFrame(explanations_list)

print(f"Generated explanations for {len(explanations_df):,} venues")

# ============================================================================
# Data Quality Check
# ============================================================================

print("\n--- Sample Explanations ---")
print(explanations_df.head(10).to_string(index=False))

# Count non-empty explanations
for i in range(1, 6):
    col = f'explain_{i}'
    non_empty = (explanations_df[col] != '').sum()
    print(f"\n{col}: {non_empty:,} venues ({non_empty/len(explanations_df)*100:.1f}%)")

# ============================================================================
# Save Output
# ============================================================================

output_file = 'venue_explanations.csv'
explanations_df.to_csv(output_file, index=False)

print(f"\n✓ Saved: {output_file}")
print(f"  Venues: {len(explanations_df):,}")

print("\nDone!")
