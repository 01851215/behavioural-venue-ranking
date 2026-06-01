"""
Add Country Display to Business Data

Infer country from state and create city_state_country_display field.

Rules:
- US states → "United States"  
- Canadian provinces → "Canada"
- Otherwise → "Unknown"

Input: business_coffee.csv
Output: business_coffee_v2.csv
"""

import pandas as pd

print("Loading data...")

# Load business data
business = pd.read_csv('business_coffee.csv')
print(f"Loaded {len(business):,} coffee businesses")

# ============================================================================
# State-to-Country Mappings
# ============================================================================

# US States (abbreviations and full names)
US_STATES = {
    # Abbreviations
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
    # Full names
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
    'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
    'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
    'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
    'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
    'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
    'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
    'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
    'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
    'West Virginia', 'Wisconsin', 'Wyoming'
}

# Canadian Provinces (abbreviations and full names)
CA_PROVINCES = {
    # Abbreviations
    'AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'PE', 'QC', 'SK', 'YT',
    # Full names
    'Alberta', 'British Columbia', 'Manitoba', 'New Brunswick',
    'Newfoundland and Labrador', 'Nova Scotia', 'Northwest Territories',
    'Nunavut', 'Ontario', 'Prince Edward Island', 'Quebec', 'Saskatchewan', 'Yukon'
}

# ============================================================================
# Infer Country Function
# ============================================================================

def infer_country(state):
    """Infer country from state field."""
    if pd.isna(state):
        return 'Unknown'
    
    state_str = str(state).strip()
    
    if state_str in US_STATES:
        return 'United States'
    elif state_str in CA_PROVINCES:
        return 'Canada'
    else:
        return 'Unknown'

# ============================================================================
# Apply Country Inference
# ============================================================================

print("\nInferring countries...")

business['country'] = business['state'].apply(infer_country)

# Count by country
country_counts = business['country'].value_counts()
print(f"\nCountry distribution:")
for country, count in country_counts.items():
    print(f"  {country}: {count:,} ({count/len(business)*100:.1f}%)")

# ============================================================================
# Create Display Field
# ============================================================================

print("\nCreating city_state_country_display field...")

def create_display_name(row):
    """Create display name in format: City, State, Country"""
    city = row.get('city', 'Unknown')
    state = row.get('state', '')
    country = row.get('country', 'Unknown')
    
    if pd.notna(state) and state != '':
        return f"{city}, {state}, {country}"
    else:
        return f"{city}, {country}"

business['city_state_country_display'] = business.apply(create_display_name, axis=1)

# Display sample
print("\n--- Sample Display Names ---")
print(business[['city', 'state', 'country', 'city_state_country_display']].head(10).to_string(index=False))

# ============================================================================
# Save Output
# ============================================================================

output_file = 'business_coffee_v2.csv'
business.to_csv(output_file, index=False)

print(f"\n✓ Saved: {output_file}")
print(f"  Businesses: {len(business):,}")
print(f"  New columns: country, city_state_country_display")

print("\nDone!")
