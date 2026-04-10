"""
Build City Search Index

Create TF-IDF-based RAG search index for fuzzy city matching.
Supports typos, abbreviations, and variations.

Input: business_coffee_v2.csv
Output: cities_index.pkl, city_aliases.json
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading data...")

# Load business data with city display names
business = pd.read_csv('business_coffee_v2.csv')
print(f"Loaded {len(business):,} businesses")

# ============================================================================
# Extract Unique Cities
# ============================================================================

print("\nExtracting unique cities...")

# Get unique city_state_country_display values
unique_cities = business['city_state_country_display'].dropna().unique()
city_list = sorted(unique_cities)

print(f"Found {len(city_list):,} unique cities")

# Display sample
print("\nSample cities:")
for city in city_list[:10]:
    print(f"  {city}")

# ============================================================================
# Build TF-IDF Index
# ============================================================================

print("\nBuilding TF-IDF search index...")

# Use character n-grams for fuzzy matching
vectorizer = TfidfVectorizer(
    analyzer='char_wb',  # Character n-grams with word boundaries
    ngram_range=(2, 4),   # 2-4 character n-grams
    lowercase=True,
    min_df=1
)

# Fit and transform city names
city_matrix = vectorizer.fit_transform(city_list)

print(f"TF-IDF matrix shape: {city_matrix.shape}")
print(f"Features (n-grams): {len(vectorizer.get_feature_names_out())}")

# ============================================================================
# Manual City Aliases
# ============================================================================

print("\nCreating city aliases...")

# Common abbreviations and variations
city_aliases = {
    # US cities
    "nyc": "New York",
    "new york city": "New York",
    "philly": "Philadelphia",
    "nola": "New Orleans",
    "sf": "San Francisco",
    "la": "Los Angeles",
    "vegas": "Las Vegas",
    "dc": "Washington", 
    "pdx": "Portland",
    "chi": "Chicago",
    "pgh": "Pittsburgh",
    "mpls": "Minneapolis",
    "stl": "Saint Louis",
    
    # Canadian cities
    "van": "Vancouver",
    "tor": "Toronto",
    "mtl": "Montreal",
    "ott": "Ottawa",
    "cal": "Calgary"
}

print(f"Added {len(city_aliases)} manual aliases")

# ============================================================================
# Test Search Function
# ============================================================================

def search_cities(query, vectorizer, city_matrix, city_list, top_n=5):
    """
    Search for cities matching the query.
    
    Args:
        query: Search string
        vectorizer: Fitted TfidfVectorizer
        city_matrix: TF-IDF matrix of city names
        city_list: List of city names
        top_n: Number of results to return
    
    Returns:
        List of (city_name, score) tuples
    """
    # Transform query
    query_vec = vectorizer.transform([query.lower()])
    
    # Compute cosine similarity
    scores = cosine_similarity(query_vec, city_matrix)[0]
    
    # Get top N indices
    top_indices = scores.argsort()[-top_n:][::-1]
    
    # Return results
    results = [(city_list[i], scores[i]) for i in top_indices]
    
    return results

print("\n--- Testing Search ---")

test_queries = [
    "philadelphia",
    "philly",
    "Philadelphi",  # typo
    "NYC",
    "newyork",  # no space
    "edmonton"
]

for query in test_queries:
    # Apply alias if exists
    search_query = city_aliases.get(query.lower(), query)
    
    results = search_cities(search_query, vectorizer, city_matrix, city_list, top_n=3)
    
    print(f"\nQuery: '{query}' (search: '{search_query}')")
    for city, score in results:
        print(f"  {score:.3f} - {city}")

# ============================================================================
# Save Outputs
# ============================================================================

print("\nSaving search index...")

# Save TF-IDF index as pickle
index_data = {
    'vectorizer': vectorizer,
    'city_matrix': city_matrix,
    'city_list': city_list
}

with open('cities_index.pkl', 'wb') as f:
    pickle.dump(index_data, f)

print("✓ Saved: cities_index.pkl")

# Save aliases as JSON
with open('city_aliases.json', 'w') as f:
    json.dump(city_aliases, f, indent=2)

print("✓ Saved: city_aliases.json")

print(f"\nDone!")
print(f"  Cities indexed: {len(city_list):,}")
print(f"  Aliases: {len(city_aliases)}")
