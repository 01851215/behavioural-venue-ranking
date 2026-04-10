import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static, st_folium
import math
from math import radians, cos, sin, asin, sqrt
import pickle
import json
import base64
import os
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import datetime

def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371
    return c * r

def patch_app_ui():
    with open('/Users/chris/Desktop/Yelp JSON/yelp_dataset/app.py', 'r') as f:
        content = f.read()

    # Find where to insert the new logic
    # Right after b_df, filtered_df, group_birank_df = load_data(domain)
    target = '    b_df, filtered_df, group_birank_df = load_data(domain)'
    
    if target not in content:
        print("Target not found.")
        return
        
    sidebar_logic = """
    # ========================================================================
    # CITY SEARCH (RAG-STYLE)
    # ========================================================================
    @st.cache_resource
    def get_city_index():
        try:
            with open(CITIES_INDEX_FILE, 'rb') as f:
                vec, mat, c_list = pickle.load(f)
            aliases = {}
            if CITY_ALIASES_FILE.exists():
                with open(CITY_ALIASES_FILE, 'r') as f:
                    aliases = json.load(f)
            return vec, mat, c_list, aliases
        except Exception:
            return None, None, None, {}

    vec, mat, c_list, aliases = get_city_index()
    
    city_query = st.sidebar.text_input("City Search", value="Philadelphia")
    
    if vec is not None and city_query:
        q = city_query.lower().strip()
        q = aliases.get(q, q)
        q_vec = vec.transform([q])
        sims = cosine_similarity(q_vec, mat).flatten()
        top_indices = sims.argsort()[-5:][::-1]
        suggestions = [c_list[i] for i in top_indices if sims[i] > 0.0]
        
        if suggestions:
            selected_city = st.sidebar.selectbox("Did you mean?", suggestions)
        else:
            selected_city = city_query
    else:
        selected_city = city_query
        
    # Filter DataFrame by city
    if 'city' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['city'].str.contains(selected_city, case=False, na=False)].copy()
        
    ranking_mode = st.sidebar.selectbox("Ranking Mode", ["Behaviour-based (BiRank)", "Rating-based", "Popularity-based", "Revisit-rate-based"])
    behavior_mode = st.sidebar.selectbox("Behavior Profile", list(BEHAVIOR_TO_GROUP.keys()))
    
    top_k = st.sidebar.select_slider(
        "Top K venues",
        options=[10, 20, 50],
        value=10
    )
    
    # Distance/Radius logic for Coffee Shops ONLY (Restaurants handle own map)
    ref_lat, ref_lon = 39.9526, -75.1652
    ref_cafe_name = ""
    radius_km = 10.0
    
    if domain == "Coffee Shops" and not filtered_df.empty:
        cafes_in_city = filtered_df['name'].tolist()
        if cafes_in_city:
            ref_cafe_name = st.sidebar.selectbox("Select a reference café", cafes_in_city)
            ref_row = filtered_df[filtered_df['name'] == ref_cafe_name].iloc[0]
            ref_lat = ref_row['latitude']
            ref_lon = ref_row['longitude']
            
            radius_km = st.sidebar.slider("Search radius (km)", 1.0, 50.0, 10.0, 1.0)
            
            # Compute distance
            coords = filtered_df[['latitude', 'longitude']].values
            dists = []
            for lat, lon in coords:
                dists.append(haversine_km(ref_lat, ref_lon, lat, lon))
            filtered_df['distance_km'] = dists
            
            # Filter by radius
            filtered_df = filtered_df[filtered_df['distance_km'] <= radius_km].copy()
    """

    # We need to replace the old sidebar block starting with city = st.sidebar.text_input
    # up to right before "    if domain == "Restaurants":"
    
    start_replace = '    city = st.sidebar.text_input("City Search", value="Philadelphia")'
    end_replace = '    if domain == "Restaurants":'
    
    s_idx = content.find(start_replace)
    e_idx = content.find(end_replace)
    
    if s_idx != -1 and e_idx != -1:
        new_content = content[:s_idx] + sidebar_logic.strip() + "\n\n" + content[e_idx:]
        
        # Also need to add the haversine_km helper to app.py if it's not there
        if 'def haversine_km' not in new_content:
            hav_func = """
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * 6371 * math.asin(math.sqrt(a))

"""
            import_idx = new_content.find('def load_data')
            new_content = new_content[:import_idx] + hav_func + new_content[import_idx:]
            
        with open('/Users/chris/Desktop/Yelp JSON/yelp_dataset/app.py', 'w') as f:
            f.write(new_content)
        print("Patched app.py sidebar!")
    else:
        print("Couldn't find replacement boundaries.")

if __name__ == "__main__":
    patch_app_ui()
