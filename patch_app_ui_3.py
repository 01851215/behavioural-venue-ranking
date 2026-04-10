import sys

def patch_app_ui():
    with open('/Users/chris/Desktop/Yelp JSON/yelp_dataset/app.py', 'r') as f:
        content = f.read()

    # 1. Missing haversine_km logic
    # During the sidebar patch, haversine_km was pushed into the sidebar but not the global scope
    # where the Map simulation loop expects it.
    
    hav_func = """
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * 6371 * math.asin(math.sqrt(a))
"""
    # Insert at top level if not already defined there outside of the sidebar patch
    if "def haversine_km(lat1, lon1, lat2, lon2):" not in content[:1000]:
        import_idx = content.find('def load_data(domain')
        content = content[:import_idx] + hav_func + "\n" + content[import_idx:]

    # 2. Key Error: distance_km for Coffee Shops
    # This happens if radius filtering doesn't run, but the columns still try to display it.
    # We will safely inject a default distance column.
    
    safety_block = """    # Sort and get top K
    if 'distance_km' not in filtered_df.columns:
        filtered_df['distance_km'] = 0.0

    top_k_df = filtered_df.nlargest(top_k, 'score').copy()"""
    
    content = content.replace("    # Sort and get top K\n    top_k_df = filtered_df.nlargest(top_k, 'score').copy()", safety_block)


    with open('/Users/chris/Desktop/Yelp JSON/yelp_dataset/app.py', 'w') as f:
        f.write(content)
    print("Patched distance_km and haversine_km.")


if __name__ == "__main__":
    patch_app_ui()
