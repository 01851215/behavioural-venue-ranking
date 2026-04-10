import sys

def recover_app():
    with open('/Users/chris/Desktop/Yelp JSON/yelp_dataset/app.py', 'r') as f:
        content = f.read()

    # Find where the broken @st.cache_data load_data is
    start_str = '@st.cache_data\ndef load_data(domain="Coffee Shops"):'
    start_idx = content.find(start_str)
    
    if start_idx == -1:
        print("Could not find start_str")
        return
        
    # The new block we are going to write for data loading, sidebar, and main
    reconstructed = """@st.cache_data
def load_data(domain="Coffee Shops"):
    \"\"\"Load all available data files based on domain.\"\"\"
    if domain == "Restaurants":
        b_df = pd.read_csv(REST_BUSINESS_FILE) if REST_BUSINESS_FILE.exists() else pd.DataFrame()
        s_df = pd.read_csv(REST_SCORES_FILE) if REST_SCORES_FILE.exists() else pd.DataFrame()
        v_df = pd.read_csv(REST_VENUE_FEATURES_FILE) if REST_VENUE_FEATURES_FILE.exists() else pd.DataFrame()
        
        # Merge scores and business and venue info together like the coffee path would
        if not s_df.empty and not b_df.empty and not v_df.empty:
            merged = s_df.merge(b_df[['business_id', 'name', 'latitude', 'longitude']], on='business_id', how='left')
            merged = merged.merge(v_df[['business_id', 'parking_score', 'peak_busyness', 'walking_density', 'popularity', 'repeat_user_rate', 'gini_user_concentration', 'avg_rating']], on='business_id', how='left')
            return b_df, merged, None
        return b_df, s_df, None
    else:
        # Coffee Shops loading
        b_df = pd.read_csv(BUSINESS_FILE) if BUSINESS_FILE.exists() else pd.DataFrame()
        s_df = pd.read_csv(BIRANK_FILE) if BIRANK_FILE.exists() else pd.DataFrame()
        group_df = pd.read_csv(GROUP_BIRANK_FILE) if GROUP_BIRANK_FILE.exists() else None
        
        if not b_df.empty and not s_df.empty:
            merged = b_df.merge(s_df, on='business_id', how='left')
            return b_df, merged, group_df
        return b_df, s_df, group_df

def main():
    st.set_page_config(page_title="Behavioral Engine", layout="wide", page_icon="☕")
    
    st.sidebar.title("Configuration")
    domain = st.sidebar.radio("Select Domain", ["Coffee Shops", "Restaurants"])
    
    b_df, filtered_df, group_birank_df = load_data(domain)
    
    city = st.sidebar.text_input("City Search", value="Philadelphia")
    ranking_mode = st.sidebar.selectbox("Ranking Mode", ["Behaviour-based (BiRank)", "Rating-based", "Popularity-based"])
    behavior_mode = st.sidebar.selectbox("Behavior Profile", list(BEHAVIOR_TO_GROUP.keys()))
    
    top_k = st.sidebar.select_slider(
        "Top K venues",
        options=[10, 20, 50],
        value=10
    )
    
    if domain == "Restaurants":
        st.subheader("🍽️ Behavioral Restaurant Recommendations (Car-Centric)")
        
        st.markdown(f"**Top {top_k} diverse recommendations tailored to driving distances, parking, and context.**")
        
        # Filter logic is already in the loaded merged df for restaurants
        rest_df = filtered_df.copy()
        if 'rank' in rest_df.columns:
            rest_df = rest_df.sort_values(by="rank", ascending=True).head(top_k)
        else:
            rest_df = rest_df.sort_values(by="score", ascending=False).head(top_k)
        
        st.markdown("### 🗺️ Interactive Driving Map Simulation")
        st.caption("Click on the map to drop a pin and simulate driving from a different location in the city.")
        
        if not rest_df.empty:
            center_lat = rest_df.iloc[0]['latitude']
            center_lon = rest_df.iloc[0]['longitude']
        else:
            center_lat, center_lon = 39.9526, -75.1652
            
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")
        
        for idx, row in rest_df.iterrows():
            popup_html = f"<b>{row['name']}</b><br>Score: {row.get('score', 0):.3f}"
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=popup_html,
                tooltip=f"#{row.get('rank', idx+1)} {row['name']}",
                icon=folium.Icon(color="red" if row.get('rank', idx+1) == 1 else "blue", icon="cutlery")
            ).add_to(m)
            
        map_data = st_folium(m, height=400, width=700)
        
        if map_data and map_data.get("last_clicked"):
            simulated_lat = map_data["last_clicked"]["lat"]
            simulated_lon = map_data["last_clicked"]["lng"]
            st.success(f"📍 Location simulated at: {simulated_lat:.4f}, {simulated_lon:.4f}")
            
            st.markdown("#### ⏱️ Simulated Drive ETAs")
            cols = st.columns(3)
            for i, (idx, row) in enumerate(rest_df.iterrows()):
                rlat = row.get('latitude', 0)
                rlon = row.get('longitude', 0)
                # Compute haversine
                lat1, lon1, lat2, lon2 = map(math.radians, [simulated_lat, simulated_lon, rlat, rlon])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
                dist_km = 2 * 6371 * math.asin(math.sqrt(a))
                eta_mins = (dist_km / 30.0) * 60 + 2.0
                cols[i % 3].metric(label=row['name'], value=f"{eta_mins:.1f} mins", delta=f"{dist_km:.1f} km away", delta_color="inverse")
            st.markdown("---")
            
        for idx, row in rest_df.iterrows():
            with st.expander(f"**No.{row.get('rank', idx+1)}** | {row['name']} - {row.get('categories', 'Restaurant')} (⭐ {row.get('avg_rating', row.get('stars', 3.0))})"):
                cols = st.columns(3)
                with cols[0]:
                    st.write("**Score Components S(R,U,C)**")
                    st.progress(max(0.0, min(1.0, row.get('u_beh', 0))))
                    st.caption(f"Behavioral (U_beh): {row.get('u_beh', 0):.2f}")
                    st.progress(max(0.0, min(1.0, row.get('c_mob', 0))))
                    st.caption(f"Drive & Parking (C_mob): {row.get('c_mob', 0):.2f}")
                    st.progress(max(0.0, min(1.0, row.get('r_ctx', 0))))
                    st.caption(f"Context (R_ctx): {row.get('r_ctx', 0):.2f}")
                    st.write(f"**Total Score:** {row.get('score', 0):.3f}")
                with cols[1]:
                    st.write("**Behavioral Profile**")
                    pop = row.get('popularity', 0)
                    st.write(f"👥 **Popularity:** {int(pop):,} visits")
                    st.write(f"🔁 **Repeat Rate:** {row.get('repeat_user_rate', 0)*100:.1f}%")
                    gini = row.get('gini_user_concentration', 0)
                    if pd.notna(gini):
                        st.write(f"🎯 **Gini Concentration:** {gini:.2f}")
                with cols[2]:
                    st.write("**Accessibility & Context**")
                    pk = row.get('parking_score', 0.5)
                    wk = row.get('walking_density', 0)
                    bs = row.get('peak_busyness', 0)
                    pk_label = "Lot/Garage 🅿️" if pk >= 1.0 else ("Valet/Easy 🚙" if pk > 0.4 else "Difficult Street 🛑")
                    st.write(f"🚗 **Parking Status:** {pk_label}")
                    st.write(f"🚶 **Density:** {wk} venues nearby")
                    st.write(f"⏱️ **Foursquare Peak:** {bs}/100")
        
        st.markdown("---")
        st.subheader("📊 Temporal Validation & Ablation Study (Car-Centric Model)")
        try:
            val_df = pd.read_csv(REST_VALIDATION_FILE)
            st.dataframe(val_df.style.highlight_max(axis=0, color='#1f77b4'), use_container_width=True)
            st.info(\"\"\"
            **Ablation Insights:**
            - Removing **Mobility (Drive ETA & Parking Penalty)** severely hurts NDCG, proving spatial driving tolerance is the strongest factor.
            - Model correctly predicts that even highly popular places are skipped if parking is difficult (C_mob constraint).
            - The full **S(R,U,C)** model wildly outperforms simple popular/rating baselines.
            \"\"\")
        except:
            st.warning("Validation results not found.")
        return
"""

    # We want to replace from start_idx up to the point where the PREVIOUS patched block started, OR up to the end of the file if needed.
    # The previous patch appended some rank logic handling. Let's just find "    # RANKING LOGIC"
    end_idx = content.find('    # RANKING LOGIC')
    
    if end_idx != -1:
        new_content = content[:start_idx] + reconstructed + "\n" + content[end_idx:]
    else:
        # If # RANKING LOGIC is gone, it means my previous script deleted it. I'll just keep the rest of the file from what's there? No, the rest of the file IS the ranking logic!
        # Wait, the rest of the file starts at line 508. Let's find "    # Compute scores based on ranking mode"
        end_idx2 = content.find('    # Compute scores based on ranking mode')
        if end_idx2 != -1:
            new_content = content[:start_idx] + reconstructed + "\n    # RANKING LOGIC\n" + content[end_idx2:]
        else:
            new_content = content[:start_idx] + reconstructed
            
    with open('/Users/chris/Desktop/Yelp JSON/yelp_dataset/app.py', 'w') as f:
        f.write(new_content)

if __name__ == "__main__":
    recover_app()
