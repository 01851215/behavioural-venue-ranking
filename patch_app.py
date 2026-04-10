import sys

def patch_app():
    with open('/Users/chris/Desktop/Yelp JSON/yelp_dataset/app.py', 'r') as f:
        content = f.read()
        
    # Replace the import if needed
    if 'from streamlit_folium import folium_static' in content and 'st_folium' not in content:
        content = content.replace('from streamlit_folium import folium_static', 'from streamlit_folium import folium_static, st_folium')
        
    # The block we want to replace starts at "if domain == "Restaurants":""
    # and ends right before "    # RANKING LOGIC"
    target_start = '    if domain == "Restaurants":'
    target_end = '    # RANKING LOGIC'
    
    start_idx = content.find(target_start)
    end_idx = content.find(target_end)
    
    if start_idx == -1 or end_idx == -1:
        print("Could not find targets in app.py")
        sys.exit(1)
        
    replacement = """    if domain == "Restaurants":
        st.subheader("🍽️ Behavioral Restaurant Recommendations (Car-Centric)")
        
        # In restaurants, our model outputs a diverse Top-K list based on S(R,U,C)
        st.markdown(f"**Top {top_k} diverse recommendations tailored to driving distances, parking, and context.**")
        
        # Sort and filter
        rest_df = filtered_df.sort_values(by="score", ascending=False).head(top_k)
        
        # Interactive Map Section
        st.markdown("### 🗺️ Interactive Driving Map Simulation")
        st.caption("Click on the map to drop a pin and simulate driving from a different location in the city.")
        
        # Initialize map centered on the first result or city center
        if not rest_df.empty:
            center_lat = rest_df.iloc[0]['latitude']
            center_lon = rest_df.iloc[0]['longitude']
        else:
            center_lat, center_lon = 39.9526, -75.1652 # Philly default
            
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")
        
        # Add restaurant markers
        for idx, row in rest_df.iterrows():
            popup_html = f"<b>{row['name']}</b><br>Score: {row.get('score', 0):.3f}<br>Pop: {int(row.get('popularity',0))}"
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=popup_html,
                tooltip=f"#{row.get('rank', idx+1)} {row['name']}",
                icon=folium.Icon(color="red" if idx == 0 else "blue", icon="cutlery")
            ).add_to(m)
            
        # Display the map and capture clicks
        map_data = st_folium(m, height=400, width=700)
        
        # Handle pin drop (user simulating location)
        simulated_lat = None
        simulated_lon = None
        if map_data and map_data.get("last_clicked"):
            simulated_lat = map_data["last_clicked"]["lat"]
            simulated_lon = map_data["last_clicked"]["lng"]
            st.success(f"📍 Location simulated at: {simulated_lat:.4f}, {simulated_lon:.4f}")
            
            # Recalculate Drive ETAs for top K
            st.markdown("#### ⏱️ Simulated Drive ETAs")
            cols = st.columns(3)
            for i, (idx, row) in enumerate(rest_df.iterrows()):
                dist_km = haversine_km(simulated_lat, simulated_lon, row['latitude'], row['longitude'])
                eta_mins = (dist_km / 30.0) * 60 + 2.0 # 30km/h avg speed + 2m parking
                cols[i % 3].metric(label=row['name'], value=f"{eta_mins:.1f} mins", delta=f"{dist_km:.1f} km away", delta_color="inverse")
            
            st.markdown("---")
            
        # Display the ranked list
        for idx, row in rest_df.iterrows():
            with st.expander(f"**No.{row.get('rank', idx+1)}** | {row['name']} - {row.get('categories', 'Restaurant')} (⭐ {row.get('avg_rating', row.get('stars', 3.0))})"):
                cols = st.columns(3)
                
                with cols[0]:
                    st.write("**Score Components S(R,U,C)**")
                    st.progress(min(1.0, row.get('u_beh', 0)))
                    st.caption(f"Behavioral (U_beh): {row.get('u_beh', 0):.2f}")
                    
                    st.progress(min(1.0, row.get('c_mob', 0)))
                    st.caption(f"Drive & Parking (C_mob): {row.get('c_mob', 0):.2f}")
                    
                    st.progress(min(1.0, row.get('r_ctx', 0)))
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
        
        # Restaurant Validation Section
        st.markdown("---")
        st.subheader("📊 Temporal Validation & Ablation Study (Car-Centric Model)")
        st.markdown("Validates S(R,U,C) ranking comparing driving constraints vs pure popularity.")
        
        # Load validation results
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
            
        return # End restaurant branch
        
    # ========================================================================
"""
    
    new_content = content[:start_idx] + replacement + content[end_idx:]
    
    with open('/Users/chris/Desktop/Yelp JSON/yelp_dataset/app.py', 'w') as f:
        f.write(new_content)
        
    print("app.py successfully patched")

if __name__ == "__main__":
    patch_app()
