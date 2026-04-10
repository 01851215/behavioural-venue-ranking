import sys

def patch_app_ui():
    with open('/Users/chris/Desktop/Yelp JSON/yelp_dataset/app.py', 'r') as f:
        content = f.read()

    # 1. Map Interaction & Sequence numbers
    old_map_logic_start = '        if not rest_df.empty:'
    old_map_logic_end = '                cols[i % 3].metric(label=row[\'name\'], value=f"{eta_mins:.1f} mins", delta=f"{dist_km:.1f} km away", delta_color="inverse")\n            st.markdown("---")'
    
    start_idx = content.find(old_map_logic_start)
    end_idx = content.find(old_map_logic_end)
    
    if start_idx == -1 or end_idx == -1:
        print("Could not find Map rendering block to replace!")
        return

    new_map_logic = """        if not rest_df.empty:
            center_lat = rest_df.iloc[0]['latitude']
            center_lon = rest_df.iloc[0]['longitude']
        else:
            center_lat, center_lon = 39.9526, -75.1652

        if "simulated_lat" not in st.session_state:
            st.session_state.simulated_lat = center_lat
            st.session_state.simulated_lon = center_lon
            
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")
        
        # Add simulated user pin and radius
        folium.Marker(
            [st.session_state.simulated_lat, st.session_state.simulated_lon],
            popup="📍 Simulated Start Location",
            tooltip="Your Location",
            icon=folium.Icon(color="black", icon="user")
        ).add_to(m)
        
        folium.Circle(
            location=[st.session_state.simulated_lat, st.session_state.simulated_lon],
            radius=5000, 
            color='#3186cc',
            fill=True,
            fill_color='#3186cc',
            popup="5 km Drive Radius"
        ).add_to(m)
        
        for i, (idx, row) in enumerate(rest_df.iterrows()):
            popup_html = f"<b>{row['name']}</b><br>Score: {row.get('score', 0):.3f}"
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=popup_html,
                tooltip=f"#{i+1} {row['name']}",
                icon=folium.Icon(color="red" if i == 0 else "blue", icon="cutlery")
            ).add_to(m)
            
        map_data = st_folium(m, height=400, width=700)
        
        if map_data and map_data.get("last_clicked"):
            lat = map_data["last_clicked"]["lat"]
            lng = map_data["last_clicked"]["lng"]
            # Only rerun if it actually changed to prevent infinite loops
            if abs(st.session_state.simulated_lat - lat) > 0.0001 or abs(st.session_state.simulated_lon - lng) > 0.0001:
                st.session_state.simulated_lat = lat
                st.session_state.simulated_lon = lng
                st.rerun()
            
        st.success(f"📍 Location simulated at: {st.session_state.simulated_lat:.4f}, {st.session_state.simulated_lon:.4f}")
        
        st.markdown("#### ⏱️ Simulated Drive ETAs")
        cols = st.columns(3)
        for i, (idx, row) in enumerate(rest_df.iterrows()):
            rlat = row.get('latitude', 0)
            rlon = row.get('longitude', 0)
            dist_km = haversine_km(st.session_state.simulated_lat, st.session_state.simulated_lon, rlat, rlon)
            eta_mins = (dist_km / 30.0) * 60 + 2.0
            cols[i % 3].metric(label=f"#{i+1} {row['name']}", value=f"{eta_mins:.1f} mins", delta=f"{dist_km:.1f} km away", delta_color="inverse")
        st.markdown("---")"""
    
    content = content[:start_idx] + new_map_logic + content[end_idx + len(old_map_logic_end):]
    
    # 2. Fix Numbering in the Expanders
    content = content.replace("for idx, row in rest_df.iterrows():\n            with st.expander(f\"**No.{row.get('rank', idx+1)}", 
                              "for i, (idx, row) in enumerate(rest_df.iterrows()):\n            with st.expander(f\"**No.{i+1}")

    # 3. Validation Explanation Inject
    validation_text = """        except:
            st.warning("Validation results not found.")
            
        st.markdown("---")
        st.markdown("##### 🔍 Why Use These Metrics?")
        with st.container(border=True):
            st.markdown("**NDCG (Normalized Discounted Cumulative Gain)**")
            st.markdown("Measures *ranking quality*. Did you put the absolute best restaurant at the very top? Score ranges from 0 to 1.")
            st.latex(r"DCG = \sum_{i=1}^{P} \\frac{rel_i}{\\log_2(i+1)}")
            
            st.markdown("**Hit@K**")
            st.markdown("Measures *recall*. Did the restaurant the user *actually visited* later appear anywhere in the Top K list? (0 = Miss, 1 = Hit).")
            
        return"""
        
    content = content.replace('        except:\n            st.warning("Validation results not found.")\n        return', validation_text)

    with open('/Users/chris/Desktop/Yelp JSON/yelp_dataset/app.py', 'w') as f:
        f.write(content)
    print("Patched map, numbering, and validations successfully.")

if __name__ == "__main__":
    patch_app_ui()
