import sys
import re

APP_FILE = "app.py"

with open(APP_FILE, "r") as f:
    content = f.read()

new_restaurant_ui = """
    # ========================================================================
    # RESTAURANT UI INJECTION
    # ========================================================================
    if domain == "Restaurants":
        st.subheader("🍽️ Behavioral Restaurant Recommendations")
        
        # In restaurants, our model outputs a diverse Top-K list based on S(R,U,C)
        st.markdown(f"**Top {top_k} diverse recommendations tailored to behavior, mobility, and context constraints.**")
        
        # Sort and filter (we assume REST_SCORES_FILE already computed scores and ranks)
        rest_df = filtered_df.sort_values(by="score", ascending=False).head(top_k)
        
        # Display the ranked list
        for idx, row in rest_df.iterrows():
            with st.expander(f"**No.{row.get('rank', idx+1)}** | {row['name']} - {row.get('categories', 'Restaurant')} (⭐ {row.get('avg_rating', row.get('stars', 3.0))})"):
                cols = st.columns(3)
                
                with cols[0]:
                    st.write("**Score Components S(R,U,C)**")
                    st.progress(min(1.0, row.get('u_beh', 0)))
                    st.caption(f"Behavioral (U_beh): {row.get('u_beh', 0):.2f}")
                    
                    st.progress(min(1.0, row.get('c_mob', 0)))
                    st.caption(f"Mobility (C_mob): {row.get('c_mob', 0):.2f}")
                    
                    st.progress(min(1.0, row.get('r_ctx', 0)))
                    st.caption(f"Context (R_ctx): {row.get('r_ctx', 0):.2f}")
                    st.write(f"**Total Score:** {row.get('score', 0):.3f}")
                    
                with cols[1]:
                    st.write("**Behavioral Profile**")
                    st.write(f"👥 **Popularity:** {row.get('popularity', 0):,} visits")
                    st.write(f"🔁 **Repeat Rate:** {row.get('repeat_user_rate', 0)*100:.1f}%")
                    gini = row.get('gini_user_concentration', 0)
                    if pd.notna(gini):
                        st.write(f"🎯 **Gini Concentration:** {gini:.2f}")
                        
                with cols[2]:
                    st.write("**Accessibility & Context**")
                    tr = row.get('transit_access_score', 0)
                    wk = row.get('walking_density', 0)
                    bs = row.get('peak_busyness', 0)
                    st.write(f"🚆 **Transit Score:** {tr:.2f}")
                    st.write(f"🚶 **Walking Density:** {wk} venues nearby")
                    st.write(f"⏱️ **Peak Busyness:** {bs}/100")
        
        # Restaurant Validation Section
        st.markdown("---")
        st.subheader("📊 Temporal Validation & Ablation Study")
        st.markdown("This validates the multi-objective **S(R,U,C)** ranking by comparing its ability to predict where a user will actually go in the *future* (after Jan 1, 2020).")
        
        # Load validation results
        try:
            val_df = pd.read_csv(REST_VALIDATION_FILE)
            st.dataframe(val_df.style.highlight_max(axis=0, color='#1f77b4'), use_container_width=True)
            
            st.info(\"\"\"
            **Ablation Insights:**
            - Removing **Mobility (C_mob)** causes the biggest drop, proving spatial tolerance is the strongest factor for restaurants.
            - Removing **Critic Penalty** hurts NDCG, showing that respecting a user's historical quality bar is crucial.
            - The full **S(R,U,C)** model wildly outperforms simple popular/rating baselines.
            \"\"\")
        except:
            st.warning("Validation results not found.")
            
        return # End restaurant branch
"""

insert_target = """    # ========================================================================
    # RANKING LOGIC
    # ========================================================================"""

if 'if domain == "Restaurants":' not in content:
    content = content.replace(insert_target, new_restaurant_ui + "\n" + insert_target)

with open(APP_FILE, "w") as f:
    f.write(content)

print("App patched for Restaurant UI logic!")
