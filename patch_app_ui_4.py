import sys

def patch_app_ui():
    with open('/Users/chris/Desktop/Yelp JSON/yelp_dataset/app.py', 'r') as f:
        content = f.read()

    # The issue:
    #     if vec is not None and city_query:
    #         ...
    #         if suggestions:
    #             selected_city = st.sidebar.selectbox("Did you mean?", suggestions)
    #         else:
    #             selected_city = city_query
    #     else:
    #             selected_city = city_query
    # 
    # But occasionally due to indentation or unhandled None states, selected_city fails to set.
    # We'll just force it to properly initialize right before the vector check.

    target_block = """    vec, mat, c_list, aliases = get_city_index()
    
    city_query = st.sidebar.text_input("City Search", value="Philadelphia")
    selected_city = city_query  # FORCE INIT
    
    if vec is not None and city_query:
        q = city_query.lower().strip()"""
    
    content = content.replace('    vec, mat, c_list, aliases = get_city_index()\n    \n    city_query = st.sidebar.text_input("City Search", value="Philadelphia")\n    \n    if vec is not None and city_query:\n        q = city_query.lower().strip()', target_block)

    with open('/Users/chris/Desktop/Yelp JSON/yelp_dataset/app.py', 'w') as f:
        f.write(content)
    print("Patched selected_city.")

if __name__ == "__main__":
    patch_app_ui()
