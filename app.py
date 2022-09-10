import pandas as pd 
from polyfuzz import PolyFuzz
from polyfuzz.models import EditDistance
import streamlit as st
import os
import time

@st.cache
def convert_df(df):
# IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
    
page = st.sidebar.selectbox("Select a Page",
["Upload csv file" ])

st.sidebar.markdown("Email: e.maximilien.k@protonmail.com")
st.sidebar.markdown("Element ID: @maximilien:matrix.org")
if page == "Upload csv file":
    st.subheader("Name matching using polyfuzzy clustering algorithm")
    shopify_file = st.file_uploader('Select csv file (default provided)')
    if shopify_file is not None:
        shopify_df = pd.read_csv(shopify_file)
        st.write('csv file loaded successfully')
        st.subheader("Displaying 10 samples of dataset")
        st.write(shopify_df.sample(10))
        st.write('Shape of the dataframe')
        st.write(shopify_df.shape)
        st.write("Displaying clean activity name without NaN value")
        test_clean = shopify_df[~shopify_df['name'].isnull()]
        st.write(test_clean)
        loaded_model = PolyFuzz.load("my_model")
        loaded_model.group(link_min_similarity=0.75)
        matches= loaded_model.get_matches()
        st.subheader("Displaying the coefficient of similarity and the grouping of match word")
        st.write(matches)
        loaded_model.get_cluster_mappings()
        cluster_prediction= pd.DataFrame.from_dict(loaded_model.get_cluster_mappings(),orient='index').reset_index()
        cluster_prediction.columns = ['Group of activities', 'Cluster']
        st.subheader("Displaying the clustered activity")
        st.write(cluster_prediction)
        st.subheader("Click on the button to download the clustered prediction in csv format ")
        csv = convert_df(cluster_prediction)
        st.download_button(label="Download cluster prediction as CSV",data=csv,file_name='Cluster_prediction.csv',mime='text/csv')
    else:
        st.balloons()
        st.progress(80)
        with st.spinner('Wait for it...'):
            time.sleep(10)
        st.warning("Warning. Waiting for csv file")
        st.stop()
