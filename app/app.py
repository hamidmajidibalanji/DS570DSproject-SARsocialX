# Implementation of a minimal streamlit app to demonstrate the use of the LLM API.
import streamlit as st
import pandas as pd   

# Streamlit Title  
st.title("Search and Rescue (SAR) Social Media Analyzer")     


st.write("Project is under development. Please check back later for updates.")


try:
    df = pd.read_csv("data/raw/data.csv")
    st.write("Sample Data:")
    st.dataframe(df.head())       
    
except:
    st.warning("NO data found. Run data download first.")
    
    
       
   