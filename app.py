"""
Root entry point for Streamlit Cloud
"""
import streamlit as st
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the actual app
from deployment.streamlit_app import main as streamlit_main

if __name__ == "__main__":
    st.set_page_config(
        page_title="Spam Detection Dashboard",
        page_icon="📧",
        layout="wide"
    )
    
    # Import the app from deployment folder
    exec(open('deployment/streamlit_app.py').read())
