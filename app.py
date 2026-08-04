"""
Spam Detection App - Entry Point for Streamlit Cloud
"""
import streamlit as st
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the actual app
from deployment.streamlit_app import main

if __name__ == "__main__":
    main()
