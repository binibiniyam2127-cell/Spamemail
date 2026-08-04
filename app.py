"""
Streamlit App Entry Point
This file redirects to deployment/streamlit_app.py
"""
import streamlit as st
import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the actual app
exec(open('deployment/streamlit_app.py').read())
