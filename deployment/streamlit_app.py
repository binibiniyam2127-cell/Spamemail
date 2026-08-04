"""
Streamlit Dashboard with FastAPI Backend
Run: streamlit run deployment/streamlit_app.py
"""
import streamlit as st
import requests
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Get API URL from environment or use default
API_URL = os.environ.get('API_URL', 'http://localhost:8000')

st.set_page_config(
    page_title="Spam Detection Dashboard",
    page_icon="📧",
    layout="wide"
)

st.title("📧 Spam Detection Dashboard")
st.markdown("### Powered by FastAPI + Random Forest (97.34% Accuracy)")

# Sidebar - API Status
with st.sidebar:
    st.title("📊 Status")
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        if response.status_code == 200:
            st.success("✅ API Online")
            st.info(f"📡 {API_URL}")
        else:
            st.error("❌ API Offline")
    except:
        st.error("❌ API Not Reachable")
        st.warning("Start FastAPI: uvicorn deployment.fastapi_app:app --reload")
    
    st.markdown("---")
    st.caption("🚀 Deployed on Streamlit Cloud")

st.subheader("🔍 Check if an email is Spam or Ham")

# Quick examples
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📌 Spam"):
        st.session_state.email_input = "Congratulations! You won $1,000,000! Click here to claim!"
        st.rerun()
with col2:
    if st.button("📌 Ham"):
        st.session_state.email_input = "Hi, how are you doing today? Can we meet tomorrow?"
        st.rerun()
with col3:
    if st.button("📌 URGENT"):
        st.session_state.email_input = "URGENT: Your account has been compromised! Verify immediately!"
        st.rerun()

email_input = st.text_area(
    "Enter email content:",
    height=150,
    placeholder="Paste email text here...",
    key="email_input"
)

if st.button("🔍 Predict", type="primary"):
    if email_input:
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"email": email_input},
                    timeout=10
                )
                if response.status_code == 200:
                    result = response.json()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if result['prediction'] == 'spam':
                            st.error(f"⚠️ SPAM (Confidence: {result['confidence']:.2%})")
                        else:
                            st.success(f"✅ HAM (Confidence: {result['confidence']:.2%})")
                    
                    with col2:
                        st.metric("Spam Probability", f"{result['spam_probability']:.2%}")
                        st.progress(result['spam_probability'])
                else:
                    st.error(f"Error: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Cannot connect to API at {API_URL}")
                st.info("Start FastAPI: uvicorn deployment.fastapi_app:app --reload")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter some text")

st.markdown("---")
st.caption(f"🚀 API: {API_URL}")
