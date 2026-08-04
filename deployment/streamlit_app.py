"""
Streamlit Dashboard for Spam Detection
Auto-detects local or deployed API
"""
import streamlit as st
import requests
import pandas as pd
import os
import json

st.set_page_config(
    page_title="Spam Detection Dashboard",
    page_icon="📧",
    layout="wide"
)

# Auto-detect API URL
def get_api_url():
    # Check if running on Render
    if os.environ.get('RENDER'):
        return os.environ.get('API_URL', 'https://spam-api.onrender.com')
    # Local development
    return 'http://localhost:8000'

API_URL = get_api_url()

st.markdown("""
<style>
    .stButton button { background-color: #ff4b4b; color: white; font-weight: bold; }
    .stButton button:hover { background-color: #ff6b6b; color: white; }
    .spam-box { background-color: #ffebee; padding: 20px; border-radius: 10px; border: 2px solid #ff1744; }
    .ham-box { background-color: #e8f5e9; padding: 20px; border-radius: 10px; border: 2px solid #00c853; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📊 Status")
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success(f"✅ API Online")
            st.info(f"🌐 {API_URL}")
        else:
            st.error("❌ API Offline")
    except:
        st.error("❌ API Not Reachable")
        st.warning("Deployed API may take a minute to start")
    
    st.markdown("---")
    st.caption("🚀 Deployed with Render")

# Main Content
st.title("📧 Spam Detection Dashboard")
st.markdown("### Powered by FastAPI + Random Forest (97.34% Accuracy)")

tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📊 Batch Analysis", "📈 Performance"])

with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📝 Enter Email Text")
        email_input = st.text_area(
            "Paste your email content here:",
            height=200,
            key="email_input"
        )
        predict_btn = st.button("🚀 Predict", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📌 Quick Examples")
        examples = {
            "Spam": "Congratulations! You've won $1,000,000! Click here!",
            "Ham": "Hi, how are you doing today?",
            "URGENT": "URGENT: Your account has been compromised!"
        }
        for label, text in examples.items():
            if st.button(label, use_container_width=True):
                st.session_state.email_input = text
                st.rerun()
    
    if predict_btn and st.session_state.email_input:
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"email": st.session_state.email_input}
                )
                if response.status_code == 200:
                    result = response.json()
                    
                    st.markdown("---")
                    st.subheader("📊 Prediction Result")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if result['prediction'] == 'spam':
                            st.error(f"⚠️ SPAM")
                        else:
                            st.success(f"✅ HAM")
                    with col2:
                        st.metric("Confidence", f"{result['confidence']:.2%}")
                    with col3:
                        if result.get('overridden', False):
                            st.info("📌 Rule Override")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Spam Prob.", f"{result['spam_probability']:.2%}")
                        st.progress(result['spam_probability'])
                    with col2:
                        st.metric("Ham Prob.", f"{result['ham_probability']:.2%}")
                        st.progress(result['ham_probability'])
                    
                    st.caption(f"Model: {result.get('model_used', 'Random Forest')}")
            except Exception as e:
                st.error(f"Connection error: {e}")

with tab2:
    st.subheader("📊 Batch Email Analysis")
    uploaded_file = st.file_uploader("Upload CSV with emails", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        
        if st.button("🔍 Analyze Batch", type="primary"):
            if 'text' in df.columns:
                with st.spinner("Analyzing..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/predict/batch",
                            json={"emails": df['text'].tolist()}
                        )
                        if response.status_code == 200:
                            result = response.json()
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📊 Total", result['total'])
                            with col2:
                                st.metric("🚨 Spam", result['spam_count'])
                            with col3:
                                st.metric("✅ Ham", result['ham_count'])
                            st.dataframe(pd.DataFrame(result['results']))
                    except Exception as e:
                        st.error(f"Error: {e}")

with tab3:
    st.subheader("📈 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Accuracy", "97.34%")
    with col2:
        st.metric("🚨 Spam Recall", "98.45%")
    with col3:
        st.metric("✅ Ham Recall", "96.19%")
    with col4:
        st.metric("📊 F1 Score", "97.32%")

st.markdown("---")
st.caption(f"🚀 API: {API_URL} | Built with FastAPI + Streamlit")
