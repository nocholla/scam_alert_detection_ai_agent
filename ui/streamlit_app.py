import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import requests
import pandas as pd
import json
import logging
import os
from src.data_loader import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()
api_endpoint = config.get('api_endpoint', 'https://api.scam-alert-detection.com')

# Initialize session state
if "setup_complete" not in st.session_state:
    st.session_state.setup_complete = False
if "profile" not in st.session_state:
    st.session_state.profile = {
        "userId": "",
        "firstName": "",
        "age": 18,
        "country": "",
        "subscribed": 0,
        "relationshipGoals": "",
        "aboutMe": ""
    }

# Setup phase
if not st.session_state.setup_complete:
    st.set_page_config(page_title="Scam Alert Detection", page_icon="🚩")
    st.title("🚩 Scam Alert Detection AI Agent")
    
    st.subheader("Enter Profile Information")
    st.session_state.profile["firstName"] = st.text_input("First Name", max_chars=40)
    st.session_state.profile["age"] = st.number_input("Age", min_value=18, max_value=100, value=18)
    st.session_state.profile["country"] = st.selectbox("Country", ["USA", "UK", "Ghana", "Kenya", "South Africa", "Tanzania"])
    st.session_state.profile["subscribed"] = 1 if st.checkbox("Subscribed") else 0
    st.session_state.profile["relationshipGoals"] = st.selectbox("Relationship Goals", ["Long-term relationship", "Casual", "Marriage", "Spiritual"])
    st.session_state.profile["aboutMe"] = st.text_area("About Me", max_chars=200)
    
    if st.button("Submit Profile"):
        st.session_state.profile["userId"] = f"user_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
        # Send profile to API Gateway
        try:
            response = requests.post(f"{api_endpoint}/preprocess", json=st.session_state.profile)
            if response.status_code == 200:
                st.session_state.setup_complete = True
                st.success("Profile submitted successfully! Processing...")
                st.rerun()
            else:
                st.error("Error submitting profile. Please try again.")
        except Exception as e:
            logger.error(f"Error submitting profile: {e}")
            st.error("Unable to connect to the service. Please try again.")

# Results phase
if st.session_state.setup_complete:
    st.title("Scam Alert Detection Results")
    
    # Fetch results from API Gateway
    try:
        response = requests.get(f"{api_endpoint}/results/{st.session_state.profile['userId']}")
        if response.status_code == 200:
            result = response.json()['result']
            st.write(f"**Anomaly Score**: {result['anomaly_score']:.2f}")
            st.write(f"**Explanation**: {result['explanation']}")
            
            if result['anomaly_score'] > 0.7:
                st.warning("🚩 This profile has a high likelihood of being a scam. Proceed with caution!")
            else:
                st.success("✅ This profile appears safe based on our analysis.")
        else:
            st.error("Error fetching results. Please try again.")
    except Exception as e:
        logger.error(f"Error fetching results: {e}")
        st.error("Unable to fetch results. Please try again.")
    
    if st.button("Restart"):
        st.session_state.setup_complete = False
        st.rerun()