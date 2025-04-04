import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the API URL and key
API_URL = os.getenv('API_URL', 'http://localhost:8000')
API_KEY = os.getenv('API_KEY')

# Headers for API requests
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

# Set page config
st.set_page_config(
    page_title="ED Clean - Project Topics",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("📚 ED Clean - Project Topics")
st.markdown("""
This tool helps you explore common topics and questions for each project part.
Choose a project part below to see the most discussed topics.
""")

if not API_KEY:
    st.error("No API key configured. Please set the API_KEY environment variable.")
    st.stop()

# Project part selector
part = st.selectbox(
    "Select Project Part",
    options=range(1, 12),
    format_func=lambda x: f"Part {x}"
)

if st.button("Get Topics"):
    try:
        # Make API request with authentication
        response = requests.get(f"{API_URL}/project/part/{part}", headers=HEADERS)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()

        if data["total_threads"] == 0:
            st.warning(f"No threads found for Part {part}")
        else:
            st.success(f"Found {data['total_threads']} total threads for Part {part}")

            # Display topics
            for i, topic in enumerate(data["topics"], 1):
                with st.expander(f"Topic {i}: {topic['title']} ({topic['thread_count']} threads)"):
                    if topic["related_threads"]:
                        st.markdown("**Related Questions:**")
                        for related in topic["related_threads"]:
                            st.markdown(f"- {related}")
                    else:
                        st.info("No related questions found for this topic.")

    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add footer with API status
st.markdown("---")
try:
    status = requests.get(f"{API_URL}/sync/status", headers=HEADERS).json()
    st.markdown(f"""
    **API Status:**
    - Sync Service: {'🟢 Running' if status['running'] else '🔴 Stopped'}
    - Sync Interval: {status['sync_interval_minutes']} minutes
    - Auto-sync: {'❌ Disabled' if status['disabled'] else '✅ Enabled'}
    """)
except:
    st.markdown("**API Status:** 🔴 Unable to connect to API")
