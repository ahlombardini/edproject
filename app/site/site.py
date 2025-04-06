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
    page_title="ED Clean - Search",
    page_icon="🔍",
    layout="wide"
)

# Function to make API requests
def make_api_request(method, endpoint, **kwargs):
    """Make an authenticated request to the API."""
    url = f"{API_URL}/{endpoint.lstrip('/')}"
    try:
        response = requests.request(
            method,
            url,
            headers=HEADERS,
            **kwargs
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response and e.response.status_code == 403:
            st.error("Invalid API key or authentication failed")
            return None
        st.error(f"Error connecting to API: {str(e)}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Function to display thread cards
def display_thread_cards(threads, show_similarity=True):
    """Display thread results in card format."""
    if not threads:
        st.warning("No threads found.")
        return

    for i, thread in enumerate(threads, 1):
        # Create HTML card with styling
        html = f"""
        <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
        """

        # Add title with similarity if available
        if show_similarity and 'similarity' in thread:
            similarity = thread.get('similarity', 0) * 100
            html += f"<h3>{i}. {thread['title']} (Similarity: {similarity:.1f}%)</h3>"
        else:
            html += f"<h3>{i}. {thread['title']}</h3>"

        # Add metadata
        html += f"<p><strong>Thread ID:</strong> {thread['ed_thread_id']}</p>"
        if thread.get('category'):
            html += f"<p><strong>Category:</strong> {thread['category']}</p>"
        if thread.get('subcategory'):
            html += f"<p><strong>Subcategory:</strong> {thread['subcategory']}</p>"

        # Add link
        html += f"<p><strong>Link:</strong> <a href=\"https://eu.edstem.org/courses/1932/discussion/{thread['ed_thread_id']}\" target=\"_blank\">View on Ed</a></p>"
        html += "</div>"

        st.markdown(html, unsafe_allow_html=True)

# Title and description
st.title("🔍 ED Clean - Search")
st.markdown("""
This tool helps you search through the ED discussions to find answers to your questions.
""")

if not API_KEY:
    st.error("No API key configured. Please set the API_KEY environment variable.")
    st.stop()

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Search Questions", "Similar Threads", "Browse by Category"])

# Tab 1: Search Questions
with tab1:
    st.header("Search Questions")
    st.markdown("Enter your question to find similar threads in the ED database.")

    search_query = st.text_input("Your question:", key="search_input")
    search_limit = st.slider("Number of results:", min_value=1, max_value=15, value=5, key="search_limit")

    if st.button("Search", key="search_button"):
        if not search_query:
            st.warning("Please enter a question to search for.")
        else:
            try:
                similar_threads = make_api_request(
                    'POST',
                    '/search/input',
                    json={"text": search_query},
                    params={"limit": search_limit}
                )

                if similar_threads:
                    st.success(f"Found {len(similar_threads)} relevant questions")
                    # Create a container for results with some spacing
                    results_container = st.container()
                    with results_container:
                        display_thread_cards(similar_threads)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Tab 2: Similar Threads
with tab2:
    st.header("Find Similar Threads")
    st.markdown("Enter a thread ID to find similar discussions in the ED database.")

    thread_id = st.text_input("Thread ID:", key="thread_id_input")
    thread_limit = st.slider("Number of results:", min_value=1, max_value=15, value=5, key="thread_limit")

    if st.button("Find Similar", key="similar_button"):
        if not thread_id:
            st.warning("Please enter a thread ID.")
        else:
            try:
                similar_threads = make_api_request('GET', f'/search/similar/{thread_id}', params={"limit": thread_limit})

                if similar_threads:
                    st.success(f"Found {len(similar_threads)} similar threads")
                    # Create a container for results with some spacing
                    results_container = st.container()
                    with results_container:
                        display_thread_cards(similar_threads)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Tab 3: Browse by Category
with tab3:
    st.header("Browse by Category")
    st.markdown("Browse threads by category and subcategory.")

    # Category selection
    category_options = ["projet", "general", "private", "announcement"]
    selected_category = st.selectbox("Select Category:", options=category_options)

    # Subcategory input field
    subcategory = st.text_input("Subcategory (optional):", placeholder="E.g., Étape 1")

    # Results per page
    results_per_page = st.slider("Results per page:", min_value=5, max_value=50, value=20, key="category_limit")

    # Page number for pagination
    page = st.number_input("Page:", min_value=1, value=1, step=1)
    skip = (page - 1) * results_per_page

    if st.button("Browse", key="category_button"):
        try:
            # Build the query parameters
            params = {
                "limit": results_per_page,
                "skip": skip
            }

            if subcategory:
                params["subcategory"] = subcategory

            # Make the API request
            result = make_api_request(
                'GET',
                f'/threads/category/{selected_category}',
                params=params
            )

            if result and result.get("total_count", 0) > 0:
                threads = result.get("threads", [])
                total_count = result.get("total_count", 0)

                # Display results count and pagination info
                total_pages = (total_count + results_per_page - 1) // results_per_page
                st.success(f"Found {total_count} threads in category '{selected_category}'" +
                          (f" with subcategory '{subcategory}'" if subcategory else "") +
                          f" (Page {page} of {total_pages})")

                # Display pagination info
                st.markdown(f"Showing results {skip + 1}-{min(skip + results_per_page, total_count)} of {total_count}")

                # Create a container for results
                results_container = st.container()
                with results_container:
                    display_thread_cards(threads, show_similarity=False)

                # Add simple pagination buttons
                cols = st.columns(2)
                with cols[0]:
                    if page > 1:
                        st.markdown(f"← [Previous Page](?page={page-1})")
                with cols[1]:
                    if page < total_pages:
                        st.markdown(f"[Next Page](?page={page+1}) →")

            elif result:
                st.warning(f"No threads found for category '{selected_category}'" +
                          (f" with subcategory '{subcategory}'" if subcategory else ""))

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
