import streamlit as st
import requests
import os
from dotenv import load_dotenv
from visualization import generate_visualization, visualize_search_results
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from collections import Counter

# Load environment variables
load_dotenv()

# Configure the API URL and key
API_URL = os.getenv('API_URL', 'http://localhost:8000')
API_KEY = os.getenv('API_KEY')

# Headers for API requests
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

# Set page config and title
st.set_page_config(
    page_title="Ed-Finder",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 0;
        color: #1E88E5;
    }
    .subtitle {
        font-size: 1.2em;
        color: #666;
        margin-bottom: 2em;
    }
    .beta-tag {
        background-color: #FF9800;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: bold;
        margin-left: 6px;
        vertical-align: middle;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<p class="main-title">Ed-Finder 🔍</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your Personal Assistant for Finding Ed Threads</p>', unsafe_allow_html=True)

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
        elif hasattr(e, 'response') and e.response and e.response.status_code == 404:
            # More specific error for 404s that includes the endpoint that wasn't found
            st.error(f"API endpoint not found: {endpoint} - The requested feature may not be deployed yet.")
            return None
        else:
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

def create_category_chart(threads):
    """Create a pie chart showing the distribution of categories/subcategories in search results."""
    # Create a list of (label, count, category) tuples
    distribution_data = []
    for thread in threads:
        category = thread['category']
        subcategory = thread.get('subcategory')
        label = f"{subcategory}" if subcategory else category
        distribution_data.append((label, 1, category))

    # Count occurrences
    counts = Counter(distribution_data)

    # Create DataFrame with the counts
    df = pd.DataFrame([
        {
            "Label": label,
            "Count": count,
            "Category": category
        }
        for (label, _, category), count in counts.items()
    ])

    # Create pie chart with improved configuration
    fig = px.pie(
        df,
        values="Count",
        names="Label",
        color="Category",  # Color by parent category
        height=200  # Reduced height for sidebar
    )

    # Update layout with explicit dimensions and settings
    fig.update_layout(
        showlegend=False,  # Remove legend
        margin=dict(l=10, r=10, t=10, b=10),  # Minimal margins for sidebar
        title=None,  # Remove title since we use markdown
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        font=dict(size=10)  # Smaller font size for sidebar
    )

    # Update traces for better hover information
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Category: %{customdata}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )

    # Add category information to hover data
    fig.update_traces(customdata=df['Category'])

    return fig

def create_similarity_histogram(threads):
    """Create a histogram of similarity scores."""
    similarities = [thread.get('similarity', 0) * 100 for thread in threads]

    fig = go.Figure(data=[go.Histogram(
        x=similarities,
        nbinsx=10,
        name="Similarity Distribution",
        hovertemplate='Score: %{x:.1f}%<br>Count: %{y}<extra></extra>',
        marker_color='rgb(55, 83, 109)'
    )])

    # Update layout with explicit dimensions and settings
    fig.update_layout(
        showlegend=False,
        xaxis_title="Similarity Score (%)",
        yaxis_title="Count",
        height=200,  # Reduced height for sidebar
        margin=dict(l=10, r=10, t=10, b=30),  # Minimal margins for sidebar
        title=None,  # Remove title since we use markdown
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)'
        ),
        xaxis=dict(
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)',
            range=[0, 100],  # Force x-axis to show 0-100%
            tickfont=dict(size=10)  # Smaller font for sidebar
        ),
        font=dict(size=10),  # Smaller font size for sidebar
        bargap=0.1  # Add some gap between bars
    )

    return fig

def create_subcategory_chart(threads):
    """Create a pie chart showing the distribution of subcategories."""
    # Count subcategories
    subcategory_counts = Counter(
        thread.get('subcategory', 'None')
        for thread in threads
    )

    # Create pie chart with improved configuration
    fig = px.pie(
        values=list(subcategory_counts.values()),
        names=list(subcategory_counts.keys()),
        height=200  # Reduced height for sidebar
    )

    # Update layout with explicit dimensions and settings
    fig.update_layout(
        showlegend=True,
        margin=dict(l=10, r=10, t=10, b=10),  # Minimal margins for sidebar
        title=None,  # Remove title since we use markdown
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)',  # Semi-transparent background
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1,
            font=dict(size=10)  # Smaller font for sidebar
        ),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        font=dict(size=10)  # Smaller font size for sidebar
    )

    # Update traces for better hover information
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
        marker=dict(line=dict(color='#FFFFFF', width=1))  # Thinner white borders between sections
    )

    return fig

def display_analytics(threads, query):
    """Display analytics and insights about the search results."""
    with st.sidebar:
        st.markdown("### 📊 Search Analytics")

        if not threads:
            st.warning("No results to analyze.")
            return

        # Create metrics row
        st.metric("Results Found", len(threads))
        st.metric("Average Similarity", f"{sum(thread.get('similarity', 0) * 100 for thread in threads) / len(threads):.1f}%")
        st.metric("Unique Categories", len(set(thread['category'] for thread in threads)))

        st.markdown("---")

        # Category distribution
        st.markdown("#### Category Distribution")
        category_fig = create_category_chart(threads)
        st.plotly_chart(
            category_fig,
            use_container_width=True,
            theme=None  # Disable Streamlit theme to ensure proper rendering
        )

        st.markdown("---")

        # Similarity histogram
        st.markdown("#### Similarity Distribution")
        similarity_fig = create_similarity_histogram(threads)
        st.plotly_chart(
            similarity_fig,
            use_container_width=True,
            theme=None  # Disable Streamlit theme to ensure proper rendering
        )

        st.markdown("---")

        # Add insights
        st.markdown("#### 🔍 Key Insights")

        # Most common category
        category_counts = Counter(thread['category'] for thread in threads)
        most_common_cat = category_counts.most_common(1)[0]
        st.markdown(f"- Most common category: **{most_common_cat[0]}** ({most_common_cat[1]} results)")

        # Highest similarity
        max_similarity = max(thread.get('similarity', 0) * 100 for thread in threads)
        st.markdown(f"- Highest similarity score: **{max_similarity:.1f}%**")

        # Category diversity
        unique_categories = len(set(thread['category'] for thread in threads))
        category_percentage = (unique_categories / len(category_counts)) * 100
        st.markdown(f"- Category diversity: **{category_percentage:.1f}%** of all possible categories")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs([
    "Search Questions",
    "Similar Threads",
    "Browse by Category"
])

# Tab 1: Search Questions
with tab1:
    st.header("Search Questions")
    st.markdown("Enter your question to find similar threads in the ED database.")

    search_query = st.text_input("Your question:", key="search_input")
    search_limit = st.slider("Number of results:", min_value=1, max_value=30, value=15, key="search_limit")

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

                    # Display results
                    st.markdown("### Search Results")
                    display_thread_cards(similar_threads)

                    # Display analytics
                    display_analytics(similar_threads, search_query)

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
    st.markdown("""
    Browse threads by category and optionally filter by subcategory and search terms.
    The search will find semantically similar content within the selected category.

    **Note:** This feature requires the latest API version to be deployed.
    """)

    # Get page from query params (for pagination) using the stable API instead of experimental
    current_page = int(st.query_params.get("page", [1])[0]) if "page" in st.query_params else 1

    # Track state for the category browse - using session state to persist across interactions
    if "category_results" not in st.session_state:
        st.session_state.category_results = None
        st.session_state.total_count = 0
        st.session_state.current_filters = {}

    # Add search query field first
    search_query = st.text_input(
        "Search within category (optional):",
        placeholder="Enter keywords or a question",
        help="Enter search terms to find semantically similar content within the selected category"
    )

    # Updated correct category options
    category_options = ["Projet", "Cours", "Exercices", "Examens", "Divers"]
    selected_category = st.selectbox("Select Category:", options=category_options)

    # Only show subcategory selector for Projet category
    subcategory = None
    if selected_category == "Projet":
        # Create dropdown options for subcategories with correct formatting
        subcategory_options = [""] + ["Étape " + str(i) for i in range(1, 13)] + ["Général"]
        subcategory = st.selectbox(
            "Subcategory:",
            options=subcategory_options,
            format_func=lambda x: "All subcategories" if x == "" else x,
            help="Select a specific subcategory to filter results"
        )
        # Convert empty string to None to maintain API compatibility
        if subcategory == "":
            subcategory = None

    # Results per page
    results_per_page = st.slider("Results per page:", min_value=5, max_value=50, value=20, key="category_limit")

    search_button = st.button("Browse", key="category_button")

    # Execute search either if button is clicked or if we're navigating pages of previous search
    if search_button or (current_page > 1 and st.session_state.category_results is not None):
        # If button clicked, reset to page 1; otherwise use current_page from URL
        page = 1 if search_button else current_page
        skip = (page - 1) * results_per_page

        try:
            # Only make API request on new search or changed page
            if search_button or (st.session_state.current_filters != {
                "category": selected_category,
                "subcategory": subcategory,
                "query": search_query,
                "limit": results_per_page,
                "page": page
            }):
                # Build the query parameters
                params = {
                    "limit": results_per_page,
                    "skip": skip
                }

                if subcategory:  # Only add if not None
                    params["subcategory"] = subcategory

                if search_query and search_query.strip():  # Only add if not empty
                    params["query"] = search_query

                # Save current filters for comparison on next render
                st.session_state.current_filters = {
                    "category": selected_category,
                    "subcategory": subcategory,
                    "query": search_query,
                    "limit": results_per_page,
                    "page": page
                }

                # Make the API request - first try the category endpoint
                result = make_api_request(
                    'GET',
                    f'/threads/category/{selected_category}',
                    params=params
                )

                # If category endpoint fails, try to fall back to the search endpoint
                if result is None and search_query and search_query.strip():
                    st.info("Attempting to fall back to general search with your query...")
                    try:
                        # Fall back to using the /search/input endpoint
                        fallback_result = make_api_request(
                            'POST',
                            '/search/input',
                            json={"text": search_query},
                            params={"limit": results_per_page}
                        )

                        if fallback_result:
                            # Convert to format similar to category endpoint result
                            result = {
                                "threads": fallback_result,
                                "total_count": len(fallback_result),
                                "category": None,
                                "subcategory": None,
                                "query": search_query
                            }
                            st.success("Successfully retrieved results using general search instead.")
                    except Exception as fallback_error:
                        st.error(f"Fallback search also failed: {str(fallback_error)}")

                # Save results to session state
                if result and result.get("total_count", 0) > 0:
                    st.session_state.category_results = result.get("threads", [])
                    st.session_state.total_count = result.get("total_count", 0)
                else:
                    st.session_state.category_results = []
                    st.session_state.total_count = 0

            # Display results from session state
            if st.session_state.category_results and len(st.session_state.category_results) > 0:
                threads = st.session_state.category_results
                total_count = st.session_state.total_count

                # Display results count and pagination info
                filter_description = f"category '{selected_category}'"
                if subcategory:
                    filter_description += f", subcategory '{subcategory}'"
                if search_query and search_query.strip():
                    filter_description += f", search: '{search_query}'"

                # Calculate total pages
                total_pages = max(1, (total_count + results_per_page - 1) // results_per_page)

                # Display results count with page info
                st.success(f"Found {total_count} threads matching {filter_description}")

                # Display pagination info
                st.markdown(f"Showing results {skip + 1}-{min(skip + results_per_page, total_count)} of {total_count} (Page {page} of {total_pages})")

                # Create a container for results
                results_container = st.container()
                with results_container:
                    # Show similarity if search query was provided
                    show_similarity = bool(search_query and search_query.strip())
                    display_thread_cards(threads, show_similarity=show_similarity)

                # Add pagination controls
                col1, col2 = st.columns(2)

                with col1:
                    if page > 1:
                        if st.button("← Previous Page"):
                            # Update query parameters and rerun to navigate to previous page
                            st.query_params.update(page=page-1)
                            st.rerun()

                with col2:
                    if page < total_pages:
                        if st.button("Next Page →"):
                            # Update query parameters and rerun to navigate to next page
                            st.query_params.update(page=page+1)
                            st.rerun()

            else:
                filter_description = f"category '{selected_category}'"
                if subcategory:
                    filter_description += f", subcategory '{subcategory}'"
                if search_query and search_query.strip():
                    filter_description += f", search: '{search_query}'"

                st.warning(f"No threads found matching {filter_description}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
