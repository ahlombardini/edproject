import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
import json
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the API URL and key
API_URL = os.getenv('API_URL', 'http://localhost:8000')
API_KEY = os.getenv('API_KEY')
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

def fetch_threads_from_api():
    """
    Fetches all threads and their embeddings from the API.
    Returns:
        tuple: (embeddings (np.ndarray), threads_df (pd.DataFrame))
    """
    try:
        # Get all threads from the API
        response = requests.get(f"{API_URL}/threads/", headers=HEADERS)
        response.raise_for_status()
        threads = response.json()

        if not threads:
            raise ValueError("No threads returned from API")

        # Extract embeddings and thread data
        embeddings = []
        thread_data = []

        for thread in threads:
            # Get the full thread details to access the embedding
            thread_response = requests.get(
                f"{API_URL}/threads/{thread['ed_thread_id']}",
                headers=HEADERS
            )
            thread_response.raise_for_status()
            full_thread = thread_response.json()

            # Convert embedding from JSON string to numpy array
            embedding = np.array(json.loads(full_thread['embedding']))
            embeddings.append(embedding)
            thread_data.append(full_thread)

        embeddings_arr = np.vstack(embeddings)
        threads_df = pd.DataFrame(thread_data)
        return embeddings_arr, threads_df

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching data from API: {str(e)}")

def visualize_embeddings(embeddings, threads_df, perplexity=30, max_iter=1000):
    """
    Visualizes high-dimensional embeddings in 3D using t-SNE.
    Args:
        embeddings (np.ndarray): Array of shape (n_samples, embedding_dim) with thread embeddings.
        threads_df (pd.DataFrame): DataFrame containing thread details.
        perplexity (int): t-SNE perplexity parameter.
        max_iter (int): Number of iterations for t-SNE.
    Returns:
        plotly.graph_objects.Figure: An interactive 3D scatter plot figure.
    """
    # Reset index to ensure unique labels
    threads_df = threads_df.reset_index(drop=True)

    # Reduce dimensions to 3D using t-SNE
    tsne = TSNE(n_components=3, perplexity=perplexity, max_iter=max_iter, random_state=42)
    embeddings_3d = tsne.fit_transform(embeddings)

    # Build a DataFrame with the 3D coordinates
    plot_df = pd.DataFrame(embeddings_3d, columns=['x', 'y', 'z'])

    # Create a combined field for category and subcategory
    plot_df['Cat_Subcat'] = threads_df['category'].astype(str) + " - " + threads_df['subcategory'].astype(str)

    # Add additional thread info for hover display
    plot_df['Title'] = threads_df['title']
    plot_df['Document'] = threads_df['document']
    plot_df['Thread_ID'] = threads_df['ed_thread_id']

    # Create a custom hover text field
    plot_df['hover_text'] = plot_df.apply(
        lambda row: f"<b>Title:</b> {row['Title']}<br><b>Category/Subcategory:</b> {row['Cat_Subcat']}<br>"
                   f"<b>Thread ID:</b> {row['Thread_ID']}<br>"
                   f"<b>Document:</b> {row['Document'][:200]}...",
        axis=1
    )

    # Create an interactive 3D scatter plot
    fig = px.scatter_3d(
        plot_df,
        x='x',
        y='y',
        z='z',
        color='Cat_Subcat',
        hover_data=['hover_text'],
        title="3D Visualization of Thread Embeddings"
    )

    # Customize hover template
    fig.update_traces(hovertemplate='%{customdata[0]}')

    return fig

def generate_visualization(perplexity=30, max_iter=1000):
    """
    Fetches thread data from API and generates visualization.
    """
    embeddings, threads_df = fetch_threads_from_api()
    fig = visualize_embeddings(embeddings, threads_df, perplexity=perplexity, max_iter=max_iter)
    return fig

def visualize_search_results(threads, query):
    """
    Visualizes search results in 3D using t-SNE, colored by similarity to query.
    Args:
        threads (list): List of thread dictionaries with similarity scores
        query (str): The search query used
    Returns:
        plotly.graph_objects.Figure: An interactive 3D scatter plot figure.
    """
    if not threads:
        raise ValueError("No threads provided for visualization")

    # Extract embeddings and create DataFrame
    embeddings = []
    thread_data = []

    try:
        for thread in threads:
            try:
                # First check if the thread already has an embedding
                if 'embedding' in thread and thread['embedding']:
                    try:
                        embedding_data = json.loads(thread['embedding'])
                        if isinstance(embedding_data, list):
                            embedding = np.array(embedding_data)
                            embeddings.append(embedding)
                            thread_data.append(thread)
                            continue
                    except json.JSONDecodeError:
                        pass  # If embedding parsing fails, try fetching from API

                # If no valid embedding in thread, get it from the API
                thread_response = requests.get(
                    f"{API_URL}/threads/{thread['ed_thread_id']}",
                    headers=HEADERS
                )
                thread_response.raise_for_status()
                full_thread = thread_response.json()

                # Validate that embedding exists and is in the correct format
                if 'embedding' not in full_thread:
                    print(f"Warning: Thread {thread['ed_thread_id']} has no embedding field")
                    continue

                if not isinstance(full_thread['embedding'], str):
                    print(f"Warning: Thread {thread['ed_thread_id']} embedding is not a string")
                    continue

                try:
                    # Convert embedding from JSON string to numpy array
                    embedding_data = json.loads(full_thread['embedding'])
                    if not isinstance(embedding_data, list):
                        print(f"Warning: Thread {thread['ed_thread_id']} embedding is not a list")
                        continue

                    embedding = np.array(embedding_data)

                    # Add similarity score from search results to full thread data
                    full_thread['similarity'] = thread.get('similarity', 0)

                    embeddings.append(embedding)
                    thread_data.append(full_thread)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse embedding for thread {thread['ed_thread_id']}: {str(e)}")
                    continue

            except requests.exceptions.RequestException as e:
                print(f"Warning: Could not fetch thread {thread['ed_thread_id']}: {str(e)}")
                continue

        if not embeddings:
            raise ValueError("No valid embeddings found in the threads")

        embeddings_arr = np.vstack(embeddings)

        # Reduce dimensions to 3D using t-SNE
        perplexity = min(30, len(embeddings)-1)
        if perplexity < 2:
            raise ValueError("Not enough valid threads for visualization (minimum 2 required)")

        tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42)
        embeddings_3d = tsne.fit_transform(embeddings_arr)

        # Build DataFrame with 3D coordinates
        plot_df = pd.DataFrame(embeddings_3d, columns=['x', 'y', 'z'])

        # Add thread info and similarity scores
        plot_df['Title'] = [t['title'] for t in thread_data]
        plot_df['Category'] = [t['category'] for t in thread_data]
        plot_df['Subcategory'] = [t['subcategory'] for t in thread_data]
        plot_df['Thread_ID'] = [t['ed_thread_id'] for t in thread_data]
        plot_df['Similarity'] = [t['similarity'] * 100 for t in thread_data]  # Convert to percentage

        # Create hover text
        plot_df['hover_text'] = plot_df.apply(
            lambda row: f"<b>Title:</b> {row['Title']}<br>"
                       f"<b>Category:</b> {row['Category']}<br>"
                       f"<b>Subcategory:</b> {row['Subcategory']}<br>"
                       f"<b>Thread ID:</b> {row['Thread_ID']}<br>"
                       f"<b>Similarity:</b> {row['Similarity']:.1f}%",
            axis=1
        )

        # Create scatter plot with similarity-based coloring
        fig = px.scatter_3d(
            plot_df,
            x='x',
            y='y',
            z='z',
            color='Similarity',
            color_continuous_scale='RdYlGn',  # Red to Yellow to Green
            hover_data=['hover_text'],
            title=f"3D Visualization of Search Results for: {query}"
        )

        # Customize hover template
        fig.update_traces(hovertemplate='%{customdata[0]}')

        # Update color axis
        fig.update_coloraxes(colorbar_title="Similarity (%)")

        return fig

    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        raise Exception(f"Error processing thread data: {str(e)}")
