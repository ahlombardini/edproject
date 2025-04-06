import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
import json
import requests
import os
from dotenv import load_dotenv
import streamlit as st

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

def visualize_embeddings(embeddings, threads_df, perplexity=30, n_iter=1000):
    """
    Visualizes high-dimensional embeddings in 3D using t-SNE.
    Args:
        embeddings (np.ndarray): Array of shape (n_samples, embedding_dim) with thread embeddings.
        threads_df (pd.DataFrame): DataFrame containing thread details.
        perplexity (int): t-SNE perplexity parameter.
        n_iter (int): Number of iterations for t-SNE.
    Returns:
        plotly.graph_objects.Figure: An interactive 3D scatter plot figure.
    """
    # Reset index to ensure unique labels
    threads_df = threads_df.reset_index(drop=True)

    # Print debug info
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Number of threads in DataFrame: {len(threads_df)}")

    # Ensure perplexity is valid (must be less than n_samples)
    n_samples = embeddings.shape[0]
    if n_samples <= 3:
        raise ValueError("Not enough samples for visualization (minimum 4 required)")

    # Adjust perplexity if needed (should be between 5 and n_samples-1)
    perplexity = min(perplexity, n_samples - 1)
    perplexity = max(5, perplexity)  # Ensure minimum of 5 for better results

    # Reduce dimensions to 3D using t-SNE
    print("Starting t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=3, perplexity=perplexity, n_iter=n_iter, random_state=42)
    embeddings_3d = tsne.fit_transform(embeddings)
    print(f"t-SNE complete. Output shape: {embeddings_3d.shape}")

    # Build a DataFrame with the 3D coordinates
    plot_df = pd.DataFrame(embeddings_3d, columns=['x', 'y', 'z'])

    # Create a combined field for category and subcategory
    plot_df['Cat_Subcat'] = threads_df['category'].astype(str) + " - " + threads_df['subcategory'].fillna('None').astype(str)

    # Add additional thread info for hover display
    plot_df['Title'] = threads_df['title']
    plot_df['Category'] = threads_df['category']
    plot_df['Subcategory'] = threads_df['subcategory'].fillna('None')
    plot_df['Thread_ID'] = threads_df['ed_thread_id']

    # Create a custom hover text field
    plot_df['hover_text'] = plot_df.apply(
        lambda row: f"<b>Title:</b> {row['Title']}<br>"
                   f"<b>Category:</b> {row['Category']}<br>"
                   f"<b>Subcategory:</b> {row['Subcategory']}<br>"
                   f"<b>Thread ID:</b> {row['Thread_ID']}",
        axis=1
    )

    # Create an interactive 3D scatter plot with adjusted layout
    fig = px.scatter_3d(
        plot_df,
        x='x',
        y='y',
        z='z',
        color='Category',  # Color by main category for better visibility
        hover_data=['hover_text'],
        title=f"3D Visualization of Thread Embeddings (Perplexity: {perplexity}, Samples: {n_samples})",
        height=600  # Set explicit height
    )

    # Customize hover template and marker size
    fig.update_traces(
        hovertemplate='%{customdata[0]}',
        marker=dict(size=8)  # Increased marker size for better visibility
    )

    # Update layout for better visibility
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=True, showgrid=True, zeroline=True, visible=True),
            yaxis=dict(showbackground=True, showgrid=True, zeroline=True, visible=True),
            zaxis=dict(showbackground=True, showgrid=True, zeroline=True, visible=True),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='cube'  # Force equal aspect ratio
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"  # Semi-transparent background
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    # Debug: Show the figure directly
    print("Showing figure directly via plotly...")
    fig.show()

    return fig

def generate_visualization(perplexity=30, n_iter=1000, num_threads=50):
    """
    Fetches thread data from API and generates visualization.
    Args:
        perplexity (int): t-SNE perplexity parameter.
        n_iter (int): Number of iterations for t-SNE.
        num_threads (int): Number of threads to visualize.
    Returns:
        plotly.graph_objects.Figure: An interactive 3D scatter plot.
    """
    try:
        # Get threads from the API with pagination
        all_threads = []
        offset = 0
        limit = 100  # Fetch in batches of 100

        while len(all_threads) < num_threads:
            print(f"Fetching threads from offset {offset}...")
            response = requests.get(
                f"{API_URL}/threads/",
                params={"skip": offset, "limit": limit},
                headers=HEADERS
            )
            response.raise_for_status()
            batch = response.json()

            if not batch:  # No more threads available
                break

            all_threads.extend(batch)
            offset += limit

            if len(batch) < limit:  # Last batch
                break

        if not all_threads:
            raise ValueError("No threads returned from API")

        print(f"Retrieved {len(all_threads)} threads from API")

        # If we have more threads than requested, select a random sample
        if len(all_threads) > num_threads:
            selected_threads = np.random.choice(all_threads, num_threads, replace=False).tolist()
        else:
            selected_threads = all_threads

        print(f"Processing {len(selected_threads)} threads...")

        # Extract embeddings and thread data
        embeddings = []
        thread_data = []

        for thread in selected_threads:
            try:
                # Get the full thread details to access the embedding
                print(f"Fetching details for thread {thread['ed_thread_id']}")
                thread_response = requests.get(
                    f"{API_URL}/threads/{thread['ed_thread_id']}",
                    headers=HEADERS
                )
                thread_response.raise_for_status()
                full_thread = thread_response.json()

                # The API might return the thread inside a 'thread' key
                if isinstance(full_thread, dict) and 'thread' in full_thread:
                    full_thread = full_thread['thread']

                if 'embedding' not in full_thread:
                    print(f"Warning: Thread {thread['ed_thread_id']} has no embedding field")
                    continue

                # Convert embedding from JSON string to numpy array
                embedding_data = json.loads(full_thread['embedding'])
                if not isinstance(embedding_data, list):
                    print(f"Warning: Thread {thread['ed_thread_id']} embedding is not a list")
                    continue

                embedding = np.array(embedding_data)
                embeddings.append(embedding)
                thread_data.append(full_thread)
                print(f"Successfully processed thread {thread['ed_thread_id']}")

            except Exception as e:
                print(f"Warning: Could not process thread {thread['ed_thread_id']}: {str(e)}")
                continue

        if not embeddings:
            raise ValueError("No valid embeddings found in the threads")

        print(f"Successfully loaded {len(embeddings)} threads with embeddings")

        # Convert to numpy array and DataFrame
        embeddings_arr = np.vstack(embeddings)
        threads_df = pd.DataFrame(thread_data)

        # Generate the visualization
        fig = visualize_embeddings(
            embeddings_arr,
            threads_df,
            perplexity=perplexity,
            n_iter=n_iter
        )
        return fig

    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        raise Exception(f"Error generating visualization: {str(e)}")

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

        tsne = TSNE(
            n_components=3,
            perplexity=perplexity,
            random_state=42,
            n_iter=1000
        )
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
            title=f"3D Visualization of Search Results for: {query}",
            height=600  # Increased height for better visibility
        )

        # Customize hover template
        fig.update_traces(
            hovertemplate='%{customdata[0]}',
            marker=dict(size=8)  # Increased marker size
        )

        # Update layout for better visibility
        fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=True, showgrid=True, zeroline=True, visible=True),
                yaxis=dict(showbackground=True, showgrid=True, zeroline=True, visible=True),
                zaxis=dict(showbackground=True, showgrid=True, zeroline=True, visible=True),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='cube'  # Force equal aspect ratio
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            coloraxis_colorbar=dict(
                title="Similarity (%)",
                tickformat=".1f"
            )
        )

        # Debug: Show the figure directly
 

        return fig

    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        raise Exception(f"Error processing thread data: {str(e)}")
