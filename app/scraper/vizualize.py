from json import load
import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px

def load_all_embeddings_and_threads(embedding_dir, thread_dir):
    """
    Loads embeddings and thread details from separate directories.
    Assumes each thread is stored in a CSV file in thread_dir and that the
    corresponding embedding is stored in embedding_dir as <thread_id>_embeddings.npy.

    Args:
        embedding_dir (str): Directory containing individual embedding files.
        thread_dir (str): Directory containing individual CSV files with thread details.

    Returns:
        tuple: (embeddings (np.ndarray), threads_df (pd.DataFrame))
            Only threads for which an embedding file exists are returned.
    """
    embeddings = []
    thread_records = []
#   /data/embedding0_100/168613_embeddings.npy
    # Loop over each CSV file (one per thread) in thread_dir
    for file in os.listdir(thread_dir):
        if file.endswith(".csv"):
            base_name = os.path.splitext(file)[0]  # e.g., "12345"

            embedding_file = os.path.join(embedding_dir, f"{base_name}_embeddings.npy")
            thread_file = os.path.join(thread_dir, file)
            print(f"Loading thread '{base_name}' from {embedding_file} and {thread_file}")
            if os.path.exists(embedding_file):
                # Load the embedding (assume each file stores a single vector)
                emb = np.load(embedding_file)
                embeddings.append(emb)

                # Load the thread details (assume CSV contains a single row for the thread)
                df_thread = pd.read_csv(thread_file)
                # If CSV has multiple rows, we take the first row (customize as needed)
                thread_records.append(df_thread.iloc[0])
            else:

                print(f"Warning: Embedding file for thread '{base_name}' not found.")

    if not embeddings:
        raise ValueError("No embeddings were loaded. Check your file paths and naming convention.")

    embeddings_arr = np.vstack(embeddings)
    threads_df = pd.DataFrame(thread_records)
    return embeddings_arr, threads_df

import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px

def visualize_embeddings(embeddings, threads_df, perplexity=30, max_iter=1000):
    """
    Visualizes high-dimensional embeddings in 3D using t-SNE and displays thread info on hover.

    Args:
        embeddings (np.ndarray): Array of shape (n_samples, embedding_dim) with thread embeddings.
        threads_df (pd.DataFrame): DataFrame containing thread details. Expected columns include
                                   'title', 'category', and 'document'.
        perplexity (int): t-SNE perplexity parameter.
        max_iter (int): Number of iterations for t-SNE.

    Returns:
        plotly.graph_objects.Figure: An interactive 3D scatter plot figure.
    """
    # Ensure the thread DataFrame has a unique index.
     # Reset index to ensure unique labels.
    threads_df = threads_df.reset_index(drop=True)

    # Reduce dimensions to 3D using t-SNE.
    tsne = TSNE(n_components=3, perplexity=perplexity, max_iter=max_iter, random_state=42)
    embeddings_3d = tsne.fit_transform(embeddings)

    # Build a DataFrame with the 3D coordinates.
    plot_df = pd.DataFrame(embeddings_3d, columns=['x', 'y', 'z'])

    # Create a combined field for category and subcategory.
    plot_df['Cat_Subcat'] = threads_df['category'].astype(str) + " - " + threads_df['subcategory'].astype(str)

    # Add additional thread info for hover display.
    plot_df['Title'] = threads_df['title']
    plot_df['Document'] = threads_df['document']
    # Create a custom hover text field.
    plot_df['hover_text'] = plot_df.apply(
        lambda row: f"<b>Title:</b> {row['Title']}<br><b>Category/Subcategory:</b> {row['Cat_Subcat']}<br>"
                    f"<b>Document:</b> {row['Document'][:200]}...",
        axis=1
    )

    # Create an interactive 3D scatter plot, coloring points by category/subcategory.
    fig = px.scatter_3d(plot_df, x='x', y='y', z='z', color='Cat_Subcat',
                        hover_data=['hover_text'],

                        title="t-SNE 3D Visualization of Thread Embeddings")

    # Customize hover template to show the custom hover text.
    fig.update_traces(hovertemplate='%{customdata[0]}')
    fig.show()
    return fig

# Example usage:
# Assuming 'stored_embeddings' is a NumPy array of your thread embeddings,
# and 'threads_df' is a DataFrame with thread details.
# fig = visualize_embeddings_3d(stored_embeddings, threads_df, perplexity=30, max_iter=1000)
def load_and_visualize_all(category_weight, text_weight, perplexity=30, max_iter=1000):
    """
    Loads per-thread embeddings and corresponding thread CSV files, then visualizes the embeddings.

    Args:
        embedding_dir (str): Directory containing embedding files.
        thread_dir (str): Directory containing thread CSV files.
        perplexity (int): t-SNE perplexity parameter.
        max_iter (int): t-SNE number of iterations.

    Returns:
        plotly.graph_objects.Figure: The interactive scatter plot figure.
    """
    embedding_dir=f"data/embedding{category_weight}_{text_weight}/"
    thread_dir="data/cleaned_text"
    embeddings, threads_df = load_all_embeddings_and_threads(embedding_dir, thread_dir)
    fig = visualize_embeddings(embeddings, threads_df, perplexity=perplexity, max_iter=max_iter)
    return fig


# Define your directories (update these paths as needed)
embedding_directory = "data/embeddings"
thread_directory = "data/cleaned_text"

# Load all per-thread embeddings and visualize them
fig = load_and_visualize_all(50, 50, perplexity=30, max_iter=1000)
