import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def load_cleaned_df(file_path):
    """Load a CSV file as a pandas DataFrame."""
    return pd.read_csv(file_path)

def embed_thread_with_category(thread, model, category_weight=0.7, text_weight=0.3):
    """
    Embeds a thread by combining its category information and text fields.

    Args:
        thread (dict): A dictionary representing a thread (row from DataFrame).
        model (SentenceTransformer): Preloaded embedding model.
        category_weight (float): Weight for the category embedding.
        text_weight (float): Weight for the text embedding.

    Returns:
        np.ndarray: Combined embedding vector.
    """
    # Combine category and subcategory into a string
    category_text = f"{thread.get('category', '')} {thread.get('subcategory', '')}".strip()

    # Combine text fields (title, document, content_and_img_desc) into one string
    text = " ".join([
        str(thread.get('title', '')),
        str(thread.get('document', '')),
        str(thread.get('content_and_img_desc', ''))
    ]).strip()

    # Encode category and text separately
    category_embedding = model.encode(category_text, show_progress_bar=False)
    text_embedding = model.encode(text, show_progress_bar=False)

    # Compute weighted combination
    combined_embedding = category_weight * category_embedding + text_weight * text_embedding
    return combined_embedding

def process_file(file_path, model, output_dir, category_weight=0.5, text_weight=0.5):
    """
    Processes a CSV file of threads, embeds each thread, and saves the embeddings.

    Args:
        file_path (str): Path to the CSV file.
        model (SentenceTransformer): Preloaded embedding model.
        output_dir (str): Directory to save the embedding file.
        category_weight (float): Weight for category embeddings.
        text_weight (float): Weight for text embeddings.
    """
    df = load_cleaned_df(file_path)
    embeddings = []

    # Iterate over each thread (row) in the DataFrame
    for idx, row in df.iterrows():
        thread_dict = row.to_dict()
        embedding = embed_thread_with_category(thread_dict, model, category_weight, text_weight)
        embeddings.append(embedding)

    # Convert list of embeddings to a NumPy array
    embeddings_arr = np.vstack(embeddings)
    output_dir = "data/prodEmbeddings"
    os.makedirs(output_dir, exist_ok=True)
    # Save embeddings to a .npy file (using the same base name as the input file)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}_embeddings.npy")
    np.save(output_file, embeddings_arr)
    print(f"Saved embeddings for {file_path} to {output_file}")

input_dir = "data/cleaned_text"
output_dir = "data/"
os.makedirs(output_dir, exist_ok=True)

# Load the model from the saved directory instead of downloading it anew.
model_dir = "models/sentence_transformer"  # Path to your saved model directory
model = SentenceTransformer(model_dir)


for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(input_dir, file)
        process_file(file_path, model, output_dir, 0.5, 0.5)
