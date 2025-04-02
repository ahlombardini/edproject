import re
import pandas as pd
import os

srcfilepath = 'data/cleaned_threads'
dstfilepath = 'data/cleaned_text'
def extract_df(file_path):
    return pd.read_csv(file_path)
def clean_text(text):
    """
    Cleans the text by removing unnecessary whitespace, formatting issues, and redundant newlines.

    Args:
        text (str): Raw text containing content and extracted image descriptions.

    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ""

    # Remove extra spaces, newlines, and tabs
    text = re.sub(r'\s+', ' ', text).strip()

    # Replace common placeholders or unwanted sequences
    text = text.replace("[Image Extracted Content: ]", "")  # If you inserted placeholders before
    text = re.sub(r'\[Image Extracted Content:\s*([^]]+)]', r'\1', text)  # Keep the extracted content

    # Normalize quotation marks and special characters
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = text.lower()  # Convert to lowercase
    return text


def parse_and_clean():
    for file in os.listdir(srcfilepath):
        if file.endswith('.csv'):
            file_path = os.path.join(srcfilepath, file)
            df = extract_df(file_path)
            df['content_and_img_desc']=df['content_and_img_desc'].apply(clean_text)
            df.to_csv(os.path.join(dstfilepath, file), index=False)
parse_and_clean()
