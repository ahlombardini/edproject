import pandas as pd
import os
import pytesseract
from PIL import Image
import io
import requests
import xml.etree.ElementTree as ET

srcfilepath = 'data/threads'
dstfilepath = 'data/cleaned_threads'
def get_image_links_from_xml(xml_str):
    root = ET.fromstring(xml_str)
    image_links = []

    # Search for <image> elements (not <img>)
    for image_elem in root.findall('.//image'):
        src = image_elem.get('src')
        if src:
            image_links.append(src)

    return image_links
def download_image(url):
    resp = requests.get(url)
    if resp.status_code == 200:
        return resp.content  # raw bytes
    return None
def extract_df(file_path):
    df = pd.read_csv(file_path)
    return df
def extract_text_from_image(image_bytes):
    if image_bytes is None:
        return ""
    img = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(img)
    return text
def get_images_and_extract_text(content):
    image_links = get_image_links_from_xml(content)
    images = [download_image(link) for link in image_links]
    texts = [extract_text_from_image(img) for img in images]
    return texts
def insert_image_text_into_document(xml_str,df):
    """
    Parses an XML document, extracts text and replaces <image> elements with their extracted OCR text.
    """
    root = ET.fromstring(xml_str)
    updated_text = []
    image_content = []
    for elem in root:
        if elem.tag == "paragraph":
            updated_text.append(elem.text if elem.text else "")
        elif elem.tag == "figure":
            # Extract the <image> element inside <figure>
            image_elem = elem.find("image")
            if image_elem is not None and "src" in image_elem.attrib:
                image_url = image_elem.attrib["src"]
                print(f"Extracting text from image: {image_url}")
                extracted_text = extract_text_from_image(download_image(image_url))
                image_content.append(extracted_text)
                updated_text.append(f"[img: {extracted_text}]")
        

    # Join paragraphs back together
    return "\n\n".join(updated_text)
def parse_dir_and_clean():
    for file in os.listdir(srcfilepath):
        if file.endswith('.csv'):
            file_path = os.path.join(srcfilepath, file)
            df = extract_df(file_path)
            df = df[['content', 'document', 'title', 'category', 'subcategory']]
            df['content_and_img_desc'] =insert_image_text_into_document(df['content'][0],df)
            df.to_csv(os.path.join(dstfilepath, file), index=False)

parse_dir_and_clean()
