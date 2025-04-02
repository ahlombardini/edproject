import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session
import logging

from app.database.database import SessionLocal, engine
from app.models.thread import Thread, Base

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

def import_data():
    """Import existing cleaned text and embeddings into the database."""
    # Define paths
    cleaned_text_dir = "data/cleaned_text"
    embeddings_dir = "data/prodEmbeddings"

    # Create database session
    db = SessionLocal()

    try:
        # Get list of all CSV files in cleaned_text
        csv_files = [f for f in os.listdir(cleaned_text_dir) if f.endswith('.csv')]
        logger.info(f"Found {len(csv_files)} CSV files to process")

        # Process each CSV file
        for i, csv_file in enumerate(csv_files):
            thread_id = csv_file.replace('.csv', '')

            # Check if thread already exists in the database
            existing_thread = db.query(Thread).filter(Thread.ed_thread_id == thread_id).first()
            if existing_thread:
                logger.info(f"Thread {thread_id} already exists in database, skipping")
                continue

            # Read the CSV file
            try:
                csv_path = os.path.join(cleaned_text_dir, csv_file)
                df = pd.read_csv(csv_path)
                if df.empty or len(df) == 0:
                    logger.warning(f"Empty CSV file: {csv_file}, skipping")
                    continue

                # Get the first row
                row = df.iloc[0]

                # Check if embedding exists
                embedding_file = f"{thread_id}_embeddings.npy"
                embedding_path = os.path.join(embeddings_dir, embedding_file)

                if not os.path.exists(embedding_path):
                    logger.warning(f"No embedding found for thread {thread_id}, skipping")
                    continue

                # Load the embedding
                embedding = np.load(embedding_path)

                # Create a new thread record
                thread = Thread(
                    ed_thread_id=thread_id,
                    title=row.get('title', ''),
                    content=row.get('content', ''),
                    document=row.get('document', ''),
                    category=row.get('category', ''),
                    subcategory=row.get('subcategory', ''),
                    content_and_img_desc=row.get('content_and_img_desc', ''),
                    embedding=json.dumps(embedding.tolist()[0]),  # Convert to list and then to JSON
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )

                # Add to database
                db.add(thread)

                # Commit every 10 records to avoid memory issues
                if i % 10 == 0:
                    db.commit()
                    logger.info(f"Processed {i+1}/{len(csv_files)} threads")

            except Exception as e:
                logger.error(f"Error processing {csv_file}: {str(e)}")
                continue

        # Final commit
        db.commit()
        logger.info("All data imported successfully")

    except Exception as e:
        logger.error(f"Error importing data: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    import_data()
