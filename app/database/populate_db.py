import os
import json
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from app.database.database import SessionLocal, engine
from app.models.thread import Thread, Base

# Create tables
Base.metadata.create_all(bind=engine)

def combine_text(thread):
    """Combine thread text fields for embedding."""
    return f"{thread.get('title', '')} {thread.get('document', '')} {thread.get('content_and_img_desc', '')}"

def populate_db():
    # Load the sentence transformer model
    model = SentenceTransformer('all-mpnet-base-v2')

    # Create database session
    db = SessionLocal()

    try:
        # Read all CSV files in the data/cleaned_text directory
        data_dir = "../python/edApi/data/cleaned_text"
        for filename in os.listdir(data_dir):
            if filename.endswith(".csv"):
                # Extract thread ID from filename (remove .csv extension)
                thread_id = filename[:-4]  # Remove .csv from the end

                # Check if thread already exists
                existing_thread = db.query(Thread).filter(
                    Thread.ed_thread_id == thread_id
                ).first()

                if existing_thread:
                    print(f"Thread {thread_id} already exists, skipping...")
                    continue

                # Read the CSV file
                df = pd.read_csv(os.path.join(data_dir, filename))

                if len(df) == 0:
                    print(f"Empty file {filename}, skipping...")
                        continue

                # Take the first row since each CSV should contain one thread
                row = df.iloc[0]

                    # Generate embedding
                    combined_text = combine_text(row)
                    embedding = model.encode(combined_text)

                    # Create new thread
                    thread = Thread(
                    ed_thread_id=thread_id,
                        title=row.get('title', ''),
                        content=row.get('content', ''),
                        document=row.get('document', ''),
                        category=row.get('category', ''),
                        subcategory=row.get('subcategory', ''),
                        content_and_img_desc=row.get('content_and_img_desc', ''),
                        embedding=json.dumps(embedding.tolist()),
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )

                    db.add(thread)
                db.commit()
                print(f"Processed thread {thread_id}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        db.rollback()
        raise  # Re-raise the exception for better error tracking

    finally:
        db.close()

if __name__ == "__main__":
    populate_db()
