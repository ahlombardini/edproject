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
        data_dir = "data/cleaned_text"
        for filename in os.listdir(data_dir):
            if filename.endswith(".csv"):
                # Read the CSV file
                df = pd.read_csv(os.path.join(data_dir, filename))

                # Process each row
                for _, row in df.iterrows():
                    # Check if thread already exists
                    existing_thread = db.query(Thread).filter(
                        Thread.ed_thread_id == str(row['ed_thread_id'])
                    ).first()

                    if existing_thread:
                        continue

                    # Generate embedding
                    combined_text = combine_text(row)
                    embedding = model.encode(combined_text)

                    # Create new thread
                    thread = Thread(
                        ed_thread_id=str(row['ed_thread_id']),
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

                # Commit after each file
                db.commit()
                print(f"Processed {filename}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        db.rollback()

    finally:
        db.close()

if __name__ == "__main__":
    populate_db()
