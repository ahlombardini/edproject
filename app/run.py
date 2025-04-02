import uvicorn
import os
import logging
from dotenv import load_dotenv
import argparse
import importlib
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Run the ED API with Telegram bot')
    parser.add_argument('--no-sync', action='store_true', help='Disable automatic syncing with ED Stem')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the API on')
    args = parser.parse_args()

    # Load environment variables from root .env file
    load_dotenv()

    if args.no_sync:
        # Create a temporary main_nosync.py file that doesn't import the sync service
        nosync_main_path = os.path.join(os.path.dirname(__file__), 'api', 'main_nosync.py')

        # Write the modified main file
        with open(nosync_main_path, 'w') as f:
            f.write('''from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

from app.database.database import get_db, engine
from app.models.thread import Thread, Base

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Load the sentence transformer model
model = SentenceTransformer('all-mpnet-base-v2')

@app.get("/")
def read_root():
    return {"message": "Welcome to the ED API (Sync Service Disabled)"}

@app.get("/threads/", response_model=List[dict])
def get_threads(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    threads = db.query(Thread).offset(skip).limit(limit).all()
    return [thread.to_dict() for thread in threads]

@app.get("/threads/{thread_id}")
def get_thread(thread_id: str, db: Session = Depends(get_db)):
    thread = db.query(Thread).filter(Thread.ed_thread_id == thread_id).first()
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread.to_dict()

@app.get("/search/similar/{thread_id}")
def find_similar_threads(thread_id: str, limit: int = 5, db: Session = Depends(get_db)):
    # Get the target thread
    thread = db.query(Thread).filter(Thread.ed_thread_id == thread_id).first()
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Get all threads
    all_threads = db.query(Thread).all()

    # Convert embeddings from JSON strings to numpy arrays
    target_embedding = np.array(json.loads(thread.embedding))
    thread_embeddings = [np.array(json.loads(t.embedding)) for t in all_threads]

    # Calculate similarities
    similarities = [np.dot(target_embedding, emb) / (np.linalg.norm(target_embedding) * np.linalg.norm(emb))
                   for emb in thread_embeddings]

    # Get indices of top similar threads (excluding the query thread)
    similar_indices = np.argsort(similarities)[::-1][1:limit+1]

    # Return similar threads with their similarity scores
    similar_threads = []
    for idx in similar_indices:
        thread_dict = all_threads[idx].to_dict()
        thread_dict['similarity'] = float(similarities[idx])
        similar_threads.append(thread_dict)

    return similar_threads
''')

        logger.info("Starting API only (sync service disabled)")
        uvicorn.run("app.api.main_nosync:app", host="0.0.0.0", port=args.port, reload=True)
    else:
        logger.info("Starting API with sync service")
        uvicorn.run("app.api.main:app", host="0.0.0.0", port=args.port, reload=True)

if __name__ == "__main__":
    main()
