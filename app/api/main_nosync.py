from fastapi import FastAPI, Depends, HTTPException
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

@app.post("/search/input")
def find_similar_from_input(query: dict, limit: int = 5, db: Session = Depends(get_db)):
    """Find threads similar to a user-provided text input."""
    # Extract the user input text
    user_text = query.get("text", "")
    if not user_text:
        raise HTTPException(status_code=400, detail="Text input is required")

    # Generate embedding for the user input
    user_embedding = model.encode(user_text)

    # Get all threads
    all_threads = db.query(Thread).all()

    # Convert thread embeddings from JSON strings to numpy arrays
    thread_embeddings = [np.array(json.loads(t.embedding)) for t in all_threads]

    # Calculate similarities
    similarities = [np.dot(user_embedding, emb) / (np.linalg.norm(user_embedding) * np.linalg.norm(emb))
                   for emb in thread_embeddings]

    # Get indices of top similar threads
    similar_indices = np.argsort(similarities)[::-1][:limit]

    # Return similar threads with their similarity scores
    similar_threads = []
    for idx in similar_indices:
        thread_dict = all_threads[idx].to_dict()
        thread_dict['similarity'] = float(similarities[idx])
        similar_threads.append(thread_dict)

    return similar_threads
