from fastapi import FastAPI, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import List, Optional
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
from app.database.database import get_db, engine
from app.models.thread import Thread, Base
from app.api.sync_service import sync_service
from app.api.auth import get_api_key

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the sentence transformer model
model = SentenceTransformer('all-mpnet-base-v2')

@app.on_event("startup")
async def startup_event():
    """Start the sync service when the API starts, unless disabled."""
    # Check if sync service is disabled
    disable_sync = os.environ.get("DISABLE_SYNC", "0")
    if disable_sync != "1":
        sync_service.start()

@app.on_event("shutdown")
async def shutdown_event():
    """Stop the sync service when the API stops."""
    # Only try to stop if it's running
    if sync_service.is_running:
        sync_service.stop()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    print(f"\n🔍 Request: {request.method} {request.url}")
    print(f"Headers: {dict(request.headers)}")
    response = await call_next(request)
    return response

@app.get("/")
def read_root():
    return {"message": "Welcome to the ED API"}

@app.get("/threads/", response_model=List[dict])
def get_threads(skip: int = 0, limit: int = 10, db: Session = Depends(get_db), api_key: str = Depends(get_api_key)):
    threads = db.query(Thread).offset(skip).limit(limit).all()
    return [thread.to_dict() for thread in threads]

@app.get("/threads/{thread_id}")
def get_thread(thread_id: str, db: Session = Depends(get_db), api_key: str = Depends(get_api_key)):
    thread = db.query(Thread).filter(Thread.ed_thread_id == thread_id).first()
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread.to_dict()

@app.get("/search/similar/{thread_id}")
def find_similar_threads(thread_id: str, limit: int = 5, db: Session = Depends(get_db), api_key: str = Depends(get_api_key)):
    # Get the target thread
    thread = db.query(Thread).filter(Thread.ed_thread_id == thread_id).first()
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Get all threads
    all_threads = db.query(Thread).all()

    # Convert embeddings from JSON strings to numpy arrays and reshape to 2D
    target_embedding = np.array(json.loads(thread.embedding)).reshape(1, -1)
    thread_embeddings = [np.array(json.loads(t.embedding)).reshape(1, -1) for t in all_threads]

    # Calculate similarities using cosine_similarity
    similarities = [float(cosine_similarity(target_embedding, emb)[0][0]) for emb in thread_embeddings]

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
def find_similar_from_input(query: dict, limit: int = 5, db: Session = Depends(get_db), api_key: str = Depends(get_api_key)):
    """Find threads similar to a user-provided text input."""
    # Extract the user input text
    user_text = query.get("text", "")
    if not user_text:
        raise HTTPException(status_code=400, detail="Text input is required")

    # Generate embedding for the user input and reshape to 2D
    user_embedding = model.encode(user_text).reshape(1, -1)

    # Get all threads
    all_threads = db.query(Thread).all()

    # Convert thread embeddings from JSON strings to numpy arrays and reshape to 2D
    thread_embeddings = [np.array(json.loads(t.embedding)).reshape(1, -1) for t in all_threads]

    # Calculate similarities using cosine_similarity
    similarities = [float(cosine_similarity(user_embedding, emb)[0][0]) for emb in thread_embeddings]

    # Get indices of top similar threads
    similar_indices = np.argsort(similarities)[::-1][:limit]

    # Return similar threads with their similarity scores
    similar_threads = []
    for idx in similar_indices:
        thread_dict = all_threads[idx].to_dict()
        thread_dict['similarity'] = float(similarities[idx])
        similar_threads.append(thread_dict)

    return similar_threads

@app.get("/sync/status")
def sync_status(api_key: str = Depends(get_api_key)):
    """Check the status of the sync service."""
    disable_sync = os.environ.get("DISABLE_SYNC", "0")
    return {
        "running": sync_service.is_running,
        "sync_interval_minutes": sync_service.sync_interval,
        "disabled": disable_sync == "1"
    }

@app.get("/project/part/{part}")
def get_project_part(part: int, db: Session = Depends(get_db), api_key: str = Depends(get_api_key)):
    """Get a summary of distinct topics and their frequency for a project part."""
    if part not in range(1, 12):
        raise HTTPException(status_code=400, detail="Invalid project part number")

    # Get all threads for this project part
    threads = db.query(Thread).filter(
        Thread.category == "projet",
        Thread.subcategory == f"Étape {part}"
    ).all()
    if not threads:
        return {"topics": [], "total_threads": 0}

    # Convert embeddings from JSON strings to numpy arrays and reshape to 2D
    embeddings = [np.array(json.loads(thread.embedding)).reshape(1, -1) for thread in threads]

    # Use cosine similarity to group similar questions
    # We'll consider questions similar if their similarity is above 0.7
    SIMILARITY_THRESHOLD = 0.7
    topics = []
    used_indices = set()

    for i, emb in enumerate(embeddings):
        if i in used_indices:
            continue

        # Find all similar threads to this one
        similar_threads = []
        for j, other_emb in enumerate(embeddings):
            if j in used_indices:
                continue

            similarity = float(cosine_similarity(emb, other_emb)[0][0])
            if similarity >= SIMILARITY_THRESHOLD:
                similar_threads.append(threads[j])
                used_indices.add(j)

        # Create a topic entry
        if similar_threads:
            # Use the first thread as the representative question
            representative = similar_threads[0]
            topics.append({
                "title": representative.title,
                "thread_count": len(similar_threads),
                "related_threads": [t.title for t in similar_threads[1:]]
            })

    # Sort topics by thread count in descending order
    topics.sort(key=lambda x: x["thread_count"], reverse=True)

    return {
        "topics": topics,
        "total_threads": len(threads)
    }

@app.post("/sync/trigger")
def trigger_sync(api_key: str = Depends(get_api_key)):
    """Manually trigger a sync."""
    disable_sync = os.environ.get("DISABLE_SYNC", "0")
    if disable_sync == "1":
        return {"status": "error", "message": "Sync service is disabled"}

    if not sync_service.is_running:
        return {"status": "error", "message": "Sync service is not running"}

    # Run sync in a separate thread
    import threading
    threading.Thread(target=sync_service._sync_threads).start()

    return {"status": "success", "message": "Sync triggered"}

@app.get("/threads/category/{category}")
def get_threads_by_category(
    category: str,
    subcategory: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 50,
    skip: int = 0,
    db: Session = Depends(get_db),
    api_key: str = Depends(get_api_key)
):
    """
    Get threads filtered by category and optionally by subcategory and text search.

    Parameters:
    - category: Category to filter by (e.g., "Projet", "Cours")
    - subcategory: Optional subcategory to further filter results
    - query: Optional text query to search for (uses semantic search with embeddings)
    - limit: Maximum number of threads to return
    - skip: Number of threads to skip (for pagination)

    Returns:
    - List of thread objects with count metadata
    """
    # Build the base query for category filters
    base_query = db.query(Thread).filter(Thread.category == category)

    # Add subcategory filter if provided
    if subcategory:
        base_query = base_query.filter(Thread.subcategory == subcategory)

    # Get filtered threads based on category/subcategory
    filtered_threads = base_query.all()

    # If there's a search query, use semantic search to find relevant threads
    if query and filtered_threads:
        # Generate embedding for the user's query
        user_embedding = model.encode(query).reshape(1, -1)

        # Process each thread to calculate similarity
        threads_with_similarity = []
        for thread in filtered_threads:
            # Convert thread embedding from JSON string to numpy array
            thread_embedding = np.array(json.loads(thread.embedding)).reshape(1, -1)

            # Calculate similarity
            similarity = float(cosine_similarity(user_embedding, thread_embedding)[0][0])

            # Create a thread dict with similarity score
            thread_dict = thread.to_dict()
            thread_dict['similarity'] = similarity
            threads_with_similarity.append(thread_dict)

        # Sort by similarity (highest first)
        threads_with_similarity.sort(key=lambda x: x['similarity'], reverse=True)

        # Apply pagination
        total_count = len(threads_with_similarity)
        result_threads = threads_with_similarity[skip:skip + limit]

    else:
        # No search query, just apply normal pagination to the filtered threads
        total_count = base_query.count()
        paginated_threads = base_query.order_by(Thread.created_at.desc()).offset(skip).limit(limit).all()
        result_threads = [thread.to_dict() for thread in paginated_threads]

    # Return the result with metadata
    return {
        "threads": result_threads,
        "total_count": total_count,
        "category": category,
        "subcategory": subcategory,
        "query": query,
        "limit": limit,
        "skip": skip
    }
