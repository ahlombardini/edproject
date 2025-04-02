import os
import json
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from threading import Thread, Lock
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from app.api.edstem_client import EdStemClient
from app.database.database import SessionLocal
from app.models.thread import Thread as ThreadModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

class RateLimiter:
    """Rate limiter to ensure we don't exceed API limits."""
    def __init__(self, max_requests_per_hour=6):
        self.max_requests = max_requests_per_hour
        self.request_timestamps = []
        self.lock = Lock()

    def can_make_request(self):
        """Check if we can make a request based on our rate limits."""
        with self.lock:
            # Remove timestamps older than 1 hour
            current_time = datetime.now()
            one_hour_ago = current_time - timedelta(hours=1)
            self.request_timestamps = [ts for ts in self.request_timestamps
                                      if ts > one_hour_ago]

            # Check if we've made fewer requests than our limit in the past hour
            return len(self.request_timestamps) < self.max_requests

    def record_request(self):
        """Record that a request was made."""
        with self.lock:
            self.request_timestamps.append(datetime.now())

    def get_next_available_time(self):
        """Get the time when the next request slot will be available."""
        with self.lock:
            if not self.request_timestamps:
                return datetime.now()

            if len(self.request_timestamps) < self.max_requests:
                return datetime.now()

            # Sort timestamps and get the oldest one
            sorted_timestamps = sorted(self.request_timestamps)
            oldest = sorted_timestamps[0]

            # Next available time is 1 hour after the oldest request
            return oldest + timedelta(hours=1)

class SyncService:
    def __init__(self):
        self.edstem_client = EdStemClient()
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.rate_limiter = RateLimiter()
        self.max_threads_per_sync = int(os.getenv('MAX_THREADS_PER_SYNC', '2'))
        self.sync_interval = int(os.getenv('SYNC_INTERVAL_MINUTES', '360'))
        self.is_running = False
        self.thread = None

    def _sync_threads(self):
        """Sync threads from ED Stem."""
        logger.info("Starting sync with ED Stem API")
        db = SessionLocal()
        try:
            # Get threads as DataFrame
            questions_df = self.edstem_client.get_threads(limit=self.max_threads_per_sync)

            logger.info(f"Retrieved {len(questions_df)} question threads from ED Stem")

            if questions_df.empty:
                logger.warning("No question threads retrieved from ED Stem")
                return

            # Process each thread
            for _, row in questions_df.iterrows():
                thread_id = str(row['id'])

                # Check if thread already exists
                existing_thread = db.query(ThreadModel).filter(
                    ThreadModel.ed_thread_id == thread_id
                ).first()

                if existing_thread:
                    continue

                # Extract thread info directly from row
                thread_info = {
                    'ed_thread_id': thread_id,
                    'title': row['title'],
                    'content': row['content'],
                    'document': row['document'],
                    'category': row['category'],
                    'subcategory': row['subcategory'],
                    'content_and_img_desc': row['content'],
                    'created_at': datetime.now(),
                    'updated_at': datetime.now()
                }

                # Generate embedding using the weighted category approach
                embedding = embed_thread_with_category(thread_info, self.model)
                thread_info['embedding'] = json.dumps(embedding.tolist())

                # Create new thread
                new_thread = ThreadModel(**thread_info)
                db.add(new_thread)
                db.commit()
                logger.info(f"Added thread {thread_id}")

                # Rate limiting
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error syncing threads: {str(e)}")
            db.rollback()
            raise
        finally:
            db.close()

    def start(self):
        """Start the sync service."""
        if self.is_running:
            logger.warning("Sync service is already running")
            return

        self.is_running = True
        self.thread = Thread(target=self._sync_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Started sync service, checking every {self.sync_interval} minutes")

    def stop(self):
        """Stop the sync service."""
        if not self.is_running:
            logger.warning("Sync service is not running")
            return

        self.is_running = False
        if self.thread:
            self.thread.join(timeout=10)
        logger.info("Stopped sync service")

    def _sync_loop(self):
        """Main sync loop."""
        while self.is_running:
            try:
                self._sync_threads()
                logger.info("Sync completed, sleeping...")
            except Exception as e:
                logger.error(f"Error during sync: {str(e)}")

            # Sleep for the specified interval
            for _ in range(self.sync_interval * 60):
                if not self.is_running:
                    break
                time.sleep(1)

# Singleton instance
sync_service = SyncService()
