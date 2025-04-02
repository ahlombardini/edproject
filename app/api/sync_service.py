import os
import json
import time
import logging
import pandas as pd
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

    def initial_sync(self):
        """Perform initial sync to populate database with all threads."""
        logger.info("Starting initial sync to populate database...")
        db = SessionLocal()
        try:
            # Fetch all threads from ED Stem without date filtering
            threads = self.edstem_client.get_threads(limit=1000)  # Get a large batch
            logger.info(f"Retrieved {len(threads)} threads from ED Stem")

            if not threads:
                logger.warning("No threads retrieved from ED Stem")
                return

            # Process the threads
            df = pd.DataFrame(threads)

            # Filter non-private questions only
            if 'is_private' in df.columns:
                df = df[df['is_private'] == False]

            # Select only relevant columns
            columns = ['id', 'content', 'document', 'title', 'type', 'category', 'subcategory']
            for col in columns:
                if col not in df.columns:
                    df[col] = None

            df = df[columns]
            questions_df = df[df['type'] == 'question']

            # Process each thread
            for _, row in questions_df.iterrows():
                thread_id = str(row['id'])

                # Check if thread already exists
                existing_thread = db.query(ThreadModel).filter(
                    ThreadModel.ed_thread_id == thread_id
                ).first()

                if existing_thread:
                    continue

                # Get thread details
                thread_details = self.edstem_client.get_thread_details(thread_id)

                # Extract thread info
                thread_info = {
                    'ed_thread_id': thread_id,
                    'title': row.get('title', ''),
                    'content': row.get('content', ''),
                    'document': row.get('document', ''),
                    'category': self._extract_name(row.get('category')),
                    'subcategory': self._extract_name(row.get('subcategory')),
                    'content_and_img_desc': self._extract_content(thread_details),
                    'created_at': datetime.now(),
                    'updated_at': datetime.now()
                }

                # Generate embedding
                combined_text = self._combine_text(thread_info)
                embedding = self.model.encode(combined_text)
                thread_info['embedding'] = json.dumps(embedding.tolist())

                # Create new thread
                thread = ThreadModel(**thread_info)
                db.add(thread)

                # Commit every few threads
                if int(thread_id) % 10 == 0:
                    db.commit()
                    logger.info(f"Processed {thread_id}")

            # Final commit
            db.commit()
            logger.info("Initial sync completed successfully")

        except Exception as e:
            logger.error(f"Error during initial sync: {str(e)}")
            db.rollback()
            raise
        finally:
            db.close()

    def _combine_text(self, thread_info):
        """Combine thread text fields for embedding."""
        return f"{thread_info['title']} {thread_info['document']} {thread_info['content_and_img_desc']}"

    def _extract_content(self, thread_details):
        """Extract content and image descriptions from thread details."""
        if not thread_details:
            return ""
        # Add your content extraction logic here
        return str(thread_details.get('content', ''))

    def start(self):
        """Start the sync service in a separate thread."""
        if self.is_running:
            logger.warning("Sync service is already running")
            return

        self.is_running = True
        self.thread = Thread(target=self._sync_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Started sync service, checking every {self.sync_interval} minutes, processing max {self.max_threads_per_sync} threads per sync")

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
                logger.info("Starting sync with ED Stem API")
                self._sync_threads()
                logger.info("Sync completed, sleeping...")
            except Exception as e:
                logger.error(f"Error during sync: {str(e)}")

            # Sleep for the specified interval
            for _ in range(self.sync_interval * 60):
                if not self.is_running:
                    break
                time.sleep(1)

    def _get_last_sync_time(self, db):
        """Get the time of the last successful sync."""
        latest_thread = db.query(ThreadModel).order_by(ThreadModel.updated_at.desc()).first()
        if latest_thread:
            # Convert to ISO format and subtract 1 day to avoid missing any threads
            last_sync = latest_thread.updated_at - timedelta(days=1)
            return last_sync.isoformat()
        return None

    def _sync_threads(self):
        """Sync threads from ED Stem to our database."""
        db = SessionLocal()
        try:
            # Check if we can make a request
            if not self.rate_limiter.can_make_request():
                next_available = self.rate_limiter.get_next_available_time()
                logger.warning(f"Rate limit reached. Next request can be made at {next_available}")
                return

            # Get last sync time
            from_date = self._get_last_sync_time(db)

            # Fetch threads from ED Stem - only make this request if we have quota
            self.rate_limiter.record_request()
            threads = self.edstem_client.get_threads(from_date=from_date, limit=self.max_threads_per_sync)
            logger.info(f"Retrieved {len(threads)} threads from ED Stem")

            # Process the threads similar to get_threads.py
            # 1. Convert to DataFrame
            df = pd.DataFrame(threads)

            # 2. Filter non-private questions only - matching get_threads.py
            if 'is_private' in df.columns:
                df = df[df['is_private'] == False]

            # 3. Select only relevant columns
            columns = ['id', 'content', 'document', 'title', 'type', 'category', 'subcategory']
            for col in columns:
                if col not in df.columns:
                    df[col] = None  # Add missing columns with None values

            df = df[columns]

            # 4. Filter only question type threads - matching get_threads.py
            questions_df = df[df['type'] == 'question']

            # We can process more threads now with the higher rate limit
            # Limit the questions to process based on our rate limit
            available_requests = self.rate_limiter.max_requests - 1  # -1 because we already used 1 request
            questions_to_process = questions_df.head(min(available_requests, len(questions_df)))

            logger.info(f"Found {len(questions_df)} question threads, processing {len(questions_to_process)}")

            # Process each question thread
            for _, row in questions_to_process.iterrows():
                thread_id = row['id']

                # Check if thread already exists
                existing_thread = db.query(ThreadModel).filter(
                    ThreadModel.ed_thread_id == str(thread_id)
                ).first()

                # If thread exists and is recent, skip it
                if existing_thread and existing_thread.updated_at > datetime.now() - timedelta(days=7):
                    logger.info(f"Skipping recently updated thread {thread_id}")
                    continue

                # Get detailed thread information only if needed and if we still have quota
                if not self.rate_limiter.can_make_request():
                    logger.info("Rate limit reached, stopping processing for now")
                    break

                # Make the API call for details and record it
                self.rate_limiter.record_request()
                thread_details = self.edstem_client.get_thread_details(thread_id)
                logger.info(f"Fetched details for thread {thread_id}")

                # Extract relevant information from both row and thread_details
                thread_info = {
                    'ed_thread_id': str(thread_id),
                    'title': row.get('title', ''),
                    'content': row.get('content', ''),
                    'document': row.get('document', ''),
                    'category': self._extract_name(row.get('category')),
                    'subcategory': self._extract_name(row.get('subcategory')),
                    'content_and_img_desc': self._extract_content(thread_details),
                    'updated_at': datetime.now()
                }

                # Generate embedding
                combined_text = self._combine_text(thread_info)
                embedding = self.model.encode(combined_text)
                thread_info['embedding'] = json.dumps(embedding.tolist())

                if existing_thread:
                    # Update existing thread
                    for key, value in thread_info.items():
                        setattr(existing_thread, key, value)
                else:
                    # Create new thread
                    thread_info['created_at'] = datetime.now()
                    thread = ThreadModel(**thread_info)
                    db.add(thread)

                # Commit after each thread to avoid losing progress
                db.commit()
                logger.info(f"Processed thread {thread_info['ed_thread_id']}")

                # Add a small delay between processing threads
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error syncing threads: {str(e)}")
            db.rollback()
            raise
        finally:
            db.close()

    def _extract_name(self, obj):
        """Extract name from category or subcategory object."""
        if isinstance(obj, dict) and 'name' in obj:
            return obj.get('name', '')
        return str(obj) if obj else ''

# Singleton instance
sync_service = SyncService()
