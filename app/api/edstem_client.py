import os
import requests
import json
from datetime import datetime
import sys
from dotenv import load_dotenv
import os.path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add scraper directory to path so we can import from it
scraper_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scraper')
sys.path.append(scraper_path)

# Load environment from the root .env file first, then scraper/.env as fallback
load_dotenv()
scraper_env_path = os.path.join(scraper_path, '.env')
if os.path.exists(scraper_env_path):
    load_dotenv(scraper_env_path)

# Try to import the existing EdAPI if available
try:
    from edapi import EdAPI
    USE_EXISTING_CLIENT = True
    logger.info("Using existing EdAPI client")
except ImportError:
    USE_EXISTING_CLIENT = False
    logger.warning("EdAPI module not found, using custom implementation")

class EdStemClient:
    def __init__(self):
        self.api_key = os.getenv("ED_API_TOKEN")
        self.base_url = os.getenv("ED_API_HOST", "https://eu.edstem.org/api")
        self.course_id = os.getenv("ED_COURSE_ID", "1932")  # Default from your get_threads.py

        if not self.api_key:
            raise ValueError("ED_API_TOKEN environment variable is not set")

        logger.info(f"Initializing EdStemClient with course_id: {self.course_id}")

        # If we can use the existing EdAPI client, initialize it
        if USE_EXISTING_CLIENT:
            try:
                self.api = EdAPI()
                # Explicitly call login
                success = self.api.login()
                if success:
                    logger.info("Successfully logged in to ED Stem API")
                else:
                    logger.error("Failed to log in to ED Stem API")
            except Exception as e:
                logger.error(f"Error initializing EdAPI client: {str(e)}")
                raise
        else:
            # Otherwise, use our own implementation
            self.headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json"
            }

    def get_threads(self, from_date=None, limit=50, offset=0):
        """
        Get threads from ED Stem API

        Args:
            from_date (str): ISO format date string to filter threads from
            limit (int): Maximum number of threads to retrieve
            offset (int): Offset for pagination

        Returns:
            list: List of thread objects
        """
        if USE_EXISTING_CLIENT:
            try:
                # Use existing client if available
                logger.info(f"Getting threads with EdAPI, course_id={self.course_id}, limit={limit}, offset={offset}")
                threads = self.api.list_threads(course_id=self.course_id, limit=limit, offset=offset)
                # Filter by date if needed
                if from_date:
                    # This is a simple filter - the actual implementation might need more logic
                    threads = [t for t in threads if t.get('created_at', '') >= from_date]
                return threads
            except Exception as e:
                logger.error(f"Error getting threads with EdAPI: {str(e)}")
                raise
        else:
            # Use our custom implementation
            url = f"{self.base_url}/courses/{self.course_id}/threads"

            params = {
                "limit": limit,
                "offset": offset
            }

            if from_date:
                params["from"] = from_date

            response = requests.get(url, headers=self.headers, params=params)

            if response.status_code != 200:
                raise Exception(f"ED Stem API error: {response.status_code} - {response.text}")

            return response.json().get("threads", [])

    def get_thread_details(self, thread_id):
        """
        Get detailed information about a specific thread

        Args:
            thread_id (str): Thread ID

        Returns:
            dict: Thread details
        """
        if USE_EXISTING_CLIENT:
            try:
                # Check if the API client has a get_thread method
                if hasattr(self.api, 'get_thread'):
                    logger.info(f"Getting thread details with EdAPI, thread_id={thread_id}")
                    return self.api.get_thread(thread_id)
                else:
                    # Fall back to our implementation
                    logger.warning("EdAPI does not have get_thread method, falling back to custom implementation")
            except Exception as e:
                logger.error(f"Error getting thread details with EdAPI: {str(e)}")
                raise

        # Use our custom implementation
        url = f"{self.base_url}/courses/{self.course_id}/threads/{thread_id}"

        response = requests.get(url, headers=self.headers)

        if response.status_code != 200:
            raise Exception(f"ED Stem API error: {response.status_code} - {response.text}")

        return response.json().get("thread", {})
