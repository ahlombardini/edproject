import os
import logging
import pandas as pd
from edapi import EdAPI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EdStemClient:
    def __init__(self):
        # Load environment variables first
        load_dotenv()

        # Initialize API and login
        self.api = EdAPI()
        self.api.login()

        self.course_id = "1932"
        logger.info(f"Successfully logged in to ED Stem API with course_id: {self.course_id}")

    def get_threads(self, from_date=None, limit=50, offset=0):
        """
        Get threads from ED Stem API
        Args:
            from_date (str): ISO format date string to filter threads from
            limit (int): Maximum number of threads to retrieve
            offset (int): Offset for pagination
        Returns:
            pd.DataFrame: DataFrame containing filtered thread data
        """
        # Get threads using the API
        threads = self.api.list_threads(course_id=self.course_id, limit=limit, offset=offset)

        # Convert to DataFrame
        df = pd.DataFrame(threads)

        # Filter non-private threads
        df = df[df['is_private'] == False]

        # Select relevant columns
        columns = ['id', 'content', 'document', 'title', 'type', 'category', 'subcategory']
        df = df[columns]

        # Filter question type threads
        questions_df = df[df['type'] == 'question']

        # Filter by date if needed
        if from_date:
            questions_df = questions_df[questions_df['created_at'] >= from_date]

        return questions_df
