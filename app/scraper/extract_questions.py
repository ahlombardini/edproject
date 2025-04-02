
from edapi import EdAPI
import pandas as pd
from dotenv import load_dotenv
import os


def extract_and_save_questions(course_id=1932,limit=100000):
    if load_dotenv():
        api = EdAPI()
        api.login()

        # 1) Fetch threads
        threads = api.list_threads(course_id=course_id,limit=limit)
        # 2) Convert to DataFrame
        df = pd.DataFrame(threads)

        # 3) Filter only rows where type is 'question' (adjust if needed)
        questions_df = df[df['type'] == 'question']
        # Filter for non-private questions
        questions_df = questions_df[questions_df['isprivate'] == 0]
        # If you want to save *each question* to its own CSV:
        for _, row in questions_df.iterrows():
            thread_id = row['id']
            # Build a directory and filename
            filename = f"../../data/threads/{thread_id}.csv"
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # Create a 1-row DataFrame for this thread
            single_thread_df = pd.DataFrame([row])

            # Write to CSV
            single_thread_df.to_csv(filename, index=False)
    else:
        print("Try again unable to load dotenv")

    # Alternatively, if you want to save *all questions* into one CSV:
    # questions_df.to_csv("../../data/threads/all_questions.csv", index=False)
