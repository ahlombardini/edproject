

from asyncio import wait
import time
from edapi import EdAPI
import pandas as pd
from dotenv import load_dotenv
import os
load_dotenv()
api = EdAPI()
api.login()

def extract_and_save_questions(course_id=1932,limit=100,offset=0):

    # 1) Fetch threads
    threads = api.list_threads(course_id=course_id,limit=limit,offset=offset)
    # 2) Convert to DataFrame
    df = pd.DataFrame(threads)
    df = df[df['is_private']==False]
    columns = ['id','content','document','title','type','category','subcategory']
    df = df[columns]
    # 3) Filter only rows where type is 'question' (adjust if needed)
    questions_df = df[df['type'] == 'question']
    # Filter for non-private questions
    # If you want to save *each question* to its own CSV:
    for _, row in questions_df.iterrows():

        thread_id = row['id']
        print(f"Extracting thread {thread_id}")
        # Build a directory and filename
        filename = f"./data/threads/{thread_id}.csv"
        print(f"Saving to {filename}")
        print(f"Directory full path {os.path.dirname(filename)}")
        # Create a 1-row DataFrame for this thread
        single_thread_df = pd.DataFrame([row])

        # Write to CSV
        single_thread_df.to_csv(filename, index=False)
        with open("data/threads/threads.txt", "a") as f:
            f.write(f"{thread_id}\n")

    # Alternatively, if you want to save *all questions* into one CSV:
    # questions_df.to_csv("../../data/threads/all_questions.csv", index=False)


for i in range(0,1000,100):
    extract_and_save_questions(offset=i)
    time.sleep(3)
