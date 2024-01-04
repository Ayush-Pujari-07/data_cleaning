import os
import ai21
import pandas as pd
import logging
import argparse
import asyncio

from datetime import datetime
from transformers import pipeline
from dotenv import load_dotenv, find_dotenv
from pymongo.mongo_client import MongoClient

# For Local system
LOG_FILE_FOLDER = f"{datetime.now().strftime('%m_%d_%Y')}"
LOG_FILE = f"{datetime.now().strftime('%H:%M:%S')}.log"
logs_path = os.path.join(os.getcwd(), 'logs', LOG_FILE_FOLDER)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d - %(filename)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s",
    level=logging.INFO,
)

load_dotenv(find_dotenv())

ai21.api_key = os.environ.get("AI21_API_KEY")

client = MongoClient(
    "mongodb+srv://tanweer:tanweer@cluster0.i5668cw.mongodb.net/?retryWrites=true&w=majority")

logging.info("Initialized mongodb client")

global summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn",
                      tokenizer="facebook/bart-large-cnn", device=0)

logging.info("Initialized summarizer")


def get_data(start, end):
    data_collection = []
    video_id = []
    for indexId in range(start, end):
        try:
            transcript_Data = client['vimeo_database']['vimeo_collection_english'].find_one({
                                                                                            "indexId": indexId})
            data_collection.append(transcript_Data['transcript'])
            video_id.append(transcript_Data['videoID'])
        except:
            pass

    logging.info(
        f"data collection: \n {video_id}  and data collection length is:  {len(data_collection)}")

    return data_collection, video_id


def generate_chunks_and_summarize(transcript, video_id):

    # Set the maximum token limit for each chunk
    max_chunk_tokens = 1020

    # Tokenize the entire transcript
    tokens = summarizer.tokenizer.encode(transcript, add_special_tokens=False)
    logging.info(f"Token size for video id {video_id} is {len(tokens)}.")

    # Initialize an empty list to store individual chunk summaries
    chunk_summaries = []

    # Generate chunks and summarize
    for i in range(0, len(tokens), max_chunk_tokens):
        # Extract a chunk of tokens
        chunk_tokens = tokens[i:i+max_chunk_tokens]

        # Decode the chunk to text
        chunk_text = summarizer.tokenizer.decode(chunk_tokens)

        # Summarize the chunk
        chunk_summary = summarizer(chunk_text, max_length=200, min_length=100,
                                   length_penalty=2.0, num_beams=4, early_stopping=True)[0]['summary_text']

        # Append the summary to the list
        chunk_summaries.append(chunk_summary)

    # Concatenate the individual summaries into a new paragraph
    summary_paragraph = ' '.join(chunk_summaries)

    logging.info(
        f"Summary paragraph length for video id {video_id} is: {len(summarizer.tokenizer.encode(summary_paragraph, add_special_tokens=False))}")

    return summary_paragraph


def generate_questions(summarized_text, video_id):
    prompt = f"""
As an IT instructor, you want to encourage students to express their doubts and seek clarification. Develop '3' doubt-oriented questions in 2nd person related to both programming and theoretical concepts based on the following text:

{summarized_text}

Ensure that the questions do not reference any specific videos, maintain a 2nd person perspective, and refrain from providing answers.

Output questions JSON format example:
{{
    "question1": "[your generated doubt-oriented question here]",
    "question2": "[your generated doubt-oriented question here]",
    "question3": "[your generated doubt-oriented question here]",
}}
"""

    question_gen = ai21.Completion.execute(
        model='j2-ultra',
        prompt=prompt,
        numResults=1,
        maxTokens=300,
        temperature=0.3,
        topKReturn=0,
        top_p=1,
        stopSequence=['##']
    )

    logging.info(
        f"Generated questions for video id {video_id}: \n{question_gen['completions'][0]['data']['text']}")
    try:
        generated_text = eval(question_gen['completions'][0]['data']['text'])

        return generated_text

    except Exception as e:
        logging.error(
            f"Error while generating questions for video id {video_id} and exception as: {e}")
        pass


def run_pipeline(start, end):
    transcript_data_list, video_id = get_data(start, end)
    # QA = []

    # summarized_list = start_summarizing(transcript_data_list)

    global ques_df
    ques_df = pd.DataFrame(
        columns=["video_id", "question1", "question2", "question3"])

    i = 0

    for text, vid_id in zip(transcript_data_list, video_id):
        i = i+1
        logging.info(f"No of videos processed: {i}")

        if len(summarizer.tokenizer.encode(text)) > 1000:
            logging.info(f"video_id : {vid_id}")
            summarized_text = generate_chunks_and_summarize(text, vid_id)
            gen_ques = generate_questions(summarized_text, vid_id)
            
            if gen_ques:

                if "question3" not in gen_ques.keys():
                    gen_ques["question3"] = "None"

                    if "question2" not in gen_ques.keys():
                        gen_ques["question2"] = "None"

                        if "question1" not in gen_ques.keys():
                            gen_ques["question1"] = "None"

                            temp_df = pd.DataFrame([{"video_id": vid_id, "question1": gen_ques["question1"],
                                                     "question2": gen_ques["question2"], "question3": gen_ques["question3"]}])
                            ques_df = pd.concat(
                                [ques_df, temp_df], axis=0, ignore_index=True)

                        temp_df = pd.DataFrame([{"video_id": vid_id, "question1": gen_ques["question1"],
                                                 "question2": gen_ques["question2"], "question3": gen_ques["question3"]}])
                        ques_df = pd.concat(
                            [ques_df, temp_df], axis=0, ignore_index=True)

                    temp_df = pd.DataFrame([{"video_id": vid_id, "question1": gen_ques["question1"],
                                             "question2": gen_ques["question2"], "question3": gen_ques["question3"]}])
                    ques_df = pd.concat([ques_df, temp_df],
                                        axis=0, ignore_index=True)

                # QA.append({"Video_id":vid_id,"QA":gen_ques})

                else:
                    temp_df = pd.DataFrame([{"video_id": vid_id, "question1": gen_ques["question1"],
                                             "question2": gen_ques["question2"], "question3": gen_ques["question3"]}])
                    ques_df = pd.concat([ques_df, temp_df],
                                        axis=0, ignore_index=True)
                    # QA.append({"Video_id":vid_id,"QA":gen_ques})
            else:
                pass
    logging.info(f"ques_df: \n{ques_df}")
    ques_df.to_csv(f"./csv_output/ques_{start}_{end}.csv", index=False)

    # return ques_df:
if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--start", type=int)
    parse.add_argument("--end", type=int)
    start = parse.parse_args().start
    end = parse.parse_args().end
    logging.info(
        f"Start value is: {start} and end value is: {end}...."
    )
    run_pipeline(start, end)
