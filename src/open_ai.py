import os
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import openai
import pandas as pd
import tiktoken
from openai import OpenAI
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src import paths
from src.utils.logger import logger

tqdm.pandas()

client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)


class Models(Enum):
    gpt_35_turbo_0125 = "gpt-3.5-turbo-0125"
    gpt_35_turbo_1106 = "gpt-3.5-turbo-1106"
    gpt_35_turbo_0613 = "gpt-3.5-turbo-0613"
    babbage_002 = "babbage-002"
    davinci_002 = "davinci-002"
    gpt_4_0613 = "gpt-4-0613"
    gpt_4_mini = "gpt-4o-mini"
    gpt_4_turbo = "gpt-4-turbo"



import re


def approximate_token_count(text: str) -> int:
    # Split the text into words using regex to handle different types of delimiters
    words = re.findall(r'\b\w+\b', text)

    # Count the number of words
    word_count = len(words)

    # Approximate additional tokens for punctuation and special characters
    punctuation_count = len(re.findall(r'[^\w\s]', text))

    # Estimate the total number of tokens
    total_tokens = word_count + punctuation_count

    # a typical word is 1.3 tokens
    total_tokens = total_tokens * 1.3

    return int(total_tokens)


# Example usage
text = "This is an example sentence, with punctuation! How many tokens?"
print(approximate_token_count(text))


def num_tokens_from_messages(messages, model: Models = Models.gpt_35_turbo_0613):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model.value)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == Models.gpt_35_turbo_0613:  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.""")


class PromptRefiner:
    _post = "post"

    @classmethod
    def add_prompt(cls, df: pd.DataFrame) -> pd.DataFrame:
        prompt = "For this reddit post, indicate whether there is suicidal 'ideation', 'behaviour', 'attempt' or 'indicator': "
        common_separator = " ->"
        logger.info(f"Adding the following prompt: \n {prompt}")
        df[cls._post] = prompt + df[cls._post] + common_separator
        return df


def prompt_engineering(data: pd.DataFrame, client, output_file_path: Path, model_name: Models):
    system_settings = """
    Based on the social media post provided in the user prompt, return the category that the post belongs to based on 
    the following four categories and their definitions.
    'indicator': The post content has no explicit expression concerning suicide.
    'ideation': The post content has explicit suicidal expression, but there is no plan to commit suicide.
    'behaviour': The post content has explicit suicidal expression and a plan to commit suicide or self-harming behaviours.
    'attempt': The post content has explicit expressions concerning historic suicide attempts. 
    
    i.e. you answer should be one of 'indicator', 'ideation', 'behaviour', 'attempt'    
    """

    posts = data.reset_index(names="original_index")

    responses = dict()
    for index, post in zip(posts['original_index'], posts.post):
        logger.info(f"processing {index + 1}/{len(posts)}")
        try:
            response = client.chat.completions.create(
                model=model_name.value,
                messages=[{"role": "system", "content": system_settings},
                          {"role": "user", "content": post}],
                temperature=1,
                n=1,
                max_tokens=50
            )
            responses[index]: openai.ChatCompletion = response
        except:
            pass

    predictions = dict()

    for index, response in responses.items():
        try:
            prediction = response.choices[0].message.content
            predictions[index] = prediction
        except:
            pass

    y_pred = [pred.lower() for pred in predictions.values()]

    posts['pred'] = posts['original_index'].map(predictions)
    posts.to_parquet(output_file_path)
    logger.info(f"Persisted to {output_file_path}")

    return posts


def create_new_posts(data: pd.DataFrame, client) -> pd.DataFrame:
    collector = dict()

    @dataclass
    class Post:
        index: int
        post: str
        post_risk: str
        new_posts: list[str] | None = None

    progress_bar = tqdm(data.iterrows(), total=data.shape[0])
    for index, row in progress_bar:
        progress_bar.set_postfix(index=index)
        post = row['post']
        post_risk = row['post_risk']

        user_msg = f"{post}"
        num_tokens = approximate_token_count(user_msg)
        try:
            response = client.chat.completions.create(
                model=Models.gpt_4_0613.value,
                messages=[{"role": "system",
                           "content": "You are an expert at rephrasing text while maintaining the original meaning and tonality. I have a set of of depressed/suicical reddit posts that I need to rephrase. Ensure that the rephrased text retains the same sentiment, style, and tone as the original"},
                          {"role": "user", "content": user_msg}],
                temperature=1,
                n=10,
                frequency_penalty=0.5,
                max_tokens=int(random.uniform(0.75, 1.25) * num_tokens)
            )

            generated_posts = [choice.message.content for choice in response.choices]
        except Exception as e:
            logger.error(f"Unexpected Error in {index}: \n{e}")
            generated_posts = []

        collector[index] = Post(index, post, post_risk, generated_posts)

    new_posts_df_list: list[pd.DataFrame] = []

    for index, value in collector.items():
        if value.new_posts:
            new_posts_df = pd.DataFrame({
                'post': value.new_posts,
                'post_risk': [value.post_risk] * len(value.new_posts)}
            )

            new_posts_df_list.append(new_posts_df)

    new_posts = pd.concat(new_posts_df_list)

    return new_posts


def get_embedding(client: OpenAI, text: str, model="text-embedding-3-large"):
    try:
        text = text.replace("\n", " ")
        return client.embeddings.create(input=[text], model=model).data[0].embedding
    except:
        return None


def create_embeddings(data: pd.DataFrame, client: OpenAI, file_name: str):
    data['embedding'] = data['post'].progress_apply(lambda x: get_embedding(client, x, model='text-embedding-3-small'))
    file_path = paths.INTERMEDIATE_DATA_PATH / file_name
    logger.info(f"Writing to {file_path}")
    data.to_parquet(file_path)

def find_substring(input_string):
    string_set =['indicator', 'behaviour', 'behavior', 'attempt', 'ideation']
    for s in string_set:
        if s in input_string:
            return s
    return None


def analyse_result(df: pd.DataFrame):
    labels = list(set(df['post_risk'].to_list()))
    df['pred_original'] = [x for x in df['pred'].values]
    df['pred'] = df['pred'].str.replace(r'[^a-zA-Z0-9]', '', regex=True)
    df['pred'] = df['pred'].apply(find_substring)

    replacements = {'behaviour': 'behavior',
                    ",": "",
                    "'": ""}

    df['pred'] = df['pred'].replace(replacements)
    pred_labels = list(set(df['pred'].to_list()))
    unknowns = [label for label in pred_labels if label not in labels]

    print(classification_report(df['post_risk'], df["pred"], labels=labels))


if __name__ == "__main__":
    from src.read_data import DataReader

    data = DataReader.get_posts_with_labels()

    model_name = Models.gpt_4_turbo

    output_file_path = Path(paths.INTERMEDIATE_DATA_PATH / f"jannics_data5_{model_name.value}_30_.parquet")

    data, _ = train_test_split(data, test_size=0.6, random_state=42, stratify=data["post_risk"])
    try:
        df = pd.read_parquet(output_file_path)
    except FileNotFoundError:
        df = prompt_engineering(
            data=data,
            client=client,
            output_file_path=output_file_path,
            model_name=model_name)

    analyse_result(df)
