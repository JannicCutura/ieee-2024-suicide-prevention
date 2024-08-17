import os
from enum import Enum
from pathlib import Path

import openai
import tiktoken
from openai import OpenAI
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src import paths
from src import prompts
from src.utils.logger import logger

tqdm.pandas()

client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Create DataFrame
np.random.seed(0)
categories = ["attempt", "indicator", "behaviour", "ideation"]


def plot_confusion_matrix(df, scale: float = 1, file_path: str = None):
    # Golden ratio
    phi = (1 + 5 ** 0.5) / 2
    width = scale * 10
    height = width / phi

    categories = list(set(df['post_risk'].to_list()))

    cm = confusion_matrix(df['post_risk'], df['pred'], labels=categories)
    plt.figure(figsize=(width, height))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if file_path:
        plt.savefig(file_path, dpi=300, bbox_inches='tight')


# Plot confusion matrix with scaling

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


def prompt_engineering(data: pd.DataFrame, client, output_file_path: Path, model_name: Models, system_prompt: str):
    posts = data.reset_index(names="original_index")

    responses = dict()
    for index, post in zip(posts['original_index'], posts.post):
        logger.info(f"processing {index + 1}/{len(posts)}")
        try:
            response = client.chat.completions.create(
                model=model_name.value,
                messages=[{"role": "system", "content": system_prompt},
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
    string_set = ['indicator', 'behaviour', 'behavior', 'attempt', 'ideation']
    for s in string_set:
        if s in input_string:
            return s
    return None


def analyse_result(df: pd.DataFrame):
    labels = list(set(df['post_risk'].to_list()))
    df['pred_original'] = [x for x in df['pred'].values]
    df['pred'] = df['pred'].str.lower()
    df['pred'] = df['pred'].str.replace(r'[^a-zA-Z0-9]', '', regex=True)
    # df['pred'] = df['pred'].apply(find_substring)

    replacements = {'behaviour': 'behavior',
                    ",": "",
                    "'": ""}

    df['pred'] = df['pred'].replace(replacements)
    pred_labels = list(set(df['pred'].to_list()))
    unknowns = [label for label in pred_labels if label not in labels]

    df['differs'] = df['pred'] != df['post_risk']
    difering = df[df['differs']]
    difering.to_excel("no_implicatsion.xlsx")
    print(classification_report(df['post_risk'], df["pred"], labels=labels))

    plot_confusion_matrix(difering, scale=0.7, file_path=paths.INTERMEDIATE_DATA_PATH / "confusion_matrix.png")


if __name__ == "__main__":
    from src.read_data import DataReader

    data = DataReader.get_posts_with_labels()

    model_name = Models.gpt_4_turbo

    prompt_fac = {'current_best': prompts.CURRENT_BEST,
                  'no_implication': prompts.NO_IMPLICATION,
                  'fxd': prompts.NO_IMPLICATION_SPELLING_FIXED}

    prompt = 'fxd'

    output_file_path = Path(paths.INTERMEDIATE_DATA_PATH / f"jannics_data5_{model_name.value}_{prompt}.parquet")

    data, _ = train_test_split(data, test_size=0.6, random_state=42, stratify=data["post_risk"])
    try:
        df = pd.read_parquet(output_file_path)
    except FileNotFoundError:
        df = prompt_engineering(
            data=data,
            client=client,
            output_file_path=output_file_path,
            model_name=model_name,
            system_prompt=prompt_fac[prompt])

    analyse_result(df)
