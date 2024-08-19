import html
import re

import pandas as pd

from src.utils.logger import logger


def clean_text(text: str) -> str:
    """
    Clean text by removing HTML tags, non-ASCII characters, and special characters.

    Parameters:
    text (str): The input text to be cleaned.

    Returns:
    str: The cleaned text.
    """

    # Unescape HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Replace \n, \t, \r with a space
    text = re.sub(r'[\n\t\r]+', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def count_words(text: str) -> int:
    """
    Count the number of words in the input text.

    Parameters:
    text (str): The input text.

    Returns:
    int: The word count.
    """

    # Clean the text: remove special characters and extra spaces
    cleaned_text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Normalize whitespace

    # Split the text into words
    words = cleaned_text.split()

    # Return the word count
    return len(words)


class DataPreprocessor:

    @staticmethod
    def _convert_to_one_token_answer(df: pd.DataFrame) -> pd.DataFrame:
        shorting_map = {
            'ideation': 'id',
            'behavior': 'be',
            'indicator': 'in',
            'attempt': 'at',
        }

        if 'post_risk' in df.columns:
            logger.info(f"Replacing cateogries with one token labels: {str(shorting_map)}")
            df['post_risk'] = df['post_risk'].map(shorting_map)

        return df

    @staticmethod
    def _clean_text(df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Removing HTML, non ASCI etc")
        df['post'] = df['post'].apply(clean_text)
        return df

    @staticmethod
    def _count_words(df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Counting words")
        df['word_count'] = df['post'].apply(count_words)
        return df

    @classmethod
    def preprocess(cls, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocesssing data")
        df = cls._clean_text(df)
        df = cls._count_words(df)
        #df = cls._convert_to_one_token_answer(df)
        return df

    @classmethod
    def to_json(cls, df: pd.DataFrame, path):
        df = df.rename(columns={'post': 'prompt', 'post_risk': 'completion'})
        df[['prompt', 'completion']].to_json(path, orient='records', lines=True)
