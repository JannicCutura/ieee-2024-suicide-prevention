import html
import re

import pandas as pd


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


import re


def count_words(text):
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
    def _clean_text(df: pd.DataFrame) -> pd.DataFrame:
        df['post'] = df['post'].apply(clean_text)
        return df

    @staticmethod
    def _count_words(df: pd.DataFrame) -> pd.DataFrame:
        df['word_count'] = df['post'].apply(count_words)
        return df

    @classmethod
    def preprocess(cls, df: pd.DataFrame):
        df = cls._clean_text(df)
        df = cls._count_words(df)
        return df
