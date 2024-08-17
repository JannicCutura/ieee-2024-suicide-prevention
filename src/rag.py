from enum import Enum
from pathlib import Path
from typing import Literal

import faiss
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from statistics import mode
from src import paths

tqdm.pandas()
client = OpenAI()


class EmbeddingModels(Enum):
    test_embedding_3_small = "text-embedding-3-small"
    test_embedding_3_large = "text-embedding-3-large"
    test_embedding_ada_002 = "text-embedding-ada-002"


def get_embedding(text: str, model: str):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def create_embeddings(df: pd.DataFrame, model: EmbeddingModels, file_path: Path) -> pd.DataFrame:
    df['embedding'] = df['post'].progress_apply(lambda x: get_embedding(x, model=model.value))
    df.to_parquet(file_path)
    return df


class FaissMatch:

    def __init__(self, labels: list[str], distance: list[float]):
        self._labels = labels
        self._distance = distance

    @property
    def closest_match(self) -> Literal['ideation', 'behavior', 'attempt', 'indicator']:
        labels = self._labels
        distances = self._distance

        class_imbalance = {
            "indicator": 0.26,
            "ideation": 0.38,
            "behavior": 0.28,
            "attempt": 0.08,
        }

        score_per_category = {
            "indicator": 0,
            "ideation": 0,
            "behavior": 0,
            "attempt": 0,
        }

        for label, distance in zip(labels, distances):
            multiplier = class_imbalance[label]
            score_per_category[label] = score_per_category[label] + multiplier*distance

        return min(score_per_category, key=score_per_category.get)


def find_closest_matches(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    test['matches'] = None
    for row_index, row in test.iterrows():
        this_embedding = row['embedding']

        # dimension of the embeddings
        index = faiss.IndexFlatL2(3072)
        index.add(np.array(train['embedding'].tolist()))

        distances, indices = index.search(np.array([this_embedding]), k=10)
        closest_matches = train.iloc[indices[0]]
        closest_matches['distance'] = distances[0]

        faiss_matches = FaissMatch(
            closest_matches['post_risk'].to_list(),
            closest_matches['distance'].to_list()
        )
        faiss_match = faiss_matches.closest_match

        test.at[row_index, "matches"] = faiss_match

    test.head()

    incorrect_ones = [x for x in test['matches'] if x not in ['ideation', 'behavior', 'attempt', 'indicator']]
    if incorrect_ones:
        print(incorrect_ones)

    print(classification_report(test['post_risk'], test["matches"], labels=list(set(test['post_risk'].tolist()))))

    return train


def analyse_results_embedding(df: pd.DataFrame):
    pass


if __name__ == "__main__":
    from src.read_data import DataReader

    data = DataReader.get_posts_with_labels().head(500)

    model = EmbeddingModels.test_embedding_3_large

    embedding_file_train = Path(paths.INTERMEDIATE_DATA_PATH / f"embedding_train_{model.value}_all.parquet")
    embedding_file_val = Path(paths.INTERMEDIATE_DATA_PATH / f"embedding_val_{model.value}_all.parquet")

    df_train, df_test = train_test_split(data, test_size=0.2, random_state=42, stratify=data["post_risk"])

    try:
        df_train = pd.read_parquet(embedding_file_train)
        df_test = pd.read_parquet(embedding_file_val)
    except FileNotFoundError:
        df_train = create_embeddings(df=df_train, model=model, file_path=embedding_file_train)
        df_test = create_embeddings(df=df_test, model=model, file_path=embedding_file_val)

    df = find_closest_matches(df_train, df_test)

    analyse_results_embedding(df)
