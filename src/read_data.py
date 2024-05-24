from pathlib import Path

import pandas as pd

from src import paths


class DataReader:

    @staticmethod
    def _read_excel(file_path: Path) -> pd.DataFrame:
        return pd.read_excel(file_path).drop(columns="index")

    @classmethod
    def get_posts_with_labels(cls) -> pd.DataFrame:
        return cls._read_excel(paths.DATA_PATH / "posts with labels.xlsx")

    @classmethod
    def get_posts_without_labels(cls) -> pd.DataFrame:
        return cls._read_excel(paths.DATA_PATH / "posts without labels.xlsx")

    @classmethod
    def get_test_set(cls) -> pd.DataFrame:
        return cls._read_excel(paths.DATA_PATH / "test set.xlsx")
