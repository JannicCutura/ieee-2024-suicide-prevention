import os
from pathlib import Path

_current_file_path = Path(os.path.abspath(__name__))
ROOT_PATH = _current_file_path.parent
DATA_PATH = ROOT_PATH / "data"
