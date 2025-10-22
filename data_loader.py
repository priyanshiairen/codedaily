from __future__ import annotations
import pandas as pd
from typing import Tuple


def load_messages(path: str, text_col: str = "message_text") -> pd.Series:
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_json(path, lines=False)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in {path}")
    return df[text_col].astype(str)
