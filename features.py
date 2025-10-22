from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_NGRAM_RANGE = (1, 2)


def build_vectorizer(ngram_range: Tuple[int, int] = DEFAULT_NGRAM_RANGE, max_features: int | None = 20000) -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=ngram_range, max_features=max_features, min_df=1)
