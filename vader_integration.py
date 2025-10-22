from __future__ import annotations
from typing import List

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def vader_compound_scores(texts: List[str]) -> List[float]:
    return [_analyzer.polarity_scores(t).get("compound", 0.0) for t in texts]


def vader_to_label(compound: float, pos_threshold: float = 0.05, neg_threshold: float = -0.05) -> str:
    if compound >= pos_threshold:
        return "positive"
    if compound <= neg_threshold:
        return "negative"
    return "neutral"
