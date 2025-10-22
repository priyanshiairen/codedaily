from __future__ import annotations
from typing import List
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def plot_sentiment_distribution(labels: List[str], save_path: str | None = None):
    counts = Counter([lbl.lower() for lbl in labels])
    labels_sorted = sorted(counts.keys())
    values = [counts[l] for l in labels_sorted]

    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels_sorted, y=values, palette="viridis")
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return counts


def plot_conflict_timeline(labels: List[str], save_path: str | None = None):
    # Map: positive=1, neutral=0, negative=-1
    mapping = {"positive": 1, "neutral": 0, "negative": -1}
    series = [mapping.get(lbl.lower(), 0) for lbl in labels]

    plt.figure(figsize=(8, 3))
    plt.plot(series, marker="o", linestyle="-", color="#cc444b")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("Sentiment Timeline (1=pos, 0=neu, -1=neg)")
    plt.xlabel("Message Index")
    plt.ylabel("Sentiment Score")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return series
