from __future__ import annotations
from typing import List, Dict

TOXIC_TERMS = {
    "idiot", "stupid", "shut up", "dumb", "useless", "hate", "angry", "annoying", "toxic",
    "trash", "wtf", "sucks", "terrible", "awful", "jerk", "lazy"
}


def is_toxic(message: str) -> bool:
    msg = message.lower()
    return any(term in msg for term in TOXIC_TERMS)


def flag_conflict(labels: List[str], messages: List[str], neg_window: int = 3) -> List[Dict]:
    flags: List[Dict] = []
    consecutive_negatives = 0
    for idx, (label, msg) in enumerate(zip(labels, messages)):
        if label.lower() == "negative":
            consecutive_negatives += 1
        else:
            consecutive_negatives = 0

        toxic = is_toxic(msg)
        conflict = consecutive_negatives >= neg_window or toxic
        flags.append({
            "index": idx,
            "label": label,
            "toxic": toxic,
            "consecutive_negative_count": consecutive_negatives,
            "conflict": conflict,
        })
    return flags
