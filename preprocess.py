import re
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from text_unidecode import unidecode

# Ensure resources (safe if already present)
for pkg in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg)

_lemmatizer = WordNetLemmatizer()
_stopwords = set(stopwords.words("english"))

EMOJI_PATTERN = re.compile(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+", re.UNICODE)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]")


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unidecode(text)
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = EMOJI_PATTERN.sub(" ", text)
    text = NON_ALPHA_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> List[str]:
    tokens = text.split()
    lemmas: List[str] = []
    for tok in tokens:
        if tok in _stopwords:
            continue
        lemma = _lemmatizer.lemmatize(tok)
        if lemma and lemma not in _stopwords:
            lemmas.append(lemma)
    return lemmas


def preprocess_text(text: str) -> str:
    norm = normalize_text(text)
    tokens = tokenize_and_lemmatize(norm)
    return " ".join(tokens)
