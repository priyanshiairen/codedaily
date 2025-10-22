from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from .features import build_vectorizer


@dataclass
class TrainResult:
    vectorizer: Any
    classifier: Any
    report: str
    metrics: Dict[str, float]


def train_lr_classifier(texts: list[str], labels: list[str], test_size: float = 0.2, random_state: int = 42) -> TrainResult:
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_size, random_state=random_state, stratify=labels)

    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    clf = LogisticRegression(max_iter=200, n_jobs=1)
    clf.fit(X_train_vec, y_train)

    X_test_vec = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)

    return TrainResult(
        vectorizer=vectorizer,
        classifier=clf,
        report=report,
        metrics={"accuracy": float(acc), "precision": float(precision), "recall": float(recall), "f1": float(f1)},
    )


def save_pipeline(vectorizer, classifier, path: str) -> None:
    joblib.dump({"vectorizer": vectorizer, "classifier": classifier}, path)


def load_pipeline(path: str):
    obj = joblib.load(path)
    return obj["vectorizer"], obj["classifier"]


def predict_labels(vectorizer, classifier, texts: list[str]) -> list[str]:
    X = vectorizer.transform(texts)
    return classifier.predict(X).tolist()
