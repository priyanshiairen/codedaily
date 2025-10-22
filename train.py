import argparse
import os
import pandas as pd
from collections import Counter

from src.preprocess import preprocess_text
from src.model import train_lr_classifier, save_pipeline
from src.visualize import plot_sentiment_distribution


def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression sentiment model")
    parser.add_argument("--data", required=True, help="Path to CSV with text and label columns")
    parser.add_argument("--text_col", default="message_text")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--out", default="model/artifacts.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    texts = df[args.text_col].astype(str).tolist()
    labels = df[args.label_col].astype(str).tolist()

    # Basic label sanity and stratification decision
    label_counts = Counter(labels)
    can_stratify = all(count >= 2 for count in label_counts.values()) and len(set(labels)) > 1

    cleaned = [preprocess_text(t) for t in texts]

    # Train with internal split logic that respects can_stratify
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned, labels, test_size=0.2 if len(labels) > 5 else 0.4, random_state=42,
        stratify=labels if can_stratify else None
    )

    # Fit vectorizer and classifier
    from src.features import build_vectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    clf = LogisticRegression(max_iter=200, n_jobs=1)
    clf.fit(X_train_vec, y_train)

    X_test_vec = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_pipeline(vectorizer, clf, args.out)

    print({"accuracy": float(acc), "precision": float(precision), "recall": float(recall), "f1": float(f1)})
    print(report)

    # Export a simple distribution plot of original labels
    try:
        from src.visualize import plot_sentiment_distribution
        plot_sentiment_distribution(labels, save_path="SentimentVisuals.png")
    except Exception:
        pass


if __name__ == "__main__":
    main()
