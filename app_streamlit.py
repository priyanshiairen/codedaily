import os
import sys
import glob

# Ensure project root is on sys.path so `src` imports work when run as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import io
from typing import List
import pandas as pd
import streamlit as st

from src.preprocess import preprocess_text
from src.model import load_pipeline, predict_labels
from src.rules import flag_conflict, is_toxic
from src.visualize import plot_sentiment_distribution, plot_conflict_timeline
from src.vader_integration import vader_compound_scores, vader_to_label

st.set_page_config(page_title="Chat Sentiment & Conflict Detector", layout="wide")

st.title("💬 AI-Powered Workplace Chat Conflict & Sentiment Detector")

# --- Quick single-text analysis ---
with st.expander("Quick check: Is this message conflict-prone?", expanded=True):
    input_text = st.text_area("Enter message text", placeholder="Type a chat message...", height=100)
    col_a, col_b = st.columns([1, 3])
    with col_a:
        analyze_click = st.button("Analyze")
    result_placeholder = col_b.empty()

# Model options
model_col, options_col = st.columns([2, 1])
with model_col:
    model_path = st.text_input("Model path (joblib)", value=os.path.join(PROJECT_ROOT, "model", "artifacts.joblib"))
with options_col:
    use_vader = st.checkbox("Use VADER (override model)", value=False)

# On analyze button, run a single-text prediction and conflict rule
if 'single_vec' not in st.session_state:
    st.session_state['single_vec'] = None
    st.session_state['single_clf'] = None

if analyze_click and input_text.strip():
    msg = input_text.strip()
    if use_vader:
        comp = vader_compound_scores([msg])[0]
        label = vader_to_label(comp)
    else:
        if st.session_state['single_vec'] is None or st.session_state['single_clf'] is None:
            try:
                vec, clf = load_pipeline(model_path)
                st.session_state['single_vec'] = vec
                st.session_state['single_clf'] = clf
            except Exception as e:
                result_placeholder.error(f"Failed to load model: {e}")
                label = "neutral"
        if st.session_state['single_vec'] is not None:
            cleaned = preprocess_text(msg)
            label = predict_labels(st.session_state['single_vec'], st.session_state['single_clf'], [cleaned])[0]

    toxic = is_toxic(msg)
    # Override sentiment if toxic terms present
    if toxic:
        label = 'negative'

    conflict = label.lower() == 'negative' or toxic
    if conflict:
        result_placeholder.error(f"Conflict: YES — sentiment={label}, toxic_terms={'yes' if toxic else 'no'}")
    else:
        result_placeholder.success(f"Conflict: NO — sentiment={label}, toxic_terms={'yes' if toxic else 'no'}")

# --- Dataset discovery ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_CANDIDATES = [
    os.path.join(DATA_DIR, "sample_chats.csv"),
    os.path.join(DATA_DIR, "sample_chats_100.csv"),
]

def discover_datasets() -> list[str]:
    paths: list[str] = []
    if os.path.isdir(DATA_DIR):
        paths.extend(sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))))
        paths.extend(sorted(glob.glob(os.path.join(DATA_DIR, "*.json"))))
    for p in DEFAULT_CANDIDATES:
        if os.path.exists(p) and p not in paths:
            paths.insert(0, p)
    return paths

# Pick dataset from repo (no uploader)
repo_datasets = discover_datasets()
if not repo_datasets:
    st.error("No datasets found in the repository under 'data/'. Add a CSV/JSON with a 'message_text' column.")
    st.stop()

choice = st.selectbox("Dataset in repo to analyze", options=repo_datasets, index=0, label_visibility="visible")

# Load chosen dataset
if choice.endswith(".csv"):
    df = pd.read_csv(choice)
else:
    df = pd.read_json(choice, lines=False)

if "message_text" not in df.columns:
    st.error("Selected dataset is missing required column 'message_text'.")
    st.stop()

st.caption(f"Using dataset: {os.path.relpath(choice, PROJECT_ROOT)}")

# --- Smoke test run on startup ---
messages: List[str] = df["message_text"].astype(str).tolist()
cleaned = [preprocess_text(m) for m in messages]

if use_vader:
    compounds = vader_compound_scores(messages)
    labels = [vader_to_label(c) for c in compounds]
    st.caption("Using VADER for sentiment labels.")
else:
    try:
        vec, clf = load_pipeline(model_path)
        labels = predict_labels(vec, clf, cleaned)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        labels = ["neutral"] * len(cleaned)

# Toxic override for batch as well
labels = ["negative" if is_toxic(msg) else lbl for msg, lbl in zip(messages, labels)]

# Attach results
out_df = df.copy()
out_df["predicted_sentiment"] = labels
flags = flag_conflict(labels, messages)
out_df["conflict_flag"] = [f["conflict"] for f in flags]
out_df["toxic"] = [f["toxic"] for f in flags]

# Show preview and results
st.subheader("Preview")
st.dataframe(out_df.head(20))

st.subheader("Visualizations")
plot_col1, plot_col2 = st.columns(2)
with plot_col1:
    counts = plot_sentiment_distribution(labels)
    st.pyplot(use_container_width=True)
with plot_col2:
    _ = plot_conflict_timeline(labels)
    st.pyplot(use_container_width=True)

# Save visuals
try:
    plot_sentiment_distribution(labels, save_path=os.path.join(PROJECT_ROOT, "SentimentVisuals.png"))
except Exception:
    pass
