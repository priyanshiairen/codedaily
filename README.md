# AI-Powered Workplace Chat Conflict & Sentiment Detector

An NLP system to analyze chat messages and detect negative sentiment and conflict patterns using TF-IDF + Logistic Regression, with optional VADER support and a Streamlit web app.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ['stopwords','wordnet','omw-1.4','punkt']]"
```

## Train
```bash
python train.py --data data/sample_chats.csv --text_col message_text --label_col label
```

## Run App
```bash
streamlit run src/app_streamlit.py
```

## Features
- TF-IDF + Logistic Regression sentiment classification
- Optional VADER scoring
- Conflict rules: consecutive negatives, toxic lexicon
- Visualizations: distribution and timeline; exports `SentimentVisuals.png`
