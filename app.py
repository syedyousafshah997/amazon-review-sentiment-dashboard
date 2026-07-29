"""
NLP & Sentiment Analysis Dashboard
Streamlit app with 3 pages: Home, Data Overview, Sentiment Predictor.
"""

import re
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction import text as sk_text

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Amazon Review Sentiment Dashboard",
    page_icon="💬",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "amazon_reviews.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
MODEL_NAME_PATH = os.path.join(BASE_DIR, "models", "best_model_name.txt")
WORDCLOUD_POS_PATH = os.path.join(BASE_DIR, "images", "wordcloud_positive.png")
WORDCLOUD_NEG_PATH = os.path.join(BASE_DIR, "images", "wordcloud_negative.png")

CUSTOM_STOPWORDS = set(sk_text.ENGLISH_STOP_WORDS)


# --------------------------------------------------------------------------
# Text cleaning — MUST match the cleaning used in notebooks/week5_nlp.ipynb
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)      # URLs
    text = re.sub(r"<.*?>", " ", text)                # HTML tags
    text = re.sub(r"&\w+;", " ", text)                # HTML entities
    text = re.sub(r"[^a-z\s]", " ", text)              # keep only letters
    text = re.sub(r"\s+", " ", text).strip()           # extra whitespace
    tokens = [w for w in text.split() if w not in CUSTOM_STOPWORDS and len(w) > 2]
    return " ".join(tokens)


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["verified_reviews"]).reset_index(drop=True)
    return df


@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    model_name = "Model"
    if os.path.exists(MODEL_NAME_PATH):
        with open(MODEL_NAME_PATH) as f:
            model_name = f.read().strip()
    return model, vectorizer, model_name


def predict_sentiment(review_text, model, vectorizer):
    cleaned = clean_text(review_text)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]

    # Confidence score: use predict_proba when available (e.g. Logistic
    # Regression, Naive Bayes). Some models (e.g. LinearSVC) don't expose
    # predict_proba, so we fall back to a sigmoid of the decision_function
    # score as a confidence proxy.
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = proba[int(pred)]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X)[0]
        sigmoid = 1 / (1 + np.exp(-score))
        confidence = sigmoid if pred == 1 else 1 - sigmoid
    else:
        confidence = None

    return pred, confidence, cleaned


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.sidebar.title("💬 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Overview", "Sentiment Predictor"])

st.sidebar.markdown("---")
st.sidebar.caption("NLP & Sentiment Analysis Dashboard")

# --------------------------------------------------------------------------
# Page: Home
# --------------------------------------------------------------------------
if page == "Home":
    st.title("💬 Amazon Review Sentiment Analysis Dashboard")
    st.markdown(
        """
        Welcome! This dashboard is for Natural
        Language Processing & Sentiment Analysis.

        ### What is Sentiment Analysis?
        Sentiment analysis is a Natural Language Processing (NLP) technique that
        automatically determines whether a piece of text expresses a **positive** or
        **negative** opinion. It powers product review summaries, customer feedback
        tools, chatbots, and social media monitoring at scale.

        ### About this project
        - **Dataset:** Amazon Product Reviews — real customer reviews
          with a verified review text and a feedback label (1 = positive, 0 = negative).
        - **Pipeline:** raw text → cleaning (lowercasing, removing noise/stopwords) →
          TF-IDF vectorization → classification model.
        - **Models compared:** Logistic Regression, Multinomial Naive Bayes, and
          Linear SVM. The best performing model (by F1-score) was saved and is used
          by the **Sentiment Predictor** page in this app.

        ### How to use this app
        Use the sidebar to navigate:
        - **Data Overview** — explore the class distribution and word clouds from the
          training data.
        - **Sentiment Predictor** — type in any review and get an instant predicted
          sentiment with a confidence score.
        """
    )

# --------------------------------------------------------------------------
# Page: Data Overview
# --------------------------------------------------------------------------
elif page == "Data Overview":
    st.title("📊 Data Overview")

    df = load_data()

    st.subheader("Dataset Snapshot")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Reviews", f"{len(df):,}")
    with col2:
        pos_pct = (df["feedback"] == 1).mean() * 100
        st.metric("Positive Reviews", f"{pos_pct:.1f}%")

    st.subheader("Class Distribution")
    class_counts = df["feedback"].value_counts().rename(
        index={0: "Negative", 1: "Positive"}
    )
    st.bar_chart(class_counts)
    st.caption(
        "The dataset is imbalanced — the large majority of reviews are positive."
    )

    st.subheader("Word Clouds")
    wc_col1, wc_col2 = st.columns(2)
    with wc_col1:
        st.markdown("**Positive Reviews**")
        if os.path.exists(WORDCLOUD_POS_PATH):
            st.image(WORDCLOUD_POS_PATH, use_container_width=True)
        else:
            st.info("Run the notebook first to generate this word cloud.")
    with wc_col2:
        st.markdown("**Negative Reviews**")
        if os.path.exists(WORDCLOUD_NEG_PATH):
            st.image(WORDCLOUD_NEG_PATH, use_container_width=True)
        else:
            st.info("Run the notebook first to generate this word cloud.")

# --------------------------------------------------------------------------
# Page: Sentiment Predictor
# --------------------------------------------------------------------------
elif page == "Sentiment Predictor":
    st.title("🔮 Sentiment Predictor")
    st.markdown(
        "Type or paste any product review below and click **Predict** to see the "
        "predicted sentiment and the model's confidence score."
    )

    try:
        model, vectorizer, model_name = load_model_and_vectorizer()
        st.caption(f"Model in use: **{model_name}**")
    except FileNotFoundError:
        st.error(
            "Model or vectorizer file not found. Please run "
            "`notebooks/week5_nlp.ipynb` first to train and save the model."
        )
        st.stop()

    review_input = st.text_area(
        "Review text",
        placeholder="e.g. This product works great and the sound quality is amazing!",
        height=150,
    )

    if st.button("Predict", type="primary"):
        if not review_input.strip():
            st.warning("Please enter a review before predicting.")
        else:
            pred, confidence, cleaned = predict_sentiment(review_input, model, vectorizer)

            if pred == 1:
                st.success("Predicted Sentiment: **Positive** 😀")
            else:
                st.error("Predicted Sentiment: **Negative** 😞")

            if confidence is not None:
                st.metric("Confidence Score", f"{confidence * 100:.1f}%")
                st.progress(min(max(confidence, 0.0), 1.0))

            with st.expander("See cleaned text used for prediction"):
                st.code(cleaned if cleaned else "(empty after cleaning)")
