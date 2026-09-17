import re
import pickle

import nltk
import torch
import torch.nn as nn
import streamlit as st


# ----------------------------
# One-time NLTK data download
# ----------------------------
@st.cache_resource
def download_nltk_data():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)


download_nltk_data()

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# ----------------------------
# Model definition (must match training exactly)
# ----------------------------
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, X):
        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        out, _ = self.rnn(X, h0)
        out = self.fc(out[:, -1, :])
        return out


# ----------------------------
# Load model + preprocessing artifacts (cached so this runs once)
# ----------------------------
@st.cache_resource
def load_artifacts():
    checkpoint = torch.load("rnn_sentiment_model.pth", map_location="cpu")

    model = RNN(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open("tfidf_vectorizer.pkl", "rb") as f:
        tf = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    return model, tf, le


model, tf, le = load_artifacts()
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()


# ----------------------------
# Preprocessing pipeline (must match training exactly)
# ----------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)          # remove URLs
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)    # remove punctuation
    text = re.sub(r"<.*?>", "", text)             # remove HTML tags

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]  # remove stopwords
    tokens = [ps.stem(t) for t in tokens]                # stemming

    return " ".join(tokens)


def predict_sentiment(text):
    cleaned = preprocess(text)

    if cleaned.strip() == "":
        return None, None

    vec = tf.transform([cleaned]).toarray()
    xb = torch.from_numpy(vec).float().unsqueeze(1)  # add sequence dimension

    with torch.no_grad():
        output = model(xb)
        prob = torch.sigmoid(output).item()

    label = le.inverse_transform([int(prob > 0.5)])[0]
    return label, prob


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Movie Review Sentiment Analysis", page_icon="🎬")

st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review below and the RNN model will predict whether it's positive or negative.")

user_input = st.text_area("Your review:", height=150, placeholder="Type or paste a movie review here...")

if st.button("Predict Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a review first.")
    else:
        label, prob = predict_sentiment(user_input)

        if label is None:
            st.warning("After cleaning, this text has no usable content. Try a longer review.")
        else:
            confidence = prob if label == "positive" else 1 - prob

            if label == "positive":
                st.success(f"**Positive** 😀 (confidence: {confidence:.1%})")
            else:
                st.error(f"**Negative** 😞 (confidence: {confidence:.1%})")

            st.progress(prob)
            st.caption(f"Raw model probability (positive class): {prob:.4f}")

st.divider()
st.caption("Model: single-layer RNN over TF-IDF features (5000 vocab) · Trained on IMDB Dataset")