# Movie Review Sentiment Analysis

A Streamlit web application that predicts the sentiment (positive or negative) of a movie review using a Recurrent Neural Network (RNN) trained on the IMDB Movie Reviews dataset.

## Table of Contents

- [About](#about)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)
- [License](#license)

## About

This project takes raw movie review text as input, processes it through a text-cleaning pipeline, converts it into a TF-IDF feature vector, and passes it through a trained RNN model to classify the review as positive or negative.

## Features

- Simple, interactive web interface built with Streamlit
- Real-time sentiment prediction with confidence score
- Preprocessing pipeline consistent with model training

## Project Structure

```
├── app.py
├── requirements.txt
├── rnn_sentiment_model.pth
├── tfidf_vectorizer.pkl
├── label_encoder.pkl
├── RNN.ipynb
└── README.md
```

- `app.py` — Main Streamlit application
- `requirements.txt` — Python dependencies
- `rnn_sentiment_model.pth` — Trained model weights and configuration
- `tfidf_vectorizer.pkl` — Fitted TF-IDF vectorizer
- `label_encoder.pkl` — Encoder mapping labels to class names
- `RNN.ipynb` — Notebook used to train the model

## Requirements

- Python 3.9 or higher
- pip

## Installation

1. Make sure all project files are in the same folder (`app.py`, `requirements.txt`, `rnn_sentiment_model.pth`, `tfidf_vectorizer.pkl`, `label_encoder.pkl`).

2. Open a terminal in that folder and install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Open the URL displayed in the terminal (typically `http://localhost:8501`).

3. Enter a movie review in the text box and click **Predict Sentiment** to see the result.

Note: On first launch, the app downloads a small set of NLTK language resources. An internet connection is required for this one-time step.

## Model Details

| Property | Value |
|---|---|
| Architecture | Single-layer RNN |
| Input features | 5,000 (TF-IDF) |
| Hidden size | 128 |
| Training data | IMDB Dataset (50,000 reviews) |
| Test accuracy | ~85.7% |

## Tech Stack

- Python
- PyTorch
- Scikit-learn
- NLTK
- Streamlit

## Future Improvements

- Replace the vanilla RNN with an LSTM or GRU
- Use word embeddings instead of TF-IDF
- Add support for batch predictions via file upload
- Deploy to Streamlit Community Cloud or Hugging Face Spaces

