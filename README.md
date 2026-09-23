Text Summarization & Sentiment Analysis App

A Streamlit NLP application that summarizes long English text and analyzes the sentiment of the generated summary.

Overview

The app combines two pretrained Transformer models in one simple interface:

Text summarization: facebook/bart-large-cnn

Sentiment analysis: distilbert-base-uncased-finetuned-sst-2-english

Users can paste a long paragraph or article, control the minimum and maximum summary length, generate a concise summary, and receive a sentiment label with a confidence score.

Screenshots

Application Interface



Example Result



Features

Long-text input through a Streamlit interface

Abstractive summarization with BART

Adjustable minimum and maximum summary length

Beam-search text generation

Automatic sentiment analysis of the generated summary

Sentiment confidence score

Cached model loading for better Streamlit performance

Input validation for empty text and summary-length settings

How It Works

User Text
   |
   v
BART Summarization
facebook/bart-large-cnn
   |
   v
Generated Summary
   |
   v
DistilBERT Sentiment Analysis
   |
   v
Sentiment Label + Confidence Score

Models

BART Summarization

The application uses:

facebook/bart-large-cnn

The input is tokenized with a maximum input length of 1024 tokens and summarized using beam search.

DistilBERT Sentiment Analysis

The generated summary is passed to:

distilbert-base-uncased-finetuned-sst-2-english

The model returns a sentiment label and confidence score.

Repository Files

app.py — Streamlit application

1783713237888.jpg — application interface screenshot

1783713238226.jpg — example summarization and sentiment result

README.md — project documentation

Technologies

Python

Streamlit

Hugging Face Transformers

BART

DistilBERT

PyTorch

NLP

Text Summarization

Sentiment Analysis

Installation

Install the required packages:

pip install streamlit transformers torch sentencepiece

Running the App

Run:

streamlit run app.py

Then open the local Streamlit URL shown in the terminal.

Notes

The models are downloaded from Hugging Face the first time the application runs, so the initial startup may take longer.

The current sentiment model is designed for English text.

Author

Mohammad Ahmad Elayyan

Email: mohamadelayyan84@gmail.com

LinkedIn: https://www.linkedin.com/in/mohammadelayyan1

GitHub: MohammadElayyan117
