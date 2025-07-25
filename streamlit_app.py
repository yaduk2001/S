import streamlit as st
import os
import onnxruntime as ort
from transformers import DistilBertTokenizerFast
import pandas as pd
import numpy as np

# Set page config for a modern look
title = "AI Sentiment Classifier"
st.set_page_config(page_title=title, layout="centered")

# Custom CSS for high-tech look
st.markdown(
    """
    <style>
    body {
        background-color: #181A1B;
    }
    .main {
        background-color: #23272A;
        border-radius: 12px;
        padding: 2rem 2rem 1.5rem 2rem;
        box-shadow: 0 4px 32px 0 rgba(0,0,0,0.25);
    }
    .stTextArea textarea {
        background-color: #23272A !important;
        color: #F8F8F2 !important;
        font-size: 1.1rem;
        border-radius: 8px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #181A1B;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
        padding: 0.5rem 1.5rem;
    }
    .sentiment-box {
        background: #23272A;
        border: 2px solid #00C9FF;
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 1.5rem;
        text-align: center;
        font-size: 1.3rem;
        color: #00C9FF;
        font-weight: bold;
        letter-spacing: 1px;
        box-shadow: 0 2px 16px 0 rgba(0,201,255,0.10);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"<h1 style='text-align:center; color:#00C9FF; margin-bottom:0.2em'>{title}</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#F8F8F2; margin-top:0'>Classify the sentiment of your product or movie review instantly.</h4>", unsafe_allow_html=True)

st.markdown("""
<div class="main">
""", unsafe_allow_html=True)

review = st.text_area("Enter your review:", height=120, key="review_input")

# Load ONNX model and tokenizer only once
@st.cache_resource
def load_model():
    model_path = './onnx_model/model_quantized.onnx'
    tokenizer = DistilBertTokenizerFast.from_pretrained('./onnx_model')
    session = ort.InferenceSession(model_path)
    return tokenizer, session

tokenizer, session = load_model()

sentiment = None
show_result = False

if st.button("Analyze Sentiment"):
    if review.strip():
        # Tokenize and prepare ONNX input
        inputs = tokenizer([review], padding='max_length', truncation=True, max_length=128, return_tensors='np')
        ort_inputs = {
            'input_ids': inputs['input_ids'].astype(np.int64),
            'attention_mask': inputs['attention_mask'].astype(np.int64)
        }
        # ONNX inference
        outputs = session.run(None, ort_inputs)
        logits = outputs[0]
        pred = int(np.argmax(logits, axis=1)[0])
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = sentiment_map.get(pred, "unknown")
        color = {"positive": "#00FFB3", "neutral": "#FFD600", "negative": "#FF4B4B"}[sentiment]
        st.markdown(f"<div class='sentiment-box' style='color:{color};border-color:{color}'>Sentiment: {sentiment.capitalize()}</div>", unsafe_allow_html=True)
        show_result = True
        # Save to local history file
        history_file = "history.csv"
        file_exists = os.path.isfile(history_file)
        import csv
        with open(history_file, mode="a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["review", "sentiment"])
            writer.writerow([review, sentiment])
    else:
        st.warning("Please enter a review to analyze.")

# Show history table
st.markdown("<hr style='margin:2em 0 1em 0;border:1px solid #222;'>", unsafe_allow_html=True)
st.subheader("Review History")
# Clear History Button
if st.button("Clear History 🗑️"):
    history_file = "history.csv"
    if os.path.isfile(history_file):
        os.remove(history_file)
    st.success("Review history cleared.")
    st.experimental_rerun()

history_file = "history.csv"
if os.path.isfile(history_file):
    try:
        df = pd.read_csv(history_file)
        if not df.empty:
            st.dataframe(df[::-1].reset_index(drop=True), use_container_width=True)
        else:
            st.info("No review history yet.")
    except Exception:
        st.info("No review history yet.")
else:
    st.info("No review history yet.")

st.markdown("""
</div>
""", unsafe_allow_html=True) 