import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

st.set_page_config(
    page_title="Text Summarization & Sentiment Analysis",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Text Summarization & Sentiment Analysis App")
st.write(
    "Enter a long text, and the app will generate a summary "
    "and analyze its sentiment."
)

@st.cache_resource
def load_summarization_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

tokenizer, model = load_summarization_model()
sentiment_analyzer = load_sentiment_model()

user_text = st.text_area(
    "Enter your text here:",
    height=250,
    placeholder="Paste a long paragraph or article here..."
)

max_length = st.slider("Summary Max Length", 50, 200, 120)
min_length = st.slider("Summary Min Length", 20, 80, 30)

if st.button("Generate Summary"):
    if user_text.strip() == "":
        st.warning("Please enter text for processing.")
    elif min_length >= max_length:
        st.warning("Summary Min Length must be smaller than Summary Max Length.")
    else:
        with st.spinner("Generating summary..."):
            inputs = tokenizer(
                user_text,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            )

            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                early_stopping=True
            )

            summarized_text = tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )

        st.subheader("Summarized Text:")
        st.success(summarized_text)

        with st.spinner("Analyzing sentiment..."):
            sentiment = sentiment_analyzer(summarized_text)[0]
            sentiment_label = sentiment["label"]
            sentiment_score = sentiment["score"]

        st.subheader("Sentiment Analysis:")
        st.info(
            f"Rating: {sentiment_label} | Score: {sentiment_score:.2f}"
        )
