import streamlit as st
from transformers import pipeline

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOJI = {"POSITIVE": "😊", "NEUTRAL": "😐", "NEGATIVE": "😠"}

st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊")

@st.cache_resource(show_spinner="Loading model (first run only)...")
def load_model():
    return pipeline("sentiment-analysis", model=MODEL)

clf = load_model()

st.title("Sentiment Analyzer")
st.caption("Paste a sentence or paragraph and the model will score its sentiment as positive, neutral or negative.")

text = st.text_area("Your text:", height=150, max_chars=5000)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please type something first.")
    else:
        try:
            with st.spinner("Analyzing..."):
                result = clf(text, truncation=True, max_length=512)[0]
            label = result["label"].upper()
            score = result["score"]
            st.subheader(f"{EMOJI.get(label, '')} {label}")
            st.progress(score)
            st.write(f"Confidence: **{score:.0%}**")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

st.divider()
st.caption(f"Model: {MODEL}. Long texts are truncated. Sarcasm and mixed feelings may be misread.")
