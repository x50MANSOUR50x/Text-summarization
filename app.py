import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- CONFIG ----------------
LOCAL_MODEL_PATH = "t5-small-cnn-tiny"  # Your fine-tuned model folder
FALLBACK_MODEL = "sshleifer/distilbart-cnn-12-6"  # Good small model for CNN/DailyMail-style summaries
# -----------------------------------------

@st.cache_resource
def load_model(model_choice):
    """Load model and tokenizer (cached for faster UI)."""
    tokenizer = AutoTokenizer.from_pretrained(model_choice)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_choice)
    model.eval()
    return tokenizer, model

def summarize_text(text, tokenizer, model, max_new_tokens=80):
    model_type = getattr(model.config, "model_type", "")
    input_text = "summarize: " + text if model_type == "t5" else text

    max_input_len = getattr(tokenizer, "model_max_length", 512)
    if max_input_len is None or max_input_len > 4096:
        max_input_len = 512

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len,
        padding="longest"
    )

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=4,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
            early_stopping=True
        )

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Text Summarizer", page_icon="📝", layout="wide")
st.title("📝 Text Summarizer")
st.write("Paste your article below and click **Summarize**.")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
model_source = st.sidebar.radio(
    "Model source",
    ("Local fine-tuned model", "Hub model"),
    index=0
)
max_tokens = st.sidebar.slider("Max new tokens", min_value=30, max_value=200, value=80, step=10)

# Load model based on user choice
model_name = LOCAL_MODEL_PATH if model_source == "Local fine-tuned model" else FALLBACK_MODEL
tokenizer, model = load_model(model_name)

# Text input area
article = st.text_area("✏️ Enter your article:", height=300)

if st.button("🚀 Summarize"):
    if article.strip():
        with st.spinner("Summarizing... please wait ⏳"):
            summary = summarize_text(article, tokenizer, model, max_new_tokens=max_tokens)
        st.subheader("✨ Summary")
        st.write(summary)
    else:
        st.warning("Please enter an article first.")