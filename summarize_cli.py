# summarize_cli.py
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ------------- CONFIG -------------
# local path (if you saved your fine-tuned model), or fallback to a hub model:
LOCAL_MODEL_PATH = "t5-small-cnn-tiny"      # replace if you saved your trainer model
FALLBACK_MODEL = "sshleifer/distilbart-cnn-12-6"  # compact, good for news; see note below
# -----------------------------------

def load_model(model_path_local=LOCAL_MODEL_PATH, fallback=FALLBACK_MODEL):
    try:
        print(f"Trying to load local model from '{model_path_local}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_path_local)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path_local)
        print("Loaded local fine-tuned model.")
    except Exception as e:
        print(f"Local model not found or failed to load ({e}). Falling back to hub model '{fallback}'...")
        tokenizer = AutoTokenizer.from_pretrained(fallback)
        model = AutoModelForSeq2SeqLM.from_pretrained(fallback)
        print("Loaded hub model:", fallback)
    model.eval()
    return tokenizer, model

def summarize_text(text, tokenizer, model, max_new_tokens=80):
    # Add T5 prefix only if model is a t5-type (T5 expects the task prefix)
    try:
        model_type = model.config.model_type
    except Exception:
        model_type = ""
    input_text = ("summarize: " + text) if model_type == "t5" else text

    # safe truncation to tokenizer/model max length
    max_input_len = getattr(tokenizer, "model_max_length", 512)
    # some tokenizers set model_max_length to a very large value (1e30) — guard against that
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

def read_multiline(prompt="Paste your article. End with a line that contains only EOF and press Enter:\n"):
    print(prompt)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().lower() == "eof":
            break
        lines.append(line)
    return "\n".join(lines)

def main():
    tokenizer, model = load_model()
    print("\nReady. Paste an article and type EOF on a new line when done.\n")
    while True:
        article = read_multiline()
        if not article.strip():
            print("No text entered. Exiting.")
            break
        print("\nSummarizing... (this may take a few seconds on CPU)\n")
        summary = summarize_text(article, tokenizer, model, max_new_tokens=80)
        print("=== SUMMARY ===\n")
        print(summary)
        print("\n----------------\n")
        cont = input("Summarize another? (y/n): ").strip().lower()
        if cont.startswith("n"):
            print("Goodbye 👋")
            break

if __name__ == "__main__":
    main()
