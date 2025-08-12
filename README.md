# 📝 Text Summarization App

A simple text summarization project with two interfaces:
- **CLI tool** (`summarize_cli.py`) — run in terminal for quick summaries.
- **Streamlit GUI** (`app.py`) — user-friendly web interface.

## 📦 Installation

1. Clone this repo:
```bash
git clone https://github.com/yourusername/text-summarization.git
cd text-summarization
```
2. Install dependencies:
``` bash
pip install -r requirements.txt
```

3. (Optional) Place your fine-tuned model in:

``` bash
t5-small-cnn-tiny/
```

🚀 Usage
CLI Mode
``` bash
python summarize_cli.py
```
Paste your article and type EOF on a new line to summarize.

Streamlit GUI
``` bash
streamlit run app.py
```
- Paste your text in the input area.
- Select Local fine-tuned model or Hub model.
- Adjust Max new tokens in the sidebar.
- Click Summarize.

🛠 Models
- Local model: Trained and saved in t5-small-cnn-tiny/.
- Fallback: sshleifer/distilbart-cnn-12-6 from Hugging Face Hub.

🤝 Credits
Developed by Mohammed Ahmed Mansour

Under guidance from Elevvo Internship Program