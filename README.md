# 🩺 Medibot – Medical AI Assistant

Medibot is a professional-grade medical chatbot built with **Flask**, **Google Gemini**, **LangChain**, and **Pinecone**. It interprets user-described symptoms, confirms interpretations, and provides structured diagnostic labels based strictly on retrieved context and clinical logic.

---

## 🚀 Features

- 🧠 Uses Gemini 2.0 Flash for natural language processing.
- 🔍 Retrieves relevant medical context via Pinecone vector search.
- 🧬 Embeds text using HuggingFace’s `MiniLM` model.
- 📋 Enforces clinical, diagnostic-style communication rules.
- 🌐 Simple Flask-based web server with CORS support.

---

## 📁 Project Structure

```text
medibot/
├── app.py
├── templates/
│   └── index.html
├── .env                 # Not included in GitHub (contains API keys)
├── .gitignore
├── requirements.txt
└── README.md
```


---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/medibot.git
cd medibot
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Create a .env File
Add your keys like this:
```
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
```
### 4. Run the App
```bash
python app.py
```

#### Access the app at: http://localhost:5000



## 🧠 Chat Behavior Logic
Confirms user symptoms using prompts like:
“Are you having Fever?”

Waits for “yes” confirmation before providing diagnoses like:
“fever”

Interprets typos like “fevt” and confirms intent before answering.

No small talk. No non-medical responses.


## 📌 Dependencies
flask

flask-cors

python-dotenv

langchain

langchain-google-genai

langchain-pinecone

langchain-huggingface

sentence-transformers

pinecone-client

----

- 📜 License
MIT © 2025 Subham Sharma
