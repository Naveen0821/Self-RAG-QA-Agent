# 📄 Self-RAG QA Agent

An intelligent question-answering agent that reads research papers and answers questions **with factual grounding**. Powered by **LangGraph**, **Groq LLM**, **FAISS**, and **Streamlit**, this project implements the Self-RAG (Self-verifying Retrieval-Augmented Generation) pattern.

---

## 🚀 Features

- 🔍 **PDF ingestion**: Upload any research paper in PDF format
- 🧠 **RAG pipeline**: Retrieve relevant chunks using FAISS
- 🤖 **Groq-powered LLM**: Use LLaMA 3 via Groq API to generate answers
- ✅ **Self-verification**: Automatically check if the generated answer is grounded in the retrieved context
- 🧠 **Memory**: Persist conversation history using LangGraph’s `MemorySaver`
- 👨‍⚕️ **Human-in-the-loop ready**: Flag unverified responses for review
- 🌐 **Streamlit UI**: Simple interface to upload, ask, and evaluate

---

## 🧱 Project Structure

```
self_rag_qa_agent/
├── src/
│ ├── graph/ # LangGraph flow
│ ├── nodes/ # Retrieval, generation, verification
│ ├── prompts/ # Prompt templates
│ ├── llm/ # Groq model loader
│ ├── state/ # Shared agent state
│ └── tools/ # PDF parser, FAISS builder
├── notebooks/
│ └── run_demo.ipynb # Jupyter interface (optional)
├── main.py # Streamlit UI
├── requirements.txt
└── .env # API keys (not checked into Git)

```

## ⚡ Setup Instructions (with `uv`)

We use [uv](https://github.com/astral-sh/uv) for fast virtual environment creation and package management.

### 🔧 1. Install `uv` (if not already)

```bash
pip install uv

```

### 🆕 2. Create a Virtual Environment

```bash
uv venv .venv

```

###  3. Create a Virtual Environment

```bash
.venv\Scripts\activate

```


### 📦 4. Install Dependencies

```bash
uv pip install -r requirements.txt

```

## 🔑 Environment Setup
Create a .env file in the root directory:

```bash
# Groq API Key
GROQ_API_KEY=your_groq_api_key_here

# Optional: Override model (e.g., LLaMA 3)
GROQ_MODEL=llama3-8b-8192
```

## ▶️ Running the App
Once the environment is set up, launch the Streamlit interface:

```bash

streamlit run main.py

```

You can also test the agent manually in notebooks/run_demo.ipynb.

