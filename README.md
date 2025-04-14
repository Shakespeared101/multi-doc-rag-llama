# 🧠 Multi-Document RAG with LLaMA 3 (via Ollama)

A production-ready **Retrieval-Augmented Generation (RAG)** system that supports multi-format document ingestion and generates highly contextual answers using **LlamaIndex**, and **LLaMA 3** via **Ollama**.

---

## 🚀 Features

- 📁 Drag-and-drop **folder ingestion** (PDF, DOCX, PPTX)
- 🔁 **Automatic re-indexing** on doc changes
- 🧠 **LLaMA 3-based RAG** via Ollama
- 🖼️ **OCR fallback** for scanned/image-based PDFs
- 💬 **Query history** logging
- 🧵 **Threaded context** support
- ❌ **Hallucination minimization** via custom prompt
- 🧹 **Verbose error logging**
- 🌐 Web UI with **Streamlit**
- 🔌 Optional **FastAPI** backend

---

## 🧰 Tech Stack

- [LlamaIndex](https://github.com/jerryjliu/llama_index)
- [Ollama](https://ollama.com)
- [Streamlit](https://streamlit.io)
- [FastAPI](https://fastapi.tiangolo.com)

---

## 📁 Project Structure

```bash
multi-doc-rag-llama/
├── app.py              # Streamlit frontend
├── main.py             # (Optional) FastAPI backend
├── doc_loader.py       # Handles PDF, DOCX, PPTX loading
├── rag_index.py        # Index creation & update
├── query_engine.py     # Handles querying logic
├── utils.py            # Misc utilities (e.g., topic extraction)
├── index/              # Saved indices
├── data/               # User-uploaded documents
├── queries/            # Query logs
└── README.md

---

**Installation & Setup**
1. Clone the repository

```bash
git clone https://github.com/your-username/multi-doc-rag-llama.git
cd multi-doc-rag-llama

2. Create a virtual environment

```bash
python -m venv env
source env/bin/activate  # or .\env\Scripts\activate on Windows

3. Install requirements

```bash
pip install -r requirements.txt

**✅ Make sure Ollama is running before starting the app.**

---

Install & Run LLaMA 3 via Ollama
1. Install Ollama

Download and install from: https://ollama.com

2. Pull the LLaMA 3 model

```bash
ollama pull llama3

3. Start the model server

```bash
ollama run llama3

**Running the App**
```bash
streamlit run app.py
