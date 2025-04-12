import os
import streamlit as st
from doc_loader import load_and_extract_text
from rag_index import build_index_from_raw_texts
from query_engine import query_rag

UPLOAD_FOLDER = "docs"
INDEX_FOLDER = "rag_index"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.set_page_config(page_title="📚 RAG Assistant", layout="wide")
st.title("📚 Multi-Document RAG Assistant")

uploaded_files = st.file_uploader("Upload your documents (PDF, PPTX, DOCX)", type=["pdf", "pptx", "docx"], accept_multiple_files=True)

if uploaded_files:
    texts = []
    st.info("Extracting text from documents...")

    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        
        text = load_and_extract_text(file_path)
        if text:
            texts.append(text)
        else:
            st.warning(f"⚠️ Failed to extract text from: {file.name}")
    
    if texts:
        st.success("✅ Documents processed successfully. Building index...")
        try:
            build_index_from_raw_texts(texts)
            st.success("🔍 Index built successfully!")
        except Exception as e:
            st.error(f"Failed to build index: {e}")
    else:
        st.error("❌ No valid content found in uploaded documents.")

st.divider()
user_query = st.text_input("📝 Ask a question based on the uploaded documents")

if user_query:
    with st.spinner("Generating response using LLaMA3..."):
        answer = query_rag(user_query)
    st.success("📥 Answer:")
    st.write(answer)
