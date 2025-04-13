import streamlit as st
from doc_loader import load_documents_from_streamlit
from rag_index import build_or_load_index
from query_engine import get_query_engine

st.set_page_config(page_title="Multi-Doc RAG with Llama 3.1", layout="wide")

st.title("📄 Multi-Document RAG with Llama 3.1")
st.markdown("Upload documents and ask questions to retrieve context-aware answers.")

# File uploader
uploaded_files = st.file_uploader(
    "Upload documents (PDF, DOCX, PPTX, TXT)",
    type=["pdf", "docx", "pptx", "txt"],
    accept_multiple_files=True
)

# Load documents
documents = []
if uploaded_files:
    with st.spinner("Loading documents..."):
        documents = load_documents_from_streamlit(uploaded_files)
    st.success(f"Loaded {len(documents)} document(s).")

# Build or load index
index = None
if documents:
    with st.spinner("Building or loading index..."):
        index = build_or_load_index(documents, index_path="index")
    st.success("Index is ready.")

# User query input
query = st.text_input("Enter your question:", placeholder="Type your question here...")

# Generate and display answer
if query and index:
    with st.spinner("Generating answer..."):
        query_engine = get_query_engine(index)
        response = query_engine.query(query)
    st.subheader("Answer:")
    st.write(response.response)
