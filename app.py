import streamlit as st
from doc_loader import load_documents_from_streamlit
from rag_index import build_or_load_index
from query_engine import load_query_engine, query_index

# Fix for 'resource' module on Windows
try:
    import resource
except ImportError:
    resource = None


st.set_page_config(page_title="Multi-Doc RAG with LLaMA 3.1", layout="wide")
st.title("📄 Multi-Document RAG with LLaMA 3.1")
st.markdown("Upload documents and ask questions to retrieve context-aware answers.")

# File uploader: allow multiple file uploads
uploaded_files = st.file_uploader(
    "Upload documents (PDF, DOCX, PPTX, TXT, etc.):",
    type=["pdf", "docx", "pptx", "txt", "md", "html", "epub", "csv", "rtf", "png", "jpg", "jpeg", "ipynb"],
    accept_multiple_files=True
)

# Initialize session state to persist across interactions
if "documents" not in st.session_state:
    st.session_state.documents = []
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

# Load and index documents only after upload
if uploaded_files:
    with st.spinner("Processing uploaded documents..."):
        st.session_state.documents = load_documents_from_streamlit(uploaded_files)
    st.success(f"Loaded {len(st.session_state.documents)} document(s).")

    with st.spinner("Building/overwriting index..."):
        index = build_or_load_index(st.session_state.documents, index_path="index")
        st.session_state.index_ready = True
    st.success("Index is ready.")

# Query input
query = st.text_input("Enter your question:", placeholder="Type your question here...")

# Check before querying
if query:
    if not uploaded_files or not st.session_state.documents:
        st.warning("⚠️ Please upload at least one document before asking a question.")
    elif not st.session_state.index_ready:
        st.warning("⚠️ Index is not ready yet. Please wait a moment.")
    else:
        with st.spinner("Generating answer..."):
            query_engine = load_query_engine("index")
            response = query_index(query_engine, query)
        st.subheader("Answer:")
        st.write(response)
