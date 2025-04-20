import streamlit as st
from doc_loader import load_documents_from_streamlit
from rag_index import build_or_load_index
from query_engine import load_query_engine, query_index
from knowledge_graph import KnowledgeGraph

try:
    import resource
except ImportError:
    resource = None

st.set_page_config(page_title="Multi-Doc RAG with LLaMA 3.1", layout="wide")
st.title("📄 Multi-Document RAG with LLaMA 3.1")
st.markdown("Upload documents and enter a topic to retrieve all related content.")

uploaded_files = st.file_uploader(
    "Upload documents (PDF, DOCX, PPTX, TXT, etc.):",
    type=["pdf", "docx", "pptx", "txt", "md", "html", "epub", "csv", "rtf", "png", "jpg", "jpeg", "ipynb"],
    accept_multiple_files=True
)

if "documents" not in st.session_state:
    st.session_state.documents = []
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "knowledge_graph" not in st.session_state:
    st.session_state.knowledge_graph = None

if uploaded_files:
    with st.spinner("Processing uploaded documents..."):
        st.session_state.documents = load_documents_from_streamlit(uploaded_files)
    st.success(f"Loaded {len(st.session_state.documents)} document(s).")

    with st.spinner("Building/overwriting index and knowledge graph..."):
        index = build_or_load_index(st.session_state.documents, index_path="index")
        st.session_state.knowledge_graph = KnowledgeGraph(st.session_state.documents, index)
        st.session_state.knowledge_graph.build_graph()
        st.session_state.index_ready = True
    st.success("Index and knowledge graph are ready.")

query = st.text_input("Enter a topic:", placeholder="e.g., machine learning")

if query:
    if not uploaded_files or not st.session_state.documents:
        st.warning("⚠️ Please upload at least one document before entering a topic.")
    elif not st.session_state.index_ready:
        st.warning("⚠️ Index is not ready yet. Please wait a moment.")
    else:
        with st.spinner("Retrieving related content..."):
            query_engine = load_query_engine("index", st.session_state.knowledge_graph)
            response = query_index(query_engine, query)
        st.subheader("Related Content:")
        st.write(response['text'])
        if response['images']:
            st.subheader("Images:")
            for img in response['images']:
                if img:
                    st.image(img, caption="Retrieved Image")
        if response['tables']:
            st.subheader("Tables:")
            for table in response['tables']:
                if table:
                    st.table(table)