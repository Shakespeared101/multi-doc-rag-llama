# app.py
import streamlit as st
from doc_loader import load_documents
from rag_index import build_index
from query_engine import query_index

def main():
    st.title("RAG System with LLaMA 3")
    
    # Input field for the document folder (default set to "documents")
    documents_path = st.text_input("Documents Folder Path", value="documents")
    
    # Button to trigger document indexing
    if st.button("Index Documents"):
        with st.spinner("Loading and indexing documents..."):
            docs = load_documents(documents_path)
            if not docs:
                st.error("No documents found. Please check the folder path.")
            else:
                build_index(docs)
                st.success("Documents indexed successfully!")
    
    # Query input
    query = st.text_input("Enter your query")
    if st.button("Get Answer") and query:
        with st.spinner("Querying the index..."):
            response = query_index(query)
            st.subheader("Response")
            st.write(response)

if __name__ == '__main__':
    main()
