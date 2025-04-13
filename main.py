import os
from doc_loader import load_documents
from rag_index import build_or_load_index
from query_engine import load_query_engine, query_index

def main():
    # Step 1: Load documents
    print("[Info] Loading documents...")
    documents = load_documents("docs")
    print(f"[Info] Loaded {len(documents)} documents.")

    # Step 2: Build or load the index
    print("[Info] Building or loading index...")
    index = build_or_load_index(documents, index_path="index")

    # Step 3: Initialize the query engine
    print("[Info] Initializing query engine...")
    query_engine = load_query_engine(index_path="index")

    # Step 4: Interactive query loop
    print("\n[Info] RAG system is ready. Enter your queries below (type 'exit' to quit):\n")
    while True:
        user_query = input(">> ")
        if user_query.lower() in ["exit", "quit"]:
            print("[Info] Exiting the RAG system.")
            break
        response = query_index(query_engine, user_query)
        print(f"\n[Response]\n{response}\n")

if __name__ == "__main__":
    main()
