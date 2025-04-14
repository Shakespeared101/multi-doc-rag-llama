import os
from doc_loader import load_documents
from rag_index import build_or_load_index
from query_engine import load_query_engine, query_index

def main():
    # Step 1: Load documents from the local 'docs' directory.
    print("[Info] Loading documents...")
    documents = load_documents("docs")
    print(f"[Info] Loaded {len(documents)} documents.")

    # Step 2: Overwrite the index with the newly loaded documents.
    print("[Info] Building (overwriting) index with current documents...")
    index = build_or_load_index(documents, index_path="index")

    # Step 3: Initialize the query engine using the new index.
    print("[Info] Initializing query engine...")
    query_engine = load_query_engine("index")

    # Step 4: Interactive query loop.
    print("\n[Info] RAG system is ready. Enter your queries (type 'exit' to quit):\n")
    while True:
        user_query = input(">> ")
        if user_query.lower() in ["exit", "quit"]:
            print("[Info] Exiting the RAG system.")
            break
        answer = query_index(query_engine, user_query)
        print(f"\n[Response]\n{answer}\n")

if __name__ == "__main__":
    main()
