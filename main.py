from doc_loader import load_documents
from rag_index import build_or_load_index
from query_engine import get_query_engine

def main():
    print("[Info] Loading documents...")
    documents = load_documents("docs")
    print(f"[Info] Loaded {len(documents)} documents.")

    print("[Info] Building or loading index...")
    index = build_or_load_index(documents)

    print("[Info] Setting up query engine...")
    query_engine = get_query_engine(index)

    while True:
        query = input("[?] Enter your question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        try:
            response = query_engine.query(query)
            print(f"[Answer]\n{response}")
        except Exception as e:
            print(f"[Error] Query failed: {e}")

if __name__ == "__main__":
    main()
