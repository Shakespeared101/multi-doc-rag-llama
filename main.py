import os
from doc_loader import load_documents
from rag_index import build_or_load_index
from query_engine import load_query_engine, query_index
from knowledge_graph import KnowledgeGraph

def main():
    """
    Main function to run the RAG system via command line with knowledge graph.
    """
    print("[Info] Loading documents...")
    documents = load_documents("docs")
    print(f"[Info] Loaded {len(documents)} documents.")

    print("[Info] Building (overwriting) index with current documents...")
    index = build_or_load_index(documents, index_path="index")
    print("[Info] Index persisted to 'index' directory")

    print("[Info] Building knowledge graph...")
    knowledge_graph = KnowledgeGraph(documents, index)
    knowledge_graph.build_graph()
    print("[Info] Knowledge graph constructed.")

    print("[Info] Initializing query engine...")
    query_engine = load_query_engine(index, knowledge_graph)

    print("\n[Info] RAG system is ready. Enter a topic (type 'exit' to quit):\n")
    while True:
        user_query = input(">> ")
        if user_query.lower() in ["exit", "quit"]:
            print("[Info] Exiting the RAG system.")
            break
        response = query_index(query_engine, user_query)
        print(f"\n[Response]\n{response['text']}")
        if response['images']:
            print("\n[Images Present] (Run Streamlit app to view)")
        if response['tables']:
            print("\n[Tables Present] (Run Streamlit app to view)")
        print()

if __name__ == "__main__":
    main()