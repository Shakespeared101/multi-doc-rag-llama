from doc_loader import load_documents_from_folder
from rag_index import build_or_load_index
from query_engine import query_rag

if __name__ == "__main__":
    folder_path = "docs"  # create this folder and add PDF, Word, PPT files
    print("[Info] Loading documents...")
    docs = load_documents_from_folder(folder_path)
    
    print(f"[Info] Loaded {len(docs)} documents.")
    index = build_or_load_index(docs)
    
    while True:
        user_input = input("\n[?] Enter your question (or type 'exit'): ")
        if user_input.lower() == "exit":
            break
        answer = query_rag(index, user_input)
        print(f"\n[Answer]\n{answer}\n")
