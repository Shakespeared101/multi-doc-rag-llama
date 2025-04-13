# main.py

from rag_index import build_or_load_index
from query_engine import query_rag

if __name__ == "__main__":
    index = build_or_load_index()
    while True:
        user_input = input("[?] Enter your question (or type 'exit'): ")
        if user_input.lower() == "exit":
            break
        response = query_rag(index, user_input)
        print(f"\n[Answer]\n{response}\n")
