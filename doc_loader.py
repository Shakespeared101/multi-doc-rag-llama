import os
from typing import List
from llama_index.core import Document
from unstructured.partition.auto import partition

def load_documents_from_folder(folder_path: str) -> List[Document]:
    docs = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        try:
            elements = partition(filename=filepath)
            text = "\n".join([str(el) for el in elements])
            docs.append(Document(text=text, metadata={"filename": filename}))
        except Exception as e:
            print(f"[Error] Failed to load {filename}: {e}")
    return docs
