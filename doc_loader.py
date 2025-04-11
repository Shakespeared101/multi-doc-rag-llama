from pathlib import Path
from llama_index.readers.file import PDFReader, DocxReader, PptxReader

def load_documents(folder_path: str):
    """
    Load documents from a folder. Supports PDF, DOCX, and PPTX formats.
    Returns a list of document objects.
    """
    documents = []
    folder = Path(folder_path)
    for file in folder.glob("*"):
        if file.suffix.lower() == ".pdf":
            reader = PDFReader()
            docs = reader.load_data(str(file))
            documents.extend(docs)
        elif file.suffix.lower() == ".docx":
            reader = DocxReader()
            docs = reader.load_data(str(file))
            documents.extend(docs)
        elif file.suffix.lower() == ".pptx":
            reader = PptxReader()
            docs = reader.load_data(str(file))
            documents.extend(docs)
    return documents

if __name__ == "__main__":
    # For testing purposes: load documents from a folder named "documents"
    folder = "documents"  # Adjust this to your documents folder
    docs = load_documents(folder)
    print(f"Loaded {len(docs)} documents.")
