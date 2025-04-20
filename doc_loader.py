import os
import tempfile
from llama_index.core import SimpleDirectoryReader
import traceback
from llama_index.core.schema import Document
from llama_index.readers.file import (
    PDFReader,
    DocxReader,
    PptxReader,
    MarkdownReader,
    HTMLTagReader,
    EpubReader,
    CSVReader,
    RTFReader,
    ImageReader,
    IPYNBReader
)

import sys

if sys.platform.startswith("win"):
    import types
    sys.modules['resource'] = types.SimpleNamespace()

from llama_index.readers.file import PyMuPDFReader

class TextReader:
    def load_data(self, file_path, **kwargs):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return [Document(text=text, metadata={"file_path": str(file_path)})]

def load_documents(directory_path="docs"):
    """
    Loads documents recursively from a directory, extracting multimedia where possible.
    """
    file_extractor = {
        ".pdf": PyMuPDFReader(),
        ".docx": DocxReader(),
        ".pptx": PptxReader(),
        ".md": MarkdownReader(),
        ".html": HTMLTagReader(),
        ".epub": EpubReader(),
        ".csv": CSVReader(),
        ".rtf": RTFReader(),
        ".png": ImageReader(),
        ".jpg": ImageReader(),
        ".jpeg": ImageReader(),
        ".ipynb": IPYNBReader(),
        ".txt": TextReader()
    }

    reader = SimpleDirectoryReader(
        input_dir=directory_path,
        recursive=True,
        file_extractor=file_extractor
    )
    documents = reader.load_data()
    for doc in documents:
        if 'images' in doc.metadata:
            doc.metadata['image'] = doc.metadata['images']
        if 'tables' in doc.metadata:
            doc.metadata['table'] = doc.metadata['tables']
    return documents

def load_documents_from_streamlit(uploaded_files):
    """
    Processes uploaded files, extracting multimedia and returning Document objects.
    """
    documents = []

    file_extractor = {
        ".pdf": PyMuPDFReader(),
        ".docx": DocxReader(),
        ".pptx": PptxReader(),
        ".md": MarkdownReader(),
        ".html": HTMLTagReader(),
        ".epub": EpubReader(),
        ".csv": CSVReader(),
        ".rtf": RTFReader(),
        ".png": ImageReader(),
        ".jpg": ImageReader(),
        ".jpeg": ImageReader(),
        ".ipynb": IPYNBReader(),
        ".txt": TextReader()
    }

    for uploaded_file in uploaded_files:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext in file_extractor:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file.flush()

                    docs = file_extractor[ext].load_data(file_path=tmp_file.name)

                    if not isinstance(docs, list):
                        raise TypeError(f"Expected a list of Document objects, got {type(docs)}")

                    for doc in docs:
                        if 'images' in doc.metadata:
                            doc.metadata['image'] = doc.metadata['images']
                        if 'tables' in doc.metadata:
                            doc.metadata['table'] = doc.metadata['tables']
                        documents.append(doc)

                os.unlink(tmp_file.name)

            except Exception as e:
                tb = traceback.format_exc()
                print(f"[Error] Failed to process {uploaded_file.name}:\n{tb}")
        else:
            print(f"[Info] Skipping unsupported file type: {uploaded_file.name}")

    return documents