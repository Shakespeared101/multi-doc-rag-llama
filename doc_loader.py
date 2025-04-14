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


# Optional: Use a more robust PDF reader
from llama_index.readers.file import PyMuPDFReader

from pathlib import Path
from llama_index.core.schema import Document

class TextReader:
    def load_data(self, file_path, **kwargs):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return [Document(text=text, metadata={"file_path": str(file_path)})]

def load_documents(directory_path="docs"):
    """
    Loads documents recursively from a directory.
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
        ".txt": TextReader()  # Added support for .txt
    }

    reader = SimpleDirectoryReader(
        input_dir=directory_path,
        recursive=True,
        file_extractor=file_extractor
    )
    documents = reader.load_data()
    return documents

def load_documents_from_streamlit(uploaded_files):
    """
    Processes a list of uploaded files (from Streamlit) and returns Document objects.
    Uses temporary files to interface with readers that expect file paths.
    """
    documents = []

    file_extractor = {
        # Use PyMuPDFReader() if default PDFReader is failing
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

                    # Try to load using the appropriate reader
                    docs = file_extractor[ext].load_data(file_path=tmp_file.name)

                    if not isinstance(docs, list):
                        raise TypeError(f"Expected a list of Document objects, got {type(docs)}")

                    documents.extend(docs)

                # Clean up
                os.unlink(tmp_file.name)

            except Exception as e:
                tb = traceback.format_exc()
                print(f"[Error] Failed to process {uploaded_file.name}:\n{tb}")
        else:
            print(f"[Info] Skipping unsupported file type: {uploaded_file.name}")

    return documents