import os
import logging
from llama_index.core import StorageContext, load_index_from_storage, Settings, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from expanded_retriever import ExpandedRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_query_engine(index_or_path="index", knowledge_graph=None):
    """
    Loads a query engine with an expanded retriever and knowledge graph for comprehensive retrieval.
    """
    Settings.llm = Ollama(model="llama3.1", request_timeout=120.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.chunk_size = 512

    if isinstance(index_or_path, str):
        persist_dir = index_or_path
        if not os.path.exists(persist_dir):
            raise FileNotFoundError(f"Index directory '{persist_dir}' does not exist.")
        try:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            logger.info(f"Index loaded from {persist_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to load index from {persist_dir}: {e}")
    elif isinstance(index_or_path, VectorStoreIndex):
        index = index_or_path
        logger.info("Using provided VectorStoreIndex instance")
    else:
        raise ValueError("Parameter must be a persist directory path (str) or a VectorStoreIndex instance.")

    retriever = ExpandedRetriever(index, knowledge_graph)
    response_synthesizer = get_response_synthesizer()
    query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)
    return query_engine

def query_index(query_engine, query_str: str) -> dict:
    """
    Executes a query and returns a structured response with deduplicated text, images, and tables.
    """
    response = query_engine.query(query_str)
    relevant_nodes = [
        node for node in response.source_nodes 
        if hasattr(node, 'score') and node.score is not None and node.score > 0.5  # Lowered threshold
    ]
    logger.info(f"Relevant nodes in response: {len(relevant_nodes)}")
    for node in relevant_nodes:
        logger.info(f"Node score: {node.score:.3f}, Text: {node.node.text[:100]}...")

    if not relevant_nodes:
        logger.warning("No relevant nodes found for the query.")
        return {'text': "No relevant information found.", 'images': [], 'tables': []}

    seen_texts = set()
    text_content = []
    images = []
    tables = []
    for node in sorted(relevant_nodes, key=lambda x: x.score, reverse=True):
        node_text = node.node.text
        text_summary = node_text[:500]  # Increased to 500 chars
        if text_summary not in seen_texts:
            seen_texts.add(text_summary)
            text_content.append(node_text)
            if 'image' in node.node.metadata:
                images.append(node.node.metadata['image'])
            if 'table' in node.node.metadata:
                tables.append(node.node.metadata['table'])

    return {
        'text': "\n\n".join(text_content),
        'images': images,
        'tables': tables
    }