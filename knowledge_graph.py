from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraph:
    def __init__(self, documents: list[Document], index: VectorStoreIndex):
        self.documents = documents
        self.index = index
        self.graph = nx.DiGraph()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.node_id_to_text = {}
        self.doc_id_to_nodes = {}

    def build_graph(self):
        """
        Builds a knowledge graph by extracting topics and subtopics and linking related nodes.
        """
        logger.info("Building knowledge graph...")
        
        for doc in self.documents:
            self.doc_id_to_nodes[doc.doc_id] = []

        for node_id, node in self.index.docstore.docs.items():
            if hasattr(node, 'ref_doc_id') and node.ref_doc_id in self.doc_id_to_nodes:
                self.doc_id_to_nodes[node.ref_doc_id].append(node)
                self.node_id_to_text[node_id] = node.text
                self.graph.add_node(node_id, text=node.text, embedding=node.embedding)
                
                first_sentence = node.text.split('.')[0].strip()
                self.graph.nodes[node_id]['topic'] = first_sentence[:100]

        node_ids = list(self.graph.nodes)
        for i, node_id in enumerate(node_ids):
            node_embedding = self.graph.nodes[node_id]['embedding']
            if node_embedding is None:
                logger.warning(f"No embedding for node {node_id}, skipping.")
                continue
            for j in range(i + 1, len(node_ids)):
                other_id = node_ids[j]
                other_embedding = self.graph.nodes[other_id]['embedding']
                if other_embedding is None:
                    continue
                similarity = self._compute_similarity(node_embedding, other_embedding)
                if similarity > 0.65:  # Lowered threshold for denser graph
                    self.graph.add_edge(node_id, other_id, weight=similarity)
                    self.graph.add_edge(other_id, node_id, weight=similarity)

        logger.info(f"Knowledge graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def get_related_nodes(self, node_id: str) -> list[str]:
        """
        Retrieves all related nodes using graph traversal.
        """
        if node_id not in self.graph:
            return []
        
        related_nodes = set()
        from queue import Queue
        q = Queue()
        q.put(node_id)
        visited = {node_id}

        while not q.empty():
            current = q.get()
            related_nodes.add(current)
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.put(neighbor)

        return list(related_nodes - {node_id})

    def _compute_similarity(self, emb1, emb2):
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(emb1, emb2) / (norm1 * norm2)