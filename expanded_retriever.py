from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpandedRetriever(BaseRetriever):
    def __init__(self, index: VectorStoreIndex, knowledge_graph, similarity_top_k=20, expansion_k=10, similarity_threshold=0.6):
        self.index = index
        self.knowledge_graph = knowledge_graph
        self.similarity_top_k = similarity_top_k
        self.expansion_k = expansion_k
        self.similarity_threshold = similarity_threshold
        self.retriever = index.as_retriever(similarity_top_k=expansion_k)
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        initial_retriever = self.index.as_retriever(similarity_top_k=self.similarity_top_k)
        initial_nodes = initial_retriever.retrieve(query_bundle)
        logger.info(f"Initial nodes retrieved: {len(initial_nodes)}")

        expanded_nodes = {}
        for node in initial_nodes:
            expanded_nodes[node.node.node_id] = node
            related_node_ids = self.knowledge_graph.get_related_nodes(node.node.node_id)
            for related_id in related_node_ids:
                related_node = self.index.docstore.get_node(related_id)
                if related_node and related_id not in expanded_nodes:
                    score = self._compute_similarity(query_bundle.embedding, related_node.embedding)
                    if score >= self.similarity_threshold:
                        expanded_nodes[related_id] = NodeWithScore(node=related_node, score=score)

            # Secondary expansion: include nodes two hops away
            for related_id in related_node_ids:
                second_hop_ids = self.knowledge_graph.get_related_nodes(related_id)
                for second_hop_id in second_hop_ids:
                    if second_hop_id not in expanded_nodes:
                        second_hop_node = self.index.docstore.get_node(second_hop_id)
                        if second_hop_node:
                            score = self._compute_similarity(query_bundle.embedding, second_hop_node.embedding)
                            if score >= self.similarity_threshold:
                                expanded_nodes[second_hop_id] = NodeWithScore(node=second_hop_node, score=score)

        for node in list(expanded_nodes.values()):
            node_embedding = node.node.embedding
            if node_embedding is None:
                continue
            node_query_bundle = QueryBundle(embedding=node_embedding)
            similar_nodes = self.retriever.retrieve(node_query_bundle)
            for sim_node in similar_nodes:
                if sim_node.score >= self.similarity_threshold and sim_node.node.node_id not in expanded_nodes:
                    expanded_nodes[sim_node.node.node_id] = sim_node
        logger.info(f"Expanded nodes: {len(expanded_nodes)}")

        query = query_bundle.query_str
        node_texts = [
            self.tokenizer.decode(self.tokenizer.encode(node.node.text, max_length=512, truncation=True))
            for node in expanded_nodes.values()
        ]
        pairs = [(query, text) for text in node_texts]
        scores = self.cross_encoder.predict(pairs)

        filtered_nodes = []
        seen_texts = set()
        for node, score in zip(expanded_nodes.values(), scores):
            node.score = float(score)
            node_text = node.node.text[:500]  # Increased to 500 chars for better deduplication
            if score > 0.8 and node_text not in seen_texts:  # Lowered threshold
                filtered_nodes.append(node)
                seen_texts.add(node_text)
            logger.info(f"Node score: {score:.3f}, Text: {node_text[:100]}...")

        logger.info(f"Filtered nodes: {len(filtered_nodes)}")
        return sorted(filtered_nodes, key=lambda x: x.score, reverse=True)  # Sort by score

    def _compute_similarity(self, query_embedding, node_embedding):
        if query_embedding is None or node_embedding is None:
            return 0.0
        query_embedding = np.array(query_embedding)
        node_embedding = np.array(node_embedding)
        norm1 = np.linalg.norm(query_embedding)
        norm2 = np.linalg.norm(node_embedding)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(query_embedding, node_embedding) / (norm1 * norm2)