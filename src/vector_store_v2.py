from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import get_leaf_nodes
import shutil
from typing import List, Dict
import os

class VectorStore:
    """
    Vector store class using LlamaIndex MilvusVectorStore
    Supports hierarchical retrieval with AutoMergingRetriever
    """

    def __init__(self, db_path="./milvus.db", collection_name="book_chunks",
                 model_name="BAAI/bge-large-en-v1.5", docstore_dir="./docstore",
                 wipe_existing: bool = False):
        self.db_path = db_path
        self.collection_name = collection_name
        self.model_name = model_name
        self.embedding_dim = 1024  # bge dimensions
        self.docstore_dir = docstore_dir

        # Load embedding model
        print(f"Loading embedding model {model_name}")
        self.embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            embed_batch_size=32
        )

        self.vector_store = MilvusVectorStore(
            uri=db_path,
            dim=self.embedding_dim,
            collection_name=collection_name,
            overwrite=wipe_existing
        )

        if wipe_existing and os.path.exists(docstore_dir):
            shutil.rmtree(docstore_dir)
            print(f"Old docstore deleted at {docstore_dir}")

        if os.path.exists(docstore_dir) and os.path.isdir(docstore_dir):
            self.docstore = SimpleDocumentStore.from_persist_dir(docstore_dir)
            print(f"Docstore loaded from {docstore_dir}")
        else:
            self.docstore = SimpleDocumentStore()
            print(f"New docstore created at {docstore_dir}")

        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            docstore=self.docstore
        )

        self._index = None
        self._retriever = None
        self.node_mapping = {}

    def create_index(self, nodes: List, node_mapping: Dict = None):
        """
        Create VectorStoreIndex from nodes

        Args:
            nodes: List of all nodes (parent + child)
            node_mapping: node_id -> node mapping (for AutoMerging)
        """
        print(f"Indexing {len(nodes)} nodes")

        # Store node mapping (required for AutoMerging)
        if node_mapping:
            self.node_mapping = node_mapping

        # This is required to preserve parent-child relationships
        for node in nodes:
            self.docstore.add_documents([node])
        print(f"All nodes added to docstore")

        # Create VectorStoreIndex
        self._index = VectorStoreIndex(
            nodes=nodes,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
            show_progress=True
        )

        # Persist storage context after creating index
        # This saves both docstore and vector_store (including parent/child relationships)
        self.storage_context.persist(persist_dir=self.docstore_dir)
        print(f"Storage context persisted at: {self.docstore_dir}")

        print(f"Index successfully created and persisted")

    def create_retriever(self, similarity_top_k: int = 10, use_auto_merging: bool = True):
        """
        Create retriever (AutoMerging or normal)

        Args:
            similarity_top_k: Number of nodes to return
            use_auto_merging: If True, use AutoMergingRetriever
        """
        if self._index is None:
            raise ValueError("Index not created. Must call create_index()")

        if use_auto_merging and self.node_mapping:
            # AutoMergingRetriever (hierarchical)
            base_retriever = self._index.as_retriever(
                similarity_top_k=similarity_top_k
            )

            self._retriever = AutoMergingRetriever(
                base_retriever,
                storage_context=self.storage_context,
                verbose=True
            )
            print("AutoMergingRetriever created (hierarchical retrieval)")
        else:
            # Normal retriever
            self._retriever = self._index.as_retriever(
                similarity_top_k=similarity_top_k
            )
            print(f"Normal retriever created (top_k={similarity_top_k})")

        return self._retriever

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Simple similarity search (for backwards compatibility)

        Returns:
            List of search results with metadata
        """
        if self._retriever is None:
            self.create_retriever(similarity_top_k=top_k, use_auto_merging=False)

        # Retrieve
        nodes = self._retriever.retrieve(query)

        results = []
        for node_with_score in nodes:
            node = node_with_score.node
            results.append({
                'text': node.text,
                'chapter': node.metadata.get('chapter', 0),
                'chapter_title': node.metadata.get('chapter_title', ''),
                'node_id': node.node_id,
                'score': node_with_score.score
            })

        return results

    def hybrid_search(self, query: str, top_parents: int = 3, top_children: int = 5) -> tuple:
        """
        Hierarchical search with AutoMergingRetriever

        Args:
            query: Search query
            top_parents: Number of nodes to retrieve in base retriever (before AutoMerge)
            top_children: Not used (for backwards compatibility)

        Returns:
            ([], retrieved_nodes) tuple - AutoMerge happens automatically in LlamaIndex
        """
        if self._retriever is None:
            self.create_retriever(
                similarity_top_k=top_parents * 3,
                use_auto_merging=True
            )

        # Retrieve (AutoMerge works automatically)
        nodes_with_scores = self._retriever.retrieve(query)

        # Format results
        results = []
        for node_with_score in nodes_with_scores:
            node = node_with_score.node
            results.append({
                'text': node.text,
                'chapter': node.metadata.get('chapter', 0),
                'chapter_title': node.metadata.get('chapter_title', ''),
                'node_id': node.node_id,
                'score': node_with_score.score,
                'is_parent': not hasattr(node, 'parent_node') or node.parent_node is None
            })

        return ([], results)

    def close(self):
        """
        Close connection
        """
        pass

    def get_stats(self) -> Dict:
   
        if self._index is None:
            return {
                'error': 'Index not created'
            }

        all_nodes = list(self.node_mapping.values()) if self.node_mapping else []
        leaf_nodes = get_leaf_nodes(all_nodes) if all_nodes else []

        return {
            'total_nodes': len(all_nodes),
            'leaf_nodes': len(leaf_nodes),
            'parent_nodes': len(all_nodes) - len(leaf_nodes),
            'collection_name': self.collection_name,
            'embedding_dim': self.embedding_dim,
            'model': self.model_name,
            'db_path': self.db_path
        }

if __name__ == "__main__":
    from chunker_v2 import HierarchicalChunker

    with open('data/children_of_new_forest.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    chunker = HierarchicalChunker(parent_size=2048, child_size=512, chunk_overlap=20)
    nodes, node_mapping = chunker.chunk_text(text)

    vs = VectorStore(db_path="./test_milvus_llama.db")
    vs.create_index(nodes, node_mapping)

    test_query = "What is the title of this story?"
    _, results = vs.hybrid_search(test_query, top_parents=3)

    print(f"AutoMerging results ({len(results)} node)")
    for i, result in enumerate(results, 1):
        node_type = "PARENT" if result.get('is_parent', False) else "CHILD"
        print(f"\n[{i}] {node_type} (Chapter {result['chapter']}, Score: {result['score']:.4f})")
        print(f"    {result['text'][:150]}")

    stats = vs.get_stats()
    print(f"Index Stats:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  leaf nodes: {stats['leaf_nodes']}")
    print(f"  parent nodes: {stats['parent_nodes']}")
    print(f"  model -> {stats['model']}")