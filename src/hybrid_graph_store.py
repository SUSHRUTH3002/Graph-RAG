"""Hybrid storage combining Neo4j graph and vector databases."""

from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

from .graph_embeddings import GraphEmbeddingGenerator, NodeEmbedding, EdgeEmbedding
from .vector_store import VectorStore
from .redis_vectorstore import RedisVectorStore

logger = logging.getLogger(__name__)

class HybridGraphStore:
    """Manages hybrid storage of graph structure and embeddings."""
    
    def __init__(
        self,
        neo4j_store,
        embeddings_client,
        use_redis: bool = False,
        redis_url: Optional[str] = None,
        vector_store_path: Optional[str] = None
    ):
        """Initialize hybrid graph store.
        
        Args:
            neo4j_store: Neo4jStore instance
            embeddings_client: EmbeddingsClient instance
            use_redis: Whether to use Redis for vector storage
            redis_url: Redis connection URL (if use_redis=True)
            vector_store_path: Path for FAISS vector store
        """
        self.neo4j_store = neo4j_store
        self.embeddings_client = embeddings_client
        self.use_redis = use_redis
        
        # Initialize embedding generator
        self.embedding_generator = GraphEmbeddingGenerator(
            embeddings_client=embeddings_client,
            neo4j_store=neo4j_store
        )
        
        # Initialize vector stores
        embedding_dim = embeddings_client.get_embedding_dimension()
        
        # FAISS vector store (always available for local operations)
        self.faiss_node_store = VectorStore(dimension=embedding_dim, index_type="flat")
        self.faiss_edge_store = VectorStore(dimension=embedding_dim, index_type="flat")
        
        # Redis vector stores (optional, for production)
        self.redis_node_store = None
        self.redis_edge_store = None
        
        if use_redis and redis_url:
            self.redis_node_store = RedisVectorStore(
                redis_url=redis_url,
                index_name="graph_nodes",
                embeddings=embeddings_client.embeddings
            )
            self.redis_edge_store = RedisVectorStore(
                redis_url=redis_url,
                index_name="graph_edges",
                embeddings=embeddings_client.embeddings
            )
            logger.info("Redis vector stores initialized")
        
        self.vector_store_path = vector_store_path
        self.is_indexed = False
    
    def index_graph(
        self,
        node_types: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Index the entire graph with embeddings.
        
        Args:
            node_types: Optional filter for node types
            relationship_types: Optional filter for relationship types
            
        Returns:
            Statistics about the indexing process
        """
        logger.info("Starting graph indexing process...")
        
        # Step 1: Generate node embeddings
        node_embeddings = self.embedding_generator.generate_node_embeddings(node_types)
        
        # Step 2: Generate edge embeddings
        edge_embeddings = self.embedding_generator.generate_edge_embeddings(relationship_types)
        
        # Step 3: Store embeddings in vector stores
        self._store_node_embeddings(node_embeddings)
        self._store_edge_embeddings(edge_embeddings)
        
        # Step 4: Save to disk if path provided
        if self.vector_store_path:
            self._save_vector_stores()
        
        self.is_indexed = True
        
        # Get statistics from all components
        embedding_stats = self.embedding_generator.get_statistics()
        faiss_node_stats = self.faiss_node_store.get_statistics()
        faiss_edge_stats = self.faiss_edge_store.get_statistics()
        
        stats = {
            'nodes_indexed': len(node_embeddings),
            'edges_indexed': len(edge_embeddings),
            'node_types': list(set(n.node_type for n in node_embeddings)),
            'relationship_types': list(set(e.relationship_type for e in edge_embeddings)),
            'embedding_dimension': self.embeddings_client.get_embedding_dimension(),
            'using_redis': self.use_redis,
            'embedding_generator': embedding_stats,
            'faiss_nodes': faiss_node_stats,
            'faiss_edges': faiss_edge_stats
        }
        
        # Add Redis stats if enabled
        if self.redis_node_store:
            stats['redis_nodes'] = self.redis_node_store.get_stats()
        if self.redis_edge_store:
            stats['redis_edges'] = self.redis_edge_store.get_stats()
        
        logger.info(f"Graph indexing completed: {stats}")
        return stats
    
    def _store_node_embeddings(self, node_embeddings: List[NodeEmbedding]):
        """Store node embeddings in vector stores."""
        if not node_embeddings:
            return
        
        embeddings = [n.embedding for n in node_embeddings]
        metadata_list = [
            {
                'node_id': n.node_id,
                'node_type': n.node_type,
                'text': n.text,
                'entity_name': n.metadata.get('name'),
                **n.metadata
            }
            for n in node_embeddings
        ]
        
        # Store in FAISS
        self.faiss_node_store.add_embeddings(embeddings, metadata_list)
        logger.info(f"Stored {len(embeddings)} node embeddings in FAISS")
        
        # Store in Redis if enabled
        if self.redis_node_store:
            chunks = [
                {
                    'id': n.node_id,
                    'text': n.text,
                    'metadata': metadata_list[i]
                }
                for i, n in enumerate(node_embeddings)
            ]
            self.redis_node_store.add_chunks(chunks)
            logger.info(f"Stored {len(chunks)} node embeddings in Redis")
    
    def _store_edge_embeddings(self, edge_embeddings: List[EdgeEmbedding]):
        """Store edge embeddings in vector stores."""
        if not edge_embeddings:
            return
        
        embeddings = [e.embedding for e in edge_embeddings]
        metadata_list = [
            {
                'edge_id': e.edge_id,
                'source': e.source,
                'target': e.target,
                'relationship_type': e.relationship_type,
                'text': e.text,
                **e.metadata
            }
            for e in edge_embeddings
        ]
        
        # Store in FAISS
        self.faiss_edge_store.add_embeddings(embeddings, metadata_list)
        logger.info(f"Stored {len(embeddings)} edge embeddings in FAISS")
        
        # Store in Redis if enabled
        if self.redis_edge_store:
            chunks = [
                {
                    'id': e.edge_id,
                    'text': e.text,
                    'metadata': metadata_list[i]
                }
                for i, e in enumerate(edge_embeddings)
            ]
            self.redis_edge_store.add_chunks(chunks)
            logger.info(f"Stored {len(chunks)} edge embeddings in Redis")
    
    def search_similar_nodes(
        self,
        query: str,
        k: int = 10,
        node_type_filter: Optional[str] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar nodes using semantic similarity.
        
        Args:
            query: Search query
            k: Number of results
            node_type_filter: Optional node type filter
            
        Returns:
            List of (node_id, score, metadata) tuples
        """
        query_embedding = self.embeddings_client.embed_query(query)
        
        # Apply filter if needed
        filter_fn = None
        if node_type_filter:
            filter_fn = lambda m: m.get('node_type') == node_type_filter
        
        results = self.faiss_node_store.search(
            query_embedding,
            k=k,
            filter_fn=filter_fn
        )
        
        return [(r[0], r[1], r[2]) for r in results]
    
    def search_similar_edges(
        self,
        query: str,
        k: int = 10,
        relationship_filter: Optional[str] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar edges using semantic similarity.
        
        Args:
            query: Search query
            k: Number of results
            relationship_filter: Optional relationship type filter
            
        Returns:
            List of (edge_id, score, metadata) tuples
        """
        query_embedding = self.embeddings_client.embed_query(query)
        
        # Apply filter if needed
        filter_fn = None
        if relationship_filter:
            filter_fn = lambda m: m.get('relationship_type') == relationship_filter
        
        results = self.faiss_edge_store.search(
            query_embedding,
            k=k,
            filter_fn=filter_fn
        )
        
        return [(r[0], r[1], r[2]) for r in results]
    
    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        search_nodes: bool = True,
        search_edges: bool = True,
        expand_graph: bool = True,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """Perform hybrid search combining vector similarity and graph traversal.
        
        Args:
            query: Search query
            k: Number of results per search type
            search_nodes: Include node search
            search_edges: Include edge search
            expand_graph: Expand results using graph structure
            max_depth: Maximum depth for graph expansion
            
        Returns:
            Dictionary with search results
        """
        results = {
            'query': query,
            'nodes': [],
            'edges': [],
            'expanded_subgraph': None
        }
        # Search similar nodes
        if search_nodes:
            node_results = self.search_similar_nodes(query, k=k)
            results['nodes'] = [
                {
                    'node_id': node_id,
                    'similarity_score': float(score),
                    'metadata': metadata
                }
                for node_id, score, metadata in node_results
            ]
        
        # Search similar edges
        if search_edges:
            edge_results = self.search_similar_edges(query, k=k)
            results['edges'] = [
                {
                    'edge_id': edge_id,
                    'similarity_score': float(score),
                    'metadata': metadata
                }
                for edge_id, score, metadata in edge_results
            ]
        
        # Expand using graph structure
        if expand_graph and results['nodes']:
            entity_names = [
                n['metadata'].get('entity_name')
                for n in results['nodes']
                if n['metadata'].get('entity_name')
            ][:5]  # Limit to top 5
            
            if entity_names:
                subgraph = self._expand_subgraph(entity_names, max_depth)
                results['expanded_subgraph'] = subgraph
        
        return results
    
    def _expand_subgraph(
        self,
        entity_names: List[str],
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """Expand a subgraph from seed entities."""
        all_nodes = set()
        all_edges = []
        
        for entity_name in entity_names:
            try:
                related = self.neo4j_store.get_related_entities(
                    entity_name,
                    max_depth=max_depth
                )
                
                # Add nodes from the related entities
                for node in related.get('nodes', []):
                    if node.get('name'):
                        all_nodes.add(node['name'])
                
                # Add edges from the related entities
                for edge in related.get('edges', []):
                    all_edges.append({
                        'source': edge.get('source'),
                        'target': edge.get('target'),
                        'type': edge.get('type')
                    })
                    
            except Exception as e:
                logger.error(f"Error expanding subgraph for {entity_name}: {e}")
        
        return {
            'nodes': list(all_nodes),
            'edges': all_edges,
            'node_count': len(all_nodes),
            'edge_count': len(all_edges)
        }
    
    def _save_vector_stores(self):
        """Save vector stores to disk."""
        if not self.vector_store_path:
            return
        
        path = Path(self.vector_store_path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS stores
        self.faiss_node_store.save(str(path / "nodes.faiss"))
        self.faiss_edge_store.save(str(path / "edges.faiss"))
        
        logger.info(f"Saved vector stores to {self.vector_store_path}")
    
    def load_vector_stores(self):
        """Load vector stores from disk."""
        if not self.vector_store_path:
            logger.warning("No vector store path specified")
            return False
        
        path = Path(self.vector_store_path)
        
        try:
            self.faiss_node_store.load(str(path / "nodes.faiss"))
            self.faiss_edge_store.load(str(path / "edges.faiss"))
            self.is_indexed = True
            logger.info(f"Loaded vector stores from {self.vector_store_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load vector stores: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            'is_indexed': self.is_indexed,
            'embedding_generator': self.embedding_generator.get_statistics(),
            'faiss_nodes': self.faiss_node_store.get_statistics(),
            'faiss_edges': self.faiss_edge_store.get_statistics(),
            'using_redis': self.use_redis
        }
        
        if self.redis_node_store:
            stats['redis_nodes'] = self.redis_node_store.get_stats()
        if self.redis_edge_store:
            stats['redis_edges'] = self.redis_edge_store.get_stats()
        
        return stats
