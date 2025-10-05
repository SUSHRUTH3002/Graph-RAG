"""Vector store for embedding-based similarity search."""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store using FAISS for efficient similarity search."""
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        """Initialize vector store.
        
        Args:
            dimension: Embedding dimension
            index_type: FAISS index type ('flat', 'ivf', 'hnsw')
        """
        import faiss
        
        self.dimension = dimension
        self.index_type = index_type
        self.faiss = faiss
        
        # Initialize FAISS index
        if index_type == "flat":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Metadata storage
        self.id_to_metadata = {}
        self.metadata_to_id = {}
        self.next_id = 0
    
    def add_embeddings(
        self,
        embeddings: List[List[float]],
        metadata_list: List[Dict[str, Any]]
    ) -> List[int]:
        """Add embeddings with metadata to the index.
        
        Args:
            embeddings: List of embedding vectors
            metadata_list: List of metadata dictionaries
            
        Returns:
            List of assigned IDs
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Embeddings and metadata lists must have same length")
        
        # Convert to numpy array
        vectors = np.array(embeddings, dtype=np.float32)
        
        # Train index if needed
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(vectors)
        
        # Add vectors to index
        start_id = self.next_id
        self.index.add(vectors)
        
        # Store metadata
        assigned_ids = []
        for i, metadata in enumerate(metadata_list):
            vector_id = start_id + i
            self.id_to_metadata[vector_id] = metadata
            
            # Create reverse mapping for chunk IDs
            if 'chunk_id' in metadata:
                self.metadata_to_id[metadata['chunk_id']] = vector_id
            elif 'entity_name' in metadata:
                self.metadata_to_id[metadata['entity_name']] = vector_id
            
            assigned_ids.append(vector_id)
        
        self.next_id += len(embeddings)
        logger.info(f"Added {len(embeddings)} vectors to index")
        
        return assigned_ids
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filter_fn: Optional[callable] = None
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Search for similar vectors.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            filter_fn: Optional function to filter results
            
        Returns:
            List of (id, distance, metadata) tuples
        """
        # Convert to numpy array
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Search
        distances, indices = self.index.search(query_vector, k * 2)  # Get more for filtering
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            metadata = self.id_to_metadata.get(int(idx), {})
            
            # Apply filter if provided
            if filter_fn and not filter_fn(metadata):
                continue
            
            results.append((int(idx), float(dist), metadata))
            
            if len(results) >= k:
                break
        
        return results
    
    def search_by_metadata(
        self,
        query_embedding: List[float],
        metadata_filter: Dict[str, Any],
        k: int = 10
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Search with metadata filtering.
        
        Args:
            query_embedding: Query vector
            metadata_filter: Dictionary of metadata key-value pairs to filter
            k: Number of results
            
        Returns:
            Filtered search results
        """
        def filter_fn(metadata):
            return all(
                metadata.get(key) == value
                for key, value in metadata_filter.items()
            )
        
        return self.search(query_embedding, k=k, filter_fn=filter_fn)
    
    def get_by_id(self, vector_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata by vector ID.
        
        Args:
            vector_id: Vector ID
            
        Returns:
            Metadata dictionary or None
        """
        return self.id_to_metadata.get(vector_id)
    
    def get_by_chunk_id(self, chunk_id: str) -> Optional[Tuple[int, Dict[str, Any]]]:
        """Get vector ID and metadata by chunk ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            (vector_id, metadata) tuple or None
        """
        vector_id = self.metadata_to_id.get(chunk_id)
        if vector_id is not None:
            return vector_id, self.id_to_metadata[vector_id]
        return None
    
    def save(self, filepath: str):
        """Save index and metadata to disk.
        
        Args:
            filepath: Path to save file
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        self.faiss.write_index(self.index, str(path))
        
        # Save metadata
        metadata_path = path.with_suffix('.metadata')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'id_to_metadata': self.id_to_metadata,
                'metadata_to_id': self.metadata_to_id,
                'next_id': self.next_id,
                'dimension': self.dimension,
                'index_type': self.index_type
            }, f)
        
        logger.info(f"Saved vector store to {filepath}")
    
    def load(self, filepath: str):
        """Load index and metadata from disk.
        
        Args:
            filepath: Path to saved file
        """
        path = Path(filepath)
        
        # Load FAISS index
        self.index = self.faiss.read_index(str(path))
        
        # Load metadata
        metadata_path = path.with_suffix('.metadata')
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.id_to_metadata = data['id_to_metadata']
            self.metadata_to_id = data['metadata_to_id']
            self.next_id = data['next_id']
            self.dimension = data['dimension']
            self.index_type = data['index_type']
        
        logger.info(f"Loaded vector store from {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metadata_count': len(self.id_to_metadata)
        }


class HybridStore:
    """Hybrid storage combining graph and vector databases."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        neo4j_store: Any,
        similarity_threshold: float = 0.7
    ):
        """Initialize hybrid store.
        
        Args:
            vector_store: VectorStore instance
            neo4j_store: Neo4jStore instance
            similarity_threshold: Threshold for similarity matching
        """
        self.vector_store = vector_store
        self.neo4j_store = neo4j_store
        self.similarity_threshold = similarity_threshold
    
    def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: Optional[str] = None,
        k: int = 10,
        expand_graph: bool = True,
        graph_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search using both vector similarity and graph structure.
        
        Args:
            query_embedding: Query embedding vector
            query_text: Optional query text for keyword matching
            k: Number of results
            expand_graph: Whether to expand results using graph
            graph_depth: Depth for graph expansion
            
        Returns:
            List of results with scores and metadata
        """
        # Step 1: Vector similarity search
        vector_results = self.vector_store.search(query_embedding, k=k * 2)
        
        results = []
        entity_names = set()
        
        for vec_id, distance, metadata in vector_results:
            # Convert distance to similarity score
            similarity = 1 / (1 + distance)
            
            if similarity < self.similarity_threshold:
                continue
            
            result = {
                'id': metadata.get('chunk_id') or metadata.get('entity_name'),
                'type': metadata.get('type', 'CHUNK'),
                'text': metadata.get('text', ''),
                'similarity_score': similarity,
                'metadata': metadata,
                'source': 'vector'
            }
            
            results.append(result)
            
            # Collect entity names for graph expansion
            if 'entity_name' in metadata:
                entity_names.add(metadata['entity_name'])
        
        # Step 2: Expand using graph relationships
        if expand_graph and entity_names and self.neo4j_store:
            for entity_name in list(entity_names)[:5]:  # Limit expansion
                try:
                    related = self.neo4j_store.get_related_entities(
                        entity_name, 
                        max_depth=graph_depth
                    )
                    
                    for rel_entity in related:
                        results.append({
                            'id': rel_entity['name'],
                            'type': rel_entity.get('type', 'ENTITY'),
                            'text': rel_entity['name'],
                            'similarity_score': 0.5 / (rel_entity['distance'] + 1),
                            'metadata': rel_entity,
                            'source': 'graph',
                            'related_to': entity_name
                        })
                        
                except Exception as e:
                    logger.error(f"Error expanding graph for {entity_name}: {e}")
        
        # Sort by similarity score
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return results[:k]
    
    def add_chunk_with_entities(
        self,
        chunk: Dict[str, Any],
        entities: List[Dict[str, Any]]
    ):
        """Add a chunk and its entities to both stores.
        
        Args:
            chunk: Chunk dictionary with embedding
            entities: List of entities in the chunk
        """
        chunk_id = chunk['id']
        embedding = chunk.get('embedding')
        
        if embedding:
            # Add to vector store
            self.vector_store.add_embeddings(
                [embedding],
                [{
                    'chunk_id': chunk_id,
                    'type': 'CHUNK',
                    'text': chunk.get('text', ''),
                    **chunk.get('metadata', {})
                }]
            )
        
        # Add to Neo4j
        if self.neo4j_store:
            doc_id = chunk.get('metadata', {}).get('doc_id', 'unknown')
            self.neo4j_store.create_chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=chunk.get('text', ''),
                metadata=chunk.get('metadata', {})
            )
            
            # Link entities
            for entity in entities:
                entity_name = entity.get('text')
                if entity_name:
                    self.neo4j_store.create_entity(
                        name=entity_name,
                        entity_type=entity.get('type', 'GENERIC'),
                        properties=entity
                    )
                    self.neo4j_store.link_chunk_to_entity(
                        chunk_id=chunk_id,
                        entity_name=entity_name
                    )
