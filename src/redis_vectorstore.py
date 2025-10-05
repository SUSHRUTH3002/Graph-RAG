"""Redis vector storage module using LangChain."""

from typing import List, Dict, Any, Optional, Tuple
import logging
import redis
from langchain_community.vectorstores import Redis
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import numpy as np

logger = logging.getLogger(__name__)

class RedisVectorStore:
    """Redis vector database interface for storing and searching document embeddings."""
    
    def __init__(self, redis_url: str, index_name: str, embeddings: Embeddings):
        """Initialize Redis vector store.
        
        Args:
            redis_url: Redis connection URL
            index_name: Name of the vector index
            embeddings: Embeddings model instance
        """
        self.redis_url = redis_url
        self.index_name = index_name
        self.embeddings = embeddings
        
        # Initialize Redis client
        self.redis_client = redis.from_url(redis_url)
        
        # Initialize LangChain Redis vectorstore
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize the Redis vectorstore."""
        try:
            # Create empty vectorstore first
            self.vectorstore = Redis(
                redis_url=self.redis_url,
                index_name=self.index_name,
                embedding=self.embeddings
            )
            logger.info(f"Redis vectorstore initialized with index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Redis vectorstore: {e}")
            raise
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Add text chunks with embeddings to Redis.
        
        Args:
            chunks: List of chunk dictionaries with 'id', 'text', and optional metadata
            
        Returns:
            List of chunk IDs that were added
        """
        documents = []
        chunk_ids = []
        
        for chunk in chunks:
            chunk_id = chunk.get('id', f"chunk_{len(documents)}")
            text = chunk.get('text', '')
            metadata = chunk.get('metadata', {})
            
            # Add chunk_id to metadata
            metadata['chunk_id'] = chunk_id
            
            doc = Document(
                page_content=text,
                metadata=metadata
            )
            documents.append(doc)
            chunk_ids.append(chunk_id)
        
        try:
            # Add documents to vectorstore
            added_ids = self.vectorstore.add_documents(documents, ids=chunk_ids)
            logger.info(f"Added {len(added_ids)} chunks to Redis vectorstore")
            return added_ids
        except Exception as e:
            logger.error(f"Failed to add chunks to Redis: {e}")
            return []
    
    def similarity_search(self, query: str, k: int = 5, 
                         filter_dict: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """Perform similarity search in Redis.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            # Perform similarity search with scores
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            logger.debug(f"Found {len(results)} similar chunks for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def similarity_search_by_vector(self, embedding: List[float], k: int = 5) -> List[Tuple[Document, float]]:
        """Search by embedding vector.
        
        Args:
            embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            results = self.vectorstore.similarity_search_by_vector_with_score(
                embedding=embedding,
                k=k
            )
            return results
        except Exception as e:
            logger.error(f"Vector similarity search failed: {e}")
            return []
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Document]:
        """Retrieve a specific chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Document if found, None otherwise
        """
        try:
            # Search with metadata filter
            results = self.vectorstore.similarity_search(
                query="",  # Empty query
                k=1,
                filter={"chunk_id": chunk_id}
            )
            return results[0] if results else None
        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {e}")
            return None
    
    def update_chunk(self, chunk_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """Update an existing chunk or add if not exists.
        
        Args:
            chunk_id: Chunk identifier
            text: Updated text content
            metadata: Updated metadata
            
        Returns:
            True if successful
        """
        try:
            # Add/update the chunk
            metadata['chunk_id'] = chunk_id
            doc = Document(page_content=text, metadata=metadata)
            
            self.vectorstore.add_documents([doc], ids=[chunk_id])
            logger.debug(f"Updated chunk {chunk_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update chunk {chunk_id}: {e}")
            return False
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk from the vectorstore.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            True if successful
        """
        try:
            # Delete by ID
            self.vectorstore.delete([chunk_id])
            logger.debug(f"Deleted chunk {chunk_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete chunk {chunk_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vectorstore statistics.
        
        Returns:
            Dictionary with vectorstore statistics
        """
        try:
            # Get index info from Redis
            info = self.redis_client.execute_command("FT.INFO", self.index_name)
            
            # Parse the info response
            stats = {}
            for i in range(0, len(info), 2):
                key = info[i].decode() if isinstance(info[i], bytes) else str(info[i])
                value = info[i + 1]
                if isinstance(value, bytes):
                    value = value.decode()
                stats[key] = value
            
            return {
                'index_name': self.index_name,
                'num_docs': stats.get('num_docs', 0),
                'index_size': stats.get('inverted_sz_mb', 0),
                'vector_index_size': stats.get('vector_index_sz_mb', 0)
            }
        except Exception as e:
            logger.error(f"Failed to get vectorstore stats: {e}")
            return {}
    
    def clear_index(self):
        """Clear all documents from the index."""
        try:
            # Drop and recreate index
            self.redis_client.execute_command("FT.DROPINDEX", self.index_name, "DD")
            self._initialize_vectorstore()
            logger.info(f"Cleared index {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
    
    def close(self):
        """Close Redis connection."""
        if self.redis_client:
            self.redis_client.close()
