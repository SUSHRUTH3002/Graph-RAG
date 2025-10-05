"""Embeddings client wrapper supporting multiple backends."""

from typing import List, Optional, Union
import numpy as np
from abc import ABC, abstractmethod

class EmbeddingClient(ABC):
    """Abstract base class for embedding clients."""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass

class HuggingFaceEmbeddings(EmbeddingClient):
    """HuggingFace Sentence Transformers embedding client."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize HuggingFace embeddings.
        
        Args:
            model_name: Name of the sentence-transformer model
        """
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents.
        
        Args:
            texts: List of text documents
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query.
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension

class OpenAIEmbeddings(EmbeddingClient):
    """OpenAI embeddings client."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """Initialize OpenAI embeddings.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI embedding model name
        """
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._dimension = 1536 if "3-small" in model else 3072
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents.
        
        Args:
            texts: List of text documents
            
        Returns:
            List of embedding vectors
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query.
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )
        return response.data[0].embedding
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension

def create_embedding_client(backend: str = "hf", **kwargs) -> EmbeddingClient:
    """Factory function to create embedding client.
    
    Args:
        backend: Either 'hf' for HuggingFace or 'openai' for OpenAI
        **kwargs: Additional arguments for the embedding client
        
    Returns:
        EmbeddingClient instance
    """
    if backend == "hf":
        model_name = kwargs.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model_name)
    elif backend == "openai":
        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError("OpenAI API key required for 'openai' backend")
        model = kwargs.get("model", "text-embedding-3-small")
        return OpenAIEmbeddings(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown embedding backend: {backend}")