"""Embeddings client wrapper supporting multiple backends."""

from typing import List, Dict, Any, Optional
import logging
from abc import ABC, abstractmethod

# LangChain imports
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

class EmbeddingsClient:
    """Unified embeddings client supporting multiple backends."""
    
    def __init__(self, backend: str = "hf", model_name: Optional[str] = None, **kwargs):
        """Initialize embeddings client.
        
        Args:
            backend: Backend type ('hf' for HuggingFace, 'openai' for OpenAI)
            model_name: Model name/path
            **kwargs: Additional arguments for the backend
        """
        self.backend = backend.lower()
        self.model_name = model_name
        self.embeddings = self._initialize_backend(**kwargs)
    
    def _initialize_backend(self, **kwargs) -> Embeddings:
        """Initialize the specified embedding backend.
        
        Returns:
            Embeddings instance
        """
        if self.backend == "hf":
            return self._initialize_huggingface(**kwargs)
        elif self.backend == "openai":
            return self._initialize_openai(**kwargs)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _initialize_huggingface(self, **kwargs) -> HuggingFaceEmbeddings:
        """Initialize HuggingFace embeddings.
        
        Returns:
            HuggingFaceEmbeddings instance
        """
        # Default model for sentence transformers
        default_model = "sentence-transformers/all-MiniLM-L6-v2"
        model_name = self.model_name or default_model
        
        # Default parameters
        model_kwargs = kwargs.get('model_kwargs', {'device': 'cpu'})
        encode_kwargs = kwargs.get('encode_kwargs', {'normalize_embeddings': True})
        
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                cache_folder=kwargs.get('cache_folder', None)
            )
            
            logger.info(f"Initialized HuggingFace embeddings with model: {model_name}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace embeddings: {e}")
            raise
    
    def _initialize_openai(self, **kwargs) -> OpenAIEmbeddings:
        """Initialize OpenAI embeddings.
        
        Returns:
            OpenAIEmbeddings instance
        """
        # Default model
        default_model = "text-embedding-ada-002"
        model_name = self.model_name or default_model
        
        try:
            embeddings = OpenAIEmbeddings(
                model=model_name,
                openai_api_key=kwargs.get('api_key'),
                openai_api_base=kwargs.get('api_base'),
                max_retries=kwargs.get('max_retries', 3),
                request_timeout=kwargs.get('request_timeout', 60)
            )
            
            logger.info(f"Initialized OpenAI embeddings with model: {model_name}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.
        
        Args:
            texts: List of text documents
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.debug(f"Embedded {len(texts)} documents")
            return embeddings
        except Exception as e:
            logger.error(f"Document embedding failed: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.embeddings.embed_query(text)
            logger.debug(f"Embedded query: {text[:50]}...")
            return embedding
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings.
        
        Returns:
            Embedding dimension
        """
        try:
            # Embed a test string to get dimension
            test_embedding = self.embed_query("test")
            return len(test_embedding)
        except Exception as e:
            logger.error(f"Failed to get embedding dimension: {e}")
            # Default dimensions for common models
            if self.backend == "hf":
                return 384  # Common for sentence-transformers
            elif self.backend == "openai":
                return 1536  # text-embedding-ada-002
            return 768  # Generic default
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Model information dictionary
        """
        return {
            'backend': self.backend,
            'model_name': self.model_name,
            'embedding_dimension': self.get_embedding_dimension()
        }

# Factory function for easy initialization
def create_embeddings_client(backend: str = "hf", **kwargs) -> EmbeddingsClient:
    """Factory function to create embeddings client.
    
    Args:
        backend: Backend type ('hf' or 'openai')
        **kwargs: Backend-specific arguments
        
    Returns:
        EmbeddingsClient instance
    """
    return EmbeddingsClient(backend=backend, **kwargs)

# Predefined configurations
EMBEDDING_CONFIGS = {
    'hf_default': {
        'backend': 'hf',
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'model_kwargs': {'device': 'cpu'},
        'encode_kwargs': {'normalize_embeddings': True}
    },
    'hf_large': {
        'backend': 'hf',
        'model_name': 'sentence-transformers/all-mpnet-base-v2',
        'model_kwargs': {'device': 'cpu'},
        'encode_kwargs': {'normalize_embeddings': True}
    },
    'openai_ada': {
        'backend': 'openai',
        'model_name': 'text-embedding-ada-002'
    },
    'openai_3_small': {
        'backend': 'openai',
        'model_name': 'text-embedding-3-small'
    }
}

def get_embeddings_from_config(config_name: str, **override_kwargs) -> EmbeddingsClient:
    """Get embeddings client from predefined configuration.
    
    Args:
        config_name: Configuration name from EMBEDDING_CONFIGS
        **override_kwargs: Arguments to override in configuration
        
    Returns:
        EmbeddingsClient instance
    """
    if config_name not in EMBEDDING_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(EMBEDDING_CONFIGS.keys())}")
    
    config = EMBEDDING_CONFIGS[config_name].copy()
    config.update(override_kwargs)
    
    return EmbeddingsClient(**config)
