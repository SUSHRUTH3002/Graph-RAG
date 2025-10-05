"""Document chunking strategies."""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import re

class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Chunk text into smaller pieces with metadata."""
        pass

class FixedSizeChunking(ChunkingStrategy):
    """Fixed-size chunking with character overlap."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize fixed-size chunking.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Chunk text into fixed-size pieces.
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            return []
        
        metadata = metadata or {}
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # Find a good breaking point (end of sentence or word)
            if end < text_length:
                # Try to break at sentence boundary
                last_period = text.rfind('.', start, end)
                last_question = text.rfind('?', start, end)
                last_exclaim = text.rfind('!', start, end)
                
                # Find the latest sentence boundary
                sentence_end = max(last_period, last_question, last_exclaim)
                
                if sentence_end > start:
                    end = sentence_end + 1
                else:
                    # Try to break at word boundary
                    last_space = text.rfind(' ', start, end)
                    if last_space > start:
                        end = last_space
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        **metadata,
                        'chunk_index': len(chunks),
                        'start_char': start,
                        'end_char': end
                    }
                })
            
            # Move start position with overlap
            start = end - self.chunk_overlap if end < text_length else end
        
        return chunks

class SentenceChunking(ChunkingStrategy):
    """Sentence-based chunking strategy."""
    
    def __init__(self, sentences_per_chunk: int = 3, sentence_overlap: int = 1):
        """Initialize sentence chunking.
        
        Args:
            sentences_per_chunk: Number of sentences per chunk
            sentence_overlap: Number of overlapping sentences
        """
        self.sentences_per_chunk = sentences_per_chunk
        self.sentence_overlap = sentence_overlap
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Chunk text by sentences.
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            return []
        
        metadata = metadata or {}
        
        # Split into sentences (simple regex approach)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        chunks = []
        i = 0
        
        while i < len(sentences):
            # Get sentences for this chunk
            chunk_sentences = sentences[i:i + self.sentences_per_chunk]
            chunk_text = ' '.join(chunk_sentences)
            
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    **metadata,
                    'chunk_index': len(chunks),
                    'sentence_start': i,
                    'sentence_end': i + len(chunk_sentences)
                }
            })
            
            # Move with overlap
            i += self.sentences_per_chunk - self.sentence_overlap
        
        return chunks

class SemanticChunking(ChunkingStrategy):
    """Semantic chunking based on paragraph breaks and topics."""
    
    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 100):
        """Initialize semantic chunking.
        
        Args:
            max_chunk_size: Maximum size of each chunk
            min_chunk_size: Minimum size of each chunk
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Chunk text semantically by paragraphs.
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            return []
        
        metadata = metadata or {}
        
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            # Fall back to fixed-size chunking
            return FixedSizeChunking(
                chunk_size=self.max_chunk_size, 
                chunk_overlap=50
            ).chunk(text, metadata)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            # If paragraph itself is too large, split it
            if para_size > self.max_chunk_size:
                # Save current chunk if any
                if current_chunk:
                    chunks.append({
                        'text': '\n\n'.join(current_chunk),
                        'metadata': {
                            **metadata,
                            'chunk_index': len(chunks),
                            'type': 'semantic'
                        }
                    })
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph
                sub_chunks = FixedSizeChunking(
                    chunk_size=self.max_chunk_size,
                    chunk_overlap=50
                ).chunk(para, metadata)
                
                for sub_chunk in sub_chunks:
                    sub_chunk['metadata']['type'] = 'semantic_split'
                    chunks.append(sub_chunk)
            
            # Check if adding this paragraph exceeds max size
            elif current_size + para_size > self.max_chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append({
                        'text': '\n\n'.join(current_chunk),
                        'metadata': {
                            **metadata,
                            'chunk_index': len(chunks),
                            'type': 'semantic'
                        }
                    })
                
                # Start new chunk
                current_chunk = [para]
                current_size = para_size
            else:
                # Add to current chunk
                current_chunk.append(para)
                current_size += para_size
        
        # Save final chunk
        if current_chunk:
            chunks.append({
                'text': '\n\n'.join(current_chunk),
                'metadata': {
                    **metadata,
                    'chunk_index': len(chunks),
                    'type': 'semantic'
                }
            })
        
        return chunks

class CSVRowChunking(ChunkingStrategy):
    """CSV row-wise chunking strategy - each row becomes a chunk."""
    
    def __init__(self, include_header: bool = True, row_format: str = "key_value"):
        """Initialize CSV row chunking.
        
        Args:
            include_header: Whether to include column names in each chunk
            row_format: Format for row text ('key_value', 'json', or 'sentence')
                - key_value: "column1: value1, column2: value2"
                - json: JSON string representation
                - sentence: "The column1 is value1. The column2 is value2."
        """
        self.include_header = include_header
        self.row_format = row_format
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Chunk CSV data by rows.
        
        Args:
            text: Input text (not used for CSV, uses metadata['rows'])
            metadata: Must contain 'rows' key with list of row dictionaries
            
        Returns:
            List of chunk dictionaries, one per CSV row
        """
        metadata = metadata or {}
        
        # Check if we have CSV row data in metadata
        if 'rows' not in metadata:
            # Fall back to text-based chunking
            return FixedSizeChunking().chunk(text, metadata)
        
        rows = metadata.get('rows', [])
        if not rows:
            return []
        
        chunks = []
        columns = list(rows[0].keys()) if rows else []
        
        for row_idx, row in enumerate(rows):
            # Format row as text based on selected format
            if self.row_format == "json":
                import json
                chunk_text = json.dumps(row, ensure_ascii=False)
            elif self.row_format == "sentence":
                # Create natural language sentences
                sentences = [f"The {key} is {value}" for key, value in row.items()]
                chunk_text = ". ".join(sentences) + "."
            else:  # key_value (default)
                chunk_text = ", ".join([f"{k}: {v}" for k, v in row.items()])
            
            # Create chunk metadata
            chunk_metadata = {
                **{k: v for k, v in metadata.items() if k != 'rows'},
                'chunk_index': row_idx,
                'row_index': row_idx,
                'chunk_type': 'csv_row',
                'columns': columns,
                'row_data': row  # Keep original row data for reference
            }
            
            chunks.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })
        
        return chunks


def create_chunking_strategy(method: str = "fixed", **kwargs) -> ChunkingStrategy:
    """Factory function to create chunking strategy.
    
    Args:
        method: Chunking method ('fixed', 'sentence', 'semantic', or 'csv_row')
        **kwargs: Additional arguments for the chunking strategy
        
    Returns:
        ChunkingStrategy instance
    """
    if method == "fixed":
        return FixedSizeChunking(
            chunk_size=kwargs.get("chunk_size", 500),
            chunk_overlap=kwargs.get("chunk_overlap", 50)
        )
    elif method == "sentence":
        return SentenceChunking(
            sentences_per_chunk=kwargs.get("sentences_per_chunk", 3),
            sentence_overlap=kwargs.get("sentence_overlap", 1)
        )
    elif method == "semantic":
        return SemanticChunking(
            max_chunk_size=kwargs.get("max_chunk_size", 1000),
            min_chunk_size=kwargs.get("min_chunk_size", 100)
        )
    elif method == "csv_row":
        return CSVRowChunking(
            include_header=kwargs.get("include_header", True),
            row_format=kwargs.get("row_format", "key_value")
        )
    else:
        raise ValueError(f"Unknown chunking method: {method}")