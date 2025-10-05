"""Data ingestion module for loading and processing various file formats."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import csv
from abc import ABC, abstractmethod


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self, file_path: str) -> Dict[str, Any]:
        """Load a document from file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing document content and metadata
        """
        pass


class TextLoader(DocumentLoader):
    """Loader for plain text files."""
    
    def __init__(self, encoding: str = "utf-8"):
        """Initialize text loader.
        
        Args:
            encoding: Text encoding to use
        """
        self.encoding = encoding
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """Load a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Dictionary with content and metadata
        """
        path = Path(file_path)
        with open(path, 'r', encoding=self.encoding) as f:
            content = f.read()
        
        return {
            'content': content,
            'metadata': {
                'source': str(path),
                'filename': path.name,
                'file_type': 'text',
                'size': len(content)
            }
        }


class JSONLoader(DocumentLoader):
    """Loader for JSON files."""
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """Load a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary with content and metadata
        """
        path = Path(file_path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert JSON to text representation
        if isinstance(data, dict):
            content = json.dumps(data, indent=2)
        elif isinstance(data, list):
            content = '\n\n'.join([json.dumps(item, indent=2) for item in data])
        else:
            content = str(data)
        
        return {
            'content': content,
            'metadata': {
                'source': str(path),
                'filename': path.name,
                'file_type': 'json',
                'data': data
            }
        }


class CSVLoader(DocumentLoader):
    """Loader for CSV files."""
    
    def __init__(self, has_header: bool = True):
        """Initialize CSV loader.
        
        Args:
            has_header: Whether the CSV has a header row
        """
        self.has_header = has_header
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """Load a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary with content and metadata
        """
        path = Path(file_path)
        rows = []
        
        with open(path, 'r', encoding='utf-8') as f:
            if self.has_header:
                reader = csv.DictReader(f)
                rows = list(reader)
            else:
                reader = csv.reader(f)
                rows = [{'col_' + str(i): val for i, val in enumerate(row)} 
                       for row in reader]
        
        # Convert rows to text representation
        content_lines = []
        for idx, row in enumerate(rows):
            row_text = ', '.join([f"{k}: {v}" for k, v in row.items()])
            content_lines.append(f"Row {idx + 1}: {row_text}")
        
        content = '\n'.join(content_lines)
        
        return {
            'content': content,
            'metadata': {
                'source': str(path),
                'filename': path.name,
                'file_type': 'csv',
                'rows': rows,
                'row_count': len(rows)
            }
        }


class MarkdownLoader(DocumentLoader):
    """Loader for Markdown files."""
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """Load a Markdown file.
        
        Args:
            file_path: Path to the Markdown file
            
        Returns:
            Dictionary with content and metadata
        """
        path = Path(file_path)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            'content': content,
            'metadata': {
                'source': str(path),
                'filename': path.name,
                'file_type': 'markdown',
                'size': len(content)
            }
        }


class DataIngestion:
    """Main data ingestion class for processing multiple documents."""
    
    def __init__(self):
        """Initialize data ingestion with supported loaders."""
        self.loaders = {
            '.txt': TextLoader(),
            '.text': TextLoader(),
            '.json': JSONLoader(),
            '.csv': CSVLoader(),
            '.md': MarkdownLoader(),
            '.markdown': MarkdownLoader()
        }
    
    def add_loader(self, extension: str, loader: DocumentLoader):
        """Add a custom loader for a file extension.
        
        Args:
            extension: File extension (e.g., '.pdf')
            loader: DocumentLoader instance
        """
        self.loaders[extension] = loader
    
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """Load a single document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Dictionary with document content and metadata
            
        Raises:
            ValueError: If file extension is not supported
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension not in self.loaders:
            raise ValueError(
                f"Unsupported file extension: {extension}. "
                f"Supported extensions: {list(self.loaders.keys())}"
            )
        
        loader = self.loaders[extension]
        return loader.load(file_path)
    
    def load_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Load multiple documents.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            List of document dictionaries
        """
        documents = []
        for file_path in file_paths:
            try:
                doc = self.load_document(file_path)
                documents.append(doc)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def load_directory(
        self, 
        directory: str, 
        pattern: str = "*.*",
        recursive: bool = False
    ) -> List[Dict[str, Any]]:
        """Load all documents from a directory.
        
        Args:
            directory: Path to directory
            pattern: File pattern to match (e.g., "*.txt")
            recursive: Whether to search subdirectories
            
        Returns:
            List of document dictionaries
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise ValueError(f"Directory not found: {directory}")
        
        if recursive:
            files = list(dir_path.rglob(pattern))
        else:
            files = list(dir_path.glob(pattern))
        
        # Filter for supported extensions
        supported_files = [
            str(f) for f in files 
            if f.suffix.lower() in self.loaders
        ]
        
        return self.load_documents(supported_files)


def create_data_ingestion() -> DataIngestion:
    """Factory function to create a DataIngestion instance.
    
    Returns:
        DataIngestion instance
    """
    return DataIngestion()
