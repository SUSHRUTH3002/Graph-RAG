"""Configuration management for Graph-RAG project."""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

@dataclass
class Neo4jConfig:
    """Neo4j database configuration."""
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")  # Bolt protocol port
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "password123")
    database: str = os.getenv("NEO4J_DATABASE", "neo4j")

@dataclass
class RedisConfig:
    """Redis database configuration."""
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    password: Optional[str] = os.getenv("REDIS_PASSWORD") or None
    decode_responses: bool = False

@dataclass
class EmbeddingConfig:
    """Embedding configuration."""
    backend: str = os.getenv("EMBEDDING_BACKEND", "hf")  # 'hf' or 'openai'
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY") or None
    hf_model_name: str = os.getenv("HF_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    dimension: int = 384  # Default for all-MiniLM-L6-v2

@dataclass
class ChunkingConfig:
    """Text chunking configuration."""
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    method: str = os.getenv("CHUNKING_METHOD", "fixed")  # 'fixed', 'sentence', 'semantic', 'csv_row'

@dataclass
class CSVChunkingConfig:
    """CSV-specific chunking configuration."""
    row_format: str = os.getenv("CSV_ROW_FORMAT", "key_value")  # 'key_value', 'json', 'sentence'
    include_header: bool = os.getenv("CSV_INCLUDE_HEADER", "true").lower() == "true"

@dataclass
class EntityExtractionConfig:
    """Entity extraction configuration."""
    method: str = os.getenv("ENTITY_EXTRACTION_METHOD", "csv")  # 'keyword', 'llm', 'csv'
    
    # Key columns for CSV entity extraction
    key_columns: List[str] = field(default_factory=lambda: [
        "section", 
        "subsection", 
        "year",
        "perspective",
        "source", 
        "target",
        "section"
    ])
    
    # Tag columns containing comma-separated values
    tag_columns: List[str] = field(default_factory=lambda: ["tags"])
    
    # Column that specifies entity type
    entity_type_column: Optional[str] = "target_type"
    
    # LLM configuration (if using LLM method)
    llm_model: str = os.getenv("ENTITY_LLM_MODEL", "gpt-3.5-turbo")
    llm_api_key: Optional[str] = os.getenv("OPENAI_API_KEY") or None
    
    # Keyword extraction configuration
    min_entity_length: int = 2
    max_entity_length: int = 50
    extract_numbers: bool = True
    extract_emails: bool = True
    extract_urls: bool = True

@dataclass
class RelationshipExtractionConfig:
    """Relationship extraction configuration."""
    method: str = os.getenv("RELATIONSHIP_EXTRACTION_METHOD", "csv")  # 'cooccurrence', 'pattern', 'csv'
    
    # Key column for relationship source
    key_column: Optional[str] = "source"
    
    # Columns that define explicit relationships
    relationship_columns: List[str] = field(default_factory=lambda: [
        "relation",
        "content"
    ])
    
    # Extract hierarchical relationships (parent-child)
    extract_hierarchical: bool = True
    
    # Co-occurrence configuration (if using cooccurrence method)
    window_size: int = 100
    min_confidence: float = 0.5
    
    # Pattern-based extraction patterns (if using pattern method)
    custom_patterns: Optional[List[dict]] = None

@dataclass
class LLMConfig:
    """LLM configuration for answer generation."""
    provider: str = "gemini"  # Options: 'gemini', 'openai', 'huggingface'
    model_name: str = "gemini-pro"  # or "gemini-1.5-flash", "gpt-3.5-turbo", etc.
    api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")  # or OPENAI_API_KEY, HUGGINGFACE_API_KEY
    temperature: float = 0.1
    max_tokens: int = 1000
    include_citations: bool = True

@dataclass
class Config:
    """Main configuration class."""
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    csv_chunking: CSVChunkingConfig = field(default_factory=CSVChunkingConfig)
    entity_extraction: EntityExtractionConfig = field(default_factory=EntityExtractionConfig)
    relationship_extraction: RelationshipExtractionConfig = field(default_factory=RelationshipExtractionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

# Global config instance
config = Config()

# File paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "csv"
OUTPUT_DIR = BASE_DIR / "output"

INPUT_FILE_PATH = DATA_DIR / "entity_relations_graph.csv"

# Entity Extraction Configuration
ENTITY_EXTRACTION_METHOD = "csv"  # Options: 'csv', 'keyword', 'gemini', 'llm'
OUTPUT_FILE_PATH_FOR_ENTITY_EXTRACTION = OUTPUT_DIR / "extracted_entities.json"
OUTPUT_FILE_PATH_FOR_ENTITY_SUMMARY = OUTPUT_DIR / "entity_summary.txt"
USE_LLM_FOR_EVIDENCE = False  # Set to True to use Gemini for evidence_note extraction
GEMINI_MODEL_NAME = "gemini-1.5-flash"
ENTITY_DEDUPLICATION_THRESHOLD = 0.85

# Entity Types
ENTITY_TYPES = [
    'ORGANIZATION',
    'PRODUCT',
    'BRAND',
    'PERSON',
    'LOCATION',
    'CONCEPT',
    'TECHNOLOGY',
    'METRIC',
    'DEPARTMENT',
    'EVENT',
    'SECTION',
    'YEAR'
]

# Relationship Extraction Configuration
RELATIONSHIP_EXTRACTION_METHOD = "csv"  # Options: 'csv', 'cooccurrence', 'pattern'
OUTPUT_FILE_PATH_FOR_RELATIONSHIP_EXTRACTION = OUTPUT_DIR / "extracted_relationships.json"

# Knowledge Graph Configuration
OUTPUT_FILE_PATH_FOR_GRAPH_STATISTICS = OUTPUT_DIR / "graph_statistics.json"

# Vector Store Configuration
VECTOR_STORE_PATH = OUTPUT_DIR / "vector_store"
REDIS_URL = f"redis://{config.redis.host}:{config.redis.port}"

# Indexing Configuration
INDEXING_STATS_PATH = OUTPUT_DIR / "indexing_stats.json"

# Retrieval Configuration
RETRIEVAL_TEST_RESULTS_OUTPUT_PATH = OUTPUT_DIR / "retrieval_test_results.json"

# Testing Configuration
TESTING_MODE = False