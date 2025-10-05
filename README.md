# Graph RAG System

A comprehensive **Graph-based Retrieval-Augmented Generation (Graph RAG)** system that combines knowledge graphs with semantic search and Large Language Models (LLMs) to provide intelligent question-answering capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Data Preparation](#data-preparation)
- [Pipeline Execution](#pipeline-execution)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)

---

## 🎯 Overview

Graph RAG is an advanced retrieval-augmented generation system that processes structured and unstructured data to build a knowledge graph enriched with semantic embeddings. It enables intelligent querying through multiple strategies:

- **Vector-based semantic search** for similarity matching
- **Graph traversal** for relationship exploration
- **Hybrid retrieval** combining both approaches
- **Adaptive routing** based on query intent
- **LLM-powered answer generation** with citations

### Key Components

1. **Data Ingestion**: Multi-format document loading (CSV, JSON, TXT, Markdown)
2. **Chunking Engine**: Context-aware text segmentation
3. **Entity Extraction**: Multi-strategy entity recognition (CSV, Keyword, LLM)
4. **Relationship Extraction**: Pattern-based and CSV-driven relationship mining
5. **Knowledge Graph**: Neo4j-based graph database
6. **Vector Store**: FAISS + optional Redis for embeddings
7. **Query Processing**: Intent classification and entity recognition
8. **Retrieval Engine**: Multi-strategy information retrieval
9. **Context Assembly**: Intelligent context preparation for LLMs
10. **Answer Generation**: Gemini-powered response generation

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Graph RAG Pipeline                        │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Step 1     │────▶│   Step 2     │────▶│   Step 3     │
│    Data      │     │   Chunking   │     │   Entity     │
│  Ingestion   │     │              │     │ Extraction   │
└──────────────┘     └──────────────┘     └──────────────┘
                                                   │
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Step 6     │◀────│   Step 5     │◀────│   Step 4     │
│   Graph      │     │   Graph      │     │ Relationship │
│  Indexing    │     │Construction  │     │ Extraction   │
└──────────────┘     └──────────────┘     └──────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│           Interactive QA System (Steps 7-10)                 │
├─────────────────────────────────────────────────────────────┤
│  Step 7: Query Processing                                    │
│  Step 8: Multi-Strategy Retrieval                           │
│  Step 9: Context Assembly                                    │
│  Step 10: LLM Answer Generation                              │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
CSV/JSON/TXT Files
       ↓
  Data Ingestion → Document Objects
       ↓
  Chunking → Text Chunks with Metadata
       ↓
  Entity Extraction → Entities (Organizations, Products, etc.)
       ↓
  Relationship Extraction → Relationships between Entities
       ↓
  Graph Construction → In-memory Graph
       ↓
  Neo4j Persistence → Knowledge Graph in Database
       ↓
  Vector Indexing → Embeddings in FAISS/Redis
       ↓
  Query Processing → Parsed Query with Intent
       ↓
  Retrieval Engine → Relevant Nodes & Edges
       ↓
  Context Assembly → Formatted Context
       ↓
  LLM Generation → Natural Language Answer
```

---

## ✨ Features

### Data Processing
- ✅ Multi-format support: CSV, JSON, TXT, Markdown
- ✅ CSV row-wise chunking for structured data
- ✅ Fixed-size, sentence-based, and semantic chunking
- ✅ Metadata preservation throughout pipeline

### Entity & Relationship Extraction
- ✅ CSV-specific entity extraction with column mapping
- ✅ Keyword-based pattern matching
- ✅ Optional Gemini LLM extraction for evidence fields
- ✅ Automatic entity deduplication
- ✅ Hierarchical relationship detection
- ✅ Temporal and spatial relationship tracking

### Knowledge Graph
- ✅ Neo4j graph database integration
- ✅ Multiple node types (Organization, Product, Section, Year, etc.)
- ✅ Rich relationship types (HAS_SUBSECTION, IN_YEAR, MENTIONS, etc.)
- ✅ Automatic index creation for performance
- ✅ Graph statistics and visualization support

### Vector Search
- ✅ Sentence transformer embeddings (HuggingFace)
- ✅ FAISS vector store for fast similarity search
- ✅ Optional Redis vector database for production
- ✅ Semantic node and edge search
- ✅ Filtered search by entity types

### Retrieval Strategies
- ✅ **Vector-only**: Pure semantic similarity search
- ✅ **Graph-only**: Graph traversal and entity expansion
- ✅ **Hybrid**: Combined vector + graph retrieval
- ✅ **Adaptive**: Automatic strategy selection based on query intent

### Query Understanding
- ✅ Intent classification (Entity Lookup, Relationship, Exploration, Factual)
- ✅ Entity recognition from queries
- ✅ Temporal and categorical filter extraction
- ✅ Query expansion with synonyms

### Answer Generation
- ✅ Google Gemini integration for LLM-powered answers
- ✅ Context-aware response generation
- ✅ Source citation and evidence tracking
- ✅ Confidence scoring
- ✅ Optional inline citation mode

---

## 💻 System Requirements

### Software Dependencies
- Python 3.8+
- Neo4j 4.x or 5.x
- Redis (optional, for production vector store)

### Hardware Recommendations
- **Minimum**: 8GB RAM, 4 CPU cores
- **Recommended**: 16GB RAM, 8 CPU cores, GPU for embeddings
- **Storage**: 5GB+ for databases and indexes

### Operating Systems
- Windows 10/11
- Linux (Ubuntu 20.04+)
- macOS 11+

---

## 🔧 Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd "Graph RAG"
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Key packages**:
- `neo4j`: Neo4j Python driver
- `langchain-community`: LangChain integrations
- `langchain-huggingface`: HuggingFace embeddings
- `langchain-core`: Core LangChain functionality
- `sentence-transformers`: Text embeddings
- `faiss-cpu` or `faiss-gpu`: Vector similarity search
- `google-generativeai`: Gemini LLM integration
- `redis`: Redis client (optional)
- `python-dotenv`: Environment variable management

### 4. Install Neo4j

#### Option A: Docker (Recommended)

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:latest
```

#### Option B: Local Installation

Download from [Neo4j Download Center](https://neo4j.com/download/)

Configure in `neo4j.conf`:
```
dbms.default_listen_address=0.0.0.0
dbms.connector.bolt.enabled=true
dbms.connector.bolt.listen_address=:7687
```

### 5. Install Redis (Optional)

```bash
# Docker
docker run -d --name redis -p 6379:6379 redis:latest

# Ubuntu
sudo apt-get install redis-server

# macOS
brew install redis
```

---

## ⚙️ Configuration

### 1. Environment Variables

Create `.env` file in project root:

```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
NEO4J_DATABASE=neo4j

# Redis Configuration (Optional)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Embeddings Configuration
EMBEDDING_BACKEND=hf
HF_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# LLM Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here

# OpenAI (Alternative)
OPENAI_API_KEY=your_openai_key_here
```

### 2. Configuration File

Edit `config.py` for advanced settings:

```python
# Entity Extraction
ENTITY_EXTRACTION_METHOD = "csv"  # csv, keyword, gemini, llm
USE_LLM_FOR_EVIDENCE = False  # Use Gemini for evidence_note extraction

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "csv_row"  # fixed, sentence, semantic, csv_row

# Relationship Extraction
RELATIONSHIP_EXTRACTION_METHOD = "csv"  # csv, cooccurrence, pattern

# Entity Types
ENTITY_TYPES = [
    'ORGANIZATION', 'PRODUCT', 'BRAND', 'PERSON', 
    'LOCATION', 'CONCEPT', 'TECHNOLOGY', 'METRIC',
    'DEPARTMENT', 'EVENT', 'SECTION', 'YEAR'
]
```

### 3. Get Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create new API key
3. Add to `.env` file

---

## 📊 Data Preparation

### Supported Formats

#### CSV Files (Recommended for Structured Data)

**Format 1: Entity Relations Graph**
```csv
source,relation,target,target_type,section,year,value,unit,evidence_note
Garudafood,produces,Chocolatos,product,Products,2023,,,Leading chocolate wafer brand
Marketing Dept,manages,Brand Strategy,concept,Organization,2023,,,Responsible for brand positioning
```

**Format 2: Subjective Sections**
```csv
section,subsection,year,perspective,content,tags
Financial Performance,Revenue Growth,2023,Positive,Company achieved 15% revenue growth,growth,revenue,performance
Sustainability,Environmental,2023,Commitment,Reduced carbon emissions by 20%,sustainability,environment
```

#### JSON Files
```json
{
  "title": "Annual Report 2023",
  "sections": [
    {
      "name": "Financial Highlights",
      "content": "Revenue increased by 15%..."
    }
  ]
}
```

#### Text/Markdown Files
Standard text documents with natural language content.

### Data Location

Place your data files in:
```
Graph RAG/
├── data/
│   ├── csv/
│   │   ├── entity_relations_graph.csv
│   │   └── subjective_sections_for_rag.csv
│   ├── json/
│   └── txt/
```

Update `config.py`:
```python
INPUT_FILE_PATH = DATA_DIR / "csv" / "entity_relations_graph.csv"
```

---

## 🚀 Pipeline Execution

### Full Pipeline

Run the complete pipeline:

```bash
python main.py
```

### Pipeline Steps

#### Step 1: Data Ingestion
```
Loading data from: e:\Projects\Graph RAG\data\csv\entity_relations_graph.csv
Document loaded: 150 rows
```

#### Step 2: Chunking
```
Creating CSV row chunking strategy...
Created 150 chunks (one per row)
```

#### Step 3: Entity Extraction
```
Extracting entities from chunks...
Extracted 450 raw entities
After deduplication: 320 unique entities

Entity Distribution:
  ORGANIZATION: 45
  PRODUCT: 78
  SECTION: 23
  YEAR: 3
  METRIC: 89
  CONCEPT: 82
```

#### Step 4: Relationship Extraction
```
Extracting relationships from chunks...
Extracted 580 raw relationships
After deduplication: 465 unique relationships

Relationship Types:
  HAS_SUBSECTION: 85
  IN_YEAR: 120
  MENTIONS_PRODUCT: 95
  IN_SECTION: 110
  RELATED_TO: 55
```

#### Step 5: Graph Construction
```
Building graph from entities and relationships...

Graph Statistics:
  Total Nodes: 470 (150 chunks + 320 entities)
  Total Edges: 615 (150 MENTIONS + 465 relationships)

Persisting to Neo4j...
  Nodes Created: 470
  Edges Created: 615
  Errors: 0
```

#### Step 6: Graph Indexing
```
Indexing graph with embeddings...

Indexing Statistics:
  Nodes indexed: 320
  Edges indexed: 465
  Embedding dimension: 384

FAISS Storage:
  Node vectors: 320
  Edge vectors: 465

Testing Hybrid Search:
Query: 'Garudafood financial performance'
  Found 10 relevant nodes
  Found 8 relevant edges
  Expanded subgraph: 25 nodes, 18 edges
```

#### Steps 7-10: Interactive QA System
```
INTERACTIVE QA MODE
Ask questions and get AI-powered answers from the knowledge graph!

Question> What is Garudafood's revenue in 2023?

Processing: What is Garudafood's revenue in 2023?
Strategy: adaptive

✓ Retrieved 8 nodes, 5 edges
  Query Intent: factual

📊 Top Retrieved Entities:
  1. [ORGANIZATION] Garudafood (score: 0.923)
  2. [METRIC] Revenue Growth (score: 0.887)
  3. [YEAR] 2023 (score: 0.856)

🤖 AI ANSWER:
Based on the knowledge graph, Garudafood reported a 15% revenue 
growth in 2023. The company's financial performance showed strong 
growth across multiple product categories...

📊 Answer Metadata:
  Confidence: 85.0%
  Entities used: 8
  Relationships: 5
  Context size: 1,234 tokens

📚 Sources (3):
  1. Section: Financial Performance
  2. entity_relations_graph.csv
  3. Section: Revenue Analysis
```

---

## 📖 Usage Guide

### Basic Query

```python
from src.hybrid_graph_store import HybridGraphStore
from src.retrieval_engine import RetrievalEngine, RetrievalStrategy

# Initialize system
hybrid_store = HybridGraphStore(...)
retrieval_engine = RetrievalEngine(hybrid_store, query_processor)

# Simple query
result = retrieval_engine.retrieve(
    query="What products does Garudafood make?",
    strategy=RetrievalStrategy.HYBRID,
    k=10
)

print(f"Found {len(result.nodes)} relevant entities")
```

### Advanced Retrieval

```python
# Vector-only search (semantic similarity)
result = retrieval_engine.retrieve(
    query="sustainability initiatives",
    strategy=RetrievalStrategy.VECTOR_ONLY,
    k=5
)

# Graph-only search (relationship traversal)
result = retrieval_engine.retrieve(
    query="Garudafood",
    strategy=RetrievalStrategy.GRAPH_ONLY,
    k=10
)

# Adaptive (automatic strategy selection)
result = retrieval_engine.retrieve(
    query="Compare revenue across products",
    strategy=RetrievalStrategy.ADAPTIVE,
    k=10
)
```

### Answer Generation

```python
from src.context_assembler import ContextAssembler
from src.answer_generator import AnswerGenerator

# Assemble context
context = context_assembler.assemble_from_retrieval(
    retrieval_result=result,
    query="What is Garudafood's strategy?",
    strategy="hybrid"
)

# Generate answer
answer = answer_generator.generate_answer(
    query="What is Garudafood's strategy?",
    context=context
)

print(answer.answer)
print(f"Confidence: {answer.confidence:.1%}")
```

### Interactive Commands

In interactive mode, use these commands:

```bash
# Basic query
Question> What is Garudafood's revenue?

# Use specific strategy
Question> strategy:vector_only financial performance

# Get citations
Question> cite sustainability initiatives

# Retrieval only (no answer generation)
Question> retrieve product portfolio

# Help
Question> help

# Exit
Question> exit
```

---

## 📚 API Reference

### Core Classes

#### `HybridGraphStore`
```python
store = HybridGraphStore(
    neo4j_store=neo4j_store,
    embeddings_client=embeddings_client,
    use_redis=False,
    vector_store_path="output/vector_store"
)

# Index graph
stats = store.index_graph()

# Search
nodes = store.search_similar_nodes("query", k=10)
edges = store.search_similar_edges("query", k=10)
results = store.hybrid_search("query", k=10, expand_graph=True)
```

#### `RetrievalEngine`
```python
engine = RetrievalEngine(hybrid_store, query_processor)

result = engine.retrieve(
    query="question",
    strategy=RetrievalStrategy.HYBRID,
    k=10,
    filters={'node_type': 'ORGANIZATION'}
)
```

#### `AnswerGenerator`
```python
generator = AnswerGenerator(
    model_name="gemini-1.5-flash",
    temperature=0.3,
    max_tokens=1024
)

answer = generator.generate_answer(query, context)
```

### Entity Extraction Methods

```python
# CSV extraction
extractor = create_entity_extractor(
    method="csv",
    key_columns=['section', 'year'],
    tag_columns=['tags']
)

# Keyword extraction
extractor = create_entity_extractor(
    method="keyword",
    min_entity_length=2,
    max_entity_length=50
)

# Gemini LLM extraction
extractor = create_entity_extractor(
    method="gemini",
    api_key="your_key",
    model_name="gemini-1.5-flash"
)
```

### Chunking Strategies

```python
# Fixed-size
chunker = create_chunking_strategy(
    method="fixed",
    chunk_size=500,
    chunk_overlap=50
)

# Sentence-based
chunker = create_chunking_strategy(
    method="sentence",
    sentences_per_chunk=3
)

# CSV row-based
chunker = create_chunking_strategy(
    method="csv_row",
    row_format="key_value"
)
```

---

## 🔍 Troubleshooting

### Neo4j Connection Issues

**Problem**: `ServiceUnavailable: Could not connect to Neo4j`

**Solution**:
```bash
# Check Neo4j is running
docker ps | grep neo4j

# Verify connection
curl http://localhost:7474

# Check credentials in .env
NEO4J_URI=bolt://localhost:7687  # Note: bolt:// not http://
NEO4J_PASSWORD=password123
```

### Gemini API Errors

**Problem**: `ValueError: Gemini API key not provided`

**Solution**:
```bash
# Add to .env
GEMINI_API_KEY=your_actual_key_here

# Verify
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('GEMINI_API_KEY'))"
```

### Memory Issues

**Problem**: Out of memory during embedding generation

**Solution**:
```python
# In config.py, reduce batch size
# Process fewer entities at once
BATCH_SIZE = 50  # Default is 100

# Or use smaller embedding model
HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims
```

### Slow Query Performance

**Problem**: Queries taking too long

**Solution**:
```bash
# Create Neo4j indexes
CREATE INDEX entity_name IF NOT EXISTS FOR (e:ENTITY) ON (e.name)
CREATE INDEX organization_name IF NOT EXISTS FOR (e:ORGANIZATION) ON (e.name)

# Use FAISS with GPU
pip install faiss-gpu

# Enable Redis for production
REDIS_ENABLED=true
```

---

## ⚡ Performance Optimization

### Indexing Performance

```python
# Use batch processing
BATCH_SIZE = 100

# Enable GPU for embeddings
embeddings_client = create_embeddings_client(
    backend='hf',
    model_kwargs={'device': 'cuda'}
)

# Use IVF index for large datasets (>100k vectors)
vector_store = VectorStore(dimension=384, index_type='ivf')
```

### Query Performance

```python
# Limit graph expansion depth
result = engine.retrieve(
    query="question",
    strategy=RetrievalStrategy.HYBRID,
    k=10,
    filters={'max_depth': 1}  # Reduce from default 2
)

# Use cached embeddings
hybrid_store.load_vector_stores()  # Load pre-computed embeddings
```

### Neo4j Optimization

```cypher
// Create indexes
CREATE INDEX entity_name IF NOT EXISTS FOR (e:ENTITY) ON (e.name);
CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_id);

// Add constraints
CREATE CONSTRAINT entity_name_unique IF NOT EXISTS 
FOR (e:ENTITY) REQUIRE e.name IS UNIQUE;

// Configure memory in neo4j.conf
dbms.memory.heap.initial_size=2G
dbms.memory.heap.max_size=4G
dbms.memory.pagecache.size=2G
```

---

## 📊 Output Files

All outputs are saved in `output/` directory:

```
output/
├── extracted_entities.json          # All extracted entities
├── entity_summary.txt               # Human-readable summary
├── extracted_relationships.json     # All relationships
├── graph_statistics.json           # Graph construction stats
├── vector_store/                   # FAISS indexes
│   ├── nodes.faiss
│   ├── nodes.faiss.metadata
│   ├── edges.faiss
│   └── edges.faiss.metadata
├── indexing_stats.json             # Indexing statistics
├── retrieval_test_results.json     # Test query results
└── full_qa_test_results.json       # Complete QA test results
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- LangChain for embedding integrations
- Neo4j for graph database
- Google Gemini for LLM capabilities
- Sentence Transformers for embeddings
- FAISS for efficient vector search

---

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the [Troubleshooting](#troubleshooting) section
- Review [Usage Guide](#usage-guide) for examples

---

**Built with ❤️ for intelligent information retrieval**
