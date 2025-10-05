"""Main execution script for Graph RAG system."""

from pathlib import Path
from src.data_ingestion import create_data_ingestion
from src.chunking import create_chunking_strategy
from src.entity_extraction import create_entity_extractor, EntityDeduplicator, save_entity_summary
from src.relationship_extraction import create_relationship_extractor, RelationshipDeduplicator
from src.graph_builder import GraphBuilder
from src.neo4j_store import create_neo4j_store, Neo4jStore
from src.embeddings_client import create_embeddings_client
from src.hybrid_graph_store import HybridGraphStore
from src.query_processor import create_query_processor
from src.retrieval_engine import create_retrieval_engine, RetrievalStrategy
from src.context_assembler import create_context_assembler
from src.answer_generator import create_answer_generator
import json
import logging
import yaml
from config import (
    INPUT_FILE_PATH,
    OUTPUT_FILE_PATH_FOR_ENTITY_EXTRACTION,
    OUTPUT_FILE_PATH_FOR_ENTITY_SUMMARY,
    OUTPUT_FILE_PATH_FOR_RELATIONSHIP_EXTRACTION,
    OUTPUT_FILE_PATH_FOR_GRAPH_STATISTICS,
    VECTOR_STORE_PATH,
    INDEXING_STATS_PATH,
    RETRIEVAL_TEST_RESULTS_OUTPUT_PATH,
    TESTING_MODE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    # Return default configuration if file doesn't exist
    logger.warning("config.yaml not found, using default configuration")
    return {
        'neo4j': {
            'uri': 'bolt://localhost:7687',
            'user': 'neo4j',
            'password': 'password'
        },
        'embeddings': {
            'backend': 'hf',
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2'
        },
        'redis': {
            'enabled': False,
            'url': 'redis://localhost:6379'
        },
        'steps': {
            'data_ingestion': True,
            'chunking': True,
            'entity_extraction': True,
            'relationship_extraction': True,
            'graph_construction': True,
            'graph_indexing': True
        }
    }


def graph_indexing(config, neo4j_connection_params, embeddings_client, hybrid_store):
    """Step 6: Graph Indexing & Storage with Vector Embeddings."""
    print("\n" + "="*80)
    print("STEP 6: GRAPH INDEXING & STORAGE")
    print("="*80)
    
    try:
        # Initialize Neo4j store
        neo4j_config = config.get('neo4j', {})
        neo4j_store = create_neo4j_store()

        # Initialize embeddings client
        embeddings_config = config.get('embeddings', {})
        # embeddings_client = create_embeddings_client(
        #     backend=embeddings_config.get('backend', 'hf'),
        #     model_name=embeddings_config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        # )
        
        print(f"\nUsing embeddings: {embeddings_config.get('backend', 'hf')} - {embeddings_config.get('model_name', 'default')}")
        print(f"Embedding dimension: {embeddings_client.get_embedding_dimension()}")
        
        # Initialize hybrid graph store
        redis_config = config.get('redis', {})
        use_redis = redis_config.get('enabled', False)
        redis_url = redis_config.get('url') if use_redis else None
        
        print(f"\nVector Storage:")
        print(f"  - FAISS: Enabled (local)")
        print(f"  - Redis: {'Enabled' if use_redis else 'Disabled'}")
        
        # hybrid_store = HybridGraphStore(
        #     neo4j_store=neo4j_store,
        #     embeddings_client=embeddings_client,
        #     use_redis=use_redis,
        #     redis_url=redis_url,
        #     vector_store_path=VECTOR_STORE_PATH
        # )

        # Check if already indexed
        print("\nChecking for existing vector stores...")
        if hybrid_store.load_vector_stores():
            print("✓ Loaded existing vector stores from disk")
            stats = hybrid_store.get_statistics()
        else:
            # Index the graph
            print("\nIndexing graph with embeddings...")
            
            stats = hybrid_store.index_graph()
            
            print("\n✓ Graph indexing completed!")
        
        # Display statistics
        print("\n" + "-"*80)
        print("INDEXING STATISTICS")
        print("-"*80)
        
        embedding_stats = stats.get('embedding_generator', {})
        print(f"  Nodes indexed: {embedding_stats.get('total_node_embeddings', 'N/A')}")
        print(f"  Edges indexed: {embedding_stats.get('total_edge_embeddings', 'N/A')}")
        print(f"  Embedding dimension: {embedding_stats.get('embedding_dimension', 'N/A')}")
        
        faiss_node_stats = stats.get('faiss_nodes', {})
        faiss_edge_stats = stats.get('faiss_edges', {})
        print(f"\nFAISS Storage:")
        print(f"  Node vectors: {faiss_node_stats.get('total_vectors', 0)}")
        print(f"  Edge vectors: {faiss_edge_stats.get('total_vectors', 0)}")
        
        if use_redis and 'redis_nodes' in stats:
            redis_node_stats = stats.get('redis_nodes', {})
            redis_edge_stats = stats.get('redis_edges', {})
            print(f"\nRedis Storage:")
            print(f"  Node documents: {redis_node_stats.get('num_docs', 0)}")
            print(f"  Edge documents: {redis_edge_stats.get('num_docs', 0)}")
        
        # Test searches
        print("\n" + "-"*80)
        print("TESTING HYBRID SEARCH")
        print("-"*80)
        
        test_query = "Garudafood financial performance"
        print(f"\nQuery: '{test_query}'")
        
        print("\n1. Semantic Node Search:")
        node_results = hybrid_store.search_similar_nodes(test_query, k=5)
        
        if node_results:
            for i, (node_id, score, metadata) in enumerate(node_results[:3], 1):
                node_type = metadata.get('node_type', 'UNKNOWN')
                entity_name = metadata.get('entity_name', metadata.get('name', 'Unknown'))
                print(f"   {i}. [{node_type}] {entity_name} (score: {score:.3f})")
        else:
            print("   No results found")
        
        print("\n2. Semantic Edge Search:")
        edge_results = hybrid_store.search_similar_edges(test_query, k=5)
        
        if edge_results:
            for i, (edge_id, score, metadata) in enumerate(edge_results[:3], 1):
                rel = metadata.get('relationship_type', 'UNKNOWN')
                source = metadata.get('source', 'Unknown')
                target = metadata.get('target', 'Unknown')
                print(f"   {i}. {source} -{rel}-> {target} (score: {score:.3f})")
        else:
            print("   No results found")
        
        print("\n3. Hybrid Search (Vector + Graph Traversal):")
        hybrid_results = hybrid_store.hybrid_search(
            query=test_query,
            k=3,
            expand_graph=True,
            max_depth=2
        )
        print(f"   Found {len(hybrid_results.get('nodes', []))} relevant nodes")
        print(f"   Found {len(hybrid_results.get('edges', []))} relevant edges")
        
        if hybrid_results.get('expanded_subgraph'):
            sg = hybrid_results['expanded_subgraph']
            print(f"   Expanded subgraph: {sg.get('node_count', 0)} nodes, {sg.get('edge_count', 0)} edges")
        
        # Save statistics
        output_path = INDEXING_STATS_PATH
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"\n✓ Statistics saved to: {output_path}")
        
        neo4j_store.close()
        print("\n" + "="*80)
        return hybrid_store
        
    except Exception as e:
        print(f"\n✗ Error in Step 6: {e}")
        logger.error(f"Step 6 failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return None


def query_processing_and_retrieval(config, hybrid_store, embeddings_client, testing=False):
    """Steps 7 & 8: Query Processing and Retrieval."""
    print("\n" + "="*80)
    print("STEPS 7 & 8: QUERY PROCESSING AND RETRIEVAL")
    print("="*80)
    
    try:
        # Initialize query processor
        print("\n[Step 7] Initializing Query Processor...")
        query_processor = create_query_processor(
            embeddings_client=embeddings_client,
            neo4j_store=create_neo4j_store()
        )
        
        # Initialize retrieval engine
        print("[Step 8] Initializing Retrieval Engine...")
        retrieval_engine = create_retrieval_engine(
            hybrid_store=hybrid_store,
            query_processor=query_processor
        )
        
        print("\n✓ Query processing and retrieval systems initialized!")

        if testing:
            # Test with sample queries
            print("\n" + "-"*80)
            print("TESTING RETRIEVAL WITH SAMPLE QUERIES")
            print("-"*80)
            
            test_queries = [
                ("What is Garudafood's financial performance in 2023?", RetrievalStrategy.HYBRID),
                ("How is Garudafood related to sustainability?", RetrievalStrategy.GRAPH_ONLY),
                ("Compare revenue across different products", RetrievalStrategy.VECTOR_ONLY),
                ("Digital transformation initiatives", RetrievalStrategy.ADAPTIVE)
            ]
            
            all_results = []
            
            for i, (query, strategy) in enumerate(test_queries, 1):
                print(f"\n{i}. Query: '{query}'")
                print(f"   Strategy: {strategy.value}")
                import pdb; pdb.set_trace()
                result = retrieval_engine.retrieve(
                    query=query,
                    strategy=strategy,
                    k=5
                )
                
                print(f"   Intent: {result.metadata.get('query_intent', 'unknown')}")
                print(f"   Retrieved: {len(result.nodes)} nodes, {len(result.edges)} edges")
                
                if result.nodes:
                    print(f"   Top Results:")
                    for j, node in enumerate(result.nodes[:3], 1):
                        name = node['metadata'].get('entity_name') or node['metadata'].get('name', 'Unknown')
                        score = node.get('final_score', node.get('score', 0))
                        method = node.get('retrieval_method', 'unknown')
                        print(f"     {j}. {name} (score: {score:.3f}, method: {method})")
                
                all_results.append({
                    'query': query,
                    'strategy': strategy.value,
                    'result': {
                        'nodes': result.nodes,
                        'edges': result.edges,
                        'scores': result.scores,
                        'metadata': result.metadata
                    }
                })

            # Save test results
            output_path = RETRIEVAL_TEST_RESULTS_OUTPUT_PATH
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, default=str)
            
            print(f"\n✓ Test results saved to: {output_path}")
        
        # Interactive query mode
        print("\n" + "-"*80)
        print("INTERACTIVE QUERY MODE")
        print("-"*80)
        print("Enter your queries (type 'exit' to quit, 'help' for options)")
        
        while True:
            try:
                query = input("\nQuery> ").strip()
                
                if not query:
                    continue
                
                if query.lower() == 'exit':
                    break
                
                if query.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  exit - Quit interactive mode")
                    print("  help - Show this help message")
                    print("  strategy:<name> <query> - Use specific strategy (hybrid/vector_only/graph_only/adaptive)")
                    print("  Example: strategy:hybrid What is Garudafood?")
                    continue
                
                # Parse strategy if specified
                strategy = RetrievalStrategy.ADAPTIVE
                if query.startswith('strategy:'):
                    parts = query.split(' ', 1)
                    strategy_name = parts[0].split(':')[1].upper()
                    query = parts[1] if len(parts) > 1 else ''
                    
                    try:
                        strategy = RetrievalStrategy[strategy_name]
                    except KeyError:
                        print(f"Unknown strategy: {strategy_name}, using ADAPTIVE")
                
                # Process query
                print(f"\nProcessing query with {strategy.value} strategy...")
                result = retrieval_engine.retrieve(
                    query=query,
                    strategy=strategy,
                    k=10
                )
                
                # Display results
                print(f"\n{'='*60}")
                print(f"Query Intent: {result.metadata.get('query_intent', 'unknown')}")
                print(f"Strategy Used: {result.strategy_used.value}")
                print(f"{'='*60}")
                
                print(f"\nRetrieved {len(result.nodes)} nodes:")
                for i, node in enumerate(result.nodes[:5], 1):
                    name = node['metadata'].get('entity_name') or node['metadata'].get('name', 'Unknown')
                    node_type = node['metadata'].get('node_type', 'UNKNOWN')
                    score = node.get('final_score', node.get('score', 0))
                    print(f"  {i}. [{node_type}] {name} (score: {score:.3f})")
                
                if result.edges:
                    print(f"\nRetrieved {len(result.edges)} edges:")
                    for i, edge in enumerate(result.edges[:3], 1):
                        source = edge['metadata'].get('source', 'Unknown')
                        target = edge['metadata'].get('target', 'Unknown')
                        rel_type = edge['metadata'].get('relationship_type', 'UNKNOWN')
                        score = edge.get('final_score', edge.get('score', 0))
                        print(f"  {i}. {source} -{rel_type}-> {target} (score: {score:.3f})")
                
                if result.scores:
                    print(f"\nScores: {result.scores}")
                
            except KeyboardInterrupt:
                print("\n\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"\nError processing query: {e}")
                logger.error(f"Query processing error: {e}", exc_info=True)
        
        print("\n" + "="*80)
        return retrieval_engine
        
    except Exception as e:
        print(f"\n✗ Error in Steps 7 & 8: {e}")
        logger.error(f"Steps 7 & 8 failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return None


def context_assembly_and_answer_generation(config, retrieval_engine, embeddings_client, testing=False):
    """Steps 9 & 10: Context Assembly and Answer Generation."""
    print("\n" + "="*80)
    print("STEPS 9 & 10: CONTEXT ASSEMBLY & ANSWER GENERATION")
    print("="*80)
    
    try:
        # Initialize context assembler
        print("\n[Step 9] Initializing Context Assembler...")
        context_assembler = create_context_assembler(
            max_tokens=4000,
            include_sources=True,
            deduplicate=True
        )
        
        # Initialize answer generator
        print("[Step 10] Initializing Answer Generator (Gemini)...")
        try:
            answer_generator = create_answer_generator(
                model_name=config.get('llm', {}).get('model_name', 'gemini-1.5-flash'),
                temperature=config.get('llm', {}).get('temperature', 0.3),
                max_tokens=config.get('llm', {}).get('max_tokens', 1024)
            )
            print("✓ Gemini LLM initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize Gemini: {e}")
            print("  Please set GEMINI_API_KEY environment variable")
            return None
        
        print("\n✓ Context assembly and answer generation systems initialized!")
        
        if testing:
            # Test with sample queries
            print("\n" + "-"*80)
            print("TESTING END-TO-END QA WITH DIFFERENT STRATEGIES")
            print("-"*80)
            
            test_queries = [
                ("What is Garudafood's financial performance?", RetrievalStrategy.HYBRID),
                ("List the main products of Garudafood", RetrievalStrategy.VECTOR_ONLY),
                ("How is Garudafood structured?", RetrievalStrategy.GRAPH_ONLY),
            ]
            
            all_qa_results = []
            
            for i, (query, strategy) in enumerate(test_queries, 1):
                print(f"\n{'='*60}")
                print(f"TEST {i}: {query}")
                print(f"Strategy: {strategy.value}")
                print("="*60)
                
                # Step 8: Retrieve
                print("\n[Retrieval]")
                retrieval_result = retrieval_engine.retrieve(
                    query=query,
                    strategy=strategy,
                    k=10
                )
                print(f"  Retrieved: {len(retrieval_result.nodes)} nodes, {len(retrieval_result.edges)} edges")
                
                # Step 9: Assemble context
                print("\n[Context Assembly]")
                context = context_assembler.assemble_from_retrieval(
                    retrieval_result=retrieval_result,
                    query=query,
                    strategy=strategy.value
                )
                print(f"  Context tokens: {context.token_count}")
                print(f"  Entities: {len(context.entities)}")
                print(f"  Relationships: {len(context.relationships)}")
                print(f"  Sources: {len(context.sources)}")
                
                # Step 10: Generate answer
                print("\n[Answer Generation]")
                answer = answer_generator.generate_answer(
                    query=query,
                    context=context,
                    include_sources=True
                )
                print(f"  Confidence: {answer.confidence:.2f}")
                
                print("\n" + "-"*60)
                print("GENERATED ANSWER:")
                print("-"*60)
                print(answer.answer)
                
                if answer.sources:
                    print("\n" + "-"*60)
                    print("SOURCES:")
                    print("-"*60)
                    for j, source in enumerate(answer.sources[:5], 1):
                        print(f"  {j}. {source}")
                
                # Store result
                all_qa_results.append({
                    'query': query,
                    'strategy': strategy.value,
                    'retrieval': {
                        'node_count': len(retrieval_result.nodes),
                        'edge_count': len(retrieval_result.edges),
                    },
                    'context': {
                        'token_count': context.token_count,
                        'entity_count': len(context.entities),
                        'relationship_count': len(context.relationships)
                    },
                    'answer': answer.answer,
                    'confidence': answer.confidence,
                    'sources': answer.sources,
                    'metadata': answer.metadata
                })
            
            # Save QA results
            output_path = Path("output/qa_test_results.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_qa_results, f, indent=2, default=str)
            
            print(f"\n✓ QA test results saved to: {output_path}")
        
        # Interactive QA mode
        print("\n" + "-"*80)
        print("INTERACTIVE QA MODE")
        print("-"*80)
        print("Ask questions and get answers from the knowledge graph!")
        print("Commands:")
        print("  exit - Quit")
        print("  help - Show help")
        print("  strategy:<name> <query> - Use specific strategy")
        print("  cite <query> - Get answer with detailed citations")
        
        while True:
            try:
                query = input("\nQuestion> ").strip()
                
                if not query:
                    continue
                
                if query.lower() == 'exit':
                    break
                
                if query.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  exit - Quit interactive mode")
                    print("  help - Show this message")
                    print("  strategy:<name> <query> - hybrid/vector_only/graph_only/adaptive")
                    print("  cite <query> - Get answer with inline citations")
                    continue
                
                # Parse commands
                use_citations = False
                strategy = RetrievalStrategy.ADAPTIVE
                
                if query.startswith('cite '):
                    use_citations = True
                    query = query[5:].strip()
                
                if query.startswith('strategy:'):
                    parts = query.split(' ', 1)
                    strategy_name = parts[0].split(':')[1].upper()
                    query = parts[1] if len(parts) > 1 else ''
                    try:
                        strategy = RetrievalStrategy[strategy_name]
                    except KeyError:
                        print(f"Unknown strategy: {strategy_name}, using ADAPTIVE")
                
                print(f"\n{'='*60}")
                print(f"Processing: {query}")
                print(f"Strategy: {strategy.value}")
                print("="*60)
                
                # Retrieve
                print("\n⏳ Retrieving relevant information...")
                retrieval_result = retrieval_engine.retrieve(
                    query=query,
                    strategy=strategy,
                    k=10
                )
                
                # Assemble context
                print("⏳ Assembling context...")
                context = context_assembler.assemble_from_retrieval(
                    retrieval_result=retrieval_result,
                    query=query,
                    strategy=strategy.value
                )
                
                # Generate answer
                print("⏳ Generating answer...")
                if use_citations:
                    answer = answer_generator.generate_with_citations(query, context)
                else:
                    answer = answer_generator.generate_answer(query, context, include_sources=True)
                
                # Display results
                print("\n" + "="*60)
                print("ANSWER:")
                print("="*60)
                print(answer.answer)
                
                print(f"\n📊 Metadata:")
                print(f"  Confidence: {answer.confidence:.2%}")
                print(f"  Entities used: {context.metadata.get('entity_count', 0)}")
                print(f"  Relationships: {context.metadata.get('relationship_count', 0)}")
                print(f"  Context tokens: {context.token_count}")
                
                if answer.sources and not use_citations:
                    print(f"\n📚 Sources ({len(answer.sources)}):")
                    for j, source in enumerate(answer.sources[:5], 1):
                        print(f"  {j}. {source}")
                    if len(answer.sources) > 5:
                        print(f"  ... and {len(answer.sources) - 5} more")
                
            except KeyboardInterrupt:
                print("\n\nExiting interactive QA mode...")
                break
            except Exception as e:
                print(f"\n✗ Error processing question: {e}")
                logger.error(f"QA error: {e}", exc_info=True)
        
        print("\n" + "="*80)
        return {
            'context_assembler': context_assembler,
            'answer_generator': answer_generator
        }
        
    except Exception as e:
        print(f"\n✗ Error in Steps 9 & 10: {e}")
        logger.error(f"Steps 9 & 10 failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return None


def interactive_qa_system(config, hybrid_store, embeddings_client, testing=False):
    """Steps 7-10: Unified Interactive QA System.
    
    Combines:
    - Step 7: Query Processing
    - Step 8: Retrieval
    - Step 9: Context Assembly
    - Step 10: Answer Generation
    """
    print("\n" + "="*80)
    print("STEPS 7-10: INTERACTIVE QA SYSTEM")
    print("="*80)
    
    try:
        # Initialize all components
        print("\n[Step 7] Initializing Query Processor...")
        query_processor = create_query_processor(
            embeddings_client=embeddings_client,
            neo4j_store=create_neo4j_store()
        )
        
        print("[Step 8] Initializing Retrieval Engine...")
        retrieval_engine = create_retrieval_engine(
            hybrid_store=hybrid_store,
            query_processor=query_processor
        )
        
        print("[Step 9] Initializing Context Assembler...")
        context_assembler = create_context_assembler(
            max_tokens=4000,
            include_sources=True,
            deduplicate=True
        )
        
        print("[Step 10] Initializing Answer Generator (Gemini)...")
        try:
            answer_generator = create_answer_generator(
                model_name=config.get('llm', {}).get('model_name', 'gemini-1.5-flash'),
                temperature=config.get('llm', {}).get('temperature', 0.3),
                max_tokens=config.get('llm', {}).get('max_tokens', 1024)
            )
            print("✓ Gemini LLM initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize Gemini: {e}")
            print("  Please set GEMINI_API_KEY environment variable")
            answer_generator = None
        
        print("\n✓ All QA system components initialized!")
        
        # Testing mode - run automated tests
        if testing:
            print("\n" + "-"*80)
            print("TESTING FULL QA PIPELINE WITH SAMPLE QUERIES")
            print("-"*80)
            
            test_queries = [
                ("What is Garudafood's financial performance in 2023?", RetrievalStrategy.HYBRID),
                ("How is Garudafood related to sustainability?", RetrievalStrategy.GRAPH_ONLY),
                ("Compare revenue across different products", RetrievalStrategy.VECTOR_ONLY),
                ("Digital transformation initiatives", RetrievalStrategy.ADAPTIVE),
                ("List the main products of Garudafood", RetrievalStrategy.HYBRID),
            ]
            
            all_qa_results = []
            
            for i, (query, strategy) in enumerate(test_queries, 1):
                print(f"\n{'='*60}")
                print(f"TEST {i}/{len(test_queries)}: {query}")
                print(f"Strategy: {strategy.value}")
                print("="*60)
                
                # Step 8: Retrieve
                print("\n[Retrieval]")
                retrieval_result = retrieval_engine.retrieve(
                    query=query,
                    strategy=strategy,
                    k=10
                )
                print(f"  Intent: {retrieval_result.metadata.get('query_intent', 'unknown')}")
                print(f"  Retrieved: {len(retrieval_result.nodes)} nodes, {len(retrieval_result.edges)} edges")
                
                if retrieval_result.nodes:
                    print(f"  Top Results:")
                    for j, node in enumerate(retrieval_result.nodes[:3], 1):
                        name = node['metadata'].get('entity_name') or node['metadata'].get('name', 'Unknown')
                        score = node.get('final_score', node.get('score', 0))
                        print(f"    {j}. {name} (score: {score:.3f})")
                
                # Step 9: Assemble context
                print("\n[Context Assembly]")
                context = context_assembler.assemble_from_retrieval(
                    retrieval_result=retrieval_result,
                    query=query,
                    strategy=strategy.value
                )
                print(f"  Context tokens: {context.token_count}")
                print(f"  Entities: {len(context.entities)}")
                print(f"  Relationships: {len(context.relationships)}")
                print(f"  Sources: {len(context.sources)}")
                
                # Step 10: Generate answer (if available)
                answer = None
                if answer_generator:
                    print("\n[Answer Generation]")
                    answer = answer_generator.generate_answer(
                        query=query,
                        context=context,
                        include_sources=True
                    )
                    print(f"  Confidence: {answer.confidence:.2f}")
                    
                    print("\n" + "-"*60)
                    print("GENERATED ANSWER:")
                    print("-"*60)
                    print(answer.answer)
                    
                    if answer.sources:
                        print("\n" + "-"*60)
                        print("SOURCES:")
                        print("-"*60)
                        for j, source in enumerate(answer.sources[:5], 1):
                            print(f"  {j}. {source}")
                else:
                    print("\n[Answer Generation] Skipped - Gemini not available")
                
                # Store result
                result_data = {
                    'query': query,
                    'strategy': strategy.value,
                    'retrieval': {
                        'intent': retrieval_result.metadata.get('query_intent'),
                        'node_count': len(retrieval_result.nodes),
                        'edge_count': len(retrieval_result.edges),
                        'nodes': retrieval_result.nodes,
                        'edges': retrieval_result.edges,
                        'scores': retrieval_result.scores
                    },
                    'context': {
                        'token_count': context.token_count,
                        'entity_count': len(context.entities),
                        'relationship_count': len(context.relationships),
                        'sources_count': len(context.sources)
                    }
                }
                
                if answer:
                    result_data['answer'] = {
                        'text': answer.answer,
                        'confidence': answer.confidence,
                        'sources': answer.sources,
                        'metadata': answer.metadata
                    }
                
                all_qa_results.append(result_data)
            
            # Save comprehensive test results
            output_path = Path("output/full_qa_test_results.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_qa_results, f, indent=2, default=str)
            
            print(f"\n✓ Full QA test results saved to: {output_path}")
        
        # Interactive QA mode
        print("\n" + "-"*80)
        print("INTERACTIVE QA MODE")
        print("-"*80)
        print("Ask questions and get AI-powered answers from the knowledge graph!")
        print("\nCommands:")
        print("  exit - Quit interactive mode")
        print("  help - Show help")
        print("  strategy:<name> <query> - Use specific strategy (hybrid/vector_only/graph_only/adaptive)")
        print("  cite <query> - Get answer with inline citations")
        print("  retrieve <query> - Show retrieval results only (no answer generation)")
        
        while True:
            try:
                query = input("\nQuestion> ").strip()
                
                if not query:
                    continue
                
                if query.lower() == 'exit':
                    print("\n👋 Exiting interactive QA mode. Goodbye!")
                    break
                
                if query.lower() == 'help':
                    print("\n" + "="*60)
                    print("HELP - Available Commands")
                    print("="*60)
                    print("\nBasic Usage:")
                    print("  Just type your question and press Enter")
                    print("\nAdvanced Commands:")
                    print("  strategy:<name> <query>")
                    print("    Available strategies: hybrid, vector_only, graph_only, adaptive")
                    print("    Example: strategy:hybrid What is Garudafood?")
                    print("\n  cite <query>")
                    print("    Get answer with inline citations")
                    print("    Example: cite Financial performance of Garudafood")
                    print("\n  retrieve <query>")
                    print("    Show only retrieval results (no answer)")
                    print("    Example: retrieve Garudafood products")
                    print("\n  exit")
                    print("    Quit interactive mode")
                    continue
                
                # Parse commands
                use_citations = False
                retrieve_only = False
                strategy = RetrievalStrategy.ADAPTIVE
                
                if query.startswith('cite '):
                    use_citations = True
                    query = query[5:].strip()
                
                if query.startswith('retrieve '):
                    retrieve_only = True
                    query = query[9:].strip()
                
                if query.startswith('strategy:'):
                    parts = query.split(' ', 1)
                    strategy_name = parts[0].split(':')[1].upper()
                    query = parts[1] if len(parts) > 1 else ''
                    try:
                        strategy = RetrievalStrategy[strategy_name]
                    except KeyError:
                        print(f"⚠️  Unknown strategy: {strategy_name}, using ADAPTIVE")
                        strategy = RetrievalStrategy.ADAPTIVE
                
                if not query:
                    print("⚠️  Please provide a query after the command")
                    continue
                
                print(f"\n{'='*60}")
                print(f"Processing: {query}")
                print(f"Strategy: {strategy.value}")
                print("="*60)
                
                # Step 8: Retrieve
                print("\n⏳ Retrieving relevant information...")
                retrieval_result = retrieval_engine.retrieve(
                    query=query,
                    strategy=strategy,
                    k=10
                )
                
                print(f"✓ Retrieved {len(retrieval_result.nodes)} nodes, {len(retrieval_result.edges)} edges")
                print(f"  Query Intent: {retrieval_result.metadata.get('query_intent', 'unknown')}")
                
                # Display retrieval results
                if retrieval_result.nodes:
                    print(f"\n📊 Top Retrieved Entities:")
                    for i, node in enumerate(retrieval_result.nodes[:5], 1):
                        name = node['metadata'].get('entity_name') or node['metadata'].get('name', 'Unknown')
                        node_type = node['metadata'].get('node_type', 'UNKNOWN')
                        score = node.get('final_score', node.get('score', 0))
                        print(f"  {i}. [{node_type}] {name} (score: {score:.3f})")
                
                if retrieval_result.edges and not retrieve_only:
                    print(f"\n🔗 Key Relationships:")
                    for i, edge in enumerate(retrieval_result.edges[:3], 1):
                        source = edge['metadata'].get('source', 'Unknown')
                        target = edge['metadata'].get('target', 'Unknown')
                        rel_type = edge['metadata'].get('relationship_type', 'UNKNOWN')
                        print(f"  {i}. {source} --[{rel_type}]--> {target}")
                
                # If retrieve_only, skip answer generation
                if retrieve_only:
                    print(f"\n✓ Retrieval complete. Use without 'retrieve' command to generate answer.")
                    continue
                
                # Skip answer generation if Gemini not available
                if not answer_generator:
                    print(f"\n⚠️  Answer generation unavailable (Gemini not initialized)")
                    print(f"   Set GEMINI_API_KEY environment variable to enable")
                    continue
                
                # Step 9: Assemble context
                print("\n⏳ Assembling context...")
                context = context_assembler.assemble_from_retrieval(
                    retrieval_result=retrieval_result,
                    query=query,
                    strategy=strategy.value
                )
                print(f"✓ Context assembled: {context.token_count} tokens, {len(context.entities)} entities")
                
                # Step 10: Generate answer
                print("⏳ Generating answer with AI...")
                if use_citations:
                    answer = answer_generator.generate_with_citations(query, context)
                else:
                    answer = answer_generator.generate_answer(query, context, include_sources=True)
                
                # Display answer
                print("\n" + "="*60)
                print("🤖 AI ANSWER:")
                print("="*60)
                print(answer.answer)
                
                print(f"\n📊 Answer Metadata:")
                print(f"  Confidence: {answer.confidence:.1%}")
                print(f"  Entities used: {context.metadata.get('entity_count', 0)}")
                print(f"  Relationships: {context.metadata.get('relationship_count', 0)}")
                print(f"  Context size: {context.token_count} tokens")
                
                if answer.sources and not use_citations:
                    print(f"\n📚 Sources ({len(answer.sources)}):")
                    for j, source in enumerate(answer.sources[:5], 1):
                        print(f"  {j}. {source}")
                    if len(answer.sources) > 5:
                        print(f"  ... and {len(answer.sources) - 5} more")
                
            except KeyboardInterrupt:
                print("\n\n👋 Exiting interactive QA mode...")
                break
            except Exception as e:
                print(f"\n❌ Error processing question: {e}")
                logger.error(f"QA error: {e}", exc_info=True)
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*80)
        
        return {
            'query_processor': query_processor,
            'retrieval_engine': retrieval_engine,
            'context_assembler': context_assembler,
            'answer_generator': answer_generator
        }
        
    except Exception as e:
        print(f"\n✗ Error in QA System: {e}")
        logger.error(f"QA System failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main execution function for Graph RAG pipeline."""

    print("=" * 80)
    print("Graph RAG System - Full Pipeline")
    print("=" * 80)

    # Load configuration
    config = load_config()
    print(f"\nConfiguration loaded:")
    print(f"  Neo4j URI: {config.get('neo4j', {}).get('uri', 'N/A')}")
    print(f"  Embeddings Backend: {config.get('embeddings', {}).get('backend', 'N/A')}")
    print(f"  Redis Enabled: {config.get('redis', {}).get('enabled', False)}")

    # Step 1: Data Ingestion
    print("\n[Step 1] Initializing Data Ingestion...")
    data_ingestion = create_data_ingestion()

    # Load the entity relations graph CSV
    csv_file_path = INPUT_FILE_PATH
    print(f"\n[Step 1] Loading data from: {csv_file_path}")

    try:
        # Load the document
        document = data_ingestion.load_document(csv_file_path)

        # Print the first row of the document for verification
        # print(document['content'].splitlines()[0])

        # Step 2: Text Chunking & Segmentation (row-based)
        print("\n[Step 2] Creating CSV row chunking strategy...")
        chunker = create_chunking_strategy(method="csv_row", row_format="key_value")

        print("[Step 2] Chunking rows into row-based chunks...")
        chunks = chunker.chunk(document.get('content', ''), metadata=document.get('metadata', {}))
        print(f"[Step 2] Created {len(chunks)} chunks")

        # Step 3: Entity Extraction
        print("\n[Step 3] Initializing Entity Extraction...")
        print("[Step 3] Using CSV entity extractor with pattern-based evidence extraction...")
        csv_extractor = create_entity_extractor(
            method="csv",
            use_llm_for_evidence=False
        )
        
        # Option 2: CSV + Gemini LLM for evidence notes (more accurate but slower)
        # Uncomment below to use Gemini for evidence_note extraction
        # print("[Step 3] Using CSV entity extractor with Gemini LLM for evidence notes...")
        # gemini_extractor = create_entity_extractor(
        #     method="gemini",
        #     api_key=None,  # Will use GEMINI_API_KEY from environment
        #     model_name="gemini-1.5-flash",
        #     temperature=0.1
        # )
        # csv_extractor = create_entity_extractor(
        #     method="csv",
        #     use_llm_for_evidence=True,
        #     llm_extractor=gemini_extractor
        # )
        
        # Extract entities from all chunks
        print("[Step 3] Extracting entities from chunks...")
        all_entities = []
        
        for i, chunk in enumerate(chunks):
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(chunks)} chunks...")
            
            entities = csv_extractor.extract_entities(
                text=chunk['text'],
                chunk_metadata=chunk.get('metadata', {})
            )
            all_entities.extend(entities)
        
        print(f"[Step 3] Extracted {len(all_entities)} raw entities")
        
        # Deduplicate entities
        print("[Step 3] Deduplicating entities...")
        deduplicator = EntityDeduplicator(similarity_threshold=0.85)
        unique_entities = deduplicator.deduplicate(all_entities)
        print(f"[Step 3] After deduplication: {len(unique_entities)} unique entities")
        
        # Save entities
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        entities_file = OUTPUT_FILE_PATH_FOR_ENTITY_EXTRACTION
        with open(entities_file, 'w', encoding='utf-8') as f:
            json.dump(unique_entities, f, indent=2, ensure_ascii=False)
        print(f"\n[Step 3] Entities saved to: {entities_file}")

        paths = save_entity_summary(
            entities=unique_entities,
            raw_entities=all_entities,
            output_dir=str(output_dir),
            entities_file=str(entities_file)
        )

        # Step 4: Relationship Extraction
        print("\n[Step 4] Initializing Relationship Extraction...")
        print("[Step 4] Using CSV relationship extractor...")
        relationship_extractor = create_relationship_extractor(method="csv")
        
        print("[Step 4] Extracting relationships from chunks...")
        all_relationships = []
        
        for i, chunk in enumerate(chunks):
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(chunks)} chunks...")
            
            relationships = relationship_extractor.extract_relationships(
                text=chunk['text'],
                entities=unique_entities,
                chunk_metadata=chunk.get('metadata', {})
            )
            all_relationships.extend(relationships)
        
        print(f"[Step 4] Extracted {len(all_relationships)} raw relationships")
        
        # Deduplicate relationships
        print("[Step 4] Deduplicating relationships...")
        relationship_deduplicator = RelationshipDeduplicator()
        unique_relationships = relationship_deduplicator.deduplicate(all_relationships)
        print(f"[Step 4] After deduplication: {len(unique_relationships)} unique relationships")
        
        # Save relationships
        relationships_file = OUTPUT_FILE_PATH_FOR_RELATIONSHIP_EXTRACTION
        with open(relationships_file, 'w', encoding='utf-8') as f:
            json.dump(unique_relationships, f, indent=2, ensure_ascii=False)
        print(f"\n[Step 4] Relationships saved to: {relationships_file}")

        # Step 5: Knowledge Graph Construction
        print("\n[Step 5] Initializing Knowledge Graph Construction...")
        
        # Build in-memory graph first
        graph_builder = GraphBuilder()
        
        print("[Step 5] Building graph from entities and relationships...")
        graph_stats = graph_builder.build_from_data(
            chunks=chunks,
            entities=unique_entities,
            relationships=unique_relationships
        )
        
        print(f"\n[Step 5] Graph Statistics:")
        print(f"  Total Nodes: {graph_stats['total_nodes']}")
        print(f"  Total Edges: {graph_stats['total_edges']}")
        
        # Save graph statistics
        graph_stats_file = OUTPUT_FILE_PATH_FOR_GRAPH_STATISTICS
        with open(graph_stats_file, 'w', encoding='utf-8') as f:
            json.dump(graph_stats, f, indent=2)
        print(f"\n[Step 5] Graph statistics saved to: {graph_stats_file}")
        
        # Persist to Neo4j
        print("\n[Step 5] Connecting to Neo4j...")
        neo4j_connection_params = None
        hybrid_store = None
        
        try:
            neo4j_store = create_neo4j_store()
            
            # Store connection params for Step 6
            neo4j_config = config.get('neo4j', {})
            neo4j_connection_params = {
                'uri': neo4j_config.get('uri', 'bolt://localhost:7687'),
                'user': neo4j_config.get('user', 'neo4j'),
                'password': neo4j_config.get('password', 'password')
            }
            
            # Optional: Clear existing data
            clear_db = input("\nClear existing Neo4j data? (yes/no): ").strip().lower()
            if clear_db == 'yes':
                print("[Step 5] Clearing Neo4j database...")
                neo4j_store.clear_database()
            
            print("[Step 5] Creating indexes...")
            neo4j_store.create_indexes()
            
            print("[Step 5] Persisting graph to Neo4j...")
            persist_stats = graph_builder.persist_to_neo4j(neo4j_store)
            
            print(f"\n[Step 5] Persistence Statistics:")
            print(f"  Nodes Created: {persist_stats['nodes_created']}")
            print(f"  Edges Created: {persist_stats['edges_created']}")
            print(f"  Errors: {persist_stats['errors']}")
            
            # Get Neo4j statistics
            print("\n[Step 5] Verifying Neo4j database...")
            db_stats = neo4j_store.get_statistics()
            print(f"  Total Nodes in DB: {db_stats['total_nodes']}")
            print(f"  Total Relationships in DB: {db_stats['total_relationships']}")
            
            neo4j_store.close()
            print("\n[Step 5] Knowledge graph successfully stored in Neo4j!")
            
            # Step 6: Graph Indexing & Storage (only if Neo4j succeeded)
            print("\n[Step 6] Starting graph indexing...")
            if config.get('steps', {}).get('graph_indexing', True):

                # Initialize embeddings client
                embeddings_config = config.get('embeddings', {})
                embeddings_client = create_embeddings_client(
                    backend=embeddings_config.get('backend', 'hf'),
                    model_name=embeddings_config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
                )

                # Create hybrid store
                redis_config = config.get('redis', {})
                use_redis = redis_config.get('enabled', False)
                redis_url = redis_config.get('url') if use_redis else None

                hybrid_store = HybridGraphStore(
                    neo4j_store=neo4j_store,
                    embeddings_client=embeddings_client,
                    use_redis=use_redis,
                    redis_url=redis_url,
                    vector_store_path=VECTOR_STORE_PATH
                )
                hybrid_store_for_indexing = graph_indexing(config, neo4j_connection_params, embeddings_client, hybrid_store)

                # # Steps 7 & 8: Query Processing and Retrieval
                # if hybrid_store and config.get('steps', {}).get('query_retrieval', True):
                #     retrieval_engine = query_processing_and_retrieval(config, hybrid_store, embeddings_client, testing=True)
                    
                #     # Steps 9 & 10: Context Assembly and Answer Generation
                #     if retrieval_engine and config.get('steps', {}).get('qa_generation', True):
                #         qa_system = context_assembly_and_answer_generation(
                #             config, retrieval_engine, embeddings_client, testing=True
                #         )

                # Steps 7-10: Unified Interactive QA System
                if hybrid_store and config.get('steps', {}).get('qa_system', True):
                    qa_system = interactive_qa_system(
                        config, hybrid_store, embeddings_client, testing=TESTING_MODE
                    )
            
        except Exception as e:
            logger.error(f"Neo4j operation failed: {e}")
            print(f"\n[Step 5] Warning: Could not persist to Neo4j: {e}")
            print("[Step 5] Graph built in-memory successfully, but Neo4j persistence failed.")
            print("[Step 5] Please check Neo4j connection settings in config.py")
            print("\n[Step 6] Skipping graph indexing due to Neo4j connection failure")

        print("\n" + "=" * 80)
        print("Pipeline Execution Complete!")
        print("=" * 80)

        return {
            'document': document,
            'chunks': chunks,
            'entities': unique_entities,
            'relationships': unique_relationships,
            'graph_stats': graph_stats,
            'hybrid_store': hybrid_store_for_indexing,
            'qa_system': qa_system if 'qa_system' in locals() else None
        }
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
