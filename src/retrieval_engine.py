"""Retrieval engine for multi-strategy knowledge graph querying."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from src.neo4j_store import create_neo4j_store
logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Retrieval strategy types."""
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


@dataclass
class RetrievalResult:
    """Container for retrieval results."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]
    scores: Dict[str, float]
    strategy_used: RetrievalStrategy
    metadata: Dict[str, Any]


class RetrievalEngine:
    """Multi-strategy retrieval engine for knowledge graphs."""
    
    def __init__(self, hybrid_store, query_processor):
        """Initialize retrieval engine.
        
        Args:
            hybrid_store: HybridGraphStore instance
            query_processor: QueryProcessor instance
        """
        self.hybrid_store = hybrid_store
        self.query_processor = query_processor
        self.neo4j_store = create_neo4j_store()
    
    def retrieve(
        self,
        query: str,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Retrieve relevant information for a query.
        
        Args:
            query: User query string
            strategy: Retrieval strategy to use
            k: Number of results to retrieve
            filters: Optional filters for retrieval
            
        Returns:
            RetrievalResult object
        """
        logger.info(f"Retrieving with strategy: {strategy.value}")
        
        # Process query
        parsed_query = self.query_processor.process_query(query)
        
        # Merge filters
        combined_filters = {**parsed_query.filters}
        if filters:
            combined_filters.update(filters)
        
        # Route to appropriate strategy
        if strategy == RetrievalStrategy.VECTOR_ONLY:
            return self._vector_retrieval(parsed_query, k, combined_filters)
        elif strategy == RetrievalStrategy.GRAPH_ONLY:
            return self._graph_retrieval(parsed_query, k, combined_filters)
        elif strategy == RetrievalStrategy.HYBRID:
            return self._hybrid_retrieval(parsed_query, k, combined_filters)
        else:  # ADAPTIVE
            return self._adaptive_retrieval(parsed_query, k, combined_filters)
    
    def _vector_retrieval(
        self,
        parsed_query,
        k: int,
        filters: Dict[str, Any]
    ) -> RetrievalResult:
        """Perform vector-based semantic search."""
        logger.info("Performing vector retrieval")
        
        # Search nodes
        node_filter = filters.get('node_type')
        node_results = self.hybrid_store.search_similar_nodes(
            parsed_query.raw_query,
            k=k,
            node_type_filter=node_filter
        )
        
        # Search edges
        edge_results = self.hybrid_store.search_similar_edges(
            parsed_query.raw_query,
            k=k
        )
        
        nodes = [
            {
                'id': node_id,
                'score': float(score),
                'metadata': metadata,
                'retrieval_method': 'vector_search'
            }
            for node_id, score, metadata in node_results
        ]
        
        edges = [
            {
                'id': edge_id,
                'score': float(score),
                'metadata': metadata,
                'retrieval_method': 'vector_search'
            }
            for edge_id, score, metadata in edge_results
        ]
        
        return RetrievalResult(
            nodes=nodes,
            edges=edges,
            chunks=[],
            scores={'avg_node_score': sum(n['score'] for n in nodes) / len(nodes) if nodes else 0},
            strategy_used=RetrievalStrategy.VECTOR_ONLY,
            metadata={'query_intent': parsed_query.intent.value}
        )
    
    def _graph_retrieval(
        self,
        parsed_query,
        k: int,
        filters: Dict[str, Any]
    ) -> RetrievalResult:
        """Perform graph traversal-based retrieval."""
        logger.info("Performing graph traversal retrieval")
        
        nodes = []
        edges = []
        
        # Start from extracted entities
        for entity in parsed_query.entities[:3]:  # Limit to top 3 entities
            try:
                # Get entity and its neighbors
                entity_data = self.neo4j_store.get_entity(entity)
                if entity_data:
                    nodes.append({
                        'id': entity,
                        'score': 1.0,
                        'metadata': entity_data,
                        'retrieval_method': 'graph_traversal'
                    })
                    
                    # Get related entities
                    related = self.neo4j_store.get_related_entities(
                        entity,
                        max_depth=2,
                        limit=k
                    )
                    
                    for node in related.get('nodes', []):
                        nodes.append({
                            'id': node.get('name'),
                            'score': 0.8,
                            'metadata': node,
                            'retrieval_method': 'graph_traversal'
                        })
                    
                    edges.extend([
                        {
                            'id': f"{e['source']}_{e['type']}_{e['target']}",
                            'score': 0.8,
                            'metadata': e,
                            'retrieval_method': 'graph_traversal'
                        }
                        for e in related.get('edges', [])
                    ])
                    
            except Exception as e:
                logger.error(f"Error retrieving entity {entity}: {e}")
        
        # Deduplicate nodes
        unique_nodes = {n['id']: n for n in nodes}
        nodes = list(unique_nodes.values())[:k]
        
        return RetrievalResult(
            nodes=nodes,
            edges=edges[:k],
            chunks=[],
            scores={'entities_found': len(nodes)},
            strategy_used=RetrievalStrategy.GRAPH_ONLY,
            metadata={'query_intent': parsed_query.intent.value}
        )
    
    def _hybrid_retrieval(
        self,
        parsed_query,
        k: int,
        filters: Dict[str, Any]
    ) -> RetrievalResult:
        """Perform hybrid retrieval combining vector and graph."""
        logger.info("Performing hybrid retrieval")
        
        # Use hybrid search from store
        results = self.hybrid_store.hybrid_search(
            query=parsed_query.raw_query,
            k=k,
            search_nodes=True,
            search_edges=True,
            expand_graph=True,
            max_depth=2
        )
        
        # Rerank and score
        ranked_nodes = self._rerank_results(
            results['nodes'],
            parsed_query,
            filters
        )
        
        ranked_edges = self._rerank_results(
            results['edges'],
            parsed_query,
            filters
        )
        
        # Calculate scores
        avg_score = sum(n['final_score'] for n in ranked_nodes) / len(ranked_nodes) if ranked_nodes else 0
        
        return RetrievalResult(
            nodes=ranked_nodes[:k],
            edges=ranked_edges[:k],
            chunks=[],
            scores={
                'avg_node_score': avg_score,
                'subgraph_nodes': results.get('expanded_subgraph', {}).get('node_count', 0),
                'subgraph_edges': results.get('expanded_subgraph', {}).get('edge_count', 0)
            },
            strategy_used=RetrievalStrategy.HYBRID,
            metadata={
                'query_intent': parsed_query.intent.value,
                'expanded_subgraph': results.get('expanded_subgraph')
            }
        )
    
    def _adaptive_retrieval(
        self,
        parsed_query,
        k: int,
        filters: Dict[str, Any]
    ) -> RetrievalResult:
        """Adaptively choose retrieval strategy based on query."""
        from .query_processor import QueryIntent
        
        # Route based on intent
        if parsed_query.intent == QueryIntent.ENTITY_LOOKUP:
            strategy = RetrievalStrategy.GRAPH_ONLY
        elif parsed_query.intent == QueryIntent.RELATIONSHIP:
            strategy = RetrievalStrategy.GRAPH_ONLY
        elif parsed_query.intent in [QueryIntent.EXPLORATION, QueryIntent.FACTUAL]:
            strategy = RetrievalStrategy.HYBRID
        else:
            strategy = RetrievalStrategy.VECTOR_ONLY
        
        logger.info(f"Adaptive routing: {parsed_query.intent.value} -> {strategy.value}")
        
        # Use selected strategy
        if strategy == RetrievalStrategy.VECTOR_ONLY:
            return self._vector_retrieval(parsed_query, k, filters)
        elif strategy == RetrievalStrategy.GRAPH_ONLY:
            return self._graph_retrieval(parsed_query, k, filters)
        else:
            return self._hybrid_retrieval(parsed_query, k, filters)
    
    def _rerank_results(
        self,
        results: List[Dict[str, Any]],
        parsed_query,
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Rerank results based on multiple signals."""
        reranked = []
        
        for result in results:
            metadata = result.get('metadata', {})
            base_score = result.get('similarity_score', 0.5)
            
            # Boost score based on filters
            boost = 1.0
            
            # Boost if node type matches filter
            if filters.get('node_type'):
                if metadata.get('node_type') == filters['node_type']:
                    boost *= 1.5
            
            # Boost if entity name in query entities
            entity_name = metadata.get('entity_name') or metadata.get('name')
            if entity_name in parsed_query.entities:
                boost *= 2.0
            
            # Boost if temporal match
            if filters.get('temporal'):
                node_year = str(metadata.get('year', ''))
                if any(t in node_year for t in filters['temporal']):
                    boost *= 1.3
            
            final_score = min(base_score * boost, 1.0)
            
            result['final_score'] = final_score
            result['boost_applied'] = boost
            reranked.append(result)
        
        # Sort by final score
        reranked.sort(key=lambda x: x['final_score'], reverse=True)
        
        return reranked
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        return {
            'hybrid_store_indexed': self.hybrid_store.is_indexed,
            'available_strategies': [s.value for s in RetrievalStrategy]
        }


def create_retrieval_engine(hybrid_store, query_processor) -> RetrievalEngine:
    """Factory function to create retrieval engine.
    
    Args:
        hybrid_store: HybridGraphStore instance
        query_processor: QueryProcessor instance
        
    Returns:
        RetrievalEngine instance
    """
    return RetrievalEngine(hybrid_store, query_processor)
