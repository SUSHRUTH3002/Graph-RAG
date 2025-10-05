"""Query processing module for parsing and understanding user queries."""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Types of query intents."""
    ENTITY_LOOKUP = "entity_lookup"  # "What is X?"
    RELATIONSHIP = "relationship"  # "How is X related to Y?"
    AGGREGATION = "aggregation"  # "Total revenue", "Count of..."
    TEMPORAL = "temporal"  # "What happened in 2023?"
    COMPARISON = "comparison"  # "Compare X and Y"
    EXPLORATION = "exploration"  # General exploration
    FACTUAL = "factual"  # Direct fact retrieval


@dataclass
class ParsedQuery:
    """Container for parsed query information."""
    raw_query: str
    intent: QueryIntent
    entities: List[str]
    keywords: List[str]
    temporal_markers: List[str]
    filters: Dict[str, Any]
    embedding: Optional[List[float]] = None


class QueryProcessor:
    """Process and parse user queries for optimal retrieval."""
    
    def __init__(self, embeddings_client, neo4j_store=None):
        """Initialize query processor.
        
        Args:
            embeddings_client: Client for generating query embeddings
            neo4j_store: Optional Neo4j store for entity validation
        """
        self.embeddings_client = embeddings_client
        self.neo4j_store = neo4j_store
        
        # Intent patterns
        self.intent_patterns = {
            QueryIntent.ENTITY_LOOKUP: [
                r'\bwhat is\b', r'\bwho is\b', r'\bdefine\b', r'\bexplain\b'
            ],
            QueryIntent.RELATIONSHIP: [
                r'\bhow.*related\b', r'\brelationship between\b', 
                r'\bconnection between\b', r'\blink between\b'
            ],
            QueryIntent.AGGREGATION: [
                r'\btotal\b', r'\bsum\b', r'\bcount\b', r'\baverage\b',
                r'\bhow many\b', r'\bhow much\b'
            ],
            QueryIntent.TEMPORAL: [
                r'\bin \d{4}\b', r'\bduring\b', r'\bwhen\b', r'\btimeline\b'
            ],
            QueryIntent.COMPARISON: [
                r'\bcompare\b', r'\bversus\b', r'\bvs\b', r'\bdifference between\b'
            ]
        }
        
        # Common entity types and indicators
        self.entity_indicators = {
            'ORGANIZATION': ['company', 'corporation', 'firm', 'business'],
            'PRODUCT': ['product', 'brand', 'line'],
            'METRIC': ['revenue', 'profit', 'sales', 'growth', 'performance'],
            'YEAR': [r'\b\d{4}\b'],
        }
    
    def process_query(self, query: str) -> ParsedQuery:
        """Process a user query.
        
        Args:
            query: Raw user query string
            
        Returns:
            ParsedQuery object with extracted information
        """
        logger.info(f"Processing query: {query}")
        
        # Extract intent
        intent = self._extract_intent(query)
        
        # Extract entities and keywords
        entities = self._extract_entities(query)
        keywords = self._extract_keywords(query)
        
        # Extract temporal markers
        temporal_markers = self._extract_temporal_markers(query)
        
        # Build filters
        filters = self._build_filters(query, entities, temporal_markers)
        
        # Generate embedding
        embedding = self.embeddings_client.embed_query(query)
        
        parsed = ParsedQuery(
            raw_query=query,
            intent=intent,
            entities=entities,
            keywords=keywords,
            temporal_markers=temporal_markers,
            filters=filters,
            embedding=embedding
        )
        
        logger.info(f"Query processed - Intent: {intent.value}, Entities: {len(entities)}")
        return parsed
    
    def _extract_intent(self, query: str) -> QueryIntent:
        """Extract query intent from patterns."""
        query_lower = query.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        # Default to exploration
        return QueryIntent.EXPLORATION
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract potential entity names from query."""
        entities = []
        
        # Extract capitalized phrases (likely proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        entities.extend(capitalized)
        
        # Extract quoted strings
        quoted = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted)
        
        # Extract years
        years = re.findall(r'\b(20\d{2})\b', query)
        entities.extend(years)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        # Remove common stop words
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or',
            'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'from'
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def _extract_temporal_markers(self, query: str) -> List[str]:
        """Extract temporal information (years, dates)."""
        temporal = []
        
        # Extract years
        years = re.findall(r'\b(20\d{2})\b', query)
        temporal.extend(years)
        
        # Extract relative time expressions
        relative_time = re.findall(
            r'\b(last year|this year|previous year|recent|latest)\b',
            query.lower()
        )
        temporal.extend(relative_time)
        
        return temporal
    
    def _build_filters(
        self,
        query: str,
        entities: List[str],
        temporal_markers: List[str]
    ) -> Dict[str, Any]:
        """Build filters for retrieval."""
        filters = {}
        
        # Add entity type filters
        query_lower = query.lower()
        for entity_type, indicators in self.entity_indicators.items():
            for indicator in indicators:
                if isinstance(indicator, str):
                    if indicator in query_lower:
                        filters['node_type'] = entity_type
                        break
                else:  # regex pattern
                    if re.search(indicator, query):
                        filters['node_type'] = entity_type
                        break
        
        # Add temporal filters
        if temporal_markers:
            filters['temporal'] = temporal_markers
        
        return filters
    
    def validate_entities(self, entities: List[str]) -> List[Tuple[str, bool]]:
        """Validate if entities exist in the knowledge graph.
        
        Args:
            entities: List of entity names
            
        Returns:
            List of (entity_name, exists) tuples
        """
        if not self.neo4j_store:
            return [(e, True) for e in entities]  # Assume valid
        
        validated = []
        for entity in entities:
            try:
                result = self.neo4j_store.get_entity(entity)
                validated.append((entity, result is not None))
            except Exception as e:
                logger.error(f"Error validating entity {entity}: {e}")
                validated.append((entity, False))
        
        return validated


def create_query_processor(embeddings_client, neo4j_store=None) -> QueryProcessor:
    """Factory function to create query processor.
    
    Args:
        embeddings_client: Embeddings client instance
        neo4j_store: Optional Neo4j store for entity validation
        
    Returns:
        QueryProcessor instance
    """
    return QueryProcessor(embeddings_client, neo4j_store)
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query.
        
        Args:
            query: Normalized query
            
        Returns:
            List of keywords
        """
        # Remove common stop words
        stop_words = {
            'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can',
            'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'about', 'as', 'into', 'through'
        }
        
        words = query.split()
        keywords = [
            word for word in words 
            if len(word) > 2 and word not in stop_words
        ]
        
        return keywords
    
    def _extract_temporal_refs(self, query: str) -> List[str]:
        """Extract temporal references from query.
        
        Args:
            query: Query string
            
        Returns:
            List of temporal references
        """
        temporal_refs = []
        
        for pattern in self.temporal_patterns:
            matches = re.findall(pattern, query.lower())
            temporal_refs.extend(matches)
        
        return temporal_refs
    
    def _determine_filters(
        self,
        query: str,
        entities: List[QueryEntity],
        temporal_refs: List[str]
    ) -> Dict[str, Any]:
        """Determine filters to apply to retrieval.
        
        Args:
            query: Query string
            entities: Extracted entities
            temporal_refs: Temporal references
            
        Returns:
            Dictionary of filters
        """
        filters = {}
        
        # Entity type filters
        if entities:
            entity_types = [e.entity_type for e in entities if e.entity_type]
            if entity_types:
                filters['entity_types'] = entity_types
        
        # Temporal filters
        if temporal_refs:
            # Extract years
            years = [int(ref) for ref in temporal_refs if ref.isdigit() and len(ref) == 4]
            if years:
                filters['years'] = years
        
        # Source filters (if mentioned in query)
        if 'report' in query.lower():
            filters['source_type'] = 'report'
        elif 'document' in query.lower():
            filters['source_type'] = 'document'
        elif 'csv' in query.lower():
            filters['source_type'] = 'csv'
        
        return filters
    
    def _requires_graph_traversal(
        self,
        intent: QueryIntent,
        entities: List[QueryEntity]
    ) -> bool:
        """Determine if query requires graph traversal.
        
        Args:
            intent: Query intent
            entities: Extracted entities
            
        Returns:
            True if graph traversal is needed
        """
        # Graph traversal useful for:
        # - Relationship queries
        # - Navigational queries with known entities
        # - Comparative queries
        
        if intent in [QueryIntent.RELATIONSHIP, QueryIntent.COMPARATIVE]:
            return True
        
        if intent == QueryIntent.NAVIGATIONAL and len(entities) > 0:
            return True
        
        return False
    
    def _requires_vector_search(self, intent: QueryIntent) -> bool:
        """Determine if query requires vector search.
        
        Args:
            intent: Query intent
            
        Returns:
            True if vector search is needed
        """
        # Vector search useful for most query types
        # Less useful for pure relationship queries
        
        if intent == QueryIntent.RELATIONSHIP:
            return False  # Pure graph traversal
        
        return True
    
    def extract_entities_for_graph_search(
        self,
        query: str
    ) -> List[str]:
        """Extract entity names for graph search.
        
        Args:
            query: Query string
            
        Returns:
            List of entity names
        """
        context = self.process_query(query)
        return [entity.text for entity in context.entities]
    
    def should_use_hybrid_search(self, query: str) -> bool:
        """Determine if hybrid search should be used.
        
        Args:
            query: Query string
            
        Returns:
            True if hybrid search is recommended
        """
        context = self.process_query(query)
        return context.requires_graph and context.requires_vector
