"""Context assembly module for preparing LLM prompts from retrieval results."""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AssembledContext:
    """Container for assembled context."""
    context_text: str
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    sources: List[str]
    metadata: Dict[str, Any]
    token_count: int


class ContextAssembler:
    """Assemble context from retrieval results for LLM prompts."""
    
    def __init__(
        self,
        max_tokens: int = 4000,
        include_sources: bool = True,
        deduplicate: bool = True
    ):
        """Initialize context assembler.
        
        Args:
            max_tokens: Maximum token limit for context
            include_sources: Include source references
            deduplicate: Remove duplicate information
        """
        self.max_tokens = max_tokens
        self.include_sources = include_sources
        self.deduplicate = deduplicate
    
    def assemble_from_retrieval(
        self,
        retrieval_result,
        query: str,
        strategy: str
    ) -> AssembledContext:
        """Assemble context from retrieval results.
        
        Args:
            retrieval_result: RetrievalResult object
            query: Original query
            strategy: Retrieval strategy used
            
        Returns:
            AssembledContext object
        """
        logger.info(f"Assembling context for strategy: {strategy}")
        
        # Route to appropriate assembly method
        if strategy == "vector_only":
            return self._assemble_vector_context(retrieval_result, query)
        elif strategy == "graph_only":
            return self._assemble_graph_context(retrieval_result, query)
        elif strategy in ["hybrid", "adaptive"]:
            return self._assemble_hybrid_context(retrieval_result, query)
        else:
            return self._assemble_default_context(retrieval_result, query)
    
    def _assemble_vector_context(
        self,
        retrieval_result,
        query: str
    ) -> AssembledContext:
        """Assemble context from vector-only retrieval."""
        nodes = retrieval_result.nodes
        edges = retrieval_result.edges
        
        # Extract entities
        entities = self._extract_entities_from_nodes(nodes)
        
        # Extract relationships
        relationships = self._extract_relationships_from_edges(edges)
        
        # Build context text
        context_parts = [
            f"Query: {query}\n",
            "\n=== RELEVANT ENTITIES ===\n"
        ]
        
        # Add entity information
        seen_entities = set()
        for i, entity in enumerate(entities[:15], 1):  # Limit to top 15
            entity_name = entity.get('name', 'Unknown')
            if entity_name in seen_entities and self.deduplicate:
                continue
            seen_entities.add(entity_name)
            
            entity_type = entity.get('type', 'ENTITY')
            score = entity.get('score', 0)
            properties = entity.get('properties', {})
            
            context_parts.append(f"\n{i}. {entity_name} ({entity_type}) [Relevance: {score:.3f}]")
            
            # Add key properties
            if 'section' in properties:
                context_parts.append(f"   Section: {properties['section']}")
            if 'year' in properties:
                context_parts.append(f"   Year: {properties['year']}")
            if 'value' in properties and 'unit' in properties:
                context_parts.append(f"   Value: {properties['value']} {properties['unit']}")
            if 'evidence' in properties:
                evidence = str(properties['evidence'])[:150]
                context_parts.append(f"   Evidence: {evidence}")
        
        # Add relationship information if available
        if relationships:
            context_parts.append("\n\n=== RELEVANT RELATIONSHIPS ===\n")
            seen_rels = set()
            for i, rel in enumerate(relationships[:10], 1):  # Limit to top 10
                rel_key = (rel.get('source'), rel.get('type'), rel.get('target'))
                if rel_key in seen_rels and self.deduplicate:
                    continue
                seen_rels.add(rel_key)
                
                source = rel.get('source', 'Unknown')
                target = rel.get('target', 'Unknown')
                rel_type = rel.get('type', 'RELATED_TO')
                score = rel.get('score', 0)
                
                context_parts.append(
                    f"\n{i}. {source} --[{rel_type}]--> {target} [Relevance: {score:.3f}]"
                )
                
                # Add relationship properties
                props = rel.get('properties', {})
                if 'evidence' in props:
                    evidence = str(props['evidence'])[:100]
                    context_parts.append(f"   Context: {evidence}")
        
        context_text = "".join(context_parts)
        
        # Extract sources
        sources = self._extract_sources(entities, relationships)
        
        return AssembledContext(
            context_text=context_text,
            entities=entities,
            relationships=relationships,
            sources=sources,
            metadata={
                'strategy': 'vector_only',
                'entity_count': len(entities),
                'relationship_count': len(relationships)
            },
            token_count=self._estimate_tokens(context_text)
        )
    
    def _assemble_graph_context(
        self,
        retrieval_result,
        query: str
    ) -> AssembledContext:
        """Assemble context from graph-only retrieval."""
        nodes = retrieval_result.nodes
        edges = retrieval_result.edges
        
        # Extract entities and relationships
        entities = self._extract_entities_from_nodes(nodes)
        relationships = self._extract_relationships_from_edges(edges)
        
        # Build graph-centric context
        context_parts = [
            f"Query: {query}\n",
            "\n=== KNOWLEDGE GRAPH CONTEXT ===\n"
        ]
        
        # Group by entity and show their connections
        entity_map = {e.get('name'): e for e in entities}
        
        for i, entity in enumerate(entities[:10], 1):
            entity_name = entity.get('name', 'Unknown')
            entity_type = entity.get('type', 'ENTITY')
            
            context_parts.append(f"\n{i}. {entity_name} ({entity_type})")
            
            # Find relationships involving this entity
            related_rels = [
                r for r in relationships
                if r.get('source') == entity_name or r.get('target') == entity_name
            ]
            
            if related_rels:
                context_parts.append("   Connections:")
                for rel in related_rels[:5]:  # Limit connections
                    source = rel.get('source')
                    target = rel.get('target')
                    rel_type = rel.get('type', 'RELATED_TO')
                    
                    if source == entity_name:
                        context_parts.append(f"   - {rel_type} → {target}")
                    else:
                        context_parts.append(f"   - {source} → {rel_type}")
                    
                    # Add evidence if available
                    props = rel.get('properties', {})
                    if 'evidence' in props:
                        evidence = str(props['evidence'])[:80]
                        context_parts.append(f"     ({evidence})")
            
            # Add entity properties
            props = entity.get('properties', {})
            if 'section' in props:
                context_parts.append(f"   From: {props['section']}")
            if 'year' in props:
                context_parts.append(f"   Year: {props['year']}")
        
        context_text = "".join(context_parts)
        sources = self._extract_sources(entities, relationships)
        
        return AssembledContext(
            context_text=context_text,
            entities=entities,
            relationships=relationships,
            sources=sources,
            metadata={
                'strategy': 'graph_only',
                'entity_count': len(entities),
                'relationship_count': len(relationships)
            },
            token_count=self._estimate_tokens(context_text)
        )
    
    def _assemble_hybrid_context(
        self,
        retrieval_result,
        query: str
    ) -> AssembledContext:
        """Assemble context from hybrid retrieval."""
        nodes = retrieval_result.nodes
        edges = retrieval_result.edges
        
        # Extract entities and relationships
        entities = self._extract_entities_from_nodes(nodes)
        relationships = self._extract_relationships_from_edges(edges)
        
        # Get subgraph information
        subgraph = retrieval_result.metadata.get('expanded_subgraph', {})
        
        # Build comprehensive context
        context_parts = [
            f"Query: {query}\n",
            "\n=== HYBRID CONTEXT (Vector + Graph) ===\n"
        ]
        
        # Section 1: Top semantically similar entities
        context_parts.append("\n--- Most Relevant Entities ---\n")
        seen_entities = set()
        
        for i, entity in enumerate(entities[:8], 1):
            entity_name = entity.get('name', 'Unknown')
            if entity_name in seen_entities and self.deduplicate:
                continue
            seen_entities.add(entity_name)
            
            entity_type = entity.get('type', 'ENTITY')
            score = entity.get('score', 0)
            boost = entity.get('boost', 1.0)
            
            context_parts.append(
                f"\n{i}. {entity_name} ({entity_type}) "
                f"[Relevance: {score:.3f}, Boost: {boost:.2f}]"
            )
            
            # Add properties
            props = entity.get('properties', {})
            if 'section' in props:
                context_parts.append(f"   Section: {props['section']}")
            if 'year' in props:
                context_parts.append(f"   Year: {props['year']}")
            if 'evidence' in props:
                evidence = str(props['evidence'])[:120]
                context_parts.append(f"   Evidence: {evidence}")
        
        # Section 2: Key relationships
        if relationships:
            context_parts.append("\n\n--- Key Relationships ---\n")
            seen_rels = set()
            
            for i, rel in enumerate(relationships[:8], 1):
                rel_key = (rel.get('source'), rel.get('type'), rel.get('target'))
                if rel_key in seen_rels and self.deduplicate:
                    continue
                seen_rels.add(rel_key)
                
                source = rel.get('source', 'Unknown')
                target = rel.get('target', 'Unknown')
                rel_type = rel.get('type', 'RELATED_TO')
                score = rel.get('score', 0)
                
                context_parts.append(
                    f"\n{i}. {source} --[{rel_type}]--> {target} "
                    f"[Relevance: {score:.3f}]"
                )
                
                props = rel.get('properties', {})
                if 'evidence' in props:
                    evidence = str(props['evidence'])[:100]
                    context_parts.append(f"   Context: {evidence}")
                if 'section' in props:
                    context_parts.append(f"   From: {props['section']}")
        
        # Section 3: Graph neighborhood summary
        if subgraph and subgraph.get('node_count', 0) > 0:
            context_parts.append("\n\n--- Connected Graph Neighborhood ---")
            context_parts.append(
                f"\n  Total connected entities: {subgraph.get('node_count', 0)}"
            )
            context_parts.append(
                f"  Total relationships: {subgraph.get('edge_count', 0)}"
            )
            
            # Add sample of connected nodes
            connected_nodes = subgraph.get('nodes', [])[:10]
            if connected_nodes:
                context_parts.append("\n  Connected entities: " + ", ".join(connected_nodes))
        
        context_text = "".join(context_parts)
        sources = self._extract_sources(entities, relationships)
        
        return AssembledContext(
            context_text=context_text,
            entities=entities,
            relationships=relationships,
            sources=sources,
            metadata={
                'strategy': 'hybrid',
                'entity_count': len(entities),
                'relationship_count': len(relationships),
                'subgraph_size': subgraph.get('node_count', 0)
            },
            token_count=self._estimate_tokens(context_text)
        )
    
    def _assemble_default_context(
        self,
        retrieval_result,
        query: str
    ) -> AssembledContext:
        """Default context assembly."""
        return self._assemble_hybrid_context(retrieval_result, query)
    
    def _extract_entities_from_nodes(
        self,
        nodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract entity information from nodes."""
        entities = []
        
        for node in nodes:
            metadata = node.get('metadata', {})
            entity = {
                'name': metadata.get('entity_name') or metadata.get('name', 'Unknown'),
                'type': metadata.get('node_type', 'ENTITY'),
                'score': node.get('final_score', node.get('score', 0)),
                'boost': node.get('boost_applied', 1.0),
                'properties': {
                    k: v for k, v in metadata.items()
                    if k not in ['entity_name', 'name', 'node_type', 'node_id']
                }
            }
            entities.append(entity)
        
        return entities
    
    def _extract_relationships_from_edges(
        self,
        edges: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract relationship information from edges."""
        relationships = []
        
        for edge in edges:
            metadata = edge.get('metadata', {})
            relationship = {
                'source': metadata.get('source', 'Unknown'),
                'target': metadata.get('target', 'Unknown'),
                'type': metadata.get('relationship_type', 'RELATED_TO'),
                'score': edge.get('final_score', edge.get('score', 0)),
                'properties': {
                    k: v for k, v in metadata.items()
                    if k not in ['source', 'target', 'relationship_type', 'edge_id']
                }
            }
            relationships.append(relationship)
        
        return relationships
    
    def _extract_sources(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract unique sources from entities and relationships."""
        sources = set()
        
        for entity in entities:
            props = entity.get('properties', {})
            if 'source' in props:
                sources.add(str(props['source']))
            if 'section' in props:
                sources.add(f"Section: {props['section']}")
        
        for rel in relationships:
            props = rel.get('properties', {})
            if 'source' in props:
                sources.add(str(props['source']))
            if 'section' in props:
                sources.add(f"Section: {props['section']}")
        
        return sorted(list(sources))
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Rough estimate: ~4 characters per token
        return len(text) // 4
    
    def truncate_context(
        self,
        context: AssembledContext,
        max_tokens: Optional[int] = None
    ) -> AssembledContext:
        """Truncate context to fit within token limit."""
        max_tokens = max_tokens or self.max_tokens
        
        if context.token_count <= max_tokens:
            return context
        
        # Simple truncation - cut entities and relationships
        target_chars = max_tokens * 4
        truncated_text = context.context_text[:target_chars] + "\n\n[... context truncated ...]"
        
        return AssembledContext(
            context_text=truncated_text,
            entities=context.entities,
            relationships=context.relationships,
            sources=context.sources,
            metadata={**context.metadata, 'truncated': True},
            token_count=self._estimate_tokens(truncated_text)
        )


def create_context_assembler(
    max_tokens: int = 4000,
    include_sources: bool = True,
    deduplicate: bool = True
) -> ContextAssembler:
    """Factory function to create context assembler.
    
    Args:
        max_tokens: Maximum token limit
        include_sources: Include source references
        deduplicate: Remove duplicates
        
    Returns:
        ContextAssembler instance
    """
    return ContextAssembler(max_tokens, include_sources, deduplicate)
