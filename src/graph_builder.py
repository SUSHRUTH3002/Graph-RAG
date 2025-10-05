"""Knowledge graph construction from entities and relationships."""

from typing import List, Dict, Any, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build knowledge graph from entities and relationships."""
    
    def __init__(self, neo4j_store=None):
        """Initialize graph builder.
        
        Args:
            neo4j_store: Optional Neo4jStore instance
        """
        self.neo4j_store = neo4j_store
        self.nodes = {}
        self.edges = []
        self.node_types = defaultdict(list)
        self.edge_types = defaultdict(list)
    
    def add_entity_as_node(
        self, 
        entity: Dict[str, Any],
        chunk_id: Optional[str] = None
    ) -> str:
        """Add an entity as a graph node.
        
        Args:
            entity: Entity dictionary
            chunk_id: Optional chunk ID for provenance
            
        Returns:
            Node ID
        """
        entity_text = entity.get('text', '')
        entity_type = entity.get('type', 'GENERIC')
        
        # Create unique node ID
        node_id = entity_text
        
        # Build node properties
        properties = {
            'name': entity_text,
            'type': entity_type,
            'confidence': entity.get('confidence', 1.0),
            'source': entity.get('source'),
            'chunk_id': entity.get('chunk_id') or chunk_id,
            'column': entity.get('column'),
            'is_key': entity.get('is_key', False)
        }
        
        # Add entity-specific properties
        if 'year' in entity:
            properties['year'] = entity['year']
        if 'section' in entity:
            properties['section'] = entity['section']
        if 'parent_section' in entity:
            properties['parent_section'] = entity['parent_section']
        if 'value' in entity:
            properties['value'] = entity['value']
        if 'unit' in entity:
            properties['unit'] = entity['unit']
        
        # Remove None values
        properties = {k: v for k, v in properties.items() if v is not None}
        
        # Store node
        self.nodes[node_id] = properties
        self.node_types[entity_type].append(node_id)
        
        return node_id
    
    def add_relationship_as_edge(
        self,
        relationship: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add a relationship as a graph edge.
        
        Args:
            relationship: Relationship dictionary
            
        Returns:
            Edge data
        """
        source = relationship.get('source')
        target = relationship.get('target')
        rel_type = relationship.get('type', 'RELATED_TO')
        
        if not source or not target:
            logger.warning(f"Skipping relationship without source/target: {relationship}")
            return None
        
        # Build edge properties
        edge_data = {
            'source': source,
            'target': target,
            'type': rel_type,
            'confidence': relationship.get('confidence', 1.0),
            'chunk_id': relationship.get('chunk_id'),
            'source_doc': relationship.get('source_doc'),
            'properties': relationship.get('properties', {})
        }
        
        # Add additional relationship properties
        if 'context' in relationship:
            edge_data['context'] = relationship['context']
        if 'year' in relationship:
            edge_data['year'] = relationship['year']
        
        # Store edge
        self.edges.append(edge_data)
        self.edge_types[rel_type].append(edge_data)
        
        return edge_data
    
    def add_chunk_as_node(
        self,
        chunk: Dict[str, Any]
    ) -> str:
        """Add a document chunk as a graph node.
        
        Args:
            chunk: Chunk dictionary
            
        Returns:
            Node ID
        """
        chunk_id = chunk.get('id')
        
        properties = {
            'name': chunk_id,
            'type': 'CHUNK',
            'text': chunk.get('text', '')[:500],  # Truncate for storage
            'full_text': chunk.get('text', ''),
            'embedding': chunk.get('embedding'),
            **chunk.get('metadata', {})
        }
        
        # Remove None and embedding from properties for Neo4j
        neo4j_properties = {
            k: v for k, v in properties.items() 
            if v is not None and k not in ['embedding', 'full_text']
        }
        
        self.nodes[chunk_id] = properties
        self.node_types['CHUNK'].append(chunk_id)
        
        return chunk_id
    
    def link_chunk_to_entities(
        self,
        chunk_id: str,
        entities: List[Dict[str, Any]]
    ):
        """Create edges from chunk to its entities.
        
        Args:
            chunk_id: Chunk node ID
            entities: List of entities mentioned in the chunk
        """
        for entity in entities:
            entity_text = entity.get('text')
            if entity_text:
                edge = {
                    'source': chunk_id,
                    'target': entity_text,
                    'type': 'MENTIONS',
                    'confidence': entity.get('confidence', 1.0),
                    'entity_type': entity.get('type')
                }
                self.edges.append(edge)
                self.edge_types['MENTIONS'].append(edge)
    
    def build_from_data(
        self,
        chunks: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build graph from processed data.
        
        Args:
            chunks: List of document chunks
            entities: List of extracted entities
            relationships: List of extracted relationships
            
        Returns:
            Graph statistics
        """
        logger.info("Building knowledge graph...")
        
        # Group entities by chunk for linking
        chunk_entity_map = defaultdict(list)
        for entity in entities:
            chunk_id = entity.get('chunk_id')
            if chunk_id:
                chunk_entity_map[chunk_id].append(entity)
        
        # Add all entities as nodes
        logger.info(f"Adding {len(entities)} entities as nodes...")
        entity_count = 0
        for entity in entities:
            # Only add key entities and non-evidence derived entities as primary nodes
            if entity.get('is_key') or entity.get('column') not in ['evidence_note']:
                self.add_entity_as_node(entity)
                entity_count += 1
        logger.info(f"Added {entity_count} entity nodes")
        
        # Add all chunks as nodes
        logger.info(f"Adding {len(chunks)} chunks as nodes...")
        for chunk in chunks:
            chunk_id = chunk.get('id') or chunk.get('metadata', {}).get('chunk_id')
            if not chunk_id:
                logger.warning(f"Chunk without ID, generating one: {chunk.get('text', '')[:50]}")
                chunk_id = f"chunk_{hash(chunk.get('text', ''))}"
                chunk['id'] = chunk_id
            
            self.add_chunk_as_node(chunk)
            
            # Link chunk to its entities
            chunk_entities = chunk_entity_map.get(chunk_id, [])
            if chunk_entities:
                self.link_chunk_to_entities(chunk_id, chunk_entities)
        
        # Add all relationships as edges
        logger.info(f"Adding {len(relationships)} relationships as edges...")
        edge_count = 0
        for relationship in relationships:
            edge = self.add_relationship_as_edge(relationship)
            if edge:
                edge_count += 1
        logger.info(f"Added {edge_count} relationship edges")
        
        stats = {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_types': {k: len(v) for k, v in self.node_types.items()},
            'edge_types': {k: len(v) for k, v in self.edge_types.items()}
        }
        
        logger.info(f"Graph built successfully: {stats}")
        return stats
    
    def persist_to_neo4j(self, neo4j_store) -> Dict[str, int]:
        """Persist the built graph to Neo4j.
        
        Args:
            neo4j_store: Neo4jStore instance
            
        Returns:
            Persistence statistics
        """
        logger.info("Persisting graph to Neo4j...")
        
        stats = {'nodes_created': 0, 'edges_created': 0, 'errors': 0}
        
        # Create all nodes
        for node_id, properties in self.nodes.items():
            try:
                node_type = properties.get('type', 'GENERIC')
                name = properties.get('name', node_id)
                
                if node_type == 'CHUNK':
                    # Handle chunks separately
                    doc_id = properties.get('doc_id') or properties.get('source', 'unknown')
                    
                    # Clean doc_id (remove path separators)
                    if '\\' in str(doc_id) or '/' in str(doc_id):
                        doc_id = str(doc_id).split('\\')[-1].split('/')[-1].replace('.csv', '')
                    
                    text = properties.get('full_text') or properties.get('text', '')
                    
                    # Build chunk metadata (exclude embedding and full_text)
                    chunk_metadata = {
                        k: v for k, v in properties.items() 
                        if v is not None and k not in ['type', 'name', 'doc_id', 'text', 'embedding', 'full_text']
                    }
                    
                    neo4j_store.create_chunk(
                        chunk_id=node_id,
                        doc_id=doc_id,
                        text=text[:1000],  # Truncate for Neo4j storage
                        metadata=chunk_metadata
                    )
                else:
                    # Create entity node
                    clean_props = {
                        k: v for k, v in properties.items() 
                        if v is not None and k not in ['type', 'name', 'embedding']
                    }
                    
                    neo4j_store.create_entity(
                        name=name,
                        entity_type=node_type,
                        properties=clean_props
                    )
                
                stats['nodes_created'] += 1
                
                if stats['nodes_created'] % 10 == 0:
                    logger.info(f"Created {stats['nodes_created']} nodes...")
                
            except Exception as e:
                logger.error(f"Error creating node {node_id}: {e}")
                logger.debug(f"Node properties: {properties}")
                stats['errors'] += 1
        
        # Create all edges
        for edge in self.edges:
            try:
                source = edge.get('source')
                target = edge.get('target')
                rel_type = edge.get('type', 'RELATED_TO')
                
                if not source or not target:
                    logger.warning(f"Skipping edge with missing source/target: {edge}")
                    stats['errors'] += 1
                    continue
                
                # Build edge properties
                properties = edge.get('properties', {}).copy()
                
                # Add edge-level metadata
                if 'confidence' in edge and edge['confidence'] is not None:
                    properties['confidence'] = edge['confidence']
                if 'context' in edge and edge['context']:
                    properties['context'] = str(edge['context'])[:500]
                if 'year' in edge and edge['year']:
                    properties['year'] = str(edge['year'])
                if 'section' in edge and edge['section']:
                    properties['section'] = str(edge['section'])
                if 'entity_type' in edge and edge['entity_type']:
                    properties['entity_type'] = str(edge['entity_type'])
                
                # Remove None values
                clean_properties = {k: v for k, v in properties.items() if v is not None}
                
                neo4j_store.create_entity_relation(
                    entity1=source,
                    entity2=target,
                    relation_type=rel_type,
                    properties=clean_properties
                )
                
                stats['edges_created'] += 1
                
                if stats['edges_created'] % 10 == 0:
                    logger.info(f"Created {stats['edges_created']} edges...")
                
            except Exception as e:
                logger.error(f"Error creating edge {source} -> {target}: {e}")
                logger.debug(f"Edge details: {edge}")
                stats['errors'] += 1
        
        logger.info(f"Persistence completed: {stats}")
        return stats
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get detailed graph statistics.
        
        Returns:
            Dictionary with graph statistics
        """
        return {
            'nodes': {
                'total': len(self.nodes),
                'by_type': {k: len(v) for k, v in self.node_types.items()}
            },
            'edges': {
                'total': len(self.edges),
                'by_type': {k: len(v) for k, v in self.edge_types.items()}
            }
        }
    
    def get_node_degrees(self) -> Dict[str, int]:
        """Calculate node degrees (number of connections).
        
        Returns:
            Dictionary mapping node IDs to their degrees
        """
        degrees = defaultdict(int)
        
        for edge in self.edges:
            degrees[edge['source']] += 1
            degrees[edge['target']] += 1
        
        return dict(degrees)
    
    def get_subgraph(
        self,
        center_node: str,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """Extract a subgraph around a center node.
        
        Args:
            center_node: Central node ID
            max_depth: Maximum traversal depth
            
        Returns:
            Subgraph data
        """
        if center_node not in self.nodes:
            return {'nodes': {}, 'edges': []}
        
        visited = {center_node}
        current_level = {center_node}
        subgraph_edges = []
        
        for _ in range(max_depth):
            next_level = set()
            
            for edge in self.edges:
                if edge['source'] in current_level:
                    next_level.add(edge['target'])
                    subgraph_edges.append(edge)
                    visited.add(edge['target'])
                elif edge['target'] in current_level:
                    next_level.add(edge['source'])
                    subgraph_edges.append(edge)
                    visited.add(edge['source'])
            
            current_level = next_level - visited
            if not current_level:
                break
        
        subgraph_nodes = {
            node_id: self.nodes[node_id]
            for node_id in visited
            if node_id in self.nodes
        }
        
        return {
            'nodes': subgraph_nodes,
            'edges': subgraph_edges,
            'center': center_node,
            'depth': max_depth
        }
