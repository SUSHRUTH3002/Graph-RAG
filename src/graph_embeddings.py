"""Module for generating and managing graph embeddings."""

from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class NodeEmbedding:
    """Container for node embedding data."""
    node_id: str
    node_type: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]

@dataclass
class EdgeEmbedding:
    """Container for edge embedding data."""
    edge_id: str
    source: str
    target: str
    relationship_type: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]

class GraphEmbeddingGenerator:
    """Generate embeddings for graph nodes and edges."""
    
    def __init__(self, embeddings_client, neo4j_store):
        """Initialize generator.
        
        Args:
            embeddings_client: EmbeddingsClient instance
            neo4j_store: Neo4jStore instance
        """
        self.embeddings_client = embeddings_client
        self.neo4j_store = neo4j_store
        self.node_embeddings: Dict[str, NodeEmbedding] = {}
        self.edge_embeddings: Dict[str, EdgeEmbedding] = {}
    
    def generate_node_embeddings(self, node_types: Optional[List[str]] = None) -> List[NodeEmbedding]:
        """Generate embeddings for all nodes in the graph.
        
        Args:
            node_types: Optional list of node types to filter
            
        Returns:
            List of NodeEmbedding objects
        """
        logger.info("Generating embeddings for graph nodes...")
        
        # Fetch all nodes from Neo4j
        nodes = self._fetch_nodes(node_types)
        
        if not nodes:
            logger.warning("No nodes found in graph")
            return []
        
        # Prepare text representations
        texts = []
        node_data = []
        
        for node in nodes:
            text = self._create_node_text(node)
            texts.append(text)
            node_data.append(node)
        
        # Generate embeddings in batch
        logger.info(f"Embedding {len(texts)} nodes...")
        embeddings = self.embeddings_client.embed_documents(texts)
        
        # Create NodeEmbedding objects
        node_embeddings = []
        for node, text, embedding in zip(node_data, texts, embeddings):
            node_emb = NodeEmbedding(
                node_id=node['id'],
                node_type=node['type'],
                text=text,
                embedding=embedding,
                metadata={
                    'name': node.get('name'),
                    'properties': node.get('properties', {}),
                    'labels': node.get('labels', [])
                }
            )
            node_embeddings.append(node_emb)
            self.node_embeddings[node['id']] = node_emb
        
        logger.info(f"Generated embeddings for {len(node_embeddings)} nodes")
        return node_embeddings
    
    def generate_edge_embeddings(self, relationship_types: Optional[List[str]] = None) -> List[EdgeEmbedding]:
        """Generate embeddings for all edges in the graph.
        
        Args:
            relationship_types: Optional list of relationship types to filter
            
        Returns:
            List of EdgeEmbedding objects
        """
        logger.info("Generating embeddings for graph edges...")
        
        # Fetch all relationships from Neo4j
        edges = self._fetch_edges(relationship_types)
        
        if not edges:
            logger.warning("No edges found in graph")
            return []
        
        # Prepare text representations
        texts = []
        edge_data = []
        
        for edge in edges:
            text = self._create_edge_text(edge)
            texts.append(text)
            edge_data.append(edge)
        
        # Generate embeddings in batch
        logger.info(f"Embedding {len(texts)} edges...")
        embeddings = self.embeddings_client.embed_documents(texts)
        
        # Create EdgeEmbedding objects
        edge_embeddings = []
        for edge, text, embedding in zip(edge_data, texts, embeddings):
            edge_id = f"{edge['source']}_{edge['type']}_{edge['target']}"
            edge_emb = EdgeEmbedding(
                edge_id=edge_id,
                source=edge['source'],
                target=edge['target'],
                relationship_type=edge['type'],
                text=text,
                embedding=embedding,
                metadata=edge.get('properties', {})
            )
            edge_embeddings.append(edge_emb)
            self.edge_embeddings[edge_id] = edge_emb
        
        logger.info(f"Generated embeddings for {len(edge_embeddings)} edges")
        return edge_embeddings
    
    def _fetch_nodes(self, node_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Fetch nodes from Neo4j."""
        query = """
        MATCH (n)
        WHERE $node_types IS NULL OR any(label IN labels(n) WHERE label IN $node_types)
        RETURN 
            id(n) as id,
            labels(n) as labels,
            n.name as name,
            properties(n) as properties
        LIMIT 10000
        """
        
        try:
            result = self.neo4j_store.query(query, {'node_types': node_types})
            nodes = []
            for record in result:
                nodes.append({
                    'id': str(record['id']),
                    'type': record['labels'][0] if record['labels'] else 'UNKNOWN',
                    'name': record['name'],
                    'properties': dict(record['properties']),
                    'labels': record['labels']
                })
            return nodes
        except Exception as e:
            logger.error(f"Failed to fetch nodes: {e}")
            return []
    
    def _fetch_edges(self, relationship_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Fetch edges from Neo4j."""
        query = """
        MATCH (source)-[r]->(target)
        WHERE $rel_types IS NULL OR type(r) IN $rel_types
        RETURN 
            id(r) as id,
            source.name as source,
            target.name as target,
            type(r) as type,
            properties(r) as properties
        LIMIT 10000
        """
        
        try:
            result = self.neo4j_store.query(query, {'rel_types': relationship_types})
            edges = []
            for record in result:
                edges.append({
                    'id': str(record['id']),
                    'source': record['source'],
                    'target': record['target'],
                    'type': record['type'],
                    'properties': dict(record['properties'])
                })
            return edges
        except Exception as e:
            logger.error(f"Failed to fetch edges: {e}")
            return []
    
    def _create_node_text(self, node: Dict[str, Any]) -> str:
        """Create text representation of a node for embedding.
        
        Format: "{type}: {name} - {key properties}"
        """
        node_type = node['type']
        name = node.get('name', 'Unknown')
        props = node.get('properties', {})
        
        # Build text based on node type
        text_parts = [f"{node_type}: {name}"]
        
        # Add relevant properties
        if 'section' in props:
            text_parts.append(f"Section: {props['section']}")
        if 'year' in props:
            text_parts.append(f"Year: {props['year']}")
        if 'value' in props and 'unit' in props:
            text_parts.append(f"Value: {props['value']} {props['unit']}")
        if 'evidence' in props:
            text_parts.append(f"Evidence: {props['evidence'][:100]}")
        
        return " | ".join(text_parts)
    
    def _create_edge_text(self, edge: Dict[str, Any]) -> str:
        """Create text representation of an edge for embedding.
        
        Format: "{source} -{relationship}-> {target} [{context}]"
        """
        source = edge['source']
        target = edge['target']
        rel_type = edge['type'].replace('_', ' ').lower()
        props = edge.get('properties', {})
        
        text_parts = [f"{source} {rel_type} {target}"]
        
        # Add context from properties
        if 'evidence' in props:
            text_parts.append(f"Context: {props['evidence'][:100]}")
        if 'section' in props:
            text_parts.append(f"in {props['section']}")
        if 'year' in props:
            text_parts.append(f"({props['year']})")
        
        return " | ".join(text_parts)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get embedding statistics."""
        return {
            'total_node_embeddings': len(self.node_embeddings),
            'total_edge_embeddings': len(self.edge_embeddings),
            'embedding_dimension': len(next(iter(self.node_embeddings.values())).embedding) if self.node_embeddings else 0
        }
