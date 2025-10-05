"""Neo4j database operations for knowledge graph storage."""

from typing import Dict, List, Any, Optional
import logging
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError

logger = logging.getLogger(__name__)

class Neo4jStore:
    """Neo4j database interface for knowledge graph operations."""
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j"
    ):
        """Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI
            user: Database username
            password: Database password
            database: Database name
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver: Optional[Driver] = None
        
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
        except AuthError as e:
            logger.error(f"Authentication failed: {e}")
            raise
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def query(self, cypher_query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results.
        
        Args:
            cypher_query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        parameters = parameters or {}
        
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher_query, parameters)
            return [dict(record) for record in result]
    
    def clear_database(self):
        """Clear all data from the database. Use with caution!"""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")
    
    def create_entity(
        self,
        name: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        """Create an entity node.
        
        Args:
            name: Entity name (unique identifier)
            entity_type: Entity type (label)
            properties: Additional properties
        """
        properties = properties or {}
        properties['name'] = name
        
        # Clean properties
        clean_props = {k: v for k, v in properties.items() if v is not None}
        
        query = f"""
        MERGE (e:{entity_type} {{name: $name}})
        SET e += $properties
        RETURN e
        """
        
        with self.driver.session(database=self.database) as session:
            session.run(query, name=name, properties=clean_props)
    
    def create_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Create a document chunk node.
        
        Args:
            chunk_id: Unique chunk identifier
            doc_id: Parent document identifier
            text: Chunk text content
            metadata: Additional metadata
        """
        metadata = metadata or {}
        metadata.update({
            'chunk_id': chunk_id,
            'doc_id': doc_id,
            'text': text
        })
        
        # Clean metadata
        clean_metadata = {k: v for k, v in metadata.items() if v is not None}
        
        query = """
        MERGE (c:Chunk {chunk_id: $chunk_id})
        SET c += $metadata
        RETURN c
        """
        
        with self.driver.session(database=self.database) as session:
            session.run(query, chunk_id=chunk_id, metadata=clean_metadata)
    
    def create_entity_relation(
        self,
        entity1: str,
        entity2: str,
        relation_type: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        """Create a relationship between two entities.
        
        Args:
            entity1: Source entity name
            entity2: Target entity name
            relation_type: Relationship type
            properties: Relationship properties
        """
        properties = properties or {}
        
        # Clean properties and convert to appropriate types
        clean_props = {}
        for k, v in properties.items():
            if v is not None:
                # Convert lists/dicts to strings for Neo4j
                if isinstance(v, (list, dict)):
                    clean_props[k] = str(v)
                else:
                    clean_props[k] = v
        
        # Sanitize relation type (remove special chars, convert to uppercase)
        safe_rel_type = relation_type.replace(' ', '_').replace('-', '_').upper()
        
        query = f"""
        MATCH (e1 {{name: $entity1}})
        MATCH (e2 {{name: $entity2}})
        MERGE (e1)-[r:{safe_rel_type}]->(e2)
        SET r += $properties
        RETURN r
        """
        
        with self.driver.session(database=self.database) as session:
            session.run(
                query,
                entity1=entity1,
                entity2=entity2,
                properties=clean_props
            )
    
    def get_entity(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve an entity by name.
        
        Args:
            name: Entity name
            
        Returns:
            Entity properties or None
        """
        query = "MATCH (e {name: $name}) RETURN e"
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, name=name)
            record = result.single()
            if record:
                return dict(record['e'])
        return None
    
    def get_entity_neighbors(
        self,
        name: str,
        max_depth: int = 1
    ) -> List[Dict[str, Any]]:
        """Get neighboring entities.
        
        Args:
            name: Entity name
            max_depth: Maximum traversal depth
            
        Returns:
            List of neighbor entities
        """
        query = f"""
        MATCH (e {{name: $name}})-[*1..{max_depth}]-(neighbor)
        RETURN DISTINCT neighbor
        LIMIT 100
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, name=name)
            return [dict(record['neighbor']) for record in result]
    
    def get_related_entities(
        self,
        entity_name: str,
        max_depth: int = 2,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Get related entities and relationships for graph expansion.
        
        Args:
            entity_name: Name of the entity to expand from
            max_depth: Maximum traversal depth
            limit: Maximum number of entities to return
            
        Returns:
            Dictionary with nodes and edges
        """
        query = f"""
        MATCH path = (start {{name: $entity_name}})-[*1..{max_depth}]-(connected)
        WITH start, connected, relationships(path) as rels
        LIMIT {limit}
        RETURN 
            collect(DISTINCT {{
                name: connected.name,
                labels: labels(connected),
                properties: properties(connected)
            }}) as nodes,
            collect(DISTINCT {{
                source: startNode(rels[0]).name,
                target: endNode(rels[0]).name,
                type: type(rels[0]),
                properties: properties(rels[0])
            }}) as edges
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, entity_name=entity_name)
                record = result.single()
                
                if record:
                    return {
                        'nodes': record['nodes'],
                        'edges': record['edges']
                    }
                return {'nodes': [], 'edges': []}
        except Exception as e:
            logger.error(f"Failed to get related entities for {entity_name}: {e}")
            return {'nodes': [], 'edges': []}
    
    def get_relationships(
        self,
        entity1: Optional[str] = None,
        entity2: Optional[str] = None,
        relation_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query relationships.
        
        Args:
            entity1: Source entity filter
            entity2: Target entity filter
            relation_type: Relationship type filter
            
        Returns:
            List of relationships
        """
        conditions = []
        params = {}
        
        if entity1:
            conditions.append("e1.name = $entity1")
            params['entity1'] = entity1
        if entity2:
            conditions.append("e2.name = $entity2")
            params['entity2'] = entity2
        
        where_clause = " AND ".join(conditions) if conditions else ""
        if where_clause:
            where_clause = f"WHERE {where_clause}"
        
        rel_pattern = f":{relation_type}" if relation_type else ""
        
        query = f"""
        MATCH (e1)-[r{rel_pattern}]->(e2)
        {where_clause}
        RETURN e1.name as source, type(r) as type, e2.name as target, properties(r) as props
        LIMIT 100
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, **params)
            return [
                {
                    'source': record['source'],
                    'type': record['type'],
                    'target': record['target'],
                    'properties': dict(record['props'])
                }
                for record in result
            ]
    
    def create_indexes(self):
        """Create indexes for better query performance."""
        indexes = [
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:ENTITY) ON (e.name)",
            "CREATE INDEX organization_name IF NOT EXISTS FOR (e:ORGANIZATION) ON (e.name)",
            "CREATE INDEX product_name IF NOT EXISTS FOR (e:PRODUCT) ON (e.name)",
            "CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_id)",
            "CREATE INDEX year IF NOT EXISTS FOR (e:YEAR) ON (e.name)",
        ]
        
        with self.driver.session(database=self.database) as session:
            for index_query in indexes:
                try:
                    session.run(index_query)
                    logger.info(f"Created index: {index_query}")
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dictionary with node and relationship counts
        """
        with self.driver.session(database=self.database) as session:
            # Count nodes by label
            node_query = """
            MATCH (n)
            RETURN labels(n) as labels, count(n) as count
            """
            node_result = session.run(node_query)
            node_counts = {}
            for record in node_result:
                labels = record['labels']
                if labels:
                    label = labels[0]  # Primary label
                    node_counts[label] = record['count']
            
            # Count relationships by type
            rel_query = """
            MATCH ()-[r]->()
            RETURN type(r) as type, count(r) as count
            """
            rel_result = session.run(rel_query)
            rel_counts = {record['type']: record['count'] for record in rel_result}
            
            return {
                'nodes': node_counts,
                'relationships': rel_counts,
                'total_nodes': sum(node_counts.values()),
                'total_relationships': sum(rel_counts.values())
            }


def create_neo4j_store(
    uri: str = None,
    user: str = None,
    password: str = None,
    database: str = "neo4j"
) -> Neo4jStore:
    """Factory function to create Neo4j store.
    
    Args:
        uri: Neo4j URI (defaults to environment variable)
        user: Username (defaults to environment variable)
        password: Password (defaults to environment variable)
        database: Database name
        
    Returns:
        Neo4jStore instance
    """
    import os
    
    uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    user = user or os.getenv('NEO4J_USER', 'neo4j')
    password = password or os.getenv('NEO4J_PASSWORD', 'password')
    
    return Neo4jStore(uri=uri, user=user, password=password, database=database)