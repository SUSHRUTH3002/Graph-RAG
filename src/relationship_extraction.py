"""Relationship extraction between entities."""

from typing import List, Dict, Any, Optional, Tuple, Set
from abc import ABC, abstractmethod
import re
from itertools import combinations


class RelationshipExtractor(ABC):
    """Abstract base class for relationship extractors."""
    
    @abstractmethod
    def extract_relationships(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        chunk_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Extract relationships between entities.
        
        Args:
            text: Input text
            entities: List of extracted entities
            chunk_metadata: Optional metadata from the chunk
            
        Returns:
            List of relationship dictionaries
        """
        pass


class CooccurrenceRelationshipExtractor(RelationshipExtractor):
    """Extract relationships based on entity co-occurrence in text."""
    
    def __init__(
        self,
        window_size: int = 100,
        min_confidence: float = 0.5
    ):
        """Initialize co-occurrence relationship extractor.
        
        Args:
            window_size: Character window for considering entities as related
            min_confidence: Minimum confidence for relationships
        """
        self.window_size = window_size
        self.min_confidence = min_confidence
    
    def extract_relationships(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        chunk_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Extract relationships based on proximity.
        
        Args:
            text: Input text
            entities: List of extracted entities
            chunk_metadata: Optional metadata from the chunk
            
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        chunk_metadata = chunk_metadata or {}
        
        # Sort entities by position
        sorted_entities = sorted(
            entities,
            key=lambda e: e.get('start', 0)
        )
        
        # Find entity pairs within window
        for i, entity1 in enumerate(sorted_entities):
            for entity2 in sorted_entities[i + 1:]:
                # Check if entities are within window
                start1 = entity1.get('start', 0)
                start2 = entity2.get('start', 0)
                
                if abs(start2 - start1) <= self.window_size:
                    # Extract context between entities
                    context = self._extract_context(text, entity1, entity2)
                    
                    relationship = {
                        'source': entity1['text'],
                        'target': entity2['text'],
                        'source_type': entity1.get('type'),
                        'target_type': entity2.get('type'),
                        'type': 'CO_OCCURS',
                        'context': context,
                        'confidence': self._calculate_confidence(entity1, entity2, context),
                        'chunk_id': chunk_metadata.get('chunk_id'),
                        'source_doc': chunk_metadata.get('source')
                    }
                    
                    if relationship['confidence'] >= self.min_confidence:
                        relationships.append(relationship)
                else:
                    break  # Entities too far apart
        
        return relationships
    
    def _extract_context(
        self,
        text: str,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any]
    ) -> str:
        """Extract text context between two entities."""
        start1 = entity1.get('start', 0)
        end1 = entity1.get('end', start1)
        start2 = entity2.get('start', 0)
        end2 = entity2.get('end', start2)
        
        # Get text between entities
        context_start = end1
        context_end = start2
        
        if context_start < context_end:
            context = text[context_start:context_end].strip()
            return context
        
        return ""
    
    def _calculate_confidence(
        self,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any],
        context: str
    ) -> float:
        """Calculate confidence score for relationship."""
        base_confidence = 0.5
        
        # Increase confidence if entities are close
        distance = abs(entity2.get('start', 0) - entity1.get('start', 0))
        proximity_score = max(0, 1 - (distance / self.window_size))
        
        # Increase confidence if context contains relationship indicators
        relationship_indicators = ['and', 'with', 'of', 'in', 'at', 'for', 'by', 'to']
        context_lower = context.lower()
        indicator_score = sum(
            0.1 for indicator in relationship_indicators
            if indicator in context_lower
        )
        
        confidence = min(1.0, base_confidence + proximity_score * 0.3 + indicator_score)
        return confidence


class PatternRelationshipExtractor(RelationshipExtractor):
    """Extract relationships using predefined patterns."""
    
    def __init__(self, patterns: Optional[List[Dict[str, Any]]] = None):
        """Initialize pattern-based relationship extractor.
        
        Args:
            patterns: List of pattern dictionaries with 'pattern' and 'type' keys
        """
        self.patterns = patterns or self._default_patterns()
    
    def _default_patterns(self) -> List[Dict[str, Any]]:
        """Get default relationship patterns."""
        return [
            {
                'pattern': r'(\w+)\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*(\w+)',
                'type': 'IS_A',
                'confidence': 0.8
            },
            {
                'pattern': r'(\w+)\s+(?:works for|employed by)\s+(\w+)',
                'type': 'WORKS_FOR',
                'confidence': 0.9
            },
            {
                'pattern': r'(\w+)\s+(?:located in|based in)\s+(\w+)',
                'type': 'LOCATED_IN',
                'confidence': 0.85
            },
            {
                'pattern': r'(\w+)\s+(?:owns|possesses|has)\s+(\w+)',
                'type': 'OWNS',
                'confidence': 0.8
            },
            {
                'pattern': r'(\w+)\s+(?:created|developed|built)\s+(\w+)',
                'type': 'CREATED',
                'confidence': 0.85
            }
        ]
    
    def extract_relationships(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        chunk_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Extract relationships using patterns.
        
        Args:
            text: Input text
            entities: List of extracted entities
            chunk_metadata: Optional metadata from the chunk
            
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        chunk_metadata = chunk_metadata or {}
        
        for pattern_def in self.patterns:
            pattern = pattern_def['pattern']
            rel_type = pattern_def['type']
            confidence = pattern_def['confidence']
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                if len(match.groups()) >= 2:
                    source = match.group(1).strip()
                    target = match.group(2).strip()
                    
                    relationship = {
                        'source': source,
                        'target': target,
                        'type': rel_type,
                        'confidence': confidence,
                        'context': match.group(0),
                        'chunk_id': chunk_metadata.get('chunk_id'),
                        'source_doc': chunk_metadata.get('source')
                    }
                    
                    relationships.append(relationship)
        
        return relationships


class CSVRelationshipExtractor(RelationshipExtractor):
    """Extract relationships from CSV data based on column relationships."""
    
    def __init__(
        self,
        key_column: Optional[str] = None,
        relationship_columns: Optional[List[str]] = None,
        extract_hierarchical: bool = True
    ):
        """Initialize CSV relationship extractor.
        
        Args:
            key_column: Primary key column name
            relationship_columns: Columns that define relationships
            extract_hierarchical: Extract hierarchical relationships (parent-child)
        """
        self.key_column = key_column
        self.relationship_columns = relationship_columns or []
        self.extract_hierarchical = extract_hierarchical
    
    def extract_relationships(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        chunk_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Extract relationships from CSV row data.
        
        Args:
            text: Input text (not used, uses chunk_metadata['row_data'])
            entities: List of extracted entities
            chunk_metadata: Must contain 'row_data' for CSV chunks
            
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        chunk_metadata = chunk_metadata or {}
        
        # Check if this is a CSV row chunk
        if chunk_metadata.get('chunk_type') != 'csv_row':
            return relationships
        
        row_data = chunk_metadata.get('row_data', {})
        columns = chunk_metadata.get('columns', [])
        
        # Detect which CSV file we're processing
        if self._is_subjective_sections_csv(columns):
            relationships = self._extract_from_subjective_sections(
                row_data, entities, chunk_metadata
            )
        elif self._is_entity_relations_csv(columns):
            relationships = self._extract_from_entity_relations(
                row_data, entities, chunk_metadata
            )
        else:
            # Generic extraction
            relationships = self._extract_generic(row_data, entities, chunk_metadata)
        
        return relationships
    
    def _is_subjective_sections_csv(self, columns: List[str]) -> bool:
        """Check if this is the subjective_sections_for_rag.csv structure."""
        required_cols = {'section', 'subsection', 'year', 'perspective', 'content', 'tags'}
        return required_cols.issubset(set(columns))
    
    def _is_entity_relations_csv(self, columns: List[str]) -> bool:
        """Check if this is the entity_relations_graph.csv structure."""
        required_cols = {'source', 'relation', 'target', 'target_type'}
        return required_cols.issubset(set(columns))
    
    def _extract_from_subjective_sections(
        self,
        row_data: Dict[str, Any],
        entities: List[Dict[str, Any]],
        chunk_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract relationships from subjective_sections_for_rag.csv format."""
        relationships = []
        
        section = row_data.get('section')
        subsection = row_data.get('subsection')
        year = str(row_data.get('year', '')).replace('.0', '')
        perspective = row_data.get('perspective')
        
        # Relationship 1: Section HAS_SUBSECTION Subsection
        if section and subsection:
            relationships.append({
                'source': str(section),
                'target': str(subsection),
                'type': 'HAS_SUBSECTION',
                'confidence': 1.0,
                'chunk_id': chunk_metadata.get('chunk_id'),
                'source_doc': chunk_metadata.get('source'),
                'year': year,
                'properties': {
                    'hierarchical': True,
                    'parent': section,
                    'child': subsection
                }
            })
        
        # Relationship 2: Subsection IN_YEAR Year
        if subsection and year:
            relationships.append({
                'source': str(subsection),
                'target': year,
                'type': 'IN_YEAR',
                'confidence': 1.0,
                'chunk_id': chunk_metadata.get('chunk_id'),
                'source_doc': chunk_metadata.get('source'),
                'properties': {
                    'temporal': True
                }
            })
        
        # Relationship 3: Subsection HAS_PERSPECTIVE Perspective
        if subsection and perspective:
            relationships.append({
                'source': str(subsection),
                'target': str(perspective),
                'type': 'HAS_PERSPECTIVE',
                'confidence': 0.9,
                'chunk_id': chunk_metadata.get('chunk_id'),
                'source_doc': chunk_metadata.get('source'),
                'year': year,
                'properties': {
                    'section': section
                }
            })
        
        # Relationship 4: Subsection TAGGED_WITH each Tag
        if 'tags' in row_data and row_data['tags']:
            tags = str(row_data['tags']).split(',')
            for tag in tags:
                tag = tag.strip()
                if tag and subsection:
                    relationships.append({
                        'source': str(subsection),
                        'target': tag,
                        'type': 'TAGGED_WITH',
                        'confidence': 0.8,
                        'chunk_id': chunk_metadata.get('chunk_id'),
                        'source_doc': chunk_metadata.get('source'),
                        'year': year,
                        'properties': {
                            'section': section,
                            'perspective': perspective
                        }
                    })
        
        # Relationship 5: Extract content-based relationships (entities mentioned in content)
        if 'content' in row_data and row_data['content']:
            content_rels = self._extract_content_relationships(
                str(row_data['content']),
                subsection or section,
                entities,
                chunk_metadata,
                year
            )
            relationships.extend(content_rels)
        
        return relationships
    
    def _extract_from_entity_relations(
        self,
        row_data: Dict[str, Any],
        entities: List[Dict[str, Any]],
        chunk_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract relationships from entity_relations_graph.csv format.
        
        This CSV explicitly defines relationships, so we extract them directly.
        """
        relationships = []
        
        source = row_data.get('source')
        relation = row_data.get('relation')
        target = row_data.get('target')
        target_type = row_data.get('target_type')
        section = row_data.get('section')
        year = str(row_data.get('year', '')).replace('.0', '')
        value = row_data.get('value')
        unit = row_data.get('unit')
        evidence = row_data.get('evidence_note')
        
        # Find entity dictionaries for source and target
        source_entity = next((e for e in entities if e['text'] == source), None)
        target_entity = next((e for e in entities if e['text'] == target), None)
        
        # Main relationship as defined in the CSV
        if source and relation and target:
            properties = {
                'target_type': target_type,
                'section': section,
                'year': year
            }
            
            # Add value and unit if present
            if value and str(value).strip():
                properties['value'] = value
                if unit and str(unit).strip():
                    properties['unit'] = unit
            
            # Add evidence note
            if evidence and str(evidence).strip():
                properties['evidence'] = evidence
            
            # Include entity types if available
            if source_entity:
                properties['source_type'] = source_entity.get('type')
            if target_entity:
                properties['target_type'] = target_entity.get('type')
            
            relationships.append({
                'source': str(source),
                'target': str(target),
                'type': str(relation).upper().replace(' ', '_'),
                'confidence': 1.0,  # High confidence as it's explicitly defined
                'chunk_id': chunk_metadata.get('chunk_id'),
                'source_doc': chunk_metadata.get('source'),
                'properties': properties
            })
        
        # Additional relationship: Source IN_SECTION Section
        if source and section:
            relationships.append({
                'source': str(source),
                'target': str(section),
                'type': 'IN_SECTION',
                'confidence': 0.9,
                'chunk_id': chunk_metadata.get('chunk_id'),
                'source_doc': chunk_metadata.get('source'),
                'properties': {
                    'year': year
                }
            })
        
        # Temporal relationship: Relation IN_YEAR Year
        if source and year:
            relationships.append({
                'source': str(source),
                'target': year,
                'type': 'IN_YEAR',
                'confidence': 1.0,
                'chunk_id': chunk_metadata.get('chunk_id'),
                'source_doc': chunk_metadata.get('source'),
                'properties': {
                    'temporal': True
                }
            })
        
        return relationships
    
    def _extract_content_relationships(
        self,
        content: str,
        subject: str,
        entities: List[Dict[str, Any]],
        chunk_metadata: Dict[str, Any],
        year: str
    ) -> List[Dict[str, Any]]:
        """Extract relationships from content text."""
        relationships = []
        
        # Find entities from this chunk in the content
        chunk_id = chunk_metadata.get('chunk_id')
        chunk_entities = [e for e in entities if e.get('chunk_id') == chunk_id]
        
        # Create MENTIONS relationships for key entities found in content
        for entity in chunk_entities:
            if entity.get('column') == 'content':
                entity_text = entity.get('text')
                entity_type = entity.get('type')
                
                # Create relationship: Subject MENTIONS Entity
                if entity_text and len(entity_text) > 2:
                    relationships.append({
                        'source': str(subject),
                        'target': entity_text,
                        'type': f'MENTIONS_{entity_type}',
                        'confidence': entity.get('confidence', 0.7),
                        'chunk_id': chunk_metadata.get('chunk_id'),
                        'source_doc': chunk_metadata.get('source'),
                        'properties': {
                            'year': year,
                            'entity_type': entity_type
                        }
                    })
        
        return relationships
    
    def _extract_generic(
        self,
        row_data: Dict[str, Any],
        entities: List[Dict[str, Any]],
        chunk_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generic relationship extraction for unknown CSV structures."""
        relationships = []
        
        # Determine key entity
        if self.key_column and self.key_column in row_data:
            key_entity = row_data[self.key_column]
        else:
            # Use first column as key
            key_entity = list(row_data.values())[0] if row_data else None
        
        if not key_entity:
            return relationships
        
        # Create relationships between key and other columns
        for column, value in row_data.items():
            if column != self.key_column and str(value).strip():
                relationship = {
                    'source': str(key_entity),
                    'target': str(value),
                    'type': f'HAS_{column.upper()}',
                    'column': column,
                    'confidence': 1.0,
                    'chunk_id': chunk_metadata.get('chunk_id'),
                    'source_doc': chunk_metadata.get('source')
                }
                
                relationships.append(relationship)
        
        return relationships


class RelationshipDeduplicator:
    """Deduplicate and normalize extracted relationships."""
    
    def deduplicate(
        self,
        relationships: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate relationships.
        
        Args:
            relationships: List of relationship dictionaries
            
        Returns:
            Deduplicated list of relationships
        """
        if not relationships:
            return []
        
        # Create unique key for each relationship
        relationship_map = {}
        
        for rel in relationships:
            key = (
                rel['source'].lower().strip(),
                rel['target'].lower().strip(),
                rel.get('type', 'UNKNOWN')
            )
            
            if key not in relationship_map:
                relationship_map[key] = rel
            else:
                # Keep relationship with higher confidence
                existing = relationship_map[key]
                if rel.get('confidence', 0) > existing.get('confidence', 0):
                    relationship_map[key] = rel
        
        return list(relationship_map.values())


def create_relationship_extractor(
    method: str = "cooccurrence",
    **kwargs
) -> RelationshipExtractor:
    """Factory function to create relationship extractor.
    
    Args:
        method: Extraction method ('cooccurrence', 'pattern', or 'csv')
        **kwargs: Additional arguments for the extractor
        
    Returns:
        RelationshipExtractor instance
    """
    if method == "cooccurrence":
        return CooccurrenceRelationshipExtractor(
            window_size=kwargs.get("window_size", 100),
            min_confidence=kwargs.get("min_confidence", 0.5)
        )
    elif method == "pattern":
        return PatternRelationshipExtractor(
            patterns=kwargs.get("patterns")
        )
    elif method == "csv":
        return CSVRelationshipExtractor(
            key_column=kwargs.get("key_column"),
            relationship_columns=kwargs.get("relationship_columns")
        )
    else:
        raise ValueError(f"Unknown relationship extraction method: {method}")
