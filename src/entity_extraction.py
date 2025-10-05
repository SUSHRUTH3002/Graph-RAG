"""Entity extraction from text chunks using various methods."""

from typing import List, Dict, Any, Optional, Set, Tuple
from abc import ABC, abstractmethod
import re
from collections import Counter
import os
import json
from pathlib import Path
from config import (
    OUTPUT_FILE_PATH_FOR_ENTITY_EXTRACTION,
    OUTPUT_FILE_PATH_FOR_ENTITY_SUMMARY
)


class EntityExtractor(ABC):
    """Abstract base class for entity extractors."""
    
    @abstractmethod
    def extract_entities(
        self, 
        text: str, 
        chunk_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Extract entities from text.
        
        Args:
            text: Input text
            chunk_metadata: Optional metadata from the chunk
            
        Returns:
            List of entity dictionaries with type, value, and metadata
        """
        pass


class KeywordEntityExtractor(EntityExtractor):
    """Simple keyword-based entity extractor using capitalization and patterns."""
    
    def __init__(
        self,
        min_entity_length: int = 2,
        max_entity_length: int = 50,
        extract_numbers: bool = True,
        extract_emails: bool = True,
        extract_urls: bool = True
    ):
        """Initialize keyword entity extractor.
        
        Args:
            min_entity_length: Minimum length for entities
            max_entity_length: Maximum length for entities
            extract_numbers: Whether to extract numeric values
            extract_emails: Whether to extract email addresses
            extract_urls: Whether to extract URLs
        """
        self.min_entity_length = min_entity_length
        self.max_entity_length = max_entity_length
        self.extract_numbers = extract_numbers
        self.extract_emails = extract_emails
        self.extract_urls = extract_urls
        
        # Common words to filter out
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their'
        }
    
    def extract_entities(
        self, 
        text: str, 
        chunk_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Extract entities from text using keyword patterns.
        
        Args:
            text: Input text
            chunk_metadata: Optional metadata from the chunk
            
        Returns:
            List of entity dictionaries
        """
        entities = []
        chunk_metadata = chunk_metadata or {}
        
        # Extract capitalized phrases (potential named entities)
        capitalized_entities = self._extract_capitalized_phrases(text)
        entities.extend(capitalized_entities)
        
        # Extract emails
        if self.extract_emails:
            email_entities = self._extract_emails(text)
            entities.extend(email_entities)
        
        # Extract URLs
        if self.extract_urls:
            url_entities = self._extract_urls(text)
            entities.extend(url_entities)
        
        # Extract numbers
        if self.extract_numbers:
            number_entities = self._extract_numbers(text)
            entities.extend(number_entities)
        
        # Add chunk metadata to all entities
        for entity in entities:
            entity['chunk_id'] = chunk_metadata.get('chunk_id')
            entity['source'] = chunk_metadata.get('source')
        
        return entities
    
    def _extract_capitalized_phrases(self, text: str) -> List[Dict[str, Any]]:
        """Extract capitalized phrases as potential named entities."""
        entities = []
        
        # Pattern for capitalized words
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.finditer(pattern, text)
        
        for match in matches:
            entity_text = match.group()
            
            # Filter by length and stopwords
            if (self.min_entity_length <= len(entity_text) <= self.max_entity_length and
                entity_text.lower() not in self.stopwords):
                
                entities.append({
                    'text': entity_text,
                    'type': 'NAMED_ENTITY',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.7
                })
        
        return entities
    
    def _extract_emails(self, text: str) -> List[Dict[str, Any]]:
        """Extract email addresses."""
        entities = []
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.finditer(pattern, text)
        
        for match in matches:
            entities.append({
                'text': match.group(),
                'type': 'EMAIL',
                'start': match.start(),
                'end': match.end(),
                'confidence': 1.0
            })
        
        return entities
    
    def _extract_urls(self, text: str) -> List[Dict[str, Any]]:
        """Extract URLs."""
        entities = []
        pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_\+.~#?&/=]*'
        matches = re.finditer(pattern, text)
        
        for match in matches:
            entities.append({
                'text': match.group(),
                'type': 'URL',
                'start': match.start(),
                'end': match.end(),
                'confidence': 1.0
            })
        
        return entities
    
    def _extract_numbers(self, text: str) -> List[Dict[str, Any]]:
        """Extract numeric values."""
        entities = []
        pattern = r'\b\d+(?:\.\d+)?\b'
        matches = re.finditer(pattern, text)
        
        for match in matches:
            entities.append({
                'text': match.group(),
                'type': 'NUMBER',
                'start': match.start(),
                'end': match.end(),
                'confidence': 1.0
            })
        
        return entities


class LLMEntityExtractor(EntityExtractor):
    """LLM-based entity extractor using language models."""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        entity_types: Optional[List[str]] = None
    ):
        """Initialize LLM entity extractor.
        
        Args:
            model_name: Name of the LLM model
            api_key: API key for the LLM service
            entity_types: List of entity types to extract
        """
        self.model_name = model_name
        self.api_key = api_key
        self.entity_types = entity_types or [
            'PERSON', 'ORGANIZATION', 'LOCATION', 'DATE', 
            'PRODUCT', 'EVENT', 'CONCEPT'
        ]
    
    def extract_entities(
        self, 
        text: str, 
        chunk_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Extract entities using LLM.
        
        Args:
            text: Input text
            chunk_metadata: Optional metadata from the chunk
            
        Returns:
            List of entity dictionaries
        """
        # Placeholder for LLM implementation
        # In production, this would call an LLM API
        entities = []
        chunk_metadata = chunk_metadata or {}
        
        # TODO: Implement LLM-based extraction
        # For now, return empty list
        return entities


class GeminiEntityExtractor(EntityExtractor):
    """Gemini LLM-based entity extractor using Google's Gemini API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
        entity_types: Optional[List[str]] = None,
        temperature: float = 0.1
    ):
        """Initialize Gemini entity extractor.
        
        Args:
            api_key: Google API key for Gemini (defaults to GEMINI_API_KEY env var)
            model_name: Gemini model name
            entity_types: List of entity types to extract
            temperature: Model temperature for consistency
        """
        try:
            import google.generativeai as genai
            self.genai = genai
        except ImportError:
            raise ImportError(
                "google-generativeai package required. "
                "Install with: pip install google-generativeai"
            )
        
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.entity_types = entity_types or [
            'ORGANIZATION', 'PRODUCT', 'BRAND', 'PERSON', 'LOCATION',
            'CONCEPT', 'TECHNOLOGY', 'METRIC', 'DEPARTMENT', 'EVENT'
        ]
        
        # Initialize the model
        self.model = self.genai.GenerativeModel(model_name)
    
    def extract_entities(
        self, 
        text: str, 
        chunk_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Extract entities using Gemini LLM.
        
        Args:
            text: Input text
            chunk_metadata: Optional metadata from the chunk
            
        Returns:
            List of entity dictionaries
        """
        if not text or not text.strip():
            return []
        
        chunk_metadata = chunk_metadata or {}
        
        # Build the prompt
        prompt = self._build_extraction_prompt(text, chunk_metadata)
        
        try:
            # Call Gemini API
            response = self.model.generate_content(
                prompt,
                generation_config=self.genai.types.GenerationConfig(
                    temperature=self.temperature,
                )
            )
            
            # Parse the response
            entities = self._parse_gemini_response(response.text, chunk_metadata)
            return entities
            
        except Exception as e:
            print(f"Warning: Gemini extraction failed: {e}")
            return []
    
    def _build_extraction_prompt(
        self, 
        text: str, 
        chunk_metadata: Dict[str, Any]
    ) -> str:
        """Build extraction prompt for Gemini."""
        entity_types_str = ", ".join(self.entity_types)
        
        prompt = f"""Extract entities from the following text. Identify entities of these types: {entity_types_str}

For each entity, provide:
1. The entity text (exact mention from the text)
2. The entity type (from the list above)
3. A confidence score (0.0 to 1.0)

Text to analyze:
{text}

Return the results in this exact format (one entity per line):
ENTITY: <text> | TYPE: <type> | CONFIDENCE: <score>

Examples:
ENTITY: PT Garudafood Putra Putri Jaya Tbk | TYPE: ORGANIZATION | CONFIDENCE: 1.0
ENTITY: ERP | TYPE: TECHNOLOGY | CONFIDENCE: 0.95
ENTITY: Internal Audit Unit | TYPE: DEPARTMENT | CONFIDENCE: 1.0

Now extract entities from the text above:"""
        
        return prompt
    
    def _parse_gemini_response(
        self, 
        response_text: str, 
        chunk_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Parse Gemini response into entity dictionaries."""
        entities = []
        
        # Parse each line of the response
        lines = response_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or not line.startswith('ENTITY:'):
                continue
            
            try:
                # Parse format: ENTITY: text | TYPE: type | CONFIDENCE: score
                parts = line.split('|')
                if len(parts) < 3:
                    continue
                
                entity_text = parts[0].replace('ENTITY:', '').strip()
                entity_type = parts[1].replace('TYPE:', '').strip()
                confidence_str = parts[2].replace('CONFIDENCE:', '').strip()
                confidence = float(confidence_str)
                
                entity = {
                    'text': entity_text,
                    'type': entity_type.upper(),
                    'confidence': confidence,
                    'chunk_id': chunk_metadata.get('chunk_id'),
                    'source': chunk_metadata.get('source'),
                    'extraction_method': 'gemini_llm'
                }
                
                # Add additional metadata if available
                if 'year' in chunk_metadata:
                    entity['year'] = chunk_metadata.get('year')
                if 'section' in chunk_metadata:
                    entity['section'] = chunk_metadata.get('section')
                
                entities.append(entity)
                
            except Exception as e:
                print(f"Warning: Failed to parse entity line: {line} - {e}")
                continue
        
        return entities


class CSVEntityExtractor(EntityExtractor):
    """CSV-specific entity extractor that treats each column value as an entity."""
    
    def __init__(
        self, 
        key_columns: Optional[List[str]] = None,
        tag_columns: Optional[List[str]] = None,
        entity_type_column: Optional[str] = None,
        use_llm_for_evidence: bool = False,
        llm_extractor: Optional[EntityExtractor] = None
    ):
        """Initialize CSV entity extractor.
        
        Args:
            key_columns: List of column names to treat as key entities
            tag_columns: Columns containing comma-separated tags
            entity_type_column: Column that specifies the entity type
            use_llm_for_evidence: Whether to use LLM for evidence_note extraction
            llm_extractor: LLM extractor instance for evidence notes
        """
        self.key_columns = key_columns or []
        self.tag_columns = tag_columns or []
        self.entity_type_column = entity_type_column
        self.use_llm_for_evidence = use_llm_for_evidence
        self.llm_extractor = llm_extractor
    
    def extract_entities(
        self, 
        text: str, 
        chunk_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Extract entities from CSV row data.
        
        Args:
            text: Input text (not used, uses chunk_metadata['row_data'])
            chunk_metadata: Must contain 'row_data' for CSV chunks
            
        Returns:
            List of entity dictionaries
        """
        entities = []
        chunk_metadata = chunk_metadata or {}
        
        # Check if this is a CSV row chunk
        if chunk_metadata.get('chunk_type') != 'csv_row':
            return entities
        
        row_data = chunk_metadata.get('row_data', {})
        columns = chunk_metadata.get('columns', [])
        filename = chunk_metadata.get('filename', '')
        
        # Detect which CSV file we're processing based on columns
        if self._is_subjective_sections_csv(columns):
            entities = self._extract_from_subjective_sections(row_data, chunk_metadata)
        elif self._is_entity_relations_csv(columns):
            entities = self._extract_from_entity_relations(row_data, chunk_metadata)
        else:
            # Generic extraction for unknown CSV structures
            entities = self._extract_generic(row_data, chunk_metadata)
        
        return entities
    
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
        chunk_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract entities from subjective_sections_for_rag.csv format."""
        entities = []
        
        # Extract section as entity (organizational structure)
        if 'section' in row_data and row_data['section']:
            entities.append({
                'text': str(row_data['section']),
                'type': 'SECTION',
                'column': 'section',
                'is_key': True,
                'confidence': 1.0,
                'chunk_id': chunk_metadata.get('chunk_id'),
                'source': chunk_metadata.get('source'),
                'year': row_data.get('year')
            })
        
        # Extract subsection as entity
        if 'subsection' in row_data and row_data['subsection']:
            entities.append({
                'text': str(row_data['subsection']),
                'type': 'SUBSECTION',
                'column': 'subsection',
                'is_key': True,
                'confidence': 1.0,
                'chunk_id': chunk_metadata.get('chunk_id'),
                'source': chunk_metadata.get('source'),
                'parent_section': row_data.get('section'),
                'year': row_data.get('year')
            })
        
        # Extract perspective as concept entity
        if 'perspective' in row_data and row_data['perspective']:
            entities.append({
                'text': str(row_data['perspective']),
                'type': 'PERSPECTIVE',
                'column': 'perspective',
                'is_key': False,
                'confidence': 0.9,
                'chunk_id': chunk_metadata.get('chunk_id'),
                'source': chunk_metadata.get('source'),
                'section': row_data.get('section'),
                'year': row_data.get('year')
            })
        
        # Extract year as temporal entity
        if 'year' in row_data and row_data['year']:
            year_str = str(row_data['year']).replace('.0', '')
            entities.append({
                'text': year_str,
                'type': 'YEAR',
                'column': 'year',
                'is_key': False,
                'confidence': 1.0,
                'chunk_id': chunk_metadata.get('chunk_id'),
                'source': chunk_metadata.get('source')
            })
        
        # Extract tags as individual concept entities
        if 'tags' in row_data and row_data['tags']:
            tags = str(row_data['tags']).split(',')
            for tag in tags:
                tag = tag.strip()
                if tag:
                    entities.append({
                        'text': tag,
                        'type': 'TAG',
                        'column': 'tags',
                        'is_key': False,
                        'confidence': 0.8,
                        'chunk_id': chunk_metadata.get('chunk_id'),
                        'source': chunk_metadata.get('source'),
                        'section': row_data.get('section'),
                        'year': row_data.get('year')
                    })
        
        # Extract key terms from content (simple keyword extraction)
        if 'content' in row_data and row_data['content']:
            content_entities = self._extract_content_entities(
                str(row_data['content']), 
                chunk_metadata,
                row_data
            )
            entities.extend(content_entities)
        
        return entities
    
    def _extract_from_entity_relations(
        self, 
        row_data: Dict[str, Any],
        chunk_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract entities from entity_relations_graph.csv format."""
        entities = []
        
        # Extract source entity - classify based on content
        if 'source' in row_data and row_data['source']:
            source_text = str(row_data['source'])
            source_type = self._classify_entity_type(source_text, 'source')
            
            entities.append({
                'text': source_text,
                'type': source_type,
                'column': 'source',
                'is_key': True,
                'confidence': 1.0,
                'chunk_id': chunk_metadata.get('chunk_id'),
                'source': chunk_metadata.get('source'),
                'section': row_data.get('section'),
                'year': row_data.get('year'),
                'relation': row_data.get('relation')
            })
        
        # Extract target entity with type from target_type column
        if 'target' in row_data and row_data['target']:
            target_text = str(row_data['target'])
            target_type_raw = row_data.get('target_type', 'entity')
            target_type = self._map_target_type(target_type_raw, target_text)
            
            entities.append({
                'text': target_text,
                'type': target_type,
                'column': 'target',
                'is_key': True,
                'confidence': 1.0,
                'chunk_id': chunk_metadata.get('chunk_id'),
                'source': chunk_metadata.get('source'),
                'entity_category': target_type_raw,
                'section': row_data.get('section'),
                'year': row_data.get('year'),
                'relation': row_data.get('relation')
            })
        
        # Extract section as context entity
        if 'section' in row_data and row_data['section']:
            entities.append({
                'text': str(row_data['section']),
                'type': 'SECTION',
                'column': 'section',
                'is_key': False,
                'confidence': 0.9,
                'chunk_id': chunk_metadata.get('chunk_id'),
                'source': chunk_metadata.get('source'),
                'year': row_data.get('year')
            })
        
        # Extract year as temporal entity
        if 'year' in row_data and row_data['year']:
            year_str = str(row_data['year']).replace('.0', '')
            entities.append({
                'text': year_str,
                'type': 'YEAR',
                'column': 'year',
                'is_key': False,
                'confidence': 1.0,
                'chunk_id': chunk_metadata.get('chunk_id'),
                'source': chunk_metadata.get('source')
            })
        
        # Extract value and unit as metric entity
        if 'value' in row_data and row_data['value'] and str(row_data['value']).strip():
            try:
                value = float(row_data['value'])
                unit = row_data.get('unit', '')
                
                # Format the metric text
                if unit:
                    metric_text = f"{value} {unit}"
                else:
                    metric_text = str(value)
                
                entities.append({
                    'text': metric_text,
                    'type': 'METRIC',
                    'column': 'value',
                    'is_key': False,
                    'confidence': 1.0,
                    'chunk_id': chunk_metadata.get('chunk_id'),
                    'source': chunk_metadata.get('source'),
                    'value': value,
                    'unit': unit,
                    'year': row_data.get('year'),
                    'section': row_data.get('section')
                })
            except (ValueError, TypeError):
                pass
        
        # Extract entities from evidence_note using LLM if enabled
        if 'evidence_note' in row_data and row_data['evidence_note']:
            evidence_text = str(row_data['evidence_note'])
            
            if self.use_llm_for_evidence and self.llm_extractor:
                # Use LLM to extract entities from evidence
                llm_entities = self.llm_extractor.extract_entities(
                    evidence_text,
                    chunk_metadata={
                        **chunk_metadata,
                        'year': row_data.get('year'),
                        'section': row_data.get('section')
                    }
                )
                
                # Mark as evidence-derived
                for entity in llm_entities:
                    entity['column'] = 'evidence_note'
                    entity['is_key'] = False
                    entity['derived_from'] = 'evidence'
                
                entities.extend(llm_entities)
            else:
                # Simple pattern-based extraction from evidence
                evidence_entities = self._extract_from_evidence(
                    evidence_text,
                    chunk_metadata,
                    row_data
                )
                entities.extend(evidence_entities)
        
        return entities
    
    def _classify_entity_type(self, text: str, column: str) -> str:
        """Classify entity type based on text patterns."""
        text_lower = text.lower()
        
        # Organization patterns
        if any(marker in text for marker in ['PT ', 'Tbk', 'Corporation', 'Inc', 'Ltd']):
            return 'ORGANIZATION'
        
        # Product/Brand patterns (capitalized single or two words)
        if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?$', text):
            if column == 'source':
                return 'ORGANIZATION'
            return 'PRODUCT'
        
        # Technology/Concept patterns
        if text.isupper() and len(text) <= 5:  # Acronyms like ERP, RPA, AI
            return 'TECHNOLOGY'
        
        # Location patterns
        if text in ['Domestic', 'Export', 'Indonesia']:
            return 'LOCATION'
        
        # Department patterns
        if 'unit' in text_lower or 'committee' in text_lower or 'department' in text_lower:
            return 'DEPARTMENT'
        
        # Default
        return 'ENTITY'
    
    def _map_target_type(self, target_type_raw: str, target_text: str) -> str:
        """Map target_type column value to standardized entity type."""
        target_type_raw = target_type_raw.lower()
        
        # Direct mappings
        type_mapping = {
            'entity': self._classify_entity_type(target_text, 'target'),
            'concept': 'CONCEPT',
            'data': 'METRIC'
        }
        
        return type_mapping.get(target_type_raw, 'ENTITY')
    
    def _extract_from_evidence(
        self,
        evidence_text: str,
        chunk_metadata: Dict[str, Any],
        row_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract entities from evidence_note using pattern matching."""
        entities = []
        
        # Extract brand mentions (capitalized words)
        brand_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b'
        brands = re.finditer(brand_pattern, evidence_text)
        
        seen_brands = set()
        for match in brands:
            brand = match.group().strip()
            if len(brand) > 2 and brand not in seen_brands:
                seen_brands.add(brand)
                entities.append({
                    'text': brand,
                    'type': 'BRAND',
                    'column': 'evidence_note',
                    'is_key': False,
                    'confidence': 0.7,
                    'chunk_id': chunk_metadata.get('chunk_id'),
                    'source': chunk_metadata.get('source'),
                    'section': row_data.get('section'),
                    'year': row_data.get('year'),
                    'derived_from': 'evidence'
                })
        
        # Extract percentages
        percentage_pattern = r'~?\d+(?:\.\d+)?%'
        percentages = re.finditer(percentage_pattern, evidence_text)
        
        for match in percentages:
            entities.append({
                'text': match.group(),
                'type': 'PERCENTAGE',
                'column': 'evidence_note',
                'is_key': False,
                'confidence': 1.0,
                'chunk_id': chunk_metadata.get('chunk_id'),
                'source': chunk_metadata.get('source'),
                'section': row_data.get('section'),
                'year': row_data.get('year'),
                'derived_from': 'evidence'
            })
        
        # Extract numbers with rank indicators (#1, #2, etc.)
        rank_pattern = r'#\d+'
        ranks = re.finditer(rank_pattern, evidence_text)
        
        for match in ranks:
            entities.append({
                'text': match.group(),
                'type': 'RANK',
                'column': 'evidence_note',
                'is_key': False,
                'confidence': 1.0,
                'chunk_id': chunk_metadata.get('chunk_id'),
                'source': chunk_metadata.get('source'),
                'section': row_data.get('section'),
                'year': row_data.get('year'),
                'derived_from': 'evidence'
            })
        
        return entities
    
    def _extract_generic(
        self, 
        row_data: Dict[str, Any],
        chunk_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generic extraction for unknown CSV structures."""
        entities = []
        columns = chunk_metadata.get('columns', [])
        
        for column, value in row_data.items():
            # Determine if this is a key column
            is_key = column in self.key_columns if self.key_columns else (columns.index(column) == 0)
            
            entity = {
                'text': str(value),
                'type': f'CSV_{column.upper()}',
                'column': column,
                'is_key': is_key,
                'confidence': 1.0,
                'chunk_id': chunk_metadata.get('chunk_id'),
                'source': chunk_metadata.get('source')
            }
            
            entities.append(entity)
        
        return entities


class EntityDeduplicator:
    """Deduplicate and normalize extracted entities."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        """Initialize entity deduplicator.
        
        Args:
            similarity_threshold: Threshold for considering entities as duplicates
        """
        self.similarity_threshold = similarity_threshold
    
    def deduplicate(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate entities based on text similarity.
        
        Args:
            entities: List of entity dictionaries
            
        Returns:
            Deduplicated list of entities
        """
        if not entities:
            return []
        
        # Group entities by type
        entities_by_type = {}
        for entity in entities:
            entity_type = entity.get('type', 'UNKNOWN')
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        
        # Deduplicate within each type
        deduplicated = []
        for entity_type, type_entities in entities_by_type.items():
            deduplicated.extend(self._deduplicate_group(type_entities))
        
        return deduplicated
    
    def _deduplicate_group(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate a group of entities of the same type."""
        if not entities:
            return []
        
        # Use text as key for simple deduplication
        entity_map = {}
        
        for entity in entities:
            text = entity['text'].lower().strip()
            
            if text not in entity_map:
                entity_map[text] = entity
            else:
                # Merge metadata (keep highest confidence, combine sources)
                existing = entity_map[text]
                existing['confidence'] = max(
                    existing.get('confidence', 0),
                    entity.get('confidence', 0)
                )
        
        return list(entity_map.values())


def create_entity_extractor(
    method: str = "keyword",
    **kwargs
) -> EntityExtractor:
    """Factory function to create entity extractor.
    
    Args:
        method: Extraction method ('keyword', 'llm', 'gemini', or 'csv')
        **kwargs: Additional arguments for the extractor
        
    Returns:
        EntityExtractor instance
    """
    if method == "keyword":
        return KeywordEntityExtractor(
            min_entity_length=kwargs.get("min_entity_length", 2),
            max_entity_length=kwargs.get("max_entity_length", 50),
            extract_numbers=kwargs.get("extract_numbers", True),
            extract_emails=kwargs.get("extract_emails", True),
            extract_urls=kwargs.get("extract_urls", True)
        )
    elif method == "llm":
        return LLMEntityExtractor(
            model_name=kwargs.get("model_name", "gpt-3.5-turbo"),
            api_key=kwargs.get("api_key"),
            entity_types=kwargs.get("entity_types")
        )
    elif method == "gemini":
        return GeminiEntityExtractor(
            api_key=kwargs.get("api_key"),
            model_name=kwargs.get("model_name", "gemini-1.5-flash"),
            entity_types=kwargs.get("entity_types"),
            temperature=kwargs.get("temperature", 0.1)
        )
    elif method == "csv":
        return CSVEntityExtractor(
            key_columns=kwargs.get("key_columns"),
            tag_columns=kwargs.get("tag_columns"),
            entity_type_column=kwargs.get("entity_type_column"),
            use_llm_for_evidence=kwargs.get("use_llm_for_evidence", False),
            llm_extractor=kwargs.get("llm_extractor")
        )
    else:
        raise ValueError(f"Unknown entity extraction method: {method}")

def save_entity_summary(
	entities: List[Dict[str, Any]],
	raw_entities: List[Dict[str, Any]],
	output_dir: Optional[str] = "output",
	entities_file: Optional[str] = None
) -> Dict[str, Any]:
	"""Save extracted entities JSON and a human-readable summary file.
	
	Args:
		entities: Deduplicated entity list
		raw_entities: Raw entity list prior to deduplication
		output_dir: Directory to write outputs (Path or str)
		entities_file: Optional path for the entities JSON file (overrides default)
	
	Returns:
		A small dict with paths written for convenience.
	"""
	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	# Entities JSON
	if entities_file:
		entities_path = Path(entities_file)
		# if a relative filename provided, place inside output_dir
		# if not entities_path.is_absolute():
		# 	entities_path = output_path / entities_path
	else:
		entities_path = OUTPUT_FILE_PATH_FOR_ENTITY_EXTRACTION
	
	with open(entities_path, "w", encoding="utf-8") as f:
		json.dump(entities, f, indent=2, ensure_ascii=False)
	
	# Compute distribution
	entity_type_counts: Dict[str, int] = {}
	for ent in entities:
		et = ent.get("type", "UNKNOWN")
		entity_type_counts[et] = entity_type_counts.get(et, 0) + 1
	
	# Summary file
	summary_path = OUTPUT_FILE_PATH_FOR_ENTITY_SUMMARY
	with open(summary_path, "w", encoding="utf-8") as f:
		f.write("Entity Extraction Summary\n")
		f.write("=" * 80 + "\n\n")
		f.write(f"Total unique entities: {len(entities)}\n")
		f.write(f"Total raw entities (before deduplication): {len(raw_entities)}\n\n")
		f.write("Entity Distribution by Type:\n")
		f.write("-" * 40 + "\n")
		for entity_type, count in sorted(entity_type_counts.items(), key=lambda x: x[1], reverse=True):
			f.write(f"{entity_type}: {count}\n")
		f.write("\n")
		f.write("Sample Entities by Type:\n")
		f.write("-" * 40 + "\n")
		
		# group samples
		entities_by_type: Dict[str, List[Dict[str, Any]]] = {}
		for ent in entities:
			et = ent.get("type", "UNKNOWN")
			entities_by_type.setdefault(et, []).append(ent)
		
		for entity_type, ents in sorted(entities_by_type.items()):
			f.write(f"\n{entity_type}:\n")
			for ent in ents[:5]:
				f.write(f"  - {ent.get('text')}\n")
	
	return {
		"entities_path": str(entities_path),
		"summary_path": str(summary_path)
	}
