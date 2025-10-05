"""Answer generation module using LLM (Gemini) with graph context."""

from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass
import logging
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

if TYPE_CHECKING:
    from .context_assembler import AssembledContext

logger = logging.getLogger(__name__)


@dataclass
class GeneratedAnswer:
    """Container for generated answer."""
    answer: str
    sources: List[str]
    confidence: float
    metadata: Dict[str, Any]


class AnswerGenerator:
    """Generate answers using LLM with graph-retrieved context."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.3,
        max_tokens: int = 1024
    ):
        """Initialize answer generator.
        
        Args:
            api_key: Gemini API key (uses env var if None)
            model_name: Gemini model name
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY environment variable.")
        
        # Initialize Gemini client
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"Initialized Gemini model: {model_name}")
        except ImportError:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini: {e}")
    
    def generate_answer(
        self,
        query: str,
        context: 'AssembledContext',
        include_sources: bool = True
    ) -> GeneratedAnswer:
        """Generate answer using LLM with retrieved context.
        
        Args:
            query: User query
            context: Assembled context from retrieval
            include_sources: Include source citations
            
        Returns:
            GeneratedAnswer object
        """
        logger.info(f"Generating answer for query: {query}")
        
        # Build prompt
        prompt = self._build_prompt(query, context, include_sources)
        
        # Generate response
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': self.temperature,
                    'max_output_tokens': self.max_tokens,
                }
            )
            
            answer_text = response.text
            
            # Extract confidence (if available)
            confidence = self._estimate_confidence(response, context)
            
            # Format sources
            sources = context.sources if include_sources else []
            
            return GeneratedAnswer(
                answer=answer_text,
                sources=sources,
                confidence=confidence,
                metadata={
                    'model': self.model_name,
                    'temperature': self.temperature,
                    'context_tokens': context.token_count,
                    'strategy': context.metadata.get('strategy'),
                    'entity_count': context.metadata.get('entity_count', 0),
                    'relationship_count': context.metadata.get('relationship_count', 0)
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return GeneratedAnswer(
                answer=f"Error generating answer: {str(e)}",
                sources=[],
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def _build_prompt(
        self,
        query: str,
        context: 'AssembledContext',
        include_sources: bool
    ) -> str:
        """Build prompt for LLM.
        
        Args:
            query: User query
            context: Assembled context
            include_sources: Include source citations
            
        Returns:
            Formatted prompt string
        """
        system_instructions = """You are an AI assistant that answers questions based on information from a knowledge graph.

Your task is to:
1. Analyze the provided context from the knowledge graph
2. Answer the user's question accurately using ONLY information from the context
3. If the context doesn't contain enough information, clearly state what is missing
4. Provide specific details and evidence when available
5. Cite relevant entities and relationships from the context
6. Be concise but comprehensive

Important guidelines:
- DO NOT make up information not present in the context
- If uncertain, express the level of confidence
- Reference specific entities, relationships, and values from the context
- Maintain factual accuracy"""

        if include_sources:
            system_instructions += "\n- Include source references when mentioning facts"
        
        prompt_parts = [
            system_instructions,
            "\n\n" + "="*80,
            "\n\nCONTEXT FROM KNOWLEDGE GRAPH:",
            "\n" + "="*80,
            f"\n{context.context_text}",
            "\n\n" + "="*80,
            "\n\nUSER QUESTION:",
            "\n" + "="*80,
            f"\n{query}",
            "\n\n" + "="*80,
            "\n\nYOUR ANSWER (based strictly on the context above):",
            "\n" + "="*80 + "\n"
        ]
        
        return "".join(prompt_parts)
    
    def _estimate_confidence(
        self,
        response,
        context: 'AssembledContext'
    ) -> float:
        """Estimate confidence in the generated answer.
        
        Args:
            response: LLM response object
            context: Assembled context
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence on context quality
        base_confidence = 0.5
        
        # Boost based on entity count
        entity_count = context.metadata.get('entity_count', 0)
        if entity_count > 5:
            base_confidence += 0.2
        elif entity_count > 2:
            base_confidence += 0.1
        
        # Boost based on relationship count
        rel_count = context.metadata.get('relationship_count', 0)
        if rel_count > 5:
            base_confidence += 0.2
        elif rel_count > 2:
            base_confidence += 0.1
        
        # Cap at 0.95 (never 100% certain)
        return min(base_confidence, 0.95)
    
    def generate_with_citations(
        self,
        query: str,
        context: 'AssembledContext'
    ) -> GeneratedAnswer:
        """Generate answer with inline citations.
        
        Args:
            query: User query
            context: Assembled context
            
        Returns:
            GeneratedAnswer with citations
        """
        # Modify prompt to request citations
        system_instructions = """You are an AI assistant that answers questions with precise citations.

When providing information:
1. Add [Entity: name] citations when mentioning entities
2. Add [Source: section] when referencing specific sections
3. Number your citations [1], [2], etc. and provide a reference list at the end

Example format:
"Garudafood [1] reported revenue growth [2] in 2024..."

References:
[1] Entity: Garudafood (ORGANIZATION)
[2] Source: Financial Performance section, 2024"""

        prompt_parts = [
            system_instructions,
            "\n\nCONTEXT:",
            f"\n{context.context_text}",
            "\n\nQUESTION:",
            f"\n{query}",
            "\n\nANSWER WITH CITATIONS:\n"
        ]
        
        prompt = "".join(prompt_parts)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': self.temperature,
                    'max_output_tokens': self.max_tokens,
                }
            )
            
            return GeneratedAnswer(
                answer=response.text,
                sources=context.sources,
                confidence=self._estimate_confidence(response, context),
                metadata={
                    'model': self.model_name,
                    'with_citations': True,
                    'strategy': context.metadata.get('strategy')
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating answer with citations: {e}")
            return GeneratedAnswer(
                answer=f"Error: {str(e)}",
                sources=[],
                confidence=0.0,
                metadata={'error': str(e)}
            )


def create_answer_generator(
    api_key: Optional[str] = None,
    model_name: str = "gemini-1.5-flash",
    temperature: float = 0.3,
    max_tokens: int = 1024
) -> AnswerGenerator:
    """Factory function to create answer generator.
    
    Args:
        api_key: Gemini API key
        model_name: Model name
        temperature: Generation temperature
        max_tokens: Max response tokens
        
    Returns:
        AnswerGenerator instance
    """
    return AnswerGenerator(api_key, model_name, temperature, max_tokens)
