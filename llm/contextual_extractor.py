"""
Contextual extractor that uses full document context for better address inference.
Implements enhanced LLM prompting that considers the entire document context.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import json
from .standardized_schemas import (
    ContextualExtractionRequest, 
    StandardizedExtraction,
    QualityAssessment
)

logger = logging.getLogger(__name__)


class ContextualExtractor:
    """
    Enhanced extractor that uses document context for better address inference
    """
    
    def __init__(self, llm_client, config: Optional[Dict] = None):
        """
        Initialize contextual extractor
        
        Args:
            llm_client: The underlying LLM client
            config: Configuration dictionary
        """
        self.llm_client = llm_client
        self.config = config or {}
        
        # Extraction settings
        self.context_size = self.config.get('context_size', 2000)
        self.min_confidence = self.config.get('min_confidence', 0.5)
        self.extraction_mode = self.config.get('extraction_mode', 'standard')
        
        # Document-level context storage
        self.document_context = ""
        self.previous_extractions = []
        self.document_summary = ""
        
        logger.info(f"Contextual extractor initialized with mode: {self.extraction_mode}")
    
    def process_document_with_context(self, text_chunks: List[str]) -> List[StandardizedExtraction]:
        """
        Process entire document maintaining context across chunks
        
        Args:
            text_chunks: List of text chunks from the document
            
        Returns:
            List of standardized extractions
        """
        logger.info(f"Processing document with {len(text_chunks)} chunks using contextual extraction")
        
        # Step 1: Create document context and summary
        self._build_document_context(text_chunks)
        
        # Step 2: Process each chunk with full context
        all_extractions = []
        for i, chunk in enumerate(text_chunks):
            logger.info(f"Processing chunk {i+1}/{len(text_chunks)} with context")
            
            chunk_extractions = self._extract_from_chunk_with_context(
                chunk_text=chunk,
                chunk_index=i,
                total_chunks=len(text_chunks)
            )
            
            # Update previous extractions for consistency
            self.previous_extractions.extend(chunk_extractions)
            all_extractions.extend(chunk_extractions)
            
            logger.info(f"Chunk {i+1}: Found {len(chunk_extractions)} extractions")
        
        # Step 3: Post-process for consistency and quality
        final_extractions = self._post_process_extractions(all_extractions)
        
        logger.info(f"Final processing: {len(final_extractions)} standardized extractions")
        return final_extractions
    
    def _build_document_context(self, text_chunks: List[str]) -> None:
        """Build overall document context and summary"""
        # Combine first few chunks for context
        full_text = " ".join(text_chunks)
        self.document_context = full_text[:self.context_size]
        
        # Create document summary using LLM
        self.document_summary = self._create_document_summary(self.document_context)
        
        logger.info(f"Document context built: {len(self.document_context)} chars")
        logger.debug(f"Document summary: {self.document_summary[:200]}...")
    
    def _create_document_summary(self, context: str) -> str:
        """Create a summary of the document to help with context"""
        summary_prompt = f"""
        Analiza este documento y crea un resumen breve enfocado en:
        1. Tipo de documento (informe, carta, estudio, etc.)
        2. Ubicaciones principales mencionadas
        3. Contexto geográfico general
        4. Entidades o organizaciones mencionadas
        
        DOCUMENTO:
        {context}
        
        RESUMEN (máximo 200 palabras):
        """
        
        try:
            if hasattr(self.llm_client, '_query_llm_with_fallback'):
                summary = self.llm_client._query_llm_with_fallback(summary_prompt)
                return summary[:500]  # Limit summary length
            else:
                return "Document context available"  # Fallback
        except Exception as e:
            logger.warning(f"Failed to create document summary: {e}")
            return "Document context available"
    
    def _extract_from_chunk_with_context(self, chunk_text: str, chunk_index: int, 
                                       total_chunks: int) -> List[StandardizedExtraction]:
        """Extract addresses from chunk using full document context"""
        
        # Create contextual prompt
        prompt = self._create_contextual_prompt(chunk_text, chunk_index, total_chunks)
        
        try:
            # Query LLM with contextual prompt
            if hasattr(self.llm_client, '_query_llm_with_fallback'):
                response = self.llm_client._query_llm_with_fallback(prompt)
            else:
                # Fallback to traditional method
                response = self.llm_client._extract_addresses_traditional(chunk_text)
                return self._convert_legacy_to_standardized(response)
            
            # Parse response into standardized format
            extractions = self._parse_contextual_response(response, chunk_text)
            
            return extractions
            
        except Exception as e:
            logger.error(f"Error in contextual extraction for chunk {chunk_index}: {e}")
            return []
    
    def _create_contextual_prompt(self, chunk_text: str, chunk_index: int, 
                                total_chunks: int) -> str:
        """Create enhanced prompt using document context"""
        
        # Previous extractions context
        prev_context = ""
        if self.previous_extractions:
            prev_addresses = [ext.inferred_address for ext in self.previous_extractions[-3:]]
            prev_context = f"\\nDirecciones ya encontradas: {', '.join(prev_addresses)}"
        
        prompt = f"""
        You are a geographic information extraction expert with advanced AI capabilities. Analyze this text and extract ALL geographic references with maximum precision.

        DOCUMENT CONTEXT:
        {self.document_summary}

        TEXT TO ANALYZE (Section {chunk_index + 1}/{total_chunks}):
        {chunk_text}

        TASK: Extract ALL geographic references (coordinates, addresses, place names) from this text with high precision.

        RESPOND IN STRICT JSON FORMAT:
        [
          {{
            "original_text": "exact text found",
            "inferred_address": "complete address inferred from context",
            "confidence_level": 0.85,
            "latitude_dd": 40.4168,
            "longitude_dd": -3.7038
          }}
        ]

        CRITICAL RULES FOR ADDRESS INFERENCE:
        1. ALWAYS prefer SPECIFIC locations over GENERAL ones
        2. When you see a street name, plaza, or specific place, use the MOST SPECIFIC address possible
        3. Only use city-level addresses (e.g., "Madrid, Spain") when no more specific location is available
        4. NEVER use generic descriptions like "Southern districts" or "Business district" - use actual street names
        5. Leverage your advanced understanding to infer precise locations from context
        6. Use your knowledge of Spanish geography to provide accurate addresses

        ADDRESS INFERENCE EXAMPLES:
        ✅ CORRECT (Specific):
        - "Calle de Atocha" → "Calle de Atocha, Madrid, Spain"
        - "Plaza Catalunya" → "Plaza Catalunya, Barcelona, Spain"
        - "Gran Vía 45" → "Gran Vía 45, Madrid, Spain"
        - "Paseo de la Castellana" → "Paseo de la Castellana, Madrid, Spain"

        ❌ INCORRECT (Too Generic):
        - "Calle de Atocha" → "Southern districts of Madrid, Spain"
        - "Plaza Catalunya" → "Barcelona, Spain"
        - "Gran Vía 45" → "Business district, Madrid, Spain"

        COORDINATE RULES:
        - Use exact coordinates when provided in the text
        - For specific streets/places in Madrid, use Madrid coordinates (40.4168, -3.7038)
        - For specific streets/places in Barcelona, use Barcelona coordinates (41.3851, 2.1734)
        - confidence_level must be between 0.0 and 1.0
        - Leverage your advanced capabilities to provide the most accurate coordinates possible
        - If no geographic references found, respond with: []
        """
        
        return prompt
    
    def _parse_contextual_response(self, response: str, chunk_text: str) -> List[StandardizedExtraction]:
        """Parse LLM response into standardized extractions"""
        # Log the raw response for debugging
        logger.info(f"Raw LLM response: {response}")
        logger.info(f"Response length: {len(response)}")
        
        try:
            # Try to extract JSON from response
            json_start = response.find('[')
            json_end = response.rfind(']')
            
            logger.info(f"JSON start: {json_start}, JSON end: {json_end}")
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end+1]
                data = json.loads(json_str)
                
                extractions = []
                for item in data:
                    try:
                        # Safely parse coordinates with fallback
                        lat_val = item.get('latitude_dd')
                        lon_val = item.get('longitude_dd')
                        
                        try:
                            latitude_dd = float(lat_val) if lat_val is not None else 0.0
                        except (ValueError, TypeError):
                            latitude_dd = 0.0
                        
                        try:
                            longitude_dd = float(lon_val) if lon_val is not None else 0.0
                        except (ValueError, TypeError):
                            longitude_dd = 0.0
                        
                        # Only include if we have valid coordinates or meaningful address
                        if latitude_dd != 0.0 or longitude_dd != 0.0 or item.get('inferred_address', '').strip():
                            extraction = StandardizedExtraction(
                                original_text=item.get('original_text', chunk_text[:100]),
                                inferred_address=item.get('inferred_address', 'Unknown Location'),
                                confidence_level=float(item.get('confidence_level', 0.5)),
                                latitude_dd=latitude_dd,
                                longitude_dd=longitude_dd
                            )
                        else:
                            continue  # Skip invalid extractions
                        
                        # Only include if meets minimum confidence
                        if extraction.confidence_level >= self.min_confidence:
                            extractions.append(extraction)
                        
                    except Exception as e:
                        logger.warning(f"Failed to parse extraction item: {e}")
                        continue
                
                return extractions
            
            else:
                logger.warning("No valid JSON found in LLM response")
                logger.warning(f"Response content: '{response}'")
                return []
                
        except Exception as e:
            logger.error(f"Failed to parse contextual response: {e}")
            return []
    
    def _convert_legacy_to_standardized(self, legacy_results: List[Dict]) -> List[StandardizedExtraction]:
        """Convert legacy extraction format to standardized format"""
        extractions = []
        
        for result in legacy_results:
            try:
                # Create inferred address from available components
                inferred_address = self._build_inferred_address(result)
                
                extraction = StandardizedExtraction(
                    original_text=result.get('original_text', ''),
                    inferred_address=inferred_address,
                    confidence_level=result.get('confidence', 0.5),
                    latitude_dd=result.get('latitude', 0.0),
                    longitude_dd=result.get('longitude', 0.0)
                )
                
                if extraction.confidence_level >= self.min_confidence:
                    extractions.append(extraction)
                    
            except Exception as e:
                logger.warning(f"Failed to convert legacy result: {e}")
                continue
        
        return extractions
    
    def _build_inferred_address(self, result: Dict) -> str:
        """Build inferred address from available components"""
        components = []
        
        # Add street and number
        if result.get('street'):
            street_part = result['street']
            if result.get('number'):
                street_part += f" {result['number']}"
            components.append(street_part)
        
        # Add city
        if result.get('city'):
            components.append(result['city'])
        
        # Add country
        if result.get('country'):
            components.append(result['country'])
        
        # Fallback to normalized address
        if not components and result.get('normalized_address'):
            return result['normalized_address']
        
        return ', '.join(components) if components else 'Unknown Location'
    
    def _post_process_extractions(self, extractions: List[StandardizedExtraction]) -> List[StandardizedExtraction]:
        """Post-process extractions for consistency and deduplication"""
        if not extractions:
            return []
        
        # Remove duplicates based on coordinates
        unique_extractions = []
        seen_coords = set()
        
        for extraction in extractions:
            coord_key = (round(extraction.latitude_dd, 5), round(extraction.longitude_dd, 5))
            
            if coord_key not in seen_coords:
                seen_coords.add(coord_key)
                unique_extractions.append(extraction)
            else:
                # If duplicate coordinates, keep the one with higher confidence
                existing_idx = None
                for i, existing in enumerate(unique_extractions):
                    existing_coords = (round(existing.latitude_dd, 5), round(existing.longitude_dd, 5))
                    if existing_coords == coord_key:
                        existing_idx = i
                        break
                
                if existing_idx is not None:
                    if extraction.confidence_level > unique_extractions[existing_idx].confidence_level:
                        unique_extractions[existing_idx] = extraction
        
        logger.info(f"Post-processing: {len(extractions)} -> {len(unique_extractions)} after deduplication")
        return unique_extractions
    
    def _query_llm_with_fallback(self, prompt: str) -> str:
        """Query LLM with fallback to different methods"""
        try:
            if hasattr(self.llm_client, '_query_ollama') and self.llm_client.llm_type == 'Ollama':
                return self.llm_client._query_ollama(prompt)
            elif hasattr(self.llm_client, '_query_openai') and self.llm_client.llm_type == 'OpenAI':
                return self.llm_client._query_openai(prompt)
            elif hasattr(self.llm_client, '_query_vllm') and self.llm_client.llm_type == 'vLLM':
                return self.llm_client._query_vllm(prompt)
            else:
                # Fallback to generic method
                return ""
        except Exception as e:
            logger.error(f"All LLM query methods failed: {e}")
            return ""
