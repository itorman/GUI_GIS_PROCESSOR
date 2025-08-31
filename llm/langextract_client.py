"""
Specialized client for using langextract with LLM services.
Provides efficient address and geographic information extraction using structured schemas.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Union
from langextract import extract

from .schemas import (
    AddressExtractionResult, 
    GeographicExtractionResult,
    Address,
    get_address_extraction_schema,
    get_geographic_extraction_schema
)

logger = logging.getLogger(__name__)


class LangextractClient:
    """
    Client for efficient address extraction using langextract.
    Replaces traditional prompt-based extraction with structured schema extraction.
    """
    
    def __init__(self, llm_client, max_retries: int = 3, timeout: int = 60):
        """
        Initialize the langextract client
        
        Args:
            llm_client: The underlying LLM client (Ollama, vLLM, OpenAI, etc.)
            max_retries: Maximum number of retry attempts
            timeout: Timeout for extraction operations
        """
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Cache schemas for efficiency
        self._address_schema = get_address_extraction_schema()
        self._geographic_schema = get_geographic_extraction_schema()
        
        logger.info("Langextract client initialized with structured schemas")
    
    def extract_addresses(self, text: str, language_hint: Optional[str] = None) -> List[Address]:
        """
        Extract addresses from text using langextract with structured schema
        
        Args:
            text: Text to process for address extraction
            language_hint: Optional language hint for better extraction
            
        Returns:
            List of extracted and validated addresses
        """
        if not text.strip():
            logger.debug("Empty text provided, returning empty result")
            return []
        
        start_time = time.time()
        
        try:
            # Use langextract with structured schema for address extraction
            # Note: langextract API may vary, using basic extraction for now
            try:
                result = extract(
                    text,
                    self._address_schema,
                    max_retries=self.max_retries
                )
            except TypeError:
                # Fallback to basic extraction if API signature doesn't match
                result = extract(
                    text,
                    self._address_schema
                )
            
            processing_time = time.time() - start_time
            
            if result and hasattr(result, 'addresses'):
                # Update processing metadata
                result.processing_time = processing_time
                result.text_chunk = text
                
                logger.info(f"Successfully extracted {len(result.addresses)} addresses in {processing_time:.2f}s")
                return result.addresses
            else:
                logger.warning("No addresses extracted from text chunk")
                return []
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in address extraction: {e} (took {processing_time:.2f}s)")
            
            # Fallback to traditional extraction if langextract fails
            return self._fallback_address_extraction(text)
    
    def extract_geographic_features(self, text: str) -> GeographicExtractionResult:
        """
        Extract all types of geographic features from text
        
        Args:
            text: Text to process for geographic feature extraction
            
        Returns:
            GeographicExtractionResult with all found features
        """
        if not text.strip():
            return GeographicExtractionResult(
                features=[],
                text_chunk=text,
                processing_time=0.0
            )
        
        start_time = time.time()
        
        try:
            # Use langextract with geographic schema
            try:
                result = extract(
                    text,
                    self._geographic_schema,
                    max_retries=self.max_retries
                )
            except TypeError:
                # Fallback to basic extraction if API signature doesn't match
                result = extract(
                    text,
                    self._geographic_schema
                )
            
            processing_time = time.time() - start_time
            
            if result and hasattr(result, 'features'):
                result.processing_time = processing_time
                result.text_chunk = text
                
                logger.info(f"Successfully extracted {len(result.features)} geographic features in {processing_time:.2f}s")
                return result
            else:
                logger.warning("No geographic features extracted")
                return GeographicExtractionResult(
                    features=[],
                    text_chunk=text,
                    processing_time=processing_time
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in geographic feature extraction: {e} (took {processing_time:.2f}s)")
            
            # Return empty result on error
            return GeographicExtractionResult(
                features=[],
                text_chunk=text,
                processing_time=processing_time
            )
    
    def extract_addresses_batch(self, text_chunks: List[str]) -> List[Address]:
        """
        Extract addresses from multiple text chunks efficiently
        
        Args:
            text_chunks: List of text chunks to process
            
        Returns:
            Combined list of all extracted addresses
        """
        if not text_chunks:
            return []
        
        logger.info(f"Processing {len(text_chunks)} text chunks for address extraction")
        
        all_addresses = []
        total_processing_time = 0.0
        
        for i, chunk in enumerate(text_chunks):
            try:
                chunk_addresses = self.extract_addresses(chunk)
                all_addresses.extend(chunk_addresses)
                
                logger.debug(f"Chunk {i+1}/{len(text_chunks)}: Found {len(chunk_addresses)} addresses")
                
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                continue
        
        logger.info(f"Batch processing complete: {len(all_addresses)} total addresses found")
        return all_addresses
    
    def _fallback_address_extraction(self, text: str) -> List[Address]:
        """
        Fallback method using traditional prompt-based extraction
        
        Args:
            text: Text to process
            
        Returns:
            List of addresses using fallback method
        """
        logger.info("Using fallback address extraction method")
        
        try:
            # Use the traditional method directly to avoid recursion
            if hasattr(self.llm_client, '_extract_addresses_traditional'):
                raw_results = self.llm_client._extract_addresses_traditional(text)
                
                # Convert raw results to Address objects
                addresses = []
                for raw_addr in raw_results:
                    try:
                        # Create coordinates from raw address data
                        coordinates = None
                        if any(raw_addr.get(field) is not None for field in ['latitude', 'longitude', 'x', 'y']):
                            from .schemas import GeographicCoordinate
                            coordinates = GeographicCoordinate(
                                latitude=raw_addr.get('latitude'),
                                longitude=raw_addr.get('longitude'),
                                x=raw_addr.get('x'),
                                y=raw_addr.get('y')
                            )
                        
                        address = Address(
                            original_text=raw_addr.get('original_text', text),
                            normalized_address=raw_addr.get('normalized_address', ''),
                            street=raw_addr.get('street'),
                            number=raw_addr.get('number'),
                            postal_code=raw_addr.get('postal_code'),
                            city=raw_addr.get('city'),
                            country=raw_addr.get('country'),
                            coordinates=coordinates,
                            confidence=raw_addr.get('confidence', 0.5)
                        )
                        addresses.append(address)
                    except Exception as e:
                        logger.warning(f"Failed to convert raw address: {e}")
                        continue
                
                return addresses
            else:
                logger.error("Traditional fallback method not available")
                return []
                
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            return []
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """
        Get statistics about extraction performance
        
        Returns:
            Dictionary with extraction statistics
        """
        return {
            "client_type": "langextract",
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "schemas_available": [
                "address_extraction",
                "geographic_extraction"
            ]
        }
