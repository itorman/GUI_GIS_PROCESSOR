"""
LLM client for address extraction from text using various LLM services.
Supports Ollama, vLLM, OpenAI, and local models.
Now enhanced with langextract for more efficient structured extraction.
"""

import json
import logging
import requests
from typing import List, Dict, Any, Optional
import time

# Import langextract components
from .langextract_client import LangextractClient
from .schemas import Address, AddressExtractionResult
from .contextual_extractor import ContextualExtractor
from .standardized_schemas import StandardizedExtraction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with various LLM services"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM client
        
        Args:
            config: Configuration dictionary with LLM settings
        """
        self.config = config
        self.llm_type = config.get('type', 'Ollama')
        self.server_url = config.get('server_url', 'http://localhost:11434')
        self.model = config.get('model', 'llama3.1:8b')
        self.api_key = config.get('api_key', None)
        self.max_retries = config.get('max_retries', 3)
        # Timeout configuration - optimized for 8B models (fast and efficient)
        self.timeout = config.get('timeout', 30)  # 30 seconds for 8B models
        
        # Initialize langextract client for efficient extraction
        self.langextract_client = None
        self._initialize_langextract()
        
        # Initialize contextual extractor for enhanced processing
        self.contextual_extractor = None
        self._initialize_contextual_extractor()
        
        # Validate configuration
        self._validate_config()
        
        # Log current configuration
        logger.info(f"LLM Client initialized: {self.llm_type} - {self.model} - Timeout: {self.timeout}s")
    
    def _initialize_langextract(self):
        """Initialize the langextract client for efficient extraction"""
        try:
            # Create a self-reference for the LLM client
            # This allows langextract to use our methods for LLM communication
            self.langextract_client = LangextractClient(
                llm_client=self,
                max_retries=self.max_retries,
                timeout=self.timeout
            )
            logger.info("Langextract client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize langextract client: {e}")
            self.langextract_client = None
    
    def _initialize_contextual_extractor(self):
        """Initialize the contextual extractor for enhanced processing"""
        try:
            extractor_config = {
                'context_size': self.config.get('context_size', 2000),
                'min_confidence': self.config.get('min_confidence', 0.5),
                'extraction_mode': self.config.get('extraction_mode', 'standard')
            }
            
            self.contextual_extractor = ContextualExtractor(
                llm_client=self,
                config=extractor_config
            )
            logger.info("Contextual extractor initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize contextual extractor: {e}")
            self.contextual_extractor = None
    
    def _validate_config(self):
        """Validate LLM configuration"""
        if not self.server_url:
            raise ValueError("Server URL is required")
        
        if not self.model:
            raise ValueError("Model name is required")
        
        if self.llm_type == 'OpenAI' and not self.api_key:
            logger.warning("OpenAI API key not provided")
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get current configuration information"""
        return {
            'llm_type': self.llm_type,
            'model': self.model,
            'server_url': self.server_url,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'has_langextract': self.langextract_client is not None,
            'has_contextual_extractor': self.contextual_extractor is not None
        }
    

    
    def extract_addresses(self, text_chunk: str) -> List[Dict[str, Any]]:
        """
        Extract addresses from text chunk using LLM with langextract optimization
        
        Args:
            text_chunk: Text to process
            
        Returns:
            List of extracted address dictionaries
        """
        if not text_chunk.strip():
            return []
        
        # Check if we're in test mode (no real LLM service)
        if self.llm_type == 'Test Mode':  # Removed automatic connection test
            logger.info("Using test mode - simulating LLM response")
            return self._simulate_llm_response(text_chunk)
        
        # Try langextract first for efficient extraction
        if self.langextract_client:
            try:
                logger.info("Using langextract for efficient address extraction")
                addresses = self.langextract_client.extract_addresses(text_chunk)
                
                # Convert Address objects back to dictionaries for compatibility
                if addresses:
                    result_dicts = []
                    for addr in addresses:
                        addr_dict = {
                            'original_text': addr.original_text,
                            'normalized_address': addr.normalized_address,
                            'street': addr.street,
                            'number': addr.number,
                            'postal_code': addr.postal_code,
                            'city': addr.city,
                            'country': addr.country,
                            'latitude': addr.coordinates.latitude if addr.coordinates else None,
                            'longitude': addr.coordinates.longitude if addr.coordinates else None,
                            'x': addr.coordinates.x if addr.coordinates else None,
                            'y': addr.coordinates.y if addr.coordinates else None,
                            'confidence': addr.confidence,
                            'language': addr.language
                        }
                        result_dicts.append(addr_dict)
                    
                    logger.info(f"Langextract successfully extracted {len(result_dicts)} addresses")
                    return result_dicts
                    
            except Exception as e:
                logger.warning(f"Langextract extraction failed, falling back to traditional method: {e}")
        
        # Fallback to traditional prompt-based extraction
        logger.info("Using traditional prompt-based extraction as fallback")
        return self._extract_addresses_traditional(text_chunk)
    
    def _extract_addresses_traditional(self, text_chunk: str) -> List[Dict[str, Any]]:
        """
        Traditional prompt-based address extraction as fallback
        
        Args:
            text_chunk: Text to process
            
        Returns:
            List of extracted address dictionaries
        """
        # Create the prompt for address extraction
        prompt = self._create_address_extraction_prompt(text_chunk)
        
        try:
            logger.info(f"Traditional extraction: Sending prompt to {self.llm_type}")
            
            # Send to LLM based on type with timeout protection
            if self.llm_type == 'Ollama':
                response = self._query_ollama(prompt)
            elif self.llm_type == 'vLLM':
                response = self._query_vllm(prompt)
            elif self.llm_type == 'OpenAI':
                response = self._query_openai(prompt)
            else:
                response = self._query_generic(prompt)
            
            if not response or not response.strip():
                logger.warning("LLM returned empty response")
                return []
            
            logger.info(f"LLM response received, length: {len(response)}")
            
            # Parse the response
            extracted_data = self._parse_llm_response(response)
            
            if extracted_data:
                logger.info(f"Traditional method successfully extracted {len(extracted_data)} addresses from chunk")
                return extracted_data
            else:
                logger.warning("No addresses extracted from chunk using traditional method")
                return []
                
        except Exception as e:
            logger.error(f"Error in traditional address extraction: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            return []
    
    def extract_addresses_batch(self, text_chunks: List[str]) -> List[Dict[str, Any]]:
        """
        Extract addresses from multiple text chunks efficiently using langextract
        
        Args:
            text_chunks: List of text chunks to process
            
        Returns:
            Combined list of all extracted addresses
        """
        if not text_chunks:
            return []
        
        # Use langextract batch processing if available and not in test mode
        if self.langextract_client and self.llm_type != 'Test Mode':
            try:
                logger.info(f"Using langextract for batch processing of {len(text_chunks)} chunks")
                addresses = self.langextract_client.extract_addresses_batch(text_chunks)
                
                # Convert Address objects to dictionaries
                result_dicts = []
                for addr in addresses:
                    addr_dict = {
                        'original_text': addr.original_text,
                        'normalized_address': addr.normalized_address,
                        'street': addr.street,
                        'number': addr.number,
                        'postal_code': addr.postal_code,
                        'city': addr.city,
                        'country': addr.country,
                        'latitude': addr.coordinates.latitude if addr.coordinates else None,
                        'longitude': addr.coordinates.longitude if addr.coordinates else None,
                        'x': addr.coordinates.x if addr.coordinates else None,
                        'y': addr.coordinates.y if addr.coordinates else None,
                        'confidence': addr.confidence,
                        'language': addr.language
                    }
                    result_dicts.append(addr_dict)
                
                logger.info(f"Langextract batch processing complete: {len(result_dicts)} total addresses")
                logger.debug(f"Sample result dict: {result_dicts[0] if result_dicts else 'None'}")
                return result_dicts
                
            except Exception as e:
                logger.warning(f"Langextract batch processing failed, falling back to traditional method: {e}")
        else:
            logger.info(f"Using traditional batch processing (langextract not available or in test mode)")
        
        # Fallback to traditional batch processing
        logger.info("Using traditional batch processing as fallback")
        all_results = []
        for i, chunk in enumerate(text_chunks):
            try:
                # Use traditional method directly to avoid recursion
                chunk_results = self._extract_addresses_traditional(chunk)
                all_results.extend(chunk_results)
                logger.debug(f"Chunk {i+1}/{len(text_chunks)}: Found {len(chunk_results)} addresses")
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                continue
        
        return all_results
    
    def extract_addresses_contextual(self, text_chunks: List[str]) -> List[StandardizedExtraction]:
        """
        Extract addresses using contextual processing for better inference
        
        Args:
            text_chunks: List of text chunks from the document
            
        Returns:
            List of standardized extractions with inferred addresses
        """
        if not text_chunks:
            return []
        
        logger.info(f"Starting contextual address extraction for {len(text_chunks)} chunks")
        
        # Use contextual extractor if available
        if self.contextual_extractor:
            try:
                extractions = self.contextual_extractor.process_document_with_context(text_chunks)
                logger.info(f"Contextual extraction completed: {len(extractions)} standardized extractions")
                return extractions
            except Exception as e:
                logger.error(f"Contextual extraction failed: {e}")
                # Fallback to traditional batch processing
                logger.info("Falling back to traditional batch processing")
        
        # Fallback: convert traditional extraction to standardized format
        logger.info("Using traditional extraction with standardization")
        traditional_results = self.extract_addresses_batch(text_chunks)
        
        # Convert to standardized format
        standardized_results = []
        for result in traditional_results:
            try:
                standardized = StandardizedExtraction(
                    original_text=result.get('original_text', ''),
                    inferred_address=self._build_inferred_address_from_legacy(result),
                    confidence_level=result.get('confidence', 0.5),
                    latitude_dd=result.get('latitude', 0.0),
                    longitude_dd=result.get('longitude', 0.0)
                )
                standardized_results.append(standardized)
            except Exception as e:
                logger.warning(f"Failed to standardize result: {e}")
                continue
        
        return standardized_results
    
    def _build_inferred_address_from_legacy(self, result: Dict[str, Any]) -> str:
        """Build inferred address from legacy extraction result"""
        components = []
        
        # Build address from components
        if result.get('street'):
            street_part = result['street']
            if result.get('number'):
                street_part += f" {result['number']}"
            components.append(street_part)
        
        if result.get('city'):
            components.append(result['city'])
        
        if result.get('country'):
            components.append(result['country'])
        
        # Fallback to normalized address
        if not components and result.get('normalized_address'):
            return result['normalized_address']
        
        return ', '.join(components) if components else 'Location from coordinates'
    
    def _query_llm_with_fallback(self, prompt: str) -> str:
        """
        Query LLM with fallback to different methods.
        Used by contextual extractor.
        """
        try:
            if self.llm_type == 'Ollama':
                return self._query_ollama(prompt)
            elif self.llm_type == 'OpenAI':
                return self._query_openai(prompt)
            elif self.llm_type == 'vLLM':
                return self._query_vllm(prompt)
            else:
                logger.warning("No suitable LLM query method found")
                return ""
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return ""
    
    def _create_address_extraction_prompt(self, text: str) -> str:
        """Create a balanced prompt for address extraction"""
        prompt = f"""Extract ALL addresses, coordinates, cities, and geographic references from this text:

{text}

Look for:
- Street addresses (e.g., "123 Main Street", "Calle Mayor 5")
- Coordinates (e.g., "40.4168, -3.7038", "40°25'21\"N")
- City names (e.g., "Madrid", "London", "Paris")
- Country names (e.g., "Spain", "UK", "France")
- Landmarks (e.g., "Times Square", "Eiffel Tower")

Return a JSON array with this format:
[
  {{
    "original_text": "exact text found",
  "normalized_address": "cleaned address or place name",
    "latitude": decimal_number or null,
    "longitude": decimal_number or null,
    "x": same as longitude,
    "y": same as latitude
  }}
]

If no geographic references found, return [].

JSON:"""
        
        return prompt
    
    def _query_ollama(self, prompt: str) -> str:
        """Query Ollama LLM service with improved error handling"""
        url = f"{self.server_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,   # Balanced temperature for 8B models
                "top_p": 0.9,        # Good quality for 8B models
                "num_predict": 2000,  # Optimal for 8B model responses
                "stop": ["```"],      # Only stop on code blocks, not newlines
                "repeat_penalty": 1.1, # Prevent repetition
                "top_k": 40          # Good quality control
            }
        }
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Ollama attempt {attempt + 1}: Querying model {self.model} with timeout {self.timeout}s")
                
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout,  # Use configured timeout (optimized per model)
                    headers={'Content-Type': 'application/json'}
                )
                response.raise_for_status()
                
                result = response.json()
                response_text = result.get('response', '')
                
                if not response_text:
                    logger.warning(f"Ollama returned empty response: {result}")
                    if attempt == self.max_retries - 1:
                        raise Exception("Ollama returned empty response")
                    continue
                
                logger.info(f"Ollama successful response length: {len(response_text)}")
                return response_text
                
            except requests.exceptions.Timeout:
                logger.warning(f"Ollama attempt {attempt + 1} timed out")
                if attempt == self.max_retries - 1:
                    raise Exception("Ollama request timed out after all attempts")
                time.sleep(2 ** attempt)
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Ollama attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise Exception(f"Ollama request failed after {self.max_retries} attempts: {e}")
                time.sleep(2 ** attempt)
                
            except Exception as e:
                logger.error(f"Unexpected error in Ollama request: {e}")
                if attempt == self.max_retries - 1:
                    raise Exception(f"Unexpected error in Ollama request: {e}")
                time.sleep(2 ** attempt)
    
    def _query_vllm(self, prompt: str) -> str:
        """Query vLLM service"""
        url = f"{self.server_url}/v1/completions"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 2000,
            "temperature": 0.1,
            "top_p": 0.9,
            "stream": False
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                    headers={'Content-Type': 'application/json'}
                )
                response.raise_for_status()
                
                result = response.json()
                return result.get('choices', [{}])[0].get('text', '')
                
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"vLLM request failed after {self.max_retries} attempts: {e}")
                logger.warning(f"vLLM attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
    
    def _query_openai(self, prompt: str) -> str:
        """Query OpenAI API"""
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        url = "https://api.openai.com/v1/chat/completions"
        
        payload = {
            "model": self.model if self.model != 'llama2:7b' else 'gpt-3.5-turbo',
            "messages": [
                {"role": "system", "content": "You are an expert in address extraction and geocoding."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                    headers=headers
                )
                response.raise_for_status()
                
                result = response.json()
                return result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"OpenAI request failed after {self.max_retries} attempts: {e}")
                logger.warning(f"OpenAI attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
    
    def _query_generic(self, prompt: str) -> str:
        """Generic LLM query method"""
        url = f"{self.server_url}/generate"
        
        payload = {
            "prompt": prompt,
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                    headers={'Content-Type': 'application/json'}
                )
                response.raise_for_status()
                
                result = response.json()
                # Try different response formats
                if 'response' in result:
                    return result['response']
                elif 'text' in result:
                    return result['text']
                elif 'generated_text' in result:
                    return result['generated_text']
                else:
                    return str(result)
                    
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"Generic LLM request failed after {self.max_retries} attempts: {e}")
                logger.warning(f"Generic LLM attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
    
    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response and extract JSON data"""
        if not response:
            return []
        
        # Clean the response
        response = response.strip()
        
        logger.info(f"Parsing LLM response, length: {len(response)}")
        logger.debug(f"Response preview: {response[:200]}...")
        
        # Try multiple strategies to find JSON
        json_data = None
        
        # Strategy 1: Look for JSON array
        json_start = response.find('[')
        json_end = response.rfind(']')
        
        if json_start != -1 and json_end != -1:
            try:
                json_str = response[json_start:json_end+1]
                json_data = json.loads(json_str)
                logger.debug("Found JSON array using bracket method")
            except json.JSONDecodeError:
                pass
        
        # Strategy 2: Look for JSON object and wrap in array
        if json_data is None:
            json_start = response.find('{')
            json_end = response.rfind('}')
            
            if json_start != -1 and json_end != -1:
                try:
                    json_str = response[json_start:json_end+1]
                    obj_data = json.loads(json_str)
                    json_data = [obj_data]  # Wrap single object in array
                    logger.debug("Found JSON object and wrapped in array")
                except json.JSONDecodeError:
                    pass
        
        # Strategy 3: Try to extract JSON from markdown code blocks
        if json_data is None:
            import re
            code_block_pattern = r'```(?:json)?\s*\n(.*?)\n```'
            matches = re.findall(code_block_pattern, response, re.DOTALL)
            
            for match in matches:
                try:
                    json_data = json.loads(match.strip())
                    logger.debug("Found JSON in markdown code block")
                    break
                except json.JSONDecodeError:
                    continue
        
        # Strategy 4: Try to find JSON after common prefixes
        if json_data is None:
            prefixes = ['JSON:', 'Output:', 'Result:', 'Response:']
            for prefix in prefixes:
                if prefix in response:
                    start_idx = response.find(prefix) + len(prefix)
                    remaining = response[start_idx:].strip()
                    
                    # Try to find JSON in remaining text
                    json_start = remaining.find('[')
                    json_end = remaining.rfind(']')
                    
                    if json_start != -1 and json_end != -1:
                        try:
                            json_str = remaining[json_start:json_end+1]
                            json_data = json.loads(json_str)
                            logger.debug(f"Found JSON after prefix '{prefix}'")
                            break
                        except json.JSONDecodeError:
                            continue
                    
                    # Try single object
                    json_start = remaining.find('{')
                    json_end = remaining.rfind('}')
                    
                    if json_start != -1 and json_end != -1:
                        try:
                            json_str = remaining[json_start:json_end+1]
                            obj_data = json.loads(json_str)
                            json_data = [obj_data]
                            logger.debug(f"Found JSON object after prefix '{prefix}'")
                            break
                        except json.JSONDecodeError:
                            continue
        
        # If still no JSON found, try to extract any text that looks like addresses
        if json_data is None:
            logger.warning("No JSON found in LLM response, attempting to extract addresses manually")
            json_data = self._extract_addresses_manually(response)
        
        # Ensure it's a list
        if isinstance(json_data, dict):
            json_data = [json_data]
        elif not isinstance(json_data, list):
            logger.warning("LLM response is not a list or dict")
            return []
        
        # Validate and clean each item
        cleaned_data = []
        for item in json_data:
            if isinstance(item, dict):
                cleaned_item = self._clean_extracted_item(item)
                if cleaned_item:
                    cleaned_data.append(cleaned_item)
        
        logger.info(f"Parsing complete: Found {len(cleaned_data)} valid addresses")
        if cleaned_data:
            logger.debug(f"Sample cleaned item: {cleaned_data[0]}")
        
        return cleaned_data
    
    def _extract_addresses_manually(self, response: str) -> List[Dict[str, Any]]:
        """Fallback method to extract addresses when JSON parsing fails"""
        addresses = []
        
        # Look for common address patterns
        import re
        
        # Pattern for street addresses
        street_patterns = [
            r'(\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Place|Pl|Court|Ct|Calle|Callejón|Avenida|Carretera|Paseo))',
            r'([A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Place|Pl|Court|Ct|Calle|Callejón|Avenida|Carretera|Paseo)\s+\d+)',
        ]
        
        # Pattern for city names
        city_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*,\s*([A-Z]{2})\b',  # City, State
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*,\s*([A-Z][a-z]+)\b',  # City, Country
        ]
        
        # Pattern for coordinates
        coord_patterns = [
            r'(\d+°\d+\'\d+\"[NS]?\s*,\s*\d+°\d+\'\d+\"[EW]?)',  # DMS format
            r'(\d+\.\d+°[NS]?\s*,\s*\d+\.\d+°[EW]?)',  # Decimal format
        ]
        
        # Extract street addresses
        for pattern in street_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                addresses.append({
                    "original_text": match,
                    "normalized_address": match,
                    "latitude": None,
                    "longitude": None,
                    "x": None,
                    "y": None
                })
        
        # Extract city names
        for pattern in city_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    city, region = match
                    addresses.append({
                        "original_text": f"{city}, {region}",
                        "normalized_address": f"{city}, {region}",
                        "latitude": None,
                        "longitude": None,
                        "x": None,
                        "y": None
                    })
        
        # Extract coordinates
        for pattern in coord_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                addresses.append({
                    "original_text": match,
                    "normalized_address": f"Coordinates: {match}",
                    "latitude": None,  # Would need conversion logic
                    "longitude": None,
                    "x": None,
                    "y": None
                })
        
        return addresses
    
    def _clean_extracted_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Clean and validate extracted address item"""
        required_fields = ['original_text', 'normalized_address']
        
        # Check required fields
        for field in required_fields:
            if field not in item or not item[field]:
                return None
        
        # Clean and validate coordinates
        cleaned_item = {
            'original_text': str(item['original_text']).strip(),
            'normalized_address': str(item['normalized_address']).strip()
        }
        
        # Handle coordinates
        for coord_field in ['latitude', 'longitude', 'x', 'y']:
            if coord_field in item and item[coord_field] is not None:
                try:
                    coord_value = float(item[coord_field])
                    # Validate coordinate ranges
                    if coord_field in ['latitude'] and -90 <= coord_value <= 90:
                        cleaned_item[coord_field] = coord_value
                    elif coord_field in ['longitude'] and -180 <= coord_value <= 180:
                        cleaned_item[coord_field] = coord_value
                    elif coord_field in ['x', 'y']:
                        cleaned_item[coord_field] = coord_value
                    else:
                        cleaned_item[coord_field] = None
                except (ValueError, TypeError):
                    cleaned_item[coord_field] = None
            else:
                cleaned_item[coord_field] = None
        
        return cleaned_item
    
    def test_connection(self) -> bool:
        """Test connection to LLM service with improved error handling"""
        try:
            logger.info(f"Testing connection to {self.llm_type}")
            
            if self.llm_type == 'Ollama':
                # Simple test prompt
                test_prompt = "Say 'OK' and nothing else."
                logger.info("Sending test prompt to Ollama...")
                
                response = self._query_ollama(test_prompt)
                logger.info(f"Ollama test response: {response[:100]}...")
                
                success = bool(response and len(response.strip()) > 0)
                logger.info(f"Ollama connection test: {'SUCCESS' if success else 'FAILED'}")
                return success
                
            elif self.llm_type == 'vLLM':
                response = self._query_vllm("Hello, respond with 'OK'")
                return bool(response and len(response.strip()) > 0)
            elif self.llm_type == 'OpenAI':
                response = self._query_openai("Hello, respond with 'OK'")
                return bool(response and len(response.strip()) > 0)
            else:
                response = self._query_generic("Hello, respond with 'OK'")
            return bool(response and len(response.strip()) > 0)
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            'type': self.llm_type,
            'model': self.model,
            'server_url': self.server_url,
            'api_key_configured': bool(self.api_key)
        } 

    def _simulate_llm_response(self, text_chunk: str) -> List[Dict[str, Any]]:
        """Simulate LLM response for testing purposes - Enhanced address detection"""
        import re
        
        addresses = []
        
        # 1. Extract coordinates in various formats first
        coord_patterns = [
            # Decimal degrees with N/S/E/W
            r'(\d+\.\d+\s*[NS]?\s*,\s*\d+\.\d+\s*[EW]?)',
            # Decimal degrees without N/S/E/W
            r'(\d+\.\d+\s*,\s*\d+\.\d+)',
            # DMS format with N/S/E/W
            r'(\d+\s*\d+\'\d+\"[NS]?\s*,\s*\d+\s*\d+\'\d+\"[EW]?)',
            # DMS format without N/S/E/W
            r'(\d+\s*\d+\'\d+\"\s*,\s*\d+\s*\d+\'\d+\")',
            # Mixed formats
            r'(\d+\.\d+\s*[NS]?\s*,\s*\d+\s*\d+\'\d+\"[EW]?)',
            r'(\d+\s*\d+\'\d+\"[NS]?\s*,\s*\d+\.\d+\s*[EW]?)',
        ]
        
        for pattern in coord_patterns:
            coord_matches = re.findall(pattern, text_chunk, re.IGNORECASE)
            for match in coord_matches:
                try:
                    # Parse coordinates
                    coords = self._parse_coordinates(match)
                    if coords:
                        lat, lon = coords
                        addresses.append({
                            "original_text": match,
                            "normalized_address": f"Coordinates: {match}",
                            "latitude": lat,
                            "longitude": lon,
                            "x": lon,
                            "y": lat
                        })
                except Exception as e:
                    logger.debug(f"Failed to parse coordinates '{match}': {e}")
        
        # 2. Extract street addresses with various patterns
        address_patterns = [
            # English patterns
            r'(\d+[A-Za-z]?\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Place|Pl|Court|Ct|Square|Sq|Circle|Cir|Terrace|Ter))',
            r'(\d+[A-Za-z]?\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Place|Pl|Court|Ct|Square|Sq|Circle|Cir|Terrace|Ter)\s*,\s*[A-Za-z\s]+)',
            r'(\d+[A-Za-z]?\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Place|Pl|Court|Ct|Square|Sq|Circle|Cir|Terrace|Ter)\s*,\s*[A-Za-z\s]+,\s*[A-Za-z\s]+)',
            
            # Spanish patterns
            r'([Cc]alle\s+[A-Za-záéíóúñÑ\s]+\s+\d+[A-Za-z]?)',
            r'([Cc]alle\s+[A-Za-záéíóúñÑ\s]+\s+\d+[A-Za-z]?\s*,\s*[A-Za-záéíóúñÑ\s]+)',
            r'([Cc]alle\s+[A-Za-záéíóúñÑ\s]+\s+\d+[A-Za-z]?\s*,\s*[A-Za-záéíóúñÑ\s]+,\s*[A-Za-záéíóúñÑ\s]+)',
            r'([Aa]venida\s+[A-Za-záéíóúñÑ\s]+\s+\d+[A-Za-z]?)',
            r'([Pp]aseo\s+[A-Za-záéíóúñÑ\s]+\s+\d+[A-Za-z]?)',
            r'([Cc]arrer\s+[A-Za-záéíóúñÑ\s]+\s+\d+[A-Za-z]?)',
            
            # French patterns
            r'([Rr]ue\s+[A-Za-zàâäéèêëïîôöùûüÿç\s]+\s+\d+[A-Za-z]?)',
            r'([Aa]venue\s+[A-Za-zàâäéèêëïîôöùûüÿç\s]+\s+\d+[A-Za-z]?)',
            r'([Bb]oulevard\s+[A-Za-zàâäéèêëïîôöùûüÿç\s]+\s+\d+[A-Za-z]?)',
            
            # German patterns
            r'([Ss]traße\s+[A-Za-zäöüß\s]+\s+\d+[A-Za-z]?)',
            r'([Ss]trasse\s+[A-Za-zäöüß\s]+\s+\d+[A-Za-z]?)',
            r'([Aa]llee\s+[A-Za-zäöüß\s]+\s+\d+[A-Za-z]?)',
            
            # Italian patterns
            r'([Vv]ia\s+[A-Za-zàèéìíîòóù\s]+\s+\d+[A-Za-z]?)',
            r'([Vv]iale\s+[A-Za-zàèéìíîòóù\s]+\s+\d+[A-Za-z]?)',
            r'([Pp]iazza\s+[A-Za-zàèéìíîòóù\s]+\s+\d+[A-Za-z]?)',
            
            # Generic patterns
            r'(\d+[A-Za-z]?\s+[A-Za-z\s]+)',
        ]
        
        for pattern in address_patterns:
            matches = re.findall(pattern, text_chunk, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 5:  # Filter out very short matches
                    # Try to geocode this address
                    coords = self._geocode_address(match)
                    addresses.append({
                        "original_text": match,
                        "normalized_address": match,
                        "latitude": coords[0] if coords else None,
                        "longitude": coords[1] if coords else None,
                        "x": coords[1] if coords else None,
                        "y": coords[0] if coords else None
                    })
        
        # 3. Extract city names and try to geocode them
        city_patterns = [
            r'\b([A-Z][a-záéíóúñÑàâäéèêëïîôöùûüÿçäöüß]+(?:\s+[A-Z][a-záéíóúñÑàâäéèêëïîôöùûüÿçäöüß]+)*)\b'
        ]
        
        # Major cities database with coordinates
        major_cities = {
            'London': (51.5074, -0.1278),
            'Paris': (48.8566, 2.3522),
            'New York': (40.7128, -74.0060),
            'Madrid': (40.4168, -3.7038),
            'Barcelona': (41.3851, 2.1734),
            'Amsterdam': (52.3676, 4.9041),
            'Venice': (45.4408, 12.3155),
            'Oslo': (59.9139, 10.7522),
            'Rome': (41.9028, 12.4964),
            'Berlin': (52.5200, 13.4050),
            'Vienna': (48.2082, 16.3738),
            'Prague': (50.0755, 14.4378),
            'Budapest': (47.4979, 19.0402),
            'Warsaw': (52.2297, 21.0122),
            'Moscow': (55.7558, 37.6176),
            'Tokyo': (35.6762, 139.6503),
            'Beijing': (39.9042, 116.4074),
            'Sydney': (-33.8688, 151.2093),
            'Cape Town': (-33.9249, 18.4241),
            'Rio de Janeiro': (-22.9068, -43.1729),
            'São Paulo': (-23.5505, -46.6333),
            'Mexico City': (19.4326, -99.1332),
            'Buenos Aires': (-34.6118, -58.3960),
            'Lima': (-12.0464, -77.0428),
            'Bogotá': (4.7110, -74.0721),
            'Caracas': (10.4806, -66.9036),
            'Santiago': (-33.4489, -70.6693),
            'Montevideo': (-34.9011, -56.1645),
            'Asunción': (-25.2637, -57.5759),
            'La Paz': (-16.4897, -68.1193),
            'Quito': (-0.1807, -78.4678),
            'Lima': (-12.0464, -77.0428),
            'Brasília': (-15.7942, -47.8822),
            'Buenos Aires': (-34.6118, -58.3960),
            'Santiago': (-33.4489, -70.6693),
            'Montevideo': (-34.9011, -56.1645),
            'Asunción': (-25.2637, -57.5759),
            'La Paz': (-16.4897, -68.1193),
            'Quito': (-0.1807, -78.4678),
            'Brasília': (-15.7942, -47.8822),
        }
        
        for city, coords in major_cities.items():
            if city in text_chunk:
                addresses.append({
                    "original_text": city,
                    "normalized_address": f"{city}",
                    "latitude": coords[0],
                    "longitude": coords[1],
                    "x": coords[1],
                    "y": coords[0]
                })
        
        # 4. Extract country names
        countries = {
            'UK': (54.2361, -4.5481),
            'USA': (39.8283, -98.5795),
            'France': (46.2276, 2.2137),
            'Germany': (51.1657, 10.4515),
            'Spain': (40.4637, -3.7492),
            'Italy': (41.8719, 12.5674),
            'Netherlands': (52.1326, 5.2913),
            'Belgium': (50.8503, 4.3517),
            'Switzerland': (46.8182, 8.2275),
            'Austria': (47.5162, 14.5501),
            'Czech Republic': (49.8175, 15.4730),
            'Poland': (51.9194, 19.1451),
            'Hungary': (47.1625, 19.5033),
            'Romania': (45.9432, 24.9668),
            'Bulgaria': (42.7339, 25.4858),
            'Greece': (39.0742, 21.8243),
            'Turkey': (38.9637, 35.2433),
            'Russia': (61.5240, 105.3188),
            'Ukraine': (48.3794, 31.1656),
            'Belarus': (53.7098, 27.9534),
            'Lithuania': (55.1694, 23.8813),
            'Latvia': (56.8796, 24.6032),
            'Estonia': (58.5953, 25.0136),
            'Finland': (61.9241, 25.7482),
            'Sweden': (60.1282, 18.6435),
            'Norway': (60.4720, 8.4689),
            'Denmark': (56.2639, 9.5018),
            'Iceland': (64.9631, -19.0208),
            'Ireland': (53.1424, -7.6921),
            'Portugal': (39.3999, -8.2245),
            'Canada': (56.1304, -106.3468),
            'Mexico': (23.6345, -102.5528),
            'Brazil': (-14.2350, -51.9253),
            'Argentina': (-38.4161, -63.6167),
            'Chile': (-35.6751, -71.5430),
            'Peru': (-9.1900, -75.0152),
            'Colombia': (4.5709, -74.2973),
            'Venezuela': (6.4238, -66.5897),
            'Ecuador': (-1.8312, -78.1834),
            'Bolivia': (-16.2902, -63.5887),
            'Paraguay': (-23.4425, -58.4438),
            'Uruguay': (-32.5228, -55.7658),
            'Guyana': (4.8604, -58.9302),
            'Suriname': (3.9193, -56.0278),
            'French Guiana': (3.9339, -53.1258),
            'Japan': (36.2048, 138.2529),
            'China': (35.8617, 104.1954),
            'India': (20.5937, 78.9629),
            'South Korea': (35.9078, 127.7669),
            'North Korea': (40.3399, 127.5101),
            'Mongolia': (46.8625, 103.8467),
            'Kazakhstan': (48.0196, 66.9237),
            'Uzbekistan': (41.3775, 64.5853),
            'Turkmenistan': (38.9697, 59.5563),
            'Tajikistan': (38.5358, 71.0965),
            'Kyrgyzstan': (41.2044, 74.7661),
            'Afghanistan': (33.9391, 67.7100),
            'Pakistan': (30.3753, 69.3451),
            'Nepal': (28.3949, 84.1240),
            'Bhutan': (27.5142, 90.4336),
            'Bangladesh': (23.6850, 90.3563),
            'Sri Lanka': (7.8731, 80.7718),
            'Maldives': (3.2028, 73.2207),
            'Myanmar': (21.9162, 95.9560),
            'Thailand': (15.8700, 100.9925),
            'Laos': (19.8563, 102.4955),
            'Cambodia': (12.5657, 104.9910),
            'Vietnam': (14.0583, 108.2772),
            'Malaysia': (4.2105, 108.9758),
            'Singapore': (1.3521, 103.8198),
            'Indonesia': (-0.7893, 113.9213),
            'Philippines': (12.8797, 121.7740),
            'Brunei': (4.5353, 114.7277),
            'East Timor': (-8.8742, 125.7275),
            'Papua New Guinea': (-6.3150, 143.9555),
            'Australia': (-25.2744, 133.7751),
            'New Zealand': (-40.9006, 174.8860),
            'Fiji': (-17.7134, 178.0650),
            'Vanuatu': (-15.3767, 166.9592),
            'New Caledonia': (-20.9043, 165.6180),
            'Solomon Islands': (-9.6457, 160.1562),
            'Samoa': (-13.7590, -172.1046),
            'Tonga': (-21.1790, -175.1982),
            'Tuvalu': (-7.1095, 177.6493),
            'Kiribati': (-3.3704, -168.7340),
            'Marshall Islands': (7.1315, 171.1845),
            'Micronesia': (7.4256, 150.5508),
            'Palau': (7.5150, 134.5825),
            'Nauru': (-0.5228, 166.9315),
            'South Africa': (-30.5595, 22.9375),
            'Namibia': (-22.9576, 18.4904),
            'Botswana': (-22.3285, 24.6849),
            'Zimbabwe': (-19.0154, 29.1549),
            'Zambia': (-13.1339, 27.8493),
            'Malawi': (-13.2543, 34.3015),
            'Mozambique': (-18.6657, 35.5296),
            'Tanzania': (-6.3690, 34.8888),
            'Kenya': (-0.0236, 37.9062),
            'Uganda': (1.3733, 32.2903),
            'Rwanda': (-1.9403, 29.8739),
            'Burundi': (-3.3731, 29.9189),
            'Democratic Republic of the Congo': (-4.0383, 21.7587),
            'Republic of the Congo': (-0.2280, 15.8277),
            'Gabon': (-0.8037, 11.6094),
            'Equatorial Guinea': (1.6508, 10.2679),
            'Cameroon': (7.3697, 12.3547),
            'Central African Republic': (6.6111, 20.9394),
            'Chad': (15.4542, 18.7322),
            'Niger': (17.6078, 8.0817),
            'Nigeria': (9.0820, 8.6753),
            'Benin': (9.3077, 2.3158),
            'Togo': (8.6195, 0.8248),
            'Ghana': (7.9465, -1.0232),
            'Ivory Coast': (7.5400, -5.5471),
            'Liberia': (6.4281, -9.4295),
            'Sierra Leone': (8.4606, -11.7799),
            'Guinea': (9.9456, -9.6966),
            'Guinea-Bissau': (11.8037, -15.1804),
            'Senegal': (14.4974, -14.4524),
            'The Gambia': (13.4432, -15.3101),
            'Mauritania': (21.0079, -10.9408),
            'Mali': (17.5707, -3.9962),
            'Burkina Faso': (12.2383, -1.5616),
            'Algeria': (28.0339, 1.6596),
            'Tunisia': (33.8869, 9.5375),
            'Libya': (26.3351, 17.2283),
            'Egypt': (26.8206, 30.8025),
            'Sudan': (12.8628, 30.2176),
            'South Sudan': (6.8770, 31.3070),
            'Ethiopia': (9.1450, 40.4897),
            'Eritrea': (15.1794, 39.7823),
            'Djibouti': (11.8251, 42.5903),
            'Somalia': (5.1521, 46.1996),
            'Morocco': (31.7917, -7.0926),
            'Western Sahara': (24.2155, -12.8858),
            'Angola': (-11.2027, 17.8739),
            'Lesotho': (-29.6099, 28.2336),
            'Eswatini': (-26.5225, 31.4659),
            'Madagascar': (-18.7669, 46.8691),
            'Comoros': (-11.6455, 43.3333),
            'Mauritius': (-20.3484, 57.5522),
            'Seychelles': (-4.6796, 55.4920),
            'Réunion': (-21.1151, 55.5364),
            'Mayotte': (-12.8275, 45.1662),
        }
        
        for country, coords in countries.items():
            if country in text_chunk:
                addresses.append({
                    "original_text": country,
                    "normalized_address": f"{country}",
                    "latitude": coords[0],
                    "longitude": coords[1],
                    "x": coords[1],
                    "y": coords[0]
                })
        
        # Remove duplicates based on normalized_address
        unique_addresses = []
        seen_addresses = set()
        
        for addr in addresses:
            if addr['normalized_address'] not in seen_addresses:
                unique_addresses.append(addr)
                seen_addresses.add(addr['normalized_address'])
        
        # Log coordinate information
        coords_count = sum(1 for addr in unique_addresses if addr.get('latitude') is not None or addr.get('longitude') is not None)
        logger.info(f"Test mode: Found {len(unique_addresses)} unique addresses, {coords_count} with coordinates")
        
        # Log sample coordinates for debugging
        if unique_addresses:
            sample_addr = unique_addresses[0]
            logger.debug(f"Sample address: {sample_addr.get('normalized_address')} - Lat: {sample_addr.get('latitude')}, Lon: {sample_addr.get('longitude')}")
        
        return unique_addresses
    
    def _parse_coordinates(self, coord_str: str) -> Optional[tuple]:
        """Parse coordinates in various formats and return (lat, lon) tuple"""
        import re
        
        try:
            # Clean the string
            coord_str = coord_str.strip()
            
            # Extract numbers
            numbers = re.findall(r'\d+[.,]\d+', coord_str)
            if len(numbers) >= 2:
                lat = float(numbers[0].replace(',', '.'))
                lon = float(numbers[1].replace(',', '.'))
                
                # Determine if coordinates are valid
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return (lat, lon)
            
            # Try DMS format
            dms_pattern = r'(\d+)\s*(\d+)\'(\d+)"\s*([NS]?)\s*,\s*(\d+)\s*(\d+)\'(\d+)"\s*([EW]?)'
            dms_match = re.search(dms_pattern, coord_str, re.IGNORECASE)
            
            if dms_match:
                lat_deg, lat_min, lat_sec, lat_dir = int(dms_match.group(1)), int(dms_match.group(2)), int(dms_match.group(3)), dms_match.group(4)
                lon_deg, lon_min, lon_sec, lon_dir = int(dms_match.group(5)), int(dms_match.group(6)), int(dms_match.group(7)), dms_match.group(8)
                
                # Convert to decimal degrees
                lat = lat_deg + lat_min/60 + lat_sec/3600
                lon = lon_deg + lon_min/60 + lon_sec/3600
                
                # Apply direction
                if lat_dir.upper() == 'S':
                    lat = -lat
                if lon_dir.upper() == 'W':
                    lon = -lon
                
                return (lat, lon)
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to parse coordinates: {e}")
            return None
    
    def _geocode_address(self, address: str) -> Optional[tuple]:
        """Enhanced geocoding for common addresses with fallback coordinates"""
        import random
        
        address_lower = address.lower()
        
        # Common address patterns with approximate coordinates
        address_coords = {
            'baker street': (51.5237, -0.1278),
            'rue de rivoli': (48.8566, 2.3522),
            'maple road': (45.4642, 9.1900),
            'piazza san marco': (45.4343, 12.3387),
            'fifth avenue': (40.7532, -73.9822),
            'carrer de mallorca': (41.3851, 2.1734),
            'lindenstrasse': (50.1109, 8.6821),
            'puerta del sol': (40.4168, -3.7038),
            'main street': (39.7817, -89.6501),
            'karl johans gate': (59.9139, 10.7522),
            'broadway': (40.7127, -74.0059),
            'rue saint-honoré': (48.8566, 2.3522),
            'markt': (50.9375, 6.9603),
            'west street': (-34.0522, 18.4241),
            'grafton street': (53.3421, -6.2597),
            'avenida paulista': (-23.5505, -46.6333),
            'grand rue': (46.2044, 6.1432),
            'ocean drive': (25.7907, -80.1300),
            'rue neuve': (50.8503, 4.3517),
            'damstraat': (52.3676, 4.9041),
        }
        
        # Try exact matches first
        for pattern, coords in address_coords.items():
            if pattern in address_lower:
                return coords
        
        # Fallback: generate realistic coordinates based on address content
        # This ensures every address gets some coordinates for testing
        # Use hash of address to make coordinates deterministic but unique
        import hashlib
        address_hash = hashlib.md5(address.encode()).hexdigest()
        
        # Convert hash to coordinates (deterministic but unique)
        lat_seed = int(address_hash[:8], 16) / (16**8)  # 0.0 to 1.0
        lon_seed = int(address_hash[8:16], 16) / (16**8)  # 0.0 to 1.0
        
        if any(word in address_lower for word in ['street', 'avenue', 'road', 'calle', 'rue', 'via', 'strasse']):
            # Generate coordinates in a realistic range
            lat = 35.0 + (lat_seed * 25.0)  # 35.0 to 60.0
            lon = -10.0 + (lon_seed * 50.0)  # -10.0 to 40.0
            return (lat, lon)
        elif any(word in address_lower for word in ['madrid', 'barcelona', 'london', 'paris', 'new york']):
            # Major cities
            lat = 40.0 + (lat_seed * 15.0)  # 40.0 to 55.0
            lon = -5.0 + (lon_seed * 20.0)  # -5.0 to 15.0
            return (lat, lon)
        else:
            # Generic fallback
            lat = 30.0 + (lat_seed * 40.0)  # 30.0 to 70.0
            lon = -180.0 + (lon_seed * 360.0)  # -180.0 to 180.0
            return (lat, lon) 