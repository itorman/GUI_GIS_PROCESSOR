"""
Processing service for document processing operations.
Handles the core document processing workflow independent of the GUI layer.
"""

from typing import Dict, Any, List, Optional, Callable
import logging
import time
from threading import Event

# Configure logging
logger = logging.getLogger(__name__)


class ProcessingService:
    """Service class for document processing operations"""
    
    def __init__(self):
        """Initialize processing service"""
        self.is_processing = False
        self.cancel_event = Event()
        self.progress_callback: Optional[Callable[[int], None]] = None
        self.status_callback: Optional[Callable[[str], None]] = None
        
    def set_progress_callback(self, callback: Callable[[int], None]):
        """Set callback for progress updates"""
        self.progress_callback = callback
        
    def set_status_callback(self, callback: Callable[[str], None]):
        """Set callback for status updates"""
        self.status_callback = callback
        
    def _update_progress(self, progress: int):
        """Update progress if callback is set"""
        if self.progress_callback:
            self.progress_callback(progress)
            
    def _update_status(self, status: str):
        """Update status if callback is set"""
        if self.status_callback:
            self.status_callback(status)
        logger.info(status)
        
    def process_document(self, document_path: str, llm_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document and extract addresses
        
        Args:
            document_path: Path to the document to process
            llm_config: LLM configuration settings
            
        Returns:
            Dictionary with processing results
        """
        result = {
            'success': False,
            'data': None,
            'error': None,
            'processing_time': 0
        }
        
        start_time = time.time()
        self.is_processing = True
        self.cancel_event.clear()
        
        try:
            self._update_status("Loading document...")
            self._update_progress(10)
            
            # Check for cancellation
            if self.cancel_event.is_set():
                result['error'] = 'Processing cancelled'
                return result
            
            # Import and initialize document processor
            try:
                from preprocessing.document_processor import DocumentProcessor
                doc_processor = DocumentProcessor()
            except ImportError as e:
                result['error'] = f'Document processor not available: {str(e)}'
                return result
            
            # Process document to extract text chunks
            try:
                text_chunks = doc_processor.process_document(document_path)
                if not text_chunks:
                    result['error'] = 'No text extracted from document'
                    return result
                    
                self._update_status(f"Document loaded. Found {len(text_chunks)} text chunks.")
                self._update_progress(30)
                
            except Exception as e:
                result['error'] = f'Document processing failed: {str(e)}'
                return result
            
            # Check for cancellation
            if self.cancel_event.is_set():
                result['error'] = 'Processing cancelled'
                return result
            
            # Initialize LLM client
            try:
                from llm.llm_client import LLMClient
                llm_client = LLMClient(llm_config)
            except ImportError as e:
                result['error'] = f'LLM client not available: {str(e)}'
                return result
            except Exception as e:
                result['error'] = f'LLM client initialization failed: {str(e)}'
                return result
            
            self._update_status("Extracting addresses with LLM...")
            self._update_progress(50)
            
            # Process with LLM using the best available method
            addresses = []
            try:
                # Try contextual processing first (if available)
                if (llm_config.get('use_contextual', True) and 
                    hasattr(llm_client, 'extract_addresses_contextual')):
                    self._update_status("Processing with contextual extraction...")
                    addresses = llm_client.extract_addresses_contextual(text_chunks)
                    
                    # Convert objects to dictionaries if needed
                    if addresses and hasattr(addresses[0], '__dict__'):
                        addresses = [addr.__dict__ for addr in addresses]
                        
                # Try batch processing
                elif hasattr(llm_client, 'extract_addresses_batch'):
                    self._update_status("Processing with batch extraction...")
                    addresses = llm_client.extract_addresses_batch(text_chunks)
                    
                # Fallback to individual chunk processing
                else:
                    self._update_status("Processing chunks individually...")
                    for i, chunk in enumerate(text_chunks):
                        if self.cancel_event.is_set():
                            result['error'] = 'Processing cancelled'
                            return result
                            
                        self._update_status(f"Processing chunk {i+1}/{len(text_chunks)}...")
                        chunk_addresses = llm_client.extract_addresses(chunk)
                        
                        if chunk_addresses:
                            addresses.extend(chunk_addresses)
                            
                        # Update progress
                        progress = 50 + int(20 * (i + 1) / len(text_chunks))
                        self._update_progress(progress)
                        
            except Exception as e:
                result['error'] = f'LLM processing failed: {str(e)}'
                return result
            
            self._update_progress(70)
            
            # Check for cancellation
            if self.cancel_event.is_set():
                result['error'] = 'Processing cancelled'
                return result
            
            # Post-process results
            self._update_status("Post-processing data...")
            try:
                from postprocessing.data_processor import DataProcessor
                data_processor = DataProcessor()
                processed_data = data_processor.process_results(addresses)
                
            except ImportError as e:
                result['error'] = f'Data processor not available: {str(e)}'
                return result
            except Exception as e:
                result['error'] = f'Post-processing failed: {str(e)}'
                return result
            
            self._update_progress(100)
            self._update_status("Processing complete!")
            
            # Prepare result
            result['success'] = True
            result['data'] = processed_data
            result['processing_time'] = time.time() - start_time
            
            return result
            
        except Exception as e:
            result['error'] = f'Unexpected error: {str(e)}'
            logger.error(f"Processing failed with unexpected error: {str(e)}")
            return result
            
        finally:
            self.is_processing = False
    
    def cancel_processing(self):
        """Cancel the current processing operation"""
        if self.is_processing:
            self.cancel_event.set()
            self._update_status("Cancelling processing...")
            logger.info("Processing cancellation requested")
    
    def is_processing_active(self) -> bool:
        """Check if processing is currently active"""
        return self.is_processing
    
    def validate_llm_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate LLM configuration
        
        Args:
            config: LLM configuration dictionary
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': False,
            'message': '',
            'missing_fields': []
        }
        
        # Required fields based on LLM type
        required_fields = {
            'Ollama': ['type', 'server_url', 'model'],
            'OpenAI': ['type', 'api_key', 'model'],
            'vLLM': ['type', 'server_url', 'model'],
            'Test': ['type']
        }
        
        llm_type = config.get('type')
        if not llm_type:
            result['message'] = 'LLM type is required'
            result['missing_fields'].append('type')
            return result
        
        if llm_type not in required_fields:
            result['message'] = f'Unsupported LLM type: {llm_type}'
            return result
        
        # Check required fields for the specific LLM type
        missing = []
        for field in required_fields[llm_type]:
            if not config.get(field):
                missing.append(field)
        
        if missing:
            result['missing_fields'] = missing
            result['message'] = f'Missing required fields: {", ".join(missing)}'
            return result
        
        # Additional validation
        if llm_type in ['Ollama', 'vLLM']:
            server_url = config.get('server_url', '')
            if not server_url.startswith(('http://', 'https://')):
                result['message'] = 'Server URL must start with http:// or https://'
                return result
        
        result['valid'] = True
        result['message'] = 'Configuration is valid'
        return result