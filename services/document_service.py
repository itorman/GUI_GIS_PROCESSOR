"""
Document service for handling document operations.
Separates document-related business logic from the GUI layer.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)


class DocumentService:
    """Service class for document operations"""
    
    def __init__(self):
        """Initialize document service"""
        self.current_document_path: Optional[str] = None
        self.supported_formats = ['.pdf', '.docx', '.xlsx', '.txt']
        
    def validate_document(self, file_path: str) -> Dict[str, Any]:
        """
        Validate a document file
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': False,
            'message': '',
            'file_size': 0,
            'file_extension': '',
            'file_name': ''
        }
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                result['message'] = 'File does not exist'
                return result
                
            # Get file info
            file_path_obj = Path(file_path)
            result['file_name'] = file_path_obj.name
            result['file_extension'] = file_path_obj.suffix.lower()
            result['file_size'] = os.path.getsize(file_path)
            
            # Check file extension
            if result['file_extension'] not in self.supported_formats:
                result['message'] = f"Unsupported file format. Supported: {', '.join(self.supported_formats)}"
                return result
                
            # Check file size (limit to 100MB)
            max_size = 100 * 1024 * 1024  # 100MB
            if result['file_size'] > max_size:
                result['message'] = f"File too large. Maximum size: {max_size / (1024*1024):.0f}MB"
                return result
                
            # Check if file is readable
            try:
                with open(file_path, 'rb') as f:
                    f.read(1024)  # Try to read first 1KB
            except Exception as e:
                result['message'] = f"Cannot read file: {str(e)}"
                return result
                
            result['valid'] = True
            result['message'] = 'File is valid'
            return result
            
        except Exception as e:
            result['message'] = f"Validation error: {str(e)}"
            logger.error(f"Document validation failed: {str(e)}")
            return result
    
    def set_current_document(self, file_path: str) -> bool:
        """
        Set the current document for processing
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if successful, False otherwise
        """
        validation = self.validate_document(file_path)
        if validation['valid']:
            self.current_document_path = file_path
            logger.info(f"Document set: {file_path}")
            return True
        else:
            logger.warning(f"Failed to set document: {validation['message']}")
            return False
    
    def get_current_document(self) -> Optional[str]:
        """
        Get the current document path
        
        Returns:
            Current document path or None if no document is set
        """
        return self.current_document_path
    
    def clear_current_document(self):
        """Clear the current document"""
        self.current_document_path = None
        logger.info("Current document cleared")
    
    def get_document_info(self) -> Dict[str, Any]:
        """
        Get information about the current document
        
        Returns:
            Dictionary with document information
        """
        if not self.current_document_path:
            return {'has_document': False}
            
        validation = self.validate_document(self.current_document_path)
        validation['has_document'] = True
        validation['file_path'] = self.current_document_path
        
        return validation
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats
        
        Returns:
            List of supported file extensions
        """
        return self.supported_formats.copy()
    
    def format_file_size(self, size_bytes: int) -> str:
        """
        Format file size in human readable format
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Formatted size string
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"