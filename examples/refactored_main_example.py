"""
Example demonstrating how to refactor main.py to use the new service layer.
This shows the improved architecture with separation of concerns.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Optional, Dict, Any

# Import the new service layer
from services.document_service import DocumentService
from services.processing_service import ProcessingService
from services.export_service import ExportService
from config.config_manager import get_config_manager
from utils.error_handler import handle_error, log_info


class DocumentProcessingController:
    """
    Controller class that coordinates between UI and services.
    This separates business logic from the GUI layer.
    """
    
    def __init__(self):
        """Initialize the controller with service instances"""
        self.document_service = DocumentService()
        self.processing_service = ProcessingService()
        self.export_service = ExportService()
        self.config_manager = get_config_manager()
        
        # Set up processing callbacks
        self.progress_callback = None
        self.status_callback = None
        self.completion_callback = None
        
    def set_callbacks(self, progress_callback=None, status_callback=None, completion_callback=None):
        """Set callbacks for UI updates"""
        self.progress_callback = progress_callback
        self.status_callback = status_callback
        self.completion_callback = completion_callback
        
        # Configure processing service callbacks
        if progress_callback:
            self.processing_service.set_progress_callback(progress_callback)
        if status_callback:
            self.processing_service.set_status_callback(status_callback)
    
    def upload_document(self, file_path: str) -> Dict[str, Any]:
        """
        Handle document upload with validation
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with operation result
        """
        try:
            log_info(f"Attempting to upload document: {file_path}")
            
            # Validate document
            validation = self.document_service.validate_document(file_path)
            if not validation['valid']:
                return {
                    'success': False,
                    'message': validation['message'],
                    'user_message': f"Document validation failed: {validation['message']}"
                }
            
            # Set as current document
            success = self.document_service.set_current_document(file_path)
            if success:
                doc_info = self.document_service.get_document_info()
                return {
                    'success': True,
                    'message': 'Document uploaded successfully',
                    'document_info': doc_info
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to set current document',
                    'user_message': 'Failed to load the document. Please try again.'
                }
                
        except Exception as e:
            return handle_error(e, {'file_path': file_path}, 
                               "Failed to upload document. Please check the file and try again.")
    
    def get_supported_formats(self) -> Dict[str, Any]:
        """Get supported formats for documents and exports"""
        return {
            'document_formats': self.document_service.get_supported_formats(),
            'export_formats': self.export_service.get_supported_formats()
        }


# Example usage and demonstration
if __name__ == '__main__':
    print("=== ARCHITECTURE IMPROVEMENT EXAMPLE ===")
    print()
    print("This example demonstrates how the monolithic main.py (1,530 lines)")
    print("can be refactored using the new service layer architecture.")
    print()
    print("KEY IMPROVEMENTS:")
    print("1. Separation of Concerns - UI, business logic, and services are separated")
    print("2. Testable Components - Each service can be tested independently")
    print("3. Consistent Error Handling - All errors go through the error handler")
    print("4. Configuration Management - Centralized configuration")
    print("5. Cleaner Code - Much more readable and maintainable")
    print()
    print("BEFORE (Original main.py issues):")
    print("- 1,530 lines in a single file")
    print("- GUI and business logic tightly coupled")
    print("- No test infrastructure")
    print("- Inconsistent error handling")
    print("- Scattered configuration")
    print()
    print("AFTER (With service layer):")
    print("- Modular services with single responsibilities")
    print("- Testable business logic")
    print("- Consistent error handling and logging")
    print("- Centralized configuration management")
    print("- Clean separation between UI and business logic")
    print()
    
    # Demonstrate the controller
    try:
        controller = DocumentProcessingController()
        formats = controller.get_supported_formats()
        print("✓ Controller created successfully")
        print(f"✓ Document formats: {formats['document_formats']}")
        print(f"✓ Export formats: {formats['export_formats']}")
    except Exception as e:
        print(f"✗ Controller demo failed: {e}")