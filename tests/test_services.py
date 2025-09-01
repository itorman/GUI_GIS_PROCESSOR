"""
Tests for service layer classes.
Tests document, processing, and export services.
"""

import unittest
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from services.document_service import DocumentService
    from services.processing_service import ProcessingService
    from services.export_service import ExportService
    SERVICES_AVAILABLE = True
except ImportError as e:
    SERVICES_AVAILABLE = False
    print(f"Services not available for testing: {e}")


class TestServices(unittest.TestCase):
    """Test cases for service layer classes"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not SERVICES_AVAILABLE:
            self.skipTest("Services not available")
        
        self.document_service = DocumentService()
        self.processing_service = ProcessingService()
        self.export_service = ExportService()
        
        # Create temporary test files
        self.temp_dir = tempfile.mkdtemp()
        self.test_txt_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.test_txt_file, 'w') as f:
            f.write("This is a test document with an address: 123 Main St, Madrid, Spain")
    
    def tearDown(self):
        """Clean up test fixtures"""
        if hasattr(self, 'temp_dir'):
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_document_service_validation(self):
        """Test document validation functionality"""
        # Test valid file
        result = self.document_service.validate_document(self.test_txt_file)
        self.assertTrue(result['valid'])
        self.assertEqual(result['file_extension'], '.txt')
        
        # Test invalid file
        invalid_file = os.path.join(self.temp_dir, "nonexistent.txt")
        result = self.document_service.validate_document(invalid_file)
        self.assertFalse(result['valid'])
        self.assertIn('does not exist', result['message'])
    
    def test_document_service_set_current(self):
        """Test setting current document"""
        success = self.document_service.set_current_document(self.test_txt_file)
        self.assertTrue(success)
        
        current = self.document_service.get_current_document()
        self.assertEqual(current, self.test_txt_file)
        
        # Test clearing
        self.document_service.clear_current_document()
        current = self.document_service.get_current_document()
        self.assertIsNone(current)
    
    def test_document_service_supported_formats(self):
        """Test supported formats"""
        formats = self.document_service.get_supported_formats()
        self.assertIn('.pdf', formats)
        self.assertIn('.txt', formats)
        self.assertIn('.docx', formats)
        self.assertIn('.xlsx', formats)
    
    def test_processing_service_config_validation(self):
        """Test LLM configuration validation"""
        # Test valid Ollama config
        valid_config = {
            'type': 'Ollama',
            'server_url': 'http://localhost:11434',
            'model': 'llama3:8b'
        }
        result = self.processing_service.validate_llm_config(valid_config)
        self.assertTrue(result['valid'])
        
        # Test invalid config (missing fields)
        invalid_config = {
            'type': 'Ollama'
        }
        result = self.processing_service.validate_llm_config(invalid_config)
        self.assertFalse(result['valid'])
        self.assertIn('server_url', result['missing_fields'])
        self.assertIn('model', result['missing_fields'])
    
    def test_processing_service_callbacks(self):
        """Test callback functionality"""
        progress_values = []
        status_messages = []
        
        def progress_callback(progress):
            progress_values.append(progress)
        
        def status_callback(status):
            status_messages.append(status)
        
        self.processing_service.set_progress_callback(progress_callback)
        self.processing_service.set_status_callback(status_callback)
        
        # Test callbacks are called
        self.processing_service._update_progress(50)
        self.processing_service._update_status("Test status")
        
        self.assertIn(50, progress_values)
        self.assertIn("Test status", status_messages)
    
    def test_export_service_supported_formats(self):
        """Test export service supported formats"""
        formats = self.export_service.get_supported_formats()
        self.assertIn('csv', formats)
        self.assertIn('xlsx', formats)
        self.assertIn('shapefile', formats)
        self.assertIn('arcgis', formats)
    
    def test_export_service_path_validation(self):
        """Test export path validation"""
        # Test valid CSV path
        csv_path = os.path.join(self.temp_dir, "test.csv")
        result = self.export_service.validate_export_path(csv_path, 'csv')
        self.assertTrue(result['valid'])
        
        # Test invalid format
        result = self.export_service.validate_export_path(csv_path, 'invalid_format')
        self.assertFalse(result['valid'])
        self.assertIn('Unsupported format', result['message'])
    
    def test_export_service_filename_suggestions(self):
        """Test filename suggestion functionality"""
        csv_filename = self.export_service.suggest_filename('csv', 'addresses')
        self.assertEqual(csv_filename, 'addresses.csv')
        
        xlsx_filename = self.export_service.suggest_filename('xlsx', 'data')
        self.assertEqual(xlsx_filename, 'data.xlsx')
    
    def test_export_service_format_descriptions(self):
        """Test format descriptions"""
        csv_desc = self.export_service.get_format_description('csv')
        self.assertIn('CSV', csv_desc)
        
        xlsx_desc = self.export_service.get_format_description('xlsx')
        self.assertIn('Excel', xlsx_desc)
    
    def test_export_service_availability_check(self):
        """Test format availability checking"""
        availability = self.export_service.check_format_availability()
        
        # CSV should always be available
        self.assertTrue(availability.get('csv', False))
        
        # Others depend on installed packages
        self.assertIn('xlsx', availability)
        self.assertIn('shapefile', availability)
        self.assertIn('arcgis', availability)


if __name__ == '__main__':
    unittest.main()