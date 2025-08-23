#!/usr/bin/env python3
"""
Test script for GIS Document Processing Application
Tests individual components without requiring LLM service
"""

import sys
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all modules can be imported"""
    logger.info("Testing module imports...")
    
    try:
        from preprocessing.document_processor import DocumentProcessor
        logger.info("✓ DocumentProcessor imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import DocumentProcessor: {e}")
        return False
    
    try:
        from llm.llm_client import LLMClient
        logger.info("✓ LLMClient imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import LLMClient: {e}")
        return False
    
    try:
        from postprocessing.data_processor import DataProcessor
        logger.info("✓ DataProcessor imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import DataProcessor: {e}")
        return False
    
    try:
        from export.data_exporter import DataExporter
        logger.info("✓ DataExporter imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import DataExporter: {e}")
        return False
    
    try:
        from utils.file_utils import FileUtils
        logger.info("✓ FileUtils imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import FileUtils: {e}")
        return False
    
    return True

def test_document_processor():
    """Test document processor functionality"""
    logger.info("Testing document processor...")
    
    try:
        from preprocessing.document_processor import DocumentProcessor
        
        # Test initialization
        processor = DocumentProcessor(chunk_size=500)
        logger.info("✓ DocumentProcessor initialized")
        
        # Test supported formats
        formats = processor.get_supported_formats()
        logger.info(f"✓ Supported formats: {formats}")
        
        # Test text chunking
        test_text = "This is a test document. " * 100  # Create long text
        chunks = processor._chunk_text(test_text)
        logger.info(f"✓ Text chunking: {len(chunks)} chunks created")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Document processor test failed: {e}")
        return False

def test_data_processor():
    """Test data processor functionality"""
    logger.info("Testing data processor...")
    
    try:
        from postprocessing.data_processor import DataProcessor
        import pandas as pd
        
        # Test initialization
        processor = DataProcessor()
        logger.info("✓ DataProcessor initialized")
        
        # Test with sample data
        sample_data = [
            {
                'original_text': 'Sample address: 123 Main St, City, Country',
                'normalized_address': '123 Main St, City, Country',
                'latitude': 40.7128,
                'longitude': -74.0060,
                'x': -74.0060,
                'y': 40.7128
            }
        ]
        
        # Process results
        df = processor.process_results(sample_data)
        logger.info(f"✓ Data processing: {len(df)} records processed")
        
        # Test statistics
        stats = processor.get_statistics(df)
        logger.info(f"✓ Statistics generated: {stats['total_records']} records")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Data processor test failed: {e}")
        return False

def test_data_exporter():
    """Test data exporter functionality"""
    logger.info("Testing data exporter...")
    
    try:
        from export.data_exporter import DataExporter
        import pandas as pd
        
        # Test initialization
        exporter = DataExporter()
        logger.info("✓ DataExporter initialized")
        
        # Test supported formats
        formats = exporter.get_export_formats()
        logger.info(f"✓ Supported export formats: {formats}")
        
        # Test validation
        sample_df = pd.DataFrame({
            'normalized_address': ['Test Address'],
            'latitude': [40.7128],
            'longitude': [-74.0060]
        })
        
        validation = exporter.validate_export_data(sample_df)
        status = validation.get('status') or ('OK' if validation.get('valid') else 'INVALID')
        logger.info(f"✓ Data validation: {status}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Data exporter test failed: {e}")
        return False

def test_file_utils():
    """Test file utility functions"""
    logger.info("Testing file utilities...")
    
    try:
        from utils.file_utils import FileUtils
        
        # Test file validation
        test_file = Path(__file__)  # Use this script as test file
        validation = FileUtils.validate_file(str(test_file))
        logger.info(f"✓ File validation: {validation['valid']}")
        
        # Test MIME type detection
        mime_type = FileUtils.get_mime_type(test_file)
        logger.info(f"✓ MIME type detection: {mime_type}")
        
        # Test file hash calculation
        file_hash = FileUtils.calculate_file_hash(str(test_file))
        if file_hash:
            logger.info(f"✓ File hash calculation: {file_hash[:8]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ File utilities test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    logger.info("Testing configuration...")
    
    try:
        import config
        
        # Test basic config
        logger.info(f"✓ App name: {config.APP_NAME}")
        logger.info(f"✓ App version: {config.APP_VERSION}")
        logger.info(f"✓ Default CRS: {config.DEFAULT_CRS}")
        
        # Test directory creation
        logger.info(f"✓ Output directory: {config.OUTPUT_DIR}")
        logger.info(f"✓ Temp directory: {config.TEMP_DIR}")
        logger.info(f"✓ Log directory: {config.LOG_DIR}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False

def create_sample_data():
    """Create sample data for testing"""
    logger.info("Creating sample data...")
    
    try:
        # Create sample text file
        sample_text = """
        Sample Document with Addresses
        
        Here are some sample addresses:
        1. 123 Main Street, New York, NY 10001, USA
        2. 456 Oak Avenue, Los Angeles, CA 90210, USA
        3. 789 Pine Road, Chicago, IL 60601, USA
        
        Coordinates found:
        - Latitude: 40.7128, Longitude: -74.0060 (New York)
        - Latitude: 34.0522, Longitude: -118.2437 (Los Angeles)
        - Latitude: 41.8781, Longitude: -87.6298 (Chicago)
        """
        
        sample_file = Path("sample_document.txt")
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        
        logger.info(f"✓ Sample document created: {sample_file}")
        return str(sample_file)
        
    except Exception as e:
        logger.error(f"✗ Failed to create sample data: {e}")
        return None

def main():
    """Run all tests"""
    logger.info("Starting GIS Document Processing Application Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_configuration),
        ("Document Processor", test_document_processor),
        ("Data Processor", test_data_processor),
        ("Data Exporter", test_data_exporter),
        ("File Utilities", test_file_utils),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
            logger.info(f"✓ {test_name} PASSED")
        else:
            logger.error(f"✗ {test_name} FAILED")
    
    # Create sample data
    logger.info("\n--- Sample Data Creation ---")
    sample_file = create_sample_data()
    if sample_file:
        logger.info("✓ Sample data created successfully")
        logger.info(f"  You can use '{sample_file}' for testing")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Application is ready to use.")
        logger.info("\nNext steps:")
        logger.info("1. Start your LLM service (Ollama, vLLM, etc.)")
        logger.info("2. Run: python main.py")
        logger.info("3. Configure LLM settings in the GUI")
        logger.info("4. Upload and process documents")
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        logger.info("\nTroubleshooting:")
        logger.info("1. Install missing dependencies: pip install -r requirements.txt")
        logger.info("2. Check Python version (3.8+ required)")
        logger.info("3. Verify all required libraries are available")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 