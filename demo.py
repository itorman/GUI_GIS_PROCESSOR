#!/usr/bin/env python3
"""
Demo script for GIS Document Processing Application
Shows how to use the application programmatically
"""

import sys
import os
from pathlib import Path
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def demo_document_processing():
    """Demonstrate document processing workflow"""
    logger.info("=== Document Processing Demo ===")
    
    try:
        from preprocessing.document_processor import DocumentProcessor
        
        # Initialize document processor
        processor = DocumentProcessor(chunk_size=500, enable_ocr=False)
        logger.info("✓ Document processor initialized")
        
        # Check if sample document exists
        sample_file = "sample_document.txt"
        if not Path(sample_file).exists():
            logger.warning(f"Sample document '{sample_file}' not found. Creating one...")
            create_sample_document(sample_file)
        
        # Process document
        logger.info(f"Processing document: {sample_file}")
        text_chunks = processor.process_document(sample_file)
        logger.info(f"✓ Document processed into {len(text_chunks)} chunks")
        
        # Show first chunk
        if text_chunks:
            logger.info(f"First chunk preview: {text_chunks[0][:100]}...")
        
        return text_chunks
        
    except Exception as e:
        logger.error(f"Document processing demo failed: {e}")
        return None

def demo_llm_simulation():
    """Simulate LLM processing (without actual LLM service)"""
    logger.info("=== LLM Processing Demo ===")
    
    try:
        # Simulate LLM response for demo purposes
        simulated_responses = [
            {
                "original_text": "Sample address: 123 Main Street, New York, NY 10001, USA",
                "normalized_address": "123 Main Street, New York, NY 10001, USA",
                "latitude": 40.7128,
                "longitude": -74.0060,
                "x": -74.0060,
                "y": 40.7128
            },
            {
                "original_text": "Another address: 456 Oak Avenue, Los Angeles, CA 90210, USA",
                "normalized_address": "456 Oak Avenue, Los Angeles, CA 90210, USA",
                "latitude": 34.0522,
                "longitude": -118.2437,
                "x": -118.2437,
                "y": 34.0522
            },
            {
                "original_text": "Third location: 789 Pine Road, Chicago, IL 60601, USA",
                "normalized_address": "789 Pine Road, Chicago, IL 60601, USA",
                "latitude": 41.8781,
                "longitude": -87.6298,
                "x": -87.6298,
                "y": 41.8781
            }
        ]
        
        logger.info("✓ Simulated LLM responses generated")
        logger.info(f"✓ {len(simulated_responses)} addresses extracted")
        
        return simulated_responses
        
    except Exception as e:
        logger.error(f"LLM simulation demo failed: {e}")
        return None

def demo_data_processing():
    """Demonstrate data processing and validation"""
    logger.info("=== Data Processing Demo ===")
    
    try:
        from postprocessing.data_processor import DataProcessor
        
        # Initialize data processor
        processor = DataProcessor(target_crs='EPSG:4326')
        logger.info("✓ Data processor initialized")
        
        # Get simulated LLM results
        llm_results = demo_llm_simulation()
        if not llm_results:
            logger.error("No LLM results to process")
            return None
        
        # Process results
        logger.info("Processing LLM results...")
        processed_data = processor.process_results(llm_results)
        logger.info(f"✓ Data processed: {len(processed_data)} records")
        
        # Show statistics
        stats = processor.get_statistics(processed_data)
        logger.info("✓ Statistics generated:")
        for key, value in stats.items():
            if key != 'coordinate_ranges':
                logger.info(f"  {key}: {value}")
        
        # Show coordinate ranges
        if 'coordinate_ranges' in stats:
            logger.info("  Coordinate ranges:")
            for coord, ranges in stats['coordinate_ranges'].items():
                logger.info(f"    {coord}: {ranges['min']:.6f} to {ranges['max']:.6f}")
        
        # Validate data quality
        quality = processor.validate_data_quality(processed_data)
        logger.info(f"✓ Data quality: {quality['status']}")
        if quality['issues']:
            logger.info("  Issues found:")
            for issue in quality['issues']:
                logger.info(f"    - {issue}")
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Data processing demo failed: {e}")
        return None

def demo_export():
    """Demonstrate data export functionality"""
    logger.info("=== Data Export Demo ===")
    
    try:
        from export.data_exporter import DataExporter
        
        # Get processed data
        processed_data = demo_data_processing()
        if processed_data is None or processed_data.empty:
            logger.error("No processed data to export")
            return False
        
        # Initialize exporter
        exporter = DataExporter()
        logger.info("✓ Data exporter initialized")
        
        # Create output directory
        output_dir = Path("demo_output")
        output_dir.mkdir(exist_ok=True)
        logger.info(f"✓ Output directory: {output_dir}")
        
        # Export to different formats
        export_results = {}
        
        # CSV export
        csv_path = output_dir / "addresses.csv"
        if exporter.export_to_csv(processed_data, str(csv_path)):
            logger.info(f"✓ CSV export: {csv_path}")
            export_results['csv'] = True
        else:
            logger.error("✗ CSV export failed")
            export_results['csv'] = False
        
        # Excel export
        excel_path = output_dir / "addresses.xlsx"
        if exporter.export_to_excel(processed_data, str(excel_path)):
            logger.info(f"✓ Excel export: {excel_path}")
            export_results['excel'] = True
        else:
            logger.error("✗ Excel export failed")
            export_results['excel'] = False
        
        # Shapefile export (if geopandas available)
        try:
            if exporter.export_to_shapefile(processed_data, str(output_dir)):
                logger.info(f"✓ Shapefile export: {output_dir}/addresses.shp")
                export_results['shapefile'] = True
            else:
                logger.warning("✗ Shapefile export failed (geopandas may not be available)")
                export_results['shapefile'] = False
        except Exception as e:
            logger.warning(f"✗ Shapefile export failed: {e}")
            export_results['shapefile'] = False
        
        # Summary
        successful_exports = sum(export_results.values())
        total_formats = len(export_results)
        logger.info(f"✓ Export summary: {successful_exports}/{total_formats} formats successful")
        
        return export_results
        
    except Exception as e:
        logger.error(f"Export demo failed: {e}")
        return False

def create_sample_document(filename: str):
    """Create a sample document for demo purposes"""
    sample_text = """
GIS Document Processing Demo Document

This is a sample document containing various addresses and geographic information.

ADDRESSES:
1. 123 Main Street, New York, NY 10001, United States
2. 456 Oak Avenue, Los Angeles, CA 90210, United States
3. 789 Pine Road, Chicago, IL 60601, United States
4. 321 Elm Street, Houston, TX 77001, United States
5. 654 Maple Drive, Phoenix, AZ 85001, United States

COORDINATES:
- New York: Latitude 40.7128, Longitude -74.0060
- Los Angeles: Latitude 34.0522, Longitude -118.2437
- Chicago: Latitude 41.8781, Longitude -87.6298
- Houston: Latitude 29.7604, Longitude -95.3698
- Phoenix: Latitude 33.4484, Longitude -112.0740

ADDITIONAL INFORMATION:
These addresses represent major cities across the United States.
The coordinates are in decimal degrees using the WGS84 coordinate system.
This document demonstrates the application's ability to extract and process geographic data.
"""
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        logger.info(f"✓ Sample document created: {filename}")
    except Exception as e:
        logger.error(f"Failed to create sample document: {e}")

def demo_coordinate_transformation():
    """Demonstrate coordinate transformation capabilities"""
    logger.info("=== Coordinate Transformation Demo ===")
    
    try:
        from postprocessing.data_processor import DataProcessor
        
        # Test different coordinate systems
        test_crs_list = ['EPSG:4326', 'EPSG:3857', 'EPSG:25830']
        
        for crs in test_crs_list:
            try:
                processor = DataProcessor(target_crs=crs)
                logger.info(f"✓ Coordinate transformer initialized for {crs}")
            except Exception as e:
                logger.warning(f"✗ Failed to initialize transformer for {crs}: {e}")
        
        logger.info("✓ Coordinate transformation demo completed")
        return True
        
    except Exception as e:
        logger.error(f"Coordinate transformation demo failed: {e}")
        return False

def demo_file_utilities():
    """Demonstrate file utility functions"""
    logger.info("=== File Utilities Demo ===")
    
    try:
        from utils.file_utils import FileUtils
        
        # Test file validation
        test_file = "demo.py"  # This script
        validation = FileUtils.validate_file(test_file)
        logger.info(f"✓ File validation: {validation['valid']}")
        
        if validation['file_info']:
            info = validation['file_info']
            logger.info(f"  File: {info['name']}")
            logger.info(f"  Size: {info['size_mb']:.2f} MB")
            logger.info(f"  Type: {info['mime_type']}")
        
        # Test file hash
        file_hash = FileUtils.calculate_file_hash(test_file)
        if file_hash:
            logger.info(f"✓ File hash: {file_hash[:16]}...")
        
        # Test MIME type detection
        mime_type = FileUtils.get_mime_type(Path(test_file))
        logger.info(f"✓ MIME type: {mime_type}")
        
        logger.info("✓ File utilities demo completed")
        return True
        
    except Exception as e:
        logger.error(f"File utilities demo failed: {e}")
        return False

def main():
    """Run the complete demo"""
    logger.info("🚀 GIS Document Processing Application Demo")
    logger.info("=" * 60)
    
    demos = [
        ("File Utilities", demo_file_utilities),
        ("Document Processing", demo_document_processing),
        ("LLM Simulation", demo_llm_simulation),
        ("Data Processing", demo_data_processing),
        ("Coordinate Transformation", demo_coordinate_transformation),
        ("Data Export", demo_export),
    ]
    
    successful_demos = 0
    total_demos = len(demos)
    
    for demo_name, demo_func in demos:
        logger.info(f"\n{'='*20} {demo_name} {'='*20}")
        try:
            if demo_func():
                successful_demos += 1
                logger.info(f"✅ {demo_name} completed successfully")
            else:
                logger.error(f"❌ {demo_name} failed")
        except Exception as e:
            logger.error(f"❌ {demo_name} crashed: {e}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info(f"Demo Results: {successful_demos}/{total_demos} demos successful")
    
    if successful_demos == total_demos:
        logger.info("🎉 All demos completed successfully!")
        logger.info("\nThe application is working correctly.")
        logger.info("You can now:")
        logger.info("1. Run the GUI: python main.py")
        logger.info("2. Process real documents")
        logger.info("3. Connect to your LLM service")
    else:
        logger.warning("⚠️  Some demos failed. Check the errors above.")
        logger.info("This may be due to missing dependencies or configuration issues.")
    
    # Show output files
    output_dir = Path("demo_output")
    if output_dir.exists():
        logger.info(f"\nDemo output files created in: {output_dir}")
        for file_path in output_dir.iterdir():
            if file_path.is_file():
                logger.info(f"  📄 {file_path.name}")
    
    return successful_demos == total_demos

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 