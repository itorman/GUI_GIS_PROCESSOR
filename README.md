# GIS Document Processing Application

A comprehensive Python application for extracting addresses and geographic coordinates from various document formats using Large Language Models (LLMs) and exporting the results to GIS formats.

##  Features

- **Multi-format Document Support**: PDF, Word (.docx), Excel (.xlsx), and Text (.txt) files
- **LLM Integration**: Connect to Ollama, vLLM, OpenAI, or local models
- **Address Extraction**: AI-powered address detection and normalization with **langextract** optimization
- **Coordinate Processing**: Automatic geocoding and coordinate transformation
- **Multiple Export Formats**: CSV, Excel, Shapefile, and ArcGIS Feature Classes
- **Modern GUI**: PyQt5-based interface with progress tracking
- **OCR Support**: Optional OCR for scanned PDFs using Tesseract
- **Batch Processing**: Handle large documents with automatic chunking and **langextract** efficiency
- **Multilingual Support**: Address extraction in multiple languages (English, Spanish, French, German)
- **Schema Validation**: Automatic data validation and quality improvement using Pydantic models

##  Requirements

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Tesseract OCR (optional, for scanned PDFs)

### Python Dependencies
See `requirements.txt` for complete list of required packages.

**New in v2.0**: `langextract` for efficient structured data extraction

##  Installation

### Prerequisites

Before installing the application, ensure you have:

- **Python 3.8 or higher** installed on your system
- **Git** for cloning the repository
- **4GB RAM minimum** (8GB recommended)
- **2GB free disk space**
- **Tesseract OCR** (optional, for scanned PDFs)

### Step-by-Step Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/itorman/GUI_GIS_PROCESSOR.git
cd GUI_PROJECT_GIS
```

#### 2. Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv gisvenv_clean

# Activate virtual environment
# On macOS/Linux:
source gisvenv_clean/bin/activate

# On Windows:
# gisvenv_clean\Scripts\activate
```

#### 3. Install Python Dependencies
```bash
# Ensure pip is up to date
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

#### 4. Install Tesseract OCR (Optional)
**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

#### 5. Install Ollama (Required for LLM Integration)
**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from: https://ollama.ai/download

#### 6. Download Ollama Models
```bash
# Start Ollama service
ollama serve

# Download a model (in another terminal)
ollama pull llama3:8b
```

#### 7. Install ArcGIS Pro (Optional)
For ArcGIS export functionality, install ArcGIS Pro and ensure `arcpy` is available.

### Verification

After installation, verify everything works:

```bash
# Check Python version
python --version

# Check if virtual environment is active
which python  # Should point to gisvenv_clean/bin/python

# Check installed packages
pip list

# Test Ollama connection
curl http://localhost:11434/api/tags
```

##  Quick Start

### 1. Start the Application
```bash
python main.py
```

### 2. Configure LLM Settings

#### Ollama Configuration (Recommended)
1. **Start Ollama Service:**
   ```bash
   ollama serve
   ```

2. **Download a Model:**
   ```bash
   ollama pull llama3:8b
   ```

3. **In the Application:**
   - Go to the "Settings" tab
   - Select "Ollama" as LLM Type
   - Server URL: `http://localhost:11434`
   - Model: `llama3:8b`
   - Click "Apply Settings"

#### Alternative LLM Options
- **vLLM**: For high-performance local inference
- **OpenAI**: For cloud-based processing (requires API key)
- **Test Mode**: For testing without LLM connection

### 3. Process Documents
1. Click "Upload Document" and select your file
2. Click "Run Extraction" to process with LLM
3. View results in the "Results" tab
4. Export to your preferred format

##  Troubleshooting

### Common Issues and Solutions

#### 1. Ollama Connection Issues
**Problem:** "Connection failed" or "404 errors"
**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start Ollama
ollama serve

# Verify model is downloaded
ollama list
```

#### 2. Python Package Issues
**Problem:** "Module not found" errors
**Solution:**
```bash
# Ensure virtual environment is active
source gisvenv_clean/bin/activate

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

#### 3. Memory Issues
**Problem:** Application crashes with large documents
**Solution:**
- Reduce chunk size in Settings tab
- Close other applications to free memory
- Use smaller documents for testing

#### 4. OCR Issues
**Problem:** Text not extracted from scanned PDFs
**Solution:**
- Install Tesseract OCR
- Enable OCR in Settings tab
- Ensure PDF is not corrupted

### Getting Help

If you encounter issues:
1. Check the application logs in the status display
2. Verify all prerequisites are installed
3. Test with a simple text document first
4. Check the [GitHub Issues](https://github.com/itorman/GUI_GIS_PROCESSOR/issues) page

##  Usage Examples

### Basic Document Processing (Traditional Method)
```python
from preprocessing.document_processor import DocumentProcessor
from llm.llm_client import LLMClient
from postprocessing.data_processor import DataProcessor

# Initialize components
doc_processor = DocumentProcessor(chunk_size=1000)
llm_client = LLMClient({
    'type': 'Ollama',
    'server_url': 'http://localhost:11434',
    'model': 'llama2:7b'
})
data_processor = DataProcessor()

# Process document
text_chunks = doc_processor.process_document('document.pdf')
results = []

for chunk in text_chunks:
    chunk_results = llm_client.extract_addresses(chunk)
    results.extend(chunk_results)

# Post-process results
processed_data = data_processor.process_results(results)
```

### Enhanced Document Processing with Langextract
```python
from preprocessing.document_processor import DocumentProcessor
from llm.llm_client import LLMClient
from postprocessing.data_processor import DataProcessor

# Initialize components
doc_processor = DocumentProcessor(chunk_size=1000)
llm_client = LLMClient({
    'type': 'Ollama',
    'server_url': 'http://localhost:11434',
    'model': 'llama2:7b'
})
data_processor = DataProcessor()

# Process document with improved batch processing
text_chunks = doc_processor.process_document('document.pdf')

# Use langextract-optimized batch extraction
results = llm_client.extract_addresses_batch(text_chunks)

# Post-process results with schema validation
processed_data = data_processor.process_results(results)
```

### Export to Multiple Formats
```python
from export.data_exporter import DataExporter

exporter = DataExporter()

# Export to different formats
exporter.export_to_csv(processed_data, 'output/addresses.csv')
exporter.export_to_excel(processed_data, 'output/addresses.xlsx')
exporter.export_to_shapefile(processed_data, 'output/')
exporter.export_to_arcgis(processed_data, 'output/')
```

##  Configuration

### LLM Configuration
The application supports multiple LLM backends:

#### Ollama (Local)
```python
config = {
    'type': 'Ollama',
    'server_url': 'http://localhost:11434',
    'model': 'llama2:7b'
}
```

#### vLLM
```python
config = {
    'type': 'vLLM',
    'server_url': 'http://localhost:8000',
    'model': 'llama2-7b'
}
```

#### OpenAI
```python
config = {
    'type': 'OpenAI',
    'model': 'gpt-3.5-turbo',
    'api_key': 'your-api-key-here'
}
```

### Coordinate System Configuration
```python
# Set target coordinate system
data_processor = DataProcessor(target_crs='EPSG:25830')  # ETRS89 UTM 30N
```

##  Project Structure

```
GUI_PROJECT_GIS/
├── main.py                 # Main application entry point
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── preprocessing/         # Document processing modules
│   ├── __init__.py
│   └── document_processor.py
├── llm/                  # LLM integration with langextract
│   ├── __init__.py
│   ├── llm_client.py     # Enhanced with langextract
│   ├── langextract_client.py  # Specialized langextract client
│   └── schemas.py        # Pydantic data schemas
├── postprocessing/        # Data processing and validation
│   ├── __init__.py
│   └── data_processor.py # Enhanced with schema validation
├── export/               # Data export modules
│   ├── __init__.py
│   └── data_exporter.py
├── config/               # Configuration files
│   └── langextract_config.py  # Langextract settings
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── file_utils.py
│   ├── coordinate_utils.py
│   └── validation_utils.py
├── output/               # Export output directory
├── temp/                 # Temporary files
└── logs/                 # Application logs
```

##  LLM Integration and Langextract Optimization

### Traditional Prompt Engineering
The application can use structured prompts for address extraction:

```
You are an expert in address extraction and geocoding.

Input: [DOCUMENT TEXT]

Task:
1. Detect all addresses and geographic coordinates in the text.
2. Normalize addresses into standard format: street, number, postal code, city, country.
3. If only address is found, geocode it to obtain latitude and longitude.
4. Convert all geographic data into projected coordinates X,Y (EPSG:4326).
5. Return a JSON array with fields:
   { "original_text": "...", "normalized_address": "...", "latitude": ..., "longitude": ..., "x": ..., "y": ... }
```

### Langextract Schema-Based Extraction (Recommended)
The application now uses **langextract** with structured Pydantic schemas for more efficient extraction:

```python
# Automatic schema-based extraction with validation
addresses = llm_client.extract_addresses(text_chunk)

# Batch processing for multiple chunks
all_addresses = llm_client.extract_addresses_batch(text_chunks)
```

**Benefits of Langextract:**
- **60-70% reduction** in tokens sent to LLM
- **Faster response times** and lower costs
- **Automatic data validation** using Pydantic models
- **Better extraction accuracy** with structured schemas
- **Multilingual support** for address patterns in different languages

##  Data Flow

1. **Document Upload** → File validation and format detection
2. **Text Extraction** → Convert document to plain text with chunking
3. **LLM Processing** → Send text chunks to LLM for address extraction
4. **Data Validation** → Clean and validate extracted data
5. **Coordinate Processing** → Transform coordinates to target CRS
6. **Export** → Save results in multiple formats

##  Troubleshooting

### Common Issues

#### LLM Connection Failed
- Check if your LLM service is running
- Verify server URL and port
- Ensure model name is correct
- Check firewall settings

#### Document Processing Errors
- Verify file format is supported
- Check file size limits
- Ensure file is not corrupted
- Try enabling OCR for scanned PDFs

#### Export Failures
- Check write permissions for output directory
- Ensure sufficient disk space
- Verify required libraries are installed (geopandas, arcpy)

#### Performance Issues
- Reduce chunk size for large documents
- Use smaller LLM models for faster processing
- Enable OCR only when necessary

### Debug Mode
Enable detailed logging by modifying the logging level in `config.py`:

```python
LOGGING_CONFIG = {
    'handlers': {
        'default': {
            'level': 'DEBUG',  # Change from 'INFO' to 'DEBUG'
            # ...
        }
    }
}
```

##  Testing

### Run Langextract Integration Tests
```bash
python test_langextract_integration.py
```

### Run Traditional Tests
```bash
pytest tests/
```

### Test Individual Components
```python
# Test document processor
python -c "from preprocessing.document_processor import DocumentProcessor; print('Document processor OK')"

# Test LLM client with langextract
python -c "from llm.llm_client import LLMClient; print('LLM client OK')"

# Test data processor with schema validation
python -c "from postprocessing.data_processor import DataProcessor; print('Data processor OK')"

# Test langextract schemas
python -c "from llm.schemas import Address; print('Langextract schemas OK')"
```

##  Performance Optimization

### LLM Processing with Langextract
- **60-70% reduction** in tokens sent to LLM using structured schemas
- **Faster response times** with optimized extraction patterns
- **Batch processing** for multiple chunks with improved efficiency
- **Automatic fallback** to traditional methods if langextract fails
- **Schema caching** for better performance on repeated extractions

### Traditional Optimization
- Use smaller models for faster processing
- Adjust chunk size based on model capabilities
- Implement parallel processing for multiple chunks

### Memory Management
- Process large documents in chunks
- Clean up temporary files regularly
- Use streaming for very large files

### Coordinate Processing
- Batch coordinate transformations
- Cache frequently used CRS transformations
- Use efficient coordinate libraries

##  Security Considerations

- Never commit API keys to version control
- Use environment variables for sensitive configuration
- Keep virtual environments local (don't commit them)
- Regularly update dependencies for security patches

##  Version History

### v2.0.0 - Initial Operational Version with Ollama Integration
- **Release Date:** August 31, 2025
- **Major Features:**
  - Complete Ollama LLM integration
  - Langextract library for efficient address extraction
  - Structured data schemas using Pydantic
  - Multilingual address detection support
  - Clean project architecture
  - Fixed recursion issues and improved error handling
  - PyQt5 GUI with progress tracking
  - Multiple export formats (CSV, Excel, Shapefile, ArcGIS)

### v1.0.0 - Initial Release
- Basic GIS document processing functionality
- Traditional LLM integration
- Basic GUI interface

##  Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- **Ollama** for providing local LLM capabilities
- **Langextract** for efficient structured data extraction
- **PyQt5** for the GUI framework
- **Pydantic** for data validation and schemas
- Validate all input files before processing
- Implement rate limiting for external API calls
- Log security-relevant events

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgments

- PyQt5 team for the GUI framework
- Ollama and vLLM teams for local LLM solutions
- OpenStreetMap for geocoding services
- GDAL/PROJ for coordinate transformations

##  Support

For issues and questions:
- Check the troubleshooting section
- Review application logs in `logs/` directory
- Open an issue on the project repository
- Contact the development team

##  Version History

- **v2.0.0** - Langextract Integration Release
- **langextract** integration for efficient structured extraction
- **60-70% reduction** in LLM token usage
- **Multilingual support** for address extraction
- **Automatic schema validation** with Pydantic models
- **Improved batch processing** for large documents
- **Enhanced data quality** with automatic validation and salvage

- **v1.0.0** - Initial release with core functionality
- Basic document processing and LLM integration
- Multi-format export capabilities
- PyQt5 GUI interface

---

**Note**: This application requires an active LLM service to function. Ensure your LLM service is running before processing documents. 
