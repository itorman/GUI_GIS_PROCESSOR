"""
Services package for business logic separation.
Contains service classes that handle core application functionality
independent of the GUI layer.
"""

# Import service classes
try:
    from .document_service import DocumentService
except ImportError:
    DocumentService = None

try:
    from .processing_service import ProcessingService
except ImportError:
    ProcessingService = None

try:
    from .export_service import ExportService
except ImportError:
    ExportService = None

__all__ = ['DocumentService', 'ProcessingService', 'ExportService']