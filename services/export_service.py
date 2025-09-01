"""
Export service for handling data export operations.
Separates export-related business logic from the GUI layer.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)


class ExportService:
    """Service class for data export operations"""
    
    def __init__(self):
        """Initialize export service"""
        self.supported_formats = ['csv', 'xlsx', 'shapefile', 'arcgis']
        self.last_export_path: Optional[str] = None
        
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported export formats
        
        Returns:
            List of supported format strings
        """
        return self.supported_formats.copy()
    
    def validate_export_path(self, file_path: str, format_type: str) -> Dict[str, Any]:
        """
        Validate export path and format
        
        Args:
            file_path: Target file path
            format_type: Export format type
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': False,
            'message': '',
            'file_path': file_path,
            'format_type': format_type
        }
        
        try:
            # Check format type
            if format_type not in self.supported_formats:
                result['message'] = f'Unsupported format: {format_type}. Supported: {", ".join(self.supported_formats)}'
                return result
            
            # Validate file path
            path_obj = Path(file_path)
            
            # Check if parent directory exists
            parent_dir = path_obj.parent
            if not parent_dir.exists():
                try:
                    parent_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    result['message'] = f'Cannot create directory: {str(e)}'
                    return result
            
            # Check write permissions
            if not os.access(parent_dir, os.W_OK):
                result['message'] = 'No write permission for target directory'
                return result
            
            # Check file extension matches format
            expected_extensions = {
                'csv': '.csv',
                'xlsx': '.xlsx',
                'shapefile': '.shp',
                'arcgis': '.gdb'
            }
            
            expected_ext = expected_extensions.get(format_type)
            if expected_ext and not file_path.lower().endswith(expected_ext):
                result['message'] = f'File extension should be {expected_ext} for {format_type} format'
                return result
            
            result['valid'] = True
            result['message'] = 'Export path is valid'
            return result
            
        except Exception as e:
            result['message'] = f'Path validation error: {str(e)}'
            logger.error(f"Export path validation failed: {str(e)}")
            return result
    
    def export_data(self, data: Any, file_path: str, format_type: str, 
                   options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Export data to specified format
        
        Args:
            data: Data to export
            file_path: Target file path
            format_type: Export format type
            options: Additional export options
            
        Returns:
            Dictionary with export results
        """
        result = {
            'success': False,
            'message': '',
            'file_path': file_path,
            'format_type': format_type
        }
        
        if options is None:
            options = {}
        
        try:
            # Validate export path
            validation = self.validate_export_path(file_path, format_type)
            if not validation['valid']:
                result['message'] = validation['message']
                return result
            
            # Check if data is valid
            if data is None:
                result['message'] = 'No data to export'
                return result
            
            # Import export modules based on format
            if format_type == 'csv':
                success, message = self._export_csv(data, file_path, options)
            elif format_type == 'xlsx':
                success, message = self._export_xlsx(data, file_path, options)
            elif format_type == 'shapefile':
                success, message = self._export_shapefile(data, file_path, options)
            elif format_type == 'arcgis':
                success, message = self._export_arcgis(data, file_path, options)
            else:
                success, message = False, f'Unsupported format: {format_type}'
            
            result['success'] = success
            result['message'] = message
            
            if success:
                self.last_export_path = file_path
                logger.info(f"Successfully exported to {file_path} as {format_type}")
            else:
                logger.error(f"Failed to export to {file_path}: {message}")
            
            return result
            
        except Exception as e:
            result['message'] = f'Export failed: {str(e)}'
            logger.error(f"Export operation failed: {str(e)}")
            return result
    
    def _export_csv(self, data: Any, file_path: str, options: Dict[str, Any]) -> tuple[bool, str]:
        """Export data to CSV format"""
        try:
            from export.data_exporter import DataExporter
            exporter = DataExporter()
            exporter.export_to_csv(data, file_path)
            return True, f"Successfully exported to CSV: {file_path}"
        except ImportError:
            return False, "CSV export functionality not available (missing dependencies)"
        except Exception as e:
            return False, f"CSV export failed: {str(e)}"
    
    def _export_xlsx(self, data: Any, file_path: str, options: Dict[str, Any]) -> tuple[bool, str]:
        """Export data to Excel format"""
        try:
            from export.data_exporter import DataExporter
            exporter = DataExporter()
            exporter.export_to_excel(data, file_path)
            return True, f"Successfully exported to Excel: {file_path}"
        except ImportError:
            return False, "Excel export functionality not available (missing dependencies)"
        except Exception as e:
            return False, f"Excel export failed: {str(e)}"
    
    def _export_shapefile(self, data: Any, file_path: str, options: Dict[str, Any]) -> tuple[bool, str]:
        """Export data to Shapefile format"""
        try:
            from export.data_exporter import DataExporter
            exporter = DataExporter()
            
            # For shapefile, we need the directory path, not the file path
            output_dir = str(Path(file_path).parent)
            exporter.export_to_shapefile(data, output_dir)
            return True, f"Successfully exported to Shapefile: {output_dir}"
        except ImportError:
            return False, "Shapefile export functionality not available (missing dependencies)"
        except Exception as e:
            return False, f"Shapefile export failed: {str(e)}"
    
    def _export_arcgis(self, data: Any, file_path: str, options: Dict[str, Any]) -> tuple[bool, str]:
        """Export data to ArcGIS format"""
        try:
            from export.standardized_exporter import StandardizedExporter
            exporter = StandardizedExporter()
            
            # For ArcGIS, we need the directory path
            output_dir = str(Path(file_path).parent)
            exporter.export_to_arcgis(data, output_dir)
            return True, f"Successfully exported to ArcGIS: {output_dir}"
        except ImportError:
            return False, "ArcGIS export functionality not available (missing arcpy or dependencies)"
        except Exception as e:
            return False, f"ArcGIS export failed: {str(e)}"
    
    def get_last_export_path(self) -> Optional[str]:
        """
        Get the path of the last successful export
        
        Returns:
            Last export path or None if no exports performed
        """
        return self.last_export_path
    
    def suggest_filename(self, format_type: str, base_name: str = "addresses") -> str:
        """
        Suggest a filename based on format and base name
        
        Args:
            format_type: Export format type
            base_name: Base name for the file
            
        Returns:
            Suggested filename with appropriate extension
        """
        extensions = {
            'csv': '.csv',
            'xlsx': '.xlsx',
            'shapefile': '.shp',
            'arcgis': '.gdb'
        }
        
        extension = extensions.get(format_type, '.txt')
        return f"{base_name}{extension}"
    
    def get_format_description(self, format_type: str) -> str:
        """
        Get description for a format type
        
        Args:
            format_type: Export format type
            
        Returns:
            Human-readable description of the format
        """
        descriptions = {
            'csv': 'Comma-Separated Values (CSV) - Universal text format',
            'xlsx': 'Microsoft Excel format - Spreadsheet with formatting',
            'shapefile': 'ESRI Shapefile - Standard GIS vector format',
            'arcgis': 'ArcGIS Feature Class - Professional GIS format'
        }
        
        return descriptions.get(format_type, f'Unknown format: {format_type}')
    
    def check_format_availability(self) -> Dict[str, bool]:
        """
        Check which export formats are available based on installed dependencies
        
        Returns:
            Dictionary mapping format names to availability status
        """
        availability = {}
        
        # Check CSV support (should always be available)
        availability['csv'] = True
        
        # Check Excel support
        try:
            import pandas as pd
            import xlsxwriter
            availability['xlsx'] = True
        except ImportError:
            availability['xlsx'] = False
        
        # Check Shapefile support
        try:
            import geopandas
            availability['shapefile'] = True
        except ImportError:
            availability['shapefile'] = False
        
        # Check ArcGIS support
        try:
            import arcpy
            availability['arcgis'] = True
        except ImportError:
            availability['arcgis'] = False
        
        return availability