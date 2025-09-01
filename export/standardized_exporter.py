"""
Standardized data exporter that produces the exact table structure required:
- original_text
- inferred_address  
- confidence_level
- latitude_dd
- longitude_dd
"""

import os
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# Import standardized schemas
try:
    from llm.standardized_schemas import (
        StandardizedExtraction, 
        ExportConfiguration,
        STANDARDIZED_COLUMN_ORDER,
        COLUMN_DESCRIPTIONS
    )
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False

# GIS libraries (optional)
try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

try:
    import arcpy
    ARCPY_AVAILABLE = True
except ImportError:
    ARCPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class StandardizedExporter:
    """
    Exporter that produces standardized table structure for all formats
    """
    
    def __init__(self):
        """Initialize the standardized exporter"""
        self.supported_formats = ['csv', 'excel', 'shapefile']
        
        # Check optional dependencies
        if not GEOPANDAS_AVAILABLE:
            logger.warning("geopandas not available - shapefile export disabled")
            if 'shapefile' in self.supported_formats:
                self.supported_formats.remove('shapefile')
        
        if not ARCPY_AVAILABLE:
            logger.warning("arcpy not available - ArcGIS export disabled")
            if 'arcgis' in self.supported_formats:
                self.supported_formats.remove('arcgis')
        else:
            self.supported_formats.append('arcgis')
    
    def export_standardized_data(self, 
                                extractions: List[Union[StandardizedExtraction, Dict]], 
                                output_path: str, 
                                export_format: str,
                                config: Optional[ExportConfiguration] = None) -> bool:
        """
        Export data in standardized format
        
        Args:
            extractions: List of StandardizedExtraction objects or dictionaries
            output_path: Output file path
            export_format: Format to export (csv, excel, shapefile, arcgis)
            config: Optional export configuration
            
        Returns:
            True if successful, False otherwise
        """
        if not extractions:
            logger.warning("No extractions provided for export")
            return False
        
        # Convert to standardized DataFrame
        df = self._create_standardized_dataframe(extractions, config)
        
        if df.empty:
            logger.warning("No valid data to export after filtering")
            return False
        
        # Export based on format
        try:
            if export_format.lower() == 'csv':
                return self._export_to_csv(df, output_path, config)
            elif export_format.lower() == 'excel':
                return self._export_to_excel(df, output_path, config)
            elif export_format.lower() == 'shapefile':
                return self._export_to_shapefile(df, output_path, config)
            elif export_format.lower() == 'arcgis':
                return self._export_to_arcgis(df, output_path, config)
            else:
                logger.error(f"Unsupported export format: {export_format}")
                return False
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def _create_standardized_dataframe(self, 
                                     extractions: List[Union[StandardizedExtraction, Dict]], 
                                     config: Optional[ExportConfiguration] = None) -> pd.DataFrame:
        """Create standardized DataFrame from extractions"""
        
        # Default config
        if config is None:
            config = ExportConfiguration(
                export_format='csv',
                min_confidence_filter=0.0,
                coordinate_precision=6
            )
        
        standardized_data = []
        
        for extraction in extractions:
            # Convert to dict if StandardizedExtraction object
            if hasattr(extraction, '__dict__'):
                data = extraction.__dict__
            else:
                data = extraction
            
            # Map fields to standardized format
            standardized_record = {
                'original_text': data.get('original_text', ''),
                'inferred_address': data.get('inferred_address', data.get('normalized_address', '')),
                'confidence_level': float(data.get('confidence_level', data.get('confidence', 0.0))),
                'latitude_dd': round(float(data.get('latitude_dd', data.get('latitude', 0.0))), 
                                   config.coordinate_precision),
                'longitude_dd': round(float(data.get('longitude_dd', data.get('longitude', 0.0))), 
                                    config.coordinate_precision)
            }
            
            # Apply confidence filter
            if standardized_record['confidence_level'] >= config.min_confidence_filter:
                standardized_data.append(standardized_record)
        
        # Create DataFrame with standardized column order
        df = pd.DataFrame(standardized_data)
        
        if not df.empty:
            # Ensure correct column order
            df = df[STANDARDIZED_COLUMN_ORDER]
            
            # Validate coordinates
            df = self._validate_coordinates(df)
            
            logger.info(f"Created standardized DataFrame: {len(df)} records, {len(df.columns)} columns")
        
        return df
    
    def _validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean coordinate data"""
        if df.empty:
            return df
        
        # Check for valid coordinate ranges
        valid_lat = (df['latitude_dd'].between(-90, 90))
        valid_lon = (df['longitude_dd'].between(-180, 180))
        valid_coords = valid_lat & valid_lon
        
        invalid_count = len(df) - valid_coords.sum()
        if invalid_count > 0:
            logger.warning(f"Removing {invalid_count} records with invalid coordinates")
            df = df[valid_coords].copy()
        
        # Remove records with zero coordinates (unless specifically valid)
        zero_coords = (df['latitude_dd'] == 0) & (df['longitude_dd'] == 0)
        if zero_coords.any():
            logger.warning(f"Found {zero_coords.sum()} records with zero coordinates")
            # Only remove if all are zero (likely invalid)
            if zero_coords.sum() == len(df):
                logger.error("All records have zero coordinates - this may indicate an issue")
        
        return df
    
    def _export_to_csv(self, df: pd.DataFrame, output_path: str, 
                      config: Optional[ExportConfiguration] = None) -> bool:
        """Export to CSV with metadata header"""
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create metadata header
            metadata_lines = [
                "# GIS Document Processing - Standardized Export",
                f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"# Records: {len(df)}",
                "# Columns:",
                *[f"#   {col}: {COLUMN_DESCRIPTIONS.get(col, 'No description')}" 
                  for col in df.columns],
                "# Coordinate System: WGS84 (EPSG:4326)",
                "# Format: Decimal Degrees",
                ""
            ]
            
            # Write metadata and data
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(metadata_lines))
                df.to_csv(f, index=False)
            
            logger.info(f"Successfully exported {len(df)} records to CSV: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            return False
    
    def _export_to_excel(self, df: pd.DataFrame, output_path: str, 
                        config: Optional[ExportConfiguration] = None) -> bool:
        """Export to Excel with multiple sheets"""
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Main data sheet
                df.to_excel(writer, sheet_name='Standardized_Data', index=False)
                
                # Metadata sheet
                metadata_df = self._create_metadata_dataframe(df, config)
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                
                # Summary statistics sheet
                summary_df = self._create_summary_dataframe(df)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Column descriptions sheet
                descriptions_df = pd.DataFrame([
                    {'Column': col, 'Description': COLUMN_DESCRIPTIONS.get(col, 'No description')}
                    for col in df.columns
                ])
                descriptions_df.to_excel(writer, sheet_name='Column_Descriptions', index=False)
            
            logger.info(f"Successfully exported {len(df)} records to Excel: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Excel export failed: {e}")
            return False
    
    def _export_to_shapefile(self, df: pd.DataFrame, output_path: str, 
                           config: Optional[ExportConfiguration] = None) -> bool:
        """Export to Shapefile format"""
        if not GEOPANDAS_AVAILABLE:
            logger.error("geopandas not available for shapefile export")
            return False
        
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create geometry from coordinates
            geometry = [Point(row['longitude_dd'], row['latitude_dd']) 
                       for _, row in df.iterrows()]
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
            
            # Ensure shapefile field name limits (10 chars)
            column_mapping = {
                'original_text': 'orig_text',
                'inferred_address': 'inf_addr',
                'confidence_level': 'conf_lvl',
                'latitude_dd': 'lat_dd',
                'longitude_dd': 'lon_dd'
            }
            
            gdf_renamed = gdf.rename(columns=column_mapping)
            
            # Truncate text fields to shapefile limits
            for col in ['orig_text', 'inf_addr']:
                if col in gdf_renamed.columns:
                    gdf_renamed[col] = gdf_renamed[col].astype(str).str[:254]
            
            # Export to shapefile
            gdf_renamed.to_file(output_path, driver='ESRI Shapefile')
            
            logger.info(f"Successfully exported {len(df)} records to Shapefile: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Shapefile export failed: {e}")
            return False
    
    def _export_to_arcgis(self, df: pd.DataFrame, output_path: str, 
                         config: Optional[ExportConfiguration] = None) -> bool:
        """Export to ArcGIS Feature Class"""
        if not ARCPY_AVAILABLE:
            logger.error("arcpy not available for ArcGIS export")
            return False
        
        try:
            # Implementation would go here
            # This is a placeholder as arcpy setup is complex
            logger.warning("ArcGIS export not fully implemented yet")
            return False
            
        except Exception as e:
            logger.error(f"ArcGIS export failed: {e}")
            return False
    
    def _create_metadata_dataframe(self, df: pd.DataFrame, 
                                 config: Optional[ExportConfiguration] = None) -> pd.DataFrame:
        """Create metadata information DataFrame"""
        metadata = [
            {'Property': 'Export Date', 'Value': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
            {'Property': 'Total Records', 'Value': len(df)},
            {'Property': 'Coordinate System', 'Value': 'WGS84 (EPSG:4326)'},
            {'Property': 'Format', 'Value': 'Decimal Degrees'},
            {'Property': 'Min Confidence', 'Value': df['confidence_level'].min() if not df.empty else 0},
            {'Property': 'Max Confidence', 'Value': df['confidence_level'].max() if not df.empty else 0},
            {'Property': 'Avg Confidence', 'Value': df['confidence_level'].mean() if not df.empty else 0},
        ]
        
        if config:
            metadata.extend([
                {'Property': 'Export Format', 'Value': config.export_format},
                {'Property': 'Confidence Filter', 'Value': config.min_confidence_filter},
                {'Property': 'Coordinate Precision', 'Value': config.coordinate_precision},
            ])
        
        return pd.DataFrame(metadata)
    
    def _create_summary_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics DataFrame"""
        if df.empty:
            return pd.DataFrame({'Statistic': ['No data'], 'Value': ['N/A']})
        
        summary = [
            {'Statistic': 'Total Records', 'Value': len(df)},
            {'Statistic': 'Records with High Confidence (>0.8)', 
             'Value': len(df[df['confidence_level'] > 0.8])},
            {'Statistic': 'Records with Medium Confidence (0.5-0.8)', 
             'Value': len(df[(df['confidence_level'] >= 0.5) & (df['confidence_level'] <= 0.8)])},
            {'Statistic': 'Records with Low Confidence (<0.5)', 
             'Value': len(df[df['confidence_level'] < 0.5])},
            {'Statistic': 'Unique Addresses', 'Value': df['inferred_address'].nunique()},
            {'Statistic': 'Latitude Range', 
             'Value': f"{df['latitude_dd'].min():.4f} to {df['latitude_dd'].max():.4f}"},
            {'Statistic': 'Longitude Range', 
             'Value': f"{df['longitude_dd'].min():.4f} to {df['longitude_dd'].max():.4f}"},
        ]
        
        return pd.DataFrame(summary)
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported export formats"""
        return self.supported_formats.copy()
    
    def validate_export_data(self, extractions: List[Union[StandardizedExtraction, Dict]]) -> Dict[str, Any]:
        """Validate data before export"""
        if not extractions:
            return {'valid': False, 'error': 'No data provided'}
        
        issues = []
        valid_count = 0
        
        for i, extraction in enumerate(extractions):
            if hasattr(extraction, '__dict__'):
                data = extraction.__dict__
            else:
                data = extraction
            
            # Check required fields
            required_fields = ['original_text', 'confidence_level', 'latitude_dd', 'longitude_dd']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                issues.append(f"Record {i}: Missing fields {missing_fields}")
                continue
            
            # Check coordinate validity
            lat = data.get('latitude_dd', 0)
            lon = data.get('longitude_dd', 0)
            
            if not (-90 <= lat <= 90):
                issues.append(f"Record {i}: Invalid latitude {lat}")
                continue
            
            if not (-180 <= lon <= 180):
                issues.append(f"Record {i}: Invalid longitude {lon}")
                continue
            
            valid_count += 1
        
        return {
            'valid': len(issues) == 0,
            'total_records': len(extractions),
            'valid_records': valid_count,
            'issues': issues[:10]  # Limit to first 10 issues
        }
