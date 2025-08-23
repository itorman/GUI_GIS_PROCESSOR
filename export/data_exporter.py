"""
Data exporter for GIS address extraction application.
Handles exporting processed data to various formats.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

# GIS libraries
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataExporter:
    """Exporter for processed address data to various formats"""
    
    def __init__(self):
        """Initialize the data exporter"""
        self.supported_formats = ['csv', 'excel', 'shapefile', 'arcgis']
        
        # Check available libraries
        if not GEOPANDAS_AVAILABLE:
            logger.warning("geopandas not available - shapefile export disabled")
            self.supported_formats.remove('shapefile')
        
        if not ARCPY_AVAILABLE:
            logger.warning("arcpy not available - ArcGIS export disabled")
            self.supported_formats.remove('arcgis')
    
    def export_to_csv(self, df: pd.DataFrame, output_path: str) -> bool:
        """
        Export data to CSV format
        
        Args:
            df: DataFrame to export
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Export to CSV
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            logger.info(f"Data exported to CSV: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            return False
    
    def export_to_excel(self, df: pd.DataFrame, output_path: str) -> bool:
        """
        Export data to Excel format
        
        Args:
            df: DataFrame to export
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create Excel writer with openpyxl engine for better compatibility
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Main data sheet
                df.to_excel(writer, sheet_name='Addresses', index=False)
                
                # Summary sheet
                summary_data = self._create_summary_data(df)
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            logger.info(f"Data exported to Excel: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export to Excel: {e}")
            return False
    
    def export_to_shapefile(self, df: pd.DataFrame, output_dir: str) -> bool:
        """
        Export data to ESRI Shapefile format
        
        Args:
            df: DataFrame to export
            output_dir: Output directory path
            
        Returns:
            True if successful, False otherwise
        """
        if not GEOPANDAS_AVAILABLE:
            logger.error("geopandas not available for shapefile export")
            return False
        
        try:
            logger.info(f"Starting shapefile export. DataFrame shape: {df.shape}")
            logger.info(f"DataFrame columns: {list(df.columns)}")
            logger.info(f"DataFrame info: {df.info()}")
            
            # Ensure output directory exists
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory created/verified: {output_path}")
            
            # Create GeoDataFrame
            logger.info("Creating GeoDataFrame...")
            gdf = self._create_geodataframe(df)
            
            if gdf is None or gdf.empty:
                logger.error("No valid geometries to export")
                return False
            
            logger.info(f"GeoDataFrame created successfully with {len(gdf)} features")
            
            # Export to shapefile
            output_file = output_path / "addresses.shp"
            logger.info(f"Exporting to shapefile: {output_file}")
            gdf.to_file(output_file, driver='ESRI Shapefile')
            
            logger.info(f"Data exported to Shapefile: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export to Shapefile: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def export_to_arcgis(self, df: pd.DataFrame, output_dir: str) -> bool:
        """
        Export data to ArcGIS Feature Class
        
        Args:
            df: DataFrame to export
            output_dir: Output directory path
            
        Returns:
            True if successful, False otherwise
        """
        if not ARCPY_AVAILABLE:
            logger.error("arcpy not available for ArcGIS export")
            return False
        
        try:
            # Ensure output directory exists
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create temporary CSV for ArcGIS import
            temp_csv = output_path / "temp_coordinates.csv"
            df.to_csv(temp_csv, index=False, encoding='utf-8')
            
            # Set up ArcGIS environment
            arcpy.env.workspace = str(output_path)
            arcpy.env.overwriteOutput = True
            
            # Create feature class from XY table
            output_fc = "addresses"
            output_fc_path = str(output_path / f"{output_fc}.shp")
            
            # Check if coordinates are available
            if 'x' in df.columns and 'y' in df.columns:
                x_field = 'x'
                y_field = 'y'
            elif 'longitude' in df.columns and 'latitude' in df.columns:
                x_field = 'longitude'
                y_field = 'latitude'
            else:
                logger.error("No valid coordinates found for ArcGIS export")
                return False
            
            # Create feature class
            arcpy.management.XYTableToPoint(
                str(temp_csv),
                output_fc_path,
                x_field,
                y_field,
                coordinate_system=arcpy.SpatialReference(4326)  # WGS84
            )
            
            # Add attribute fields
            self._add_arcgis_attributes(output_fc_path, df)
            
            # Clean up temporary file
            if temp_csv.exists():
                temp_csv.unlink()
            
            logger.info(f"Data exported to ArcGIS Feature Class: {output_fc_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export to ArcGIS: {e}")
            return False
    
    def _create_geodataframe(self, df: pd.DataFrame) -> Optional['gpd.GeoDataFrame']:
        """Create a GeoDataFrame from the DataFrame"""
        logger.info(f"_create_geodataframe called with DataFrame shape: {df.shape}")
        
        if df.empty:
            logger.warning("DataFrame is empty")
            return None
        
        # Determine coordinate columns with better error handling
        x_col = None
        y_col = None
        
        logger.info(f"Available columns: {list(df.columns)}")
        logger.info(f"Column types: {df.dtypes.to_dict()}")
        
        if 'x' in df.columns and 'y' in df.columns:
            x_col, y_col = 'x', 'y'
            logger.info("Using 'x' and 'y' columns for coordinates")
        elif 'longitude' in df.columns and 'latitude' in df.columns:
            x_col, y_col = 'longitude', 'latitude'
            logger.info("Using 'longitude' and 'latitude' columns for coordinates")
        else:
            logger.error("No valid coordinate columns found. Available columns: %s", list(df.columns))
            return None
        
        # Verify columns exist and are not None
        if x_col is None or y_col is None:
            logger.error("Coordinate columns could not be determined")
            return None
        
        logger.info(f"Selected coordinate columns: x_col='{x_col}', y_col='{y_col}'")
        
        # Check coordinate column contents
        logger.info(f"Sample x values: {df[x_col].head().tolist()}")
        logger.info(f"Sample y values: {df[y_col].head().tolist()}")
        
        # Filter rows with valid coordinates
        valid_coords = df[x_col].notna() & df[y_col].notna()
        valid_df = df[valid_coords].copy()
        
        if valid_df.empty:
            logger.error("No rows with valid coordinates")
            return None
        
        logger.info(f"Found {len(valid_df)} rows with valid coordinates")
        
        # Create Point geometries with better error handling
        geometries = []
        valid_rows = []
        
        for idx, row in valid_df.iterrows():
            try:
                # Get coordinate values safely
                x_val = row[x_col]
                y_val = row[y_col]
                
                logger.debug(f"Processing row {idx}: x={x_val} (type: {type(x_val)}), y={y_val} (type: {type(y_val)})")
                
                # Convert to float if needed
                if not isinstance(x_val, (int, float)):
                    x_val = float(x_val)
                if not isinstance(y_val, (int, float)):
                    y_val = float(y_val)
                
                # Create Point geometry
                point = Point(x_val, y_val)
                geometries.append(point)
                valid_rows.append(idx)
                
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to create geometry for row {idx}: {e} (x={row[x_col]}, y={row[y_col]})")
                continue
            except Exception as e:
                logger.warning(f"Unexpected error creating geometry for row {idx}: {e}")
                continue
        
        if not geometries:
            logger.error("No valid geometries could be created")
            return None
        
        # Create GeoDataFrame with only valid rows
        valid_df_filtered = valid_df.loc[valid_rows]
        gdf = gpd.GeoDataFrame(valid_df_filtered, geometry=geometries, crs='EPSG:4326')
        
        logger.info(f"Successfully created GeoDataFrame with {len(gdf)} features")
        
        return gdf
    
    def _create_summary_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create summary data for Excel export"""
        summary = {
            'Metric': [
                'Total Records',
                'Records with Coordinates',
                'Records with Lat/Lon',
                'Records with X/Y',
                'Unique Addresses'
            ],
            'Value': [
                len(df),
                df[['latitude', 'longitude', 'x', 'y']].notna().any(axis=1).sum(),
                df[['latitude', 'longitude']].notna().all(axis=1).sum(),
                df[['x', 'y']].notna().all(axis=1).sum(),
                df['normalized_address'].nunique()
            ]
        }
        
        # Add coordinate ranges
        for coord_col in ['latitude', 'longitude', 'x', 'y']:
            if coord_col in df.columns and df[coord_col].notna().any():
                summary['Metric'].extend([
                    f'{coord_col} Min',
                    f'{coord_col} Max',
                    f'{coord_col} Mean'
                ])
                summary['Value'].extend([
                    f"{df[coord_col].min():.6f}",
                    f"{df[coord_col].max():.6f}",
                    f"{df[coord_col].mean():.6f}"
                ])
        
        return summary
    
    def _add_arcgis_attributes(self, fc_path: str, df: pd.DataFrame) -> None:
        """Add attribute fields to ArcGIS feature class"""
        try:
            # Add text fields
            if 'normalized_address' in df.columns:
                arcpy.management.AddField(fc_path, "Address", "TEXT", field_length=255)
            
            if 'original_text' in df.columns:
                arcpy.management.AddField(fc_path, "OriginalText", "TEXT", field_length=1000)
            
            # Add numeric fields
            if 'latitude' in df.columns:
                arcpy.management.AddField(fc_path, "Latitude", "DOUBLE")
            
            if 'longitude' in df.columns:
                arcpy.management.AddField(fc_path, "Longitude", "DOUBLE")
            
            # Update attributes
            with arcpy.da.UpdateCursor(fc_path, ["SHAPE@", "Address", "OriginalText", "Latitude", "Longitude"]) as cursor:
                for i, row in enumerate(cursor):
                    if i < len(df):
                        row_data = df.iloc[i]
                        
                        # Update fields
                        if 'normalized_address' in df.columns:
                            row[1] = str(row_data.get('normalized_address', ''))[:255]
                        
                        if 'original_text' in df.columns:
                            row[2] = str(row_data.get('original_text', ''))[:1000]
                        
                        if 'latitude' in df.columns:
                            row[3] = row_data.get('latitude')
                        
                        if 'longitude' in df.columns:
                            row[4] = row_data.get('longitude')
                        
                        cursor.updateRow(row)
            
            logger.info("ArcGIS attributes added successfully")
            
        except Exception as e:
            logger.warning(f"Failed to add ArcGIS attributes: {e}")
    
    def get_export_formats(self) -> list:
        """Get list of supported export formats"""
        return self.supported_formats.copy()
    
    def validate_export_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data before export"""
        validation = {
            'valid': True,
            'status': 'OK',
            'warnings': [],
            'errors': []
        }
        
        if df.empty:
            validation['valid'] = False
            validation['status'] = 'EMPTY'
            validation['errors'].append("DataFrame is empty")
            return validation
        
        # Check for required columns
        required_cols = ['normalized_address']
        for col in required_cols:
            if col not in df.columns:
                validation['valid'] = False
                validation['errors'].append(f"Missing required column: {col}")
        
        # Check for coordinate columns
        coord_cols = ['latitude', 'longitude', 'x', 'y']
        available_coords = [col for col in coord_cols if col in df.columns]
        
        if not available_coords:
            validation['warnings'].append("No coordinate columns found")
        else:
            # Check coordinate validity
            for col in available_coords:
                if df[col].notna().any():
                    invalid_coords = df[col].isna().sum()
                    if invalid_coords > 0:
                        validation['warnings'].append(f"{invalid_coords} rows have missing {col} values")
        
        # Check for duplicate addresses
        if 'normalized_address' in df.columns:
            duplicates = df['normalized_address'].duplicated().sum()
            if duplicates > 0:
                validation['warnings'].append(f"{duplicates} duplicate addresses found")
        
        # Derive status
        if not validation['valid']:
            validation['status'] = 'INVALID'
        elif validation['warnings']:
            validation['status'] = 'WARN'
        else:
            validation['status'] = 'OK'
        
        return validation
    
    def export_all_formats(self, df: pd.DataFrame, base_output_dir: str) -> Dict[str, bool]:
        """
        Export data to all supported formats
        
        Args:
            df: DataFrame to export
            base_output_dir: Base output directory
            
        Returns:
            Dictionary with export results for each format
        """
        results = {}
        base_path = Path(base_output_dir)
        
        # Validate data first
        validation = self.validate_export_data(df)
        if not validation['valid']:
            logger.error("Data validation failed, cannot export")
            return {format: False for format in self.supported_formats}
        
        # Export to each format
        for export_format in self.supported_formats:
            try:
                if export_format == 'csv':
                    output_path = base_path / "addresses.csv"
                    results[export_format] = self.export_to_csv(df, str(output_path))
                
                elif export_format == 'excel':
                    output_path = base_path / "addresses.xlsx"
                    results[export_format] = self.export_to_excel(df, str(output_path))
                
                elif export_format == 'shapefile':
                    results[export_format] = self.export_to_shapefile(df, str(base_path))
                
                elif export_format == 'arcgis':
                    results[export_format] = self.export_to_arcgis(df, str(base_path))
                
            except Exception as e:
                logger.error(f"Export to {export_format} failed: {e}")
                results[export_format] = False
        
        # Log summary
        successful_exports = sum(results.values())
        total_formats = len(results)
        logger.info(f"Export completed: {successful_exports}/{total_formats} formats successful")
        
        return results 