"""
Data processor for post-processing LLM extraction results.
Handles validation, cleaning, and coordinate processing.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

# GIS libraries
try:
    import pyproj
    from pyproj import Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False

try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Post-processor for LLM extraction results"""
    
    def __init__(self, target_crs: str = 'EPSG:4326'):
        """
        Initialize data processor
        
        Args:
            target_crs: Target coordinate reference system
        """
        self.target_crs = target_crs
        self.transformer = None
        
        # Initialize coordinate transformer if pyproj is available
        if PYPROJ_AVAILABLE:
            try:
                self.transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
                logger.info(f"Coordinate transformer initialized: WGS84 -> {target_crs}")
            except Exception as e:
                logger.warning(f"Failed to initialize coordinate transformer: {e}")
    
    def process_results(self, llm_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process LLM extraction results into a clean DataFrame
        
        Args:
            llm_results: List of dictionaries from LLM extraction
            
        Returns:
            Processed pandas DataFrame
        """
        if not llm_results:
            logger.warning("No results to process")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(llm_results)
        
        # Clean and validate data
        df = self._clean_dataframe(df)
        
        # Process coordinates
        df = self._process_coordinates(df)
        
        # Remove duplicates
        df = self._remove_duplicates(df)
        
        # Sort by address for better organization
        df = df.sort_values('normalized_address').reset_index(drop=True)
        
        logger.info(f"Processed {len(df)} records successfully")
        return df
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate DataFrame data"""
        if df.empty:
            return df
        
        # Ensure required columns exist
        required_columns = ['original_text', 'normalized_address']
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''
        
        # Clean text fields
        text_columns = ['original_text', 'normalized_address']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                # Remove empty or very short entries
                df = df[df[col].str.len() > 2]
        
        # Clean coordinate columns
        coord_columns = ['latitude', 'longitude', 'x', 'y']
        for col in coord_columns:
            if col in df.columns:
                # Convert to numeric, invalid values become NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with no valid coordinates
        coord_mask = (
            df['latitude'].notna() | 
            df['longitude'].notna() | 
            df['x'].notna() | 
            df['y'].notna()
        )
        df = df[coord_mask]
        
        return df
    
    def _process_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate coordinates"""
        if df.empty:
            return df
        
        # Fill missing lat/lon from x/y if available
        df = self._fill_missing_coordinates(df)
        
        # Validate coordinate ranges
        df = self._validate_coordinate_ranges(df)
        
        # Transform coordinates if needed
        if self.transformer and self.target_crs != 'EPSG:4326':
            df = self._transform_coordinates(df)
        
        # Ensure x,y are always populated (use lat,lon as fallback)
        df = self._ensure_xy_coordinates(df)
        
        return df
    
    def _fill_missing_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing latitude/longitude from x/y coordinates"""
        if df.empty:
            return df
        
        # If we have x,y but missing lat,lon, assume x,y are in WGS84
        missing_lat = df['latitude'].isna() & df['x'].notna()
        missing_lon = df['longitude'].isna() & df['y'].notna()
        
        if missing_lat.any():
            df.loc[missing_lat, 'latitude'] = df.loc[missing_lat, 'x']
        
        if missing_lon.any():
            df.loc[missing_lon, 'longitude'] = df.loc[missing_lon, 'y']
        
        # If we have lat,lon but missing x,y, copy them
        missing_x = df['x'].isna() & df['longitude'].notna()
        missing_y = df['y'].isna() & df['latitude'].notna()
        
        if missing_x.any():
            df.loc[missing_x, 'x'] = df.loc[missing_x, 'longitude']
        
        if missing_y.any():
            df.loc[missing_y, 'y'] = df.loc[missing_y, 'latitude']
        
        return df
    
    def _validate_coordinate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate coordinate ranges and remove invalid entries"""
        if df.empty:
            return df
        
        # Validate latitude (-90 to 90)
        valid_lat = (df['latitude'] >= -90) & (df['latitude'] <= 90)
        invalid_lat = ~valid_lat & df['latitude'].notna()
        
        if invalid_lat.any():
            logger.warning(f"Found {invalid_lat.sum()} invalid latitude values")
            df.loc[invalid_lat, 'latitude'] = np.nan
        
        # Validate longitude (-180 to 180)
        valid_lon = (df['longitude'] >= -180) & (df['longitude'] <= 180)
        invalid_lon = ~valid_lon & df['longitude'].notna()
        
        if invalid_lon.any():
            logger.warning(f"Found {invalid_lon.sum()} invalid longitude values")
            df.loc[invalid_lon, 'longitude'] = np.nan
        
        # Remove rows with no valid coordinates after validation
        valid_coords = (
            df['latitude'].notna() | 
            df['longitude'].notna() | 
            df['x'].notna() | 
            df['y'].notna()
        )
        df = df[valid_coords]
        
        return df
    
    def _transform_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform coordinates to target CRS"""
        if not self.transformer or df.empty:
            return df
        
        # Transform lat,lon to target CRS
        valid_coords = df['latitude'].notna() & df['longitude'].notna()
        
        if valid_coords.any():
            try:
                # Get coordinate values
                lon_values = df.loc[valid_coords, 'longitude'].values
                lat_values = df.loc[valid_coords, 'latitude'].values
                
                logger.debug(f"Transforming {len(lon_values)} coordinates from WGS84 to {self.target_crs}")
                
                # Transform coordinates with better error handling
                transformed_result = self.transformer.transform(lon_values, lat_values)
                
                # Check if we got the expected result
                if isinstance(transformed_result, tuple) and len(transformed_result) == 2:
                    x_transformed, y_transformed = transformed_result
                    
                    # Update x,y with transformed coordinates
                    df.loc[valid_coords, 'x'] = x_transformed
                    df.loc[valid_coords, 'y'] = y_transformed
                    
                    logger.info(f"Transformed {valid_coords.sum()} coordinates to {self.target_crs}")
                else:
                    logger.warning(f"Unexpected transformer result format: {type(transformed_result)}")
                    # Fallback: copy lat,lon to x,y
                    df.loc[valid_coords, 'x'] = df.loc[valid_coords, 'longitude']
                    df.loc[valid_coords, 'y'] = df.loc[valid_coords, 'latitude']
                
            except Exception as e:
                logger.error(f"Coordinate transformation failed: {e}")
                # Fallback: copy lat,lon to x,y
                df.loc[valid_coords, 'x'] = df.loc[valid_coords, 'longitude']
                df.loc[valid_coords, 'y'] = df.loc[valid_coords, 'latitude']
        
        return df
    
    def _ensure_xy_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure x,y coordinates are always populated"""
        if df.empty:
            return df
        
        # If x,y are missing but lat,lon available, copy them
        missing_x = df['x'].isna() & df['longitude'].notna()
        missing_y = df['y'].isna() & df['latitude'].notna()
        
        if missing_x.any():
            df.loc[missing_x, 'x'] = df.loc[missing_x, 'longitude']
        
        if missing_y.any():
            df.loc[missing_y, 'y'] = df.loc[missing_y, 'latitude']
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate entries based on address and coordinates"""
        if df.empty:
            return df
        
        initial_count = len(df)
        
        # Remove exact duplicates
        df = df.drop_duplicates()
        
        # Remove duplicates based on normalized address (keep first occurrence)
        df = df.drop_duplicates(subset=['normalized_address'], keep='first')
        
        # Remove duplicates based on coordinates (keep first occurrence)
        coord_cols = ['latitude', 'longitude', 'x', 'y']
        available_coord_cols = [col for col in coord_cols if col in df.columns]
        
        if available_coord_cols:
            df = df.drop_duplicates(subset=available_coord_cols, keep='first')
        
        final_count = len(df)
        if initial_count != final_count:
            logger.info(f"Removed {initial_count - final_count} duplicate entries")
        
        return df
    
    def get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about the processed data"""
        if df.empty:
            return {}
        
        stats = {
            'total_records': len(df),
            'records_with_coordinates': df[['latitude', 'longitude', 'x', 'y']].notna().any(axis=1).sum(),
            'records_with_lat_lon': df[['latitude', 'longitude']].notna().all(axis=1).sum(),
            'records_with_xy': df[['x', 'y']].notna().all(axis=1).sum(),
            'unique_addresses': df['normalized_address'].nunique(),
            'coordinate_ranges': {}
        }
        
        # Coordinate ranges
        for coord_col in ['latitude', 'longitude', 'x', 'y']:
            if coord_col in df.columns and df[coord_col].notna().any():
                stats['coordinate_ranges'][coord_col] = {
                    'min': float(df[coord_col].min()),
                    'max': float(df[coord_col].max()),
                    'mean': float(df[coord_col].mean())
                }
        
        return stats
    
    def export_summary(self, df: pd.DataFrame, output_path: str) -> None:
        """Export a summary report of the processed data"""
        if df.empty:
            logger.warning("No data to export summary for")
            return
        
        stats = self.get_statistics(df)
        
        summary_lines = [
            "GIS Document Processing Summary Report",
            "=" * 50,
            f"Total Records Processed: {stats['total_records']}",
            f"Records with Coordinates: {stats['records_with_coordinates']}",
            f"Records with Lat/Lon: {stats['records_with_lat_lon']}",
            f"Records with X/Y: {stats['records_with_xy']}",
            f"Unique Addresses: {stats['unique_addresses']}",
            "",
            "Coordinate Ranges:"
        ]
        
        for coord_col, ranges in stats['coordinate_ranges'].items():
            summary_lines.append(f"  {coord_col}:")
            summary_lines.append(f"    Min: {ranges['min']:.6f}")
            summary_lines.append(f"    Max: {ranges['max']:.6f}")
            summary_lines.append(f"    Mean: {ranges['mean']:.6f}")
        
        summary_lines.extend([
            "",
            "Sample Records:",
            "-" * 30
        ])
        
        # Add sample records
        sample_size = min(5, len(df))
        sample_df = df.head(sample_size)
        
        for idx, row in sample_df.iterrows():
            summary_lines.append(f"Record {idx + 1}:")
            summary_lines.append(f"  Address: {row.get('normalized_address', 'N/A')}")
            summary_lines.append(f"  Lat: {row.get('latitude', 'N/A')}")
            summary_lines.append(f"  Lon: {row.get('longitude', 'N/A')}")
            summary_lines.append(f"  X: {row.get('x', 'N/A')}")
            summary_lines.append(f"  Y: {row.get('y', 'N/A')}")
            summary_lines.append("")
        
        # Write summary to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary_lines))
            logger.info(f"Summary report exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export summary report: {e}")
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and return issues found"""
        if df.empty:
            return {'status': 'no_data', 'issues': []}
        
        issues = []
        
        # Check for missing coordinates
        missing_coords = df[['latitude', 'longitude', 'x', 'y']].isna().all(axis=1)
        if missing_coords.any():
            issues.append(f"{missing_coords.sum()} records have no coordinates")
        
        # Check for invalid coordinate ranges
        invalid_lat = (df['latitude'] < -90) | (df['latitude'] > 90)
        invalid_lon = (df['longitude'] < -180) | (df['longitude'] > 180)
        
        if invalid_lat.any():
            issues.append(f"{invalid_lat.sum()} records have invalid latitude values")
        
        if invalid_lon.any():
            issues.append(f"{invalid_lon.sum()} records have invalid longitude values")
        
        # Check for very short addresses
        short_addresses = df['normalized_address'].str.len() < 10
        if short_addresses.any():
            issues.append(f"{short_addresses.sum()} records have very short addresses")
        
        # Determine overall quality status
        if not issues:
            quality_status = 'excellent'
        elif len(issues) <= 2:
            quality_status = 'good'
        elif len(issues) <= 4:
            quality_status = 'fair'
        else:
            quality_status = 'poor'
        
        return {
            'status': quality_status,
            'issues': issues,
            'total_issues': len(issues)
        } 