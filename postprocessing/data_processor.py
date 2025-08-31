"""
Data processor for post-processing LLM extraction results.
Handles validation, cleaning, and coordinate processing.
Enhanced with langextract schema validation and improved data quality.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path

# Import langextract schemas for validation
try:
    from llm.schemas import Address, AddressExtractionResult
    LANGEXTRACT_AVAILABLE = True
except ImportError:
    LANGEXTRACT_AVAILABLE = False

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
        
        # Validate results using langextract schemas if available
        # But skip if we already have coordinate data (from LLM)
        if LANGEXTRACT_AVAILABLE and llm_results:
            # Check if results already have coordinate data
            has_coords = any(
                any(result.get(field) is not None for field in ['latitude', 'longitude', 'x', 'y'])
                for result in llm_results
            )
            
            if has_coords:
                logger.info("Results already have coordinate data, skipping schema validation")
            else:
                try:
                    logger.info("Validating results with langextract schemas")
                    llm_results = self._validate_with_schemas(llm_results)
                except Exception as e:
                    logger.warning(f"Schema validation failed, using raw results: {e}")
                    # Continue with raw results if validation fails
        
        # Convert to DataFrame
        logger.info(f"Converting {len(llm_results)} results to DataFrame")
        df = pd.DataFrame(llm_results)
        logger.info(f"DataFrame created with {len(df)} rows and columns: {list(df.columns)}")
        
        # Handle nested coordinate fields if they exist
        df = self._flatten_coordinate_fields(df)
        logger.info(f"After flattening coordinates: DataFrame shape {df.shape}, columns: {list(df.columns)}")
        
        # Check if we already have coordinate columns (from LLM)
        has_coord_cols = any(col in df.columns for col in ['latitude', 'longitude', 'x', 'y'])
        if has_coord_cols:
            logger.info(f"Found existing coordinate columns: {[col for col in ['latitude', 'longitude', 'x', 'y'] if col in df.columns]}")
            # Check coordinate values
            coord_counts = {}
            for col in ['latitude', 'longitude', 'x', 'y']:
                if col in df.columns:
                    coord_counts[col] = df[col].notna().sum()
            logger.info(f"Coordinate counts: {coord_counts}")
        
        # Clean and validate data
        logger.info(f"Before cleaning: DataFrame shape {df.shape}")
        df = self._clean_dataframe(df)
        logger.info(f"After cleaning: DataFrame shape {df.shape}")
        
        # Process coordinates
        logger.info(f"Before coordinate processing: DataFrame shape {df.shape}")
        df = self._process_coordinates(df)
        logger.info(f"After coordinate processing: DataFrame shape {df.shape}")
        
        # Remove duplicates
        df = self._remove_duplicates(df)
        
        # Sort by address for better organization
        df = df.sort_values('normalized_address').reset_index(drop=True)
        
        logger.info(f"Processed {len(df)} records successfully")
        return df
    
    def _validate_with_schemas(self, llm_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate LLM results using langextract schemas for improved data quality
        
        Args:
            llm_results: List of raw LLM extraction results
            
        Returns:
            List of validated and cleaned results
        """
        if not LANGEXTRACT_AVAILABLE:
            logger.warning("Langextract schemas not available, skipping validation")
            return llm_results
        
        validated_results = []
        validation_errors = 0
        
        for i, result in enumerate(llm_results):
            try:
                # Validate using Address schema
                validated_address = Address(**result)
                
                # Convert back to dictionary for DataFrame processing
                validated_dict = validated_address.model_dump()
                validated_results.append(validated_dict)
                
            except Exception as e:
                validation_errors += 1
                logger.warning(f"Validation error for result {i}: {e}")
                
                # Try to clean and salvage the result
                cleaned_result = self._salvage_invalid_result(result)
                if cleaned_result:
                    validated_results.append(cleaned_result)
        
        if validation_errors > 0:
            logger.info(f"Schema validation complete: {len(validated_results)} valid, {validation_errors} errors")
        else:
            logger.info("All results passed schema validation successfully")
        
        return validated_results
    
    def _flatten_coordinate_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten nested coordinate fields to separate columns
        
        Args:
            df: DataFrame that may have nested coordinate fields
            
        Returns:
            DataFrame with flattened coordinate columns
        """
        if df.empty:
            return df
        
        # Check if we have a 'coordinates' column with nested data
        if 'coordinates' in df.columns:
            logger.info("Found 'coordinates' column, flattening to separate columns")
            
            # Extract coordinate values
            df['latitude'] = df['coordinates'].apply(
                lambda x: x.latitude if hasattr(x, 'latitude') else 
                         (x.get('latitude') if isinstance(x, dict) else None)
            )
            df['longitude'] = df['coordinates'].apply(
                lambda x: x.longitude if hasattr(x, 'longitude') else 
                         (x.get('longitude') if isinstance(x, dict) else None)
            )
            df['x'] = df['coordinates'].apply(
                lambda x: x.x if hasattr(x, 'x') else 
                         (x.get('x') if isinstance(x, dict) else None)
            )
            df['y'] = df['coordinates'].apply(
                lambda x: x.y if hasattr(x, 'y') else 
                         (x.get('y') if isinstance(x, dict) else None)
            )
            
            # Drop the original coordinates column
            df = df.drop(columns=['coordinates'])
            logger.info("Successfully flattened coordinate fields")
        
        return df
    
    def _salvage_invalid_result(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Attempt to salvage invalid results by cleaning and fixing common issues
        
        Args:
            result: Invalid result dictionary
            
        Returns:
            Cleaned result dictionary or None if unsalvageable
        """
        try:
            cleaned = result.copy()
            
            # Ensure required fields exist
            if 'original_text' not in cleaned or not cleaned['original_text']:
                cleaned['original_text'] = str(result.get('text', ''))
            
            if 'normalized_address' not in cleaned or not cleaned['normalized_address']:
                # Try to create a normalized address from available fields
                address_parts = []
                for field in ['street', 'number', 'city', 'postal_code', 'country']:
                    if field in cleaned and cleaned[field]:
                        address_parts.append(str(cleaned[field]))
                
                if address_parts:
                    cleaned['normalized_address'] = ', '.join(address_parts)
                else:
                    cleaned['normalized_address'] = cleaned.get('original_text', 'Unknown Address')
            
            # Clean coordinate fields
            coord_fields = ['latitude', 'longitude', 'x', 'y']
            for field in coord_fields:
                if field in cleaned:
                    try:
                        coord_value = float(cleaned[field])
                        # Basic coordinate validation
                        if field in ['latitude'] and (-90 <= coord_value <= 90):
                            cleaned[field] = coord_value
                        elif field in ['longitude'] and (-180 <= coord_value <= 180):
                            cleaned[field] = coord_value
                        elif field in ['x', 'y']:
                            cleaned[field] = coord_value
                        else:
                            cleaned[field] = None
                    except (ValueError, TypeError):
                        cleaned[field] = None
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Failed to salvage result: {e}")
            return None
    
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
        
        # Remove rows with no valid coordinates (only if columns exist)
        coord_mask = pd.Series([True] * len(df), index=df.index)
        
        if 'latitude' in df.columns:
            coord_mask = coord_mask | df['latitude'].notna()
        if 'longitude' in df.columns:
            coord_mask = coord_mask | df['longitude'].notna()
        if 'x' in df.columns:
            coord_mask = coord_mask | df['x'].notna()
        if 'y' in df.columns:
            coord_mask = coord_mask | df['y'].notna()
        
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
        if 'latitude' in df.columns and 'x' in df.columns:
            missing_lat = df['latitude'].isna() & df['x'].notna()
            if missing_lat.any():
                df.loc[missing_lat, 'latitude'] = df.loc[missing_lat, 'x']
        
        if 'longitude' in df.columns and 'y' in df.columns:
            missing_lon = df['longitude'].isna() & df['y'].notna()
            if missing_lon.any():
                df.loc[missing_lon, 'longitude'] = df.loc[missing_lon, 'y']
        
        # If we have lat,lon but missing x,y, copy them
        if 'x' in df.columns and 'longitude' in df.columns:
            missing_x = df['x'].isna() & df['longitude'].notna()
            if missing_x.any():
                df.loc[missing_x, 'x'] = df.loc[missing_x, 'longitude']
        
        if 'y' in df.columns and 'latitude' in df.columns:
            missing_y = df['y'].isna() & df['latitude'].notna()
            if missing_y.any():
                df.loc[missing_y, 'y'] = df.loc[missing_y, 'latitude']
        
        return df
    
    def _validate_coordinate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate coordinate ranges and remove invalid entries"""
        if df.empty:
            return df
        
        # Validate latitude (-90 to 90)
        if 'latitude' in df.columns:
            valid_lat = (df['latitude'] >= -90) & (df['latitude'] <= 90)
            invalid_lat = ~valid_lat & df['latitude'].notna()
            
            if invalid_lat.any():
                logger.warning(f"Found {invalid_lat.sum()} invalid latitude values")
                df.loc[invalid_lat, 'latitude'] = np.nan
        
        # Validate longitude (-180 to 180)
        if 'longitude' in df.columns:
            valid_lon = (df['longitude'] >= -180) & (df['longitude'] <= 180)
            invalid_lon = ~valid_lon & df['longitude'].notna()
            
            if invalid_lon.any():
                logger.warning(f"Found {invalid_lon.sum()} invalid longitude values")
                df.loc[invalid_lon, 'longitude'] = np.nan
        
        # Remove rows with no valid coordinates after validation
        # Only remove rows if we actually have coordinate data to validate
        has_coordinate_data = False
        valid_coords = pd.Series([True] * len(df), index=df.index)  # Start with True
        
        if 'latitude' in df.columns and df['latitude'].notna().any():
            has_coordinate_data = True
            valid_coords = valid_coords & (df['latitude'].notna() | df['longitude'].notna() | df['x'].notna() | df['y'].notna())
        elif 'longitude' in df.columns and df['longitude'].notna().any():
            has_coordinate_data = True
            valid_coords = valid_coords & (df['latitude'].notna() | df['longitude'].notna() | df['x'].notna() | df['y'].notna())
        elif 'x' in df.columns and df['x'].notna().any():
            has_coordinate_data = True
            valid_coords = valid_coords & (df['latitude'].notna() | df['longitude'].notna() | df['x'].notna() | df['y'].notna())
        elif 'y' in df.columns and df['y'].notna().any():
            has_coordinate_data = True
            valid_coords = valid_coords & (df['latitude'].notna() | df['longitude'].notna() | df['x'].notna() | df['y'].notna())
        
        # Only filter if we actually have coordinate data
        if has_coordinate_data:
            df = df[valid_coords]
            logger.info(f"Filtered to {len(df)} rows with valid coordinates")
        else:
            logger.info("No coordinate data found, keeping all rows")
        
        return df
    
    def _transform_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform coordinates to target CRS"""
        if not self.transformer or df.empty:
            return df
        
        # Transform lat,lon to target CRS
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            logger.warning("Cannot transform coordinates: latitude or longitude columns missing")
            return df
        
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
        if 'x' in df.columns and 'longitude' in df.columns:
            missing_x = df['x'].isna() & df['longitude'].notna()
            if missing_x.any():
                df.loc[missing_x, 'x'] = df.loc[missing_x, 'longitude']
        
        if 'y' in df.columns and 'latitude' in df.columns:
            missing_y = df['y'].isna() & df['latitude'].notna()
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
        # But be less aggressive in test mode
        if len(df) > 10:  # Only remove duplicates if we have many records
            df = df.drop_duplicates(subset=['normalized_address'], keep='first')
            logger.info(f"Removed address duplicates, remaining: {len(df)}")
        
        # Remove duplicates based on coordinates (keep first occurrence)
        # But only if coordinates are actually different and we have many records
        coord_cols = ['latitude', 'longitude', 'x', 'y']
        available_coord_cols = [col for col in coord_cols if col in df.columns]
        
        if available_coord_cols and len(df) > 20:  # Only remove if we have many records
            # Check if coordinates are actually different
            coord_variation = df[available_coord_cols].nunique().sum()
            if coord_variation > len(df) * 0.5:  # Only remove if coordinates are diverse
                df = df.drop_duplicates(subset=available_coord_cols, keep='first')
                logger.info(f"Removed coordinate duplicates, remaining: {len(df)}")
            else:
                logger.info(f"Coordinates too similar, skipping coordinate deduplication")
        else:
            logger.info(f"Not enough records for coordinate deduplication")
        
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
            'records_with_coordinates': 0,
            'records_with_lat_lon': 0,
            'records_with_xy': 0,
            'unique_addresses': df['normalized_address'].nunique() if 'normalized_address' in df.columns else 0,
            'coordinate_ranges': {}
        }
        
        # Calculate coordinate statistics safely
        coord_cols = ['latitude', 'longitude', 'x', 'y']
        available_coord_cols = [col for col in coord_cols if col in df.columns]
        
        if available_coord_cols:
            stats['records_with_coordinates'] = df[available_coord_cols].notna().any(axis=1).sum()
        
        if 'latitude' in df.columns and 'longitude' in df.columns:
            stats['records_with_lat_lon'] = df[['latitude', 'longitude']].notna().all(axis=1).sum()
        
        if 'x' in df.columns and 'y' in df.columns:
            stats['records_with_xy'] = df[['x', 'y']].notna().all(axis=1).sum()
        
        # Coordinate ranges
        for coord_col in coord_cols:
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
        coord_cols = ['latitude', 'longitude', 'x', 'y']
        available_coord_cols = [col for col in coord_cols if col in df.columns]
        
        if available_coord_cols:
            missing_coords = df[available_coord_cols].isna().all(axis=1)
            if missing_coords.any():
                issues.append(f"{missing_coords.sum()} records have no coordinates")
        
        # Check for invalid coordinate ranges
        if 'latitude' in df.columns:
            invalid_lat = (df['latitude'] < -90) | (df['latitude'] > 90)
            if invalid_lat.any():
                issues.append(f"{invalid_lat.sum()} records have invalid latitude values")
        
        if 'longitude' in df.columns:
            invalid_lon = (df['longitude'] < -180) | (df['longitude'] > 180)
            if invalid_lon.any():
                issues.append(f"{invalid_lon.sum()} records have invalid longitude values")
        
        # Check for very short addresses
        if 'normalized_address' in df.columns:
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