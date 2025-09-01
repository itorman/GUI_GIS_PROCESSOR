"""
Standardized schemas for data export according to project requirements.
Defines the exact structure needed for CSV, Excel, and Shapefile exports.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import re


class StandardizedExtraction(BaseModel):
    """
    Standardized structure for final data export.
    Matches the exact requirements: original_text, inferred_address, confidence_level, latitude_dd, longitude_dd
    """
    original_text: str = Field(
        ..., 
        description="Texto original del cual se obtienen las coordenadas o direcciones"
    )
    inferred_address: str = Field(
        ..., 
        description="Dirección inferida por el LLM basada en el contexto completo del documento"
    )
    confidence_level: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Nivel de seguridad en la dirección inferida (0.0-1.0)"
    )
    latitude_dd: float = Field(
        ..., 
        ge=-90.0, 
        le=90.0, 
        description="Latitud en grados decimales WGS84"
    )
    longitude_dd: float = Field(
        ..., 
        ge=-180.0, 
        le=180.0, 
        description="Longitud en grados decimales WGS84"
    )
    
    @validator('original_text')
    def validate_original_text(cls, v):
        """Ensure original text is not empty"""
        if not v.strip():
            raise ValueError("Original text cannot be empty")
        return v.strip()
    
    @validator('inferred_address')
    def validate_inferred_address(cls, v):
        """Ensure inferred address is meaningful"""
        if not v.strip():
            raise ValueError("Inferred address cannot be empty")
        return v.strip()
    
    @validator('latitude_dd', 'longitude_dd')
    def validate_coordinate_precision(cls, v):
        """Limit coordinate precision to 6 decimal places (~1 meter accuracy)"""
        return round(v, 6)


class ContextualExtractionRequest(BaseModel):
    """
    Request schema for contextual extraction that uses document context
    """
    chunk_text: str = Field(..., description="Current text chunk to analyze")
    document_context: str = Field(..., description="Overall document context")
    previous_extractions: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Previous extractions for consistency"
    )
    extraction_mode: str = Field(
        default="standard",
        description="Extraction mode: standard, high_context, conservative, aggressive"
    )
    min_confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for extractions"
    )


class QualityAssessment(BaseModel):
    """
    Quality assessment for extracted addresses and coordinates
    """
    extraction_id: str = Field(..., description="Unique identifier for the extraction")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="Original confidence level")
    assessment_factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Individual assessment factors"
    )
    quality_label: str = Field(
        ..., 
        description="Human-readable quality label: Excellent, Good, Fair, Poor"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improving quality"
    )
    
    @validator('quality_label')
    def validate_quality_label(cls, v):
        """Ensure quality label is valid"""
        valid_labels = ['Excellent', 'Good', 'Fair', 'Poor']
        if v not in valid_labels:
            raise ValueError(f"Quality label must be one of: {valid_labels}")
        return v


class ExportConfiguration(BaseModel):
    """
    Configuration for data export operations
    """
    export_format: str = Field(..., description="Export format: csv, excel, shapefile, arcgis")
    include_quality_assessment: bool = Field(
        default=True,
        description="Include quality assessment in export"
    )
    coordinate_precision: int = Field(
        default=6,
        ge=1,
        le=10,
        description="Number of decimal places for coordinates"
    )
    min_confidence_filter: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Filter out extractions below this confidence level"
    )
    include_metadata: bool = Field(
        default=False,
        description="Include processing metadata in export"
    )
    
    @validator('export_format')
    def validate_export_format(cls, v):
        """Ensure export format is supported"""
        valid_formats = ['csv', 'excel', 'shapefile', 'arcgis']
        if v.lower() not in valid_formats:
            raise ValueError(f"Export format must be one of: {valid_formats}")
        return v.lower()


def get_standardized_extraction_schema() -> dict:
    """Get the schema for standardized extraction export"""
    return StandardizedExtraction.schema()


def get_contextual_request_schema() -> dict:
    """Get the schema for contextual extraction requests"""
    return ContextualExtractionRequest.schema()


def get_quality_assessment_schema() -> dict:
    """Get the schema for quality assessment"""
    return QualityAssessment.schema()


# Export column mapping for backward compatibility
LEGACY_TO_STANDARDIZED_MAPPING = {
    'original_text': 'original_text',
    'normalized_address': 'inferred_address',
    'confidence': 'confidence_level',
    'latitude': 'latitude_dd',
    'longitude': 'longitude_dd'
}

STANDARDIZED_COLUMN_ORDER = [
    'original_text',
    'inferred_address', 
    'confidence_level',
    'latitude_dd',
    'longitude_dd'
]

COLUMN_DESCRIPTIONS = {
    'original_text': 'Texto original del documento donde se encontró la referencia geográfica',
    'inferred_address': 'Dirección completa inferida por IA usando contexto del documento',
    'confidence_level': 'Nivel de confianza de la inferencia (0.0 = baja, 1.0 = alta)',
    'latitude_dd': 'Latitud en grados decimales, sistema WGS84',
    'longitude_dd': 'Longitud en grados decimales, sistema WGS84'
}
