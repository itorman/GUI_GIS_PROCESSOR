"""
Structured data schemas for address and geographic information extraction.
Uses Pydantic models for validation and langextract for efficient LLM processing.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union
import re


class GeographicCoordinate(BaseModel):
    """Represents a geographic coordinate with validation"""
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Longitude in decimal degrees")
    x: Optional[float] = Field(None, description="Projected X coordinate")
    y: Optional[float] = Field(None, description="Projected Y coordinate")
    crs: Optional[str] = Field("EPSG:4326", description="Coordinate reference system")
    
    @validator('latitude', 'longitude')
    def validate_coordinate_precision(cls, v):
        """Validate coordinate precision to reasonable limits"""
        if v is not None:
            # Limit to 6 decimal places (approximately 1 meter precision)
            return round(v, 6)
        return v


class Address(BaseModel):
    """Represents a structured address with validation"""
    original_text: str = Field(..., description="Original text where address was found")
    normalized_address: str = Field(..., description="Normalized address string")
    
    # Address components
    street: Optional[str] = Field(None, description="Street name")
    number: Optional[str] = Field(None, description="Street number or building identifier")
    postal_code: Optional[str] = Field(None, description="Postal/ZIP code")
    city: Optional[str] = Field(None, description="City or municipality name")
    country: Optional[str] = Field(None, description="Country name")
    
    # Geographic coordinates
    coordinates: Optional[GeographicCoordinate] = Field(None, description="Geographic coordinates")
    
    # Confidence and metadata
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Extraction confidence score")
    language: Optional[str] = Field(None, description="Detected language of the address")
    
    @validator('normalized_address')
    def validate_normalized_address(cls, v):
        """Ensure normalized address is not empty"""
        if not v.strip():
            raise ValueError("Normalized address cannot be empty")
        return v.strip()
    
    @validator('language')
    def validate_language_code(cls, v):
        """Validate language code format"""
        if v and not re.match(r'^[a-z]{2}(-[A-Z]{2})?$', v):
            raise ValueError("Language must be in ISO 639-1 format (e.g., 'en', 'es-ES')")
        return v


class AddressExtractionResult(BaseModel):
    """Result of address extraction from a text chunk"""
    addresses: List[Address] = Field(default_factory=list, description="List of extracted addresses")
    text_chunk: str = Field(..., description="Original text chunk processed")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    extraction_method: str = Field("langextract", description="Method used for extraction")
    
    @validator('addresses')
    def validate_addresses_not_empty(cls, v):
        """Ensure at least one address is found or result is marked as empty"""
        return v


class GeographicFeature(BaseModel):
    """Represents any geographic feature found in text"""
    feature_type: str = Field(..., description="Type of geographic feature (address, coordinate, landmark, etc.)")
    original_text: str = Field(..., description="Original text where feature was found")
    normalized_text: str = Field(..., description="Normalized representation of the feature")
    coordinates: Optional[GeographicCoordinate] = Field(None, description="Geographic coordinates if available")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Extraction confidence score")
    
    @validator('feature_type')
    def validate_feature_type(cls, v):
        """Validate feature type is one of the allowed values"""
        allowed_types = ['address', 'coordinate', 'landmark', 'boundary', 'waterbody', 'other']
        if v not in allowed_types:
            raise ValueError(f"Feature type must be one of: {allowed_types}")
        return v


class GeographicExtractionResult(BaseModel):
    """Result of geographic information extraction from a text chunk"""
    features: List[GeographicFeature] = Field(default_factory=list, description="List of extracted geographic features")
    text_chunk: str = Field(..., description="Original text chunk processed")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    extraction_method: str = Field("langextract", description="Method used for extraction")
    
    def get_addresses(self) -> List[Address]:
        """Extract addresses from geographic features"""
        addresses = []
        for feature in self.features:
            if feature.feature_type == 'address':
                # Convert geographic feature to address format
                address = Address(
                    original_text=feature.original_text,
                    normalized_address=feature.normalized_text,
                    coordinates=feature.coordinates,
                    confidence=feature.confidence
                )
                addresses.append(address)
        return addresses


# Schema definitions for langextract
def get_address_extraction_schema() -> dict:
    """Get the schema for address extraction using langextract"""
    return AddressExtractionResult.model_json_schema()


def get_geographic_extraction_schema() -> dict:
    """Get the schema for general geographic information extraction using langextract"""
    return GeographicExtractionResult.model_json_schema()


def get_simple_address_schema() -> dict:
    """Get a simplified schema for basic address extraction"""
    return Address.model_json_schema()
