"""
Quality assessment system for address extractions and coordinate validation.
Provides confidence scoring, validation, and quality recommendations.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union
import hashlib
from datetime import datetime

# Import standardized schemas
try:
    from llm.standardized_schemas import (
        StandardizedExtraction, 
        QualityAssessment
    )
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class QualityAssessor:
    """
    Comprehensive quality assessment for address extractions and coordinates
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize quality assessor
        
        Args:
            config: Configuration dictionary for assessment parameters
        """
        self.config = config or {}
        
        # Quality thresholds
        self.excellent_threshold = self.config.get('excellent_threshold', 0.9)
        self.good_threshold = self.config.get('good_threshold', 0.7)
        self.fair_threshold = self.config.get('fair_threshold', 0.5)
        
        # Assessment weights
        self.weights = {
            'confidence': self.config.get('confidence_weight', 0.3),
            'address_specificity': self.config.get('address_weight', 0.25),
            'coordinate_precision': self.config.get('coordinate_weight', 0.25),
            'consistency': self.config.get('consistency_weight', 0.2)
        }
        
        # Pattern definitions for address quality
        self._initialize_quality_patterns()
        
        logger.info("Quality assessor initialized with comprehensive validation")
    
    def _initialize_quality_patterns(self):
        """Initialize patterns for quality assessment"""
        
        # High-quality address indicators
        self.high_quality_patterns = [
            r'\b\d+\s+[A-Za-z\s]+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|boulevard|blvd)\b',
            r'\b\d{5}(?:-\d{4})?\b',  # ZIP codes
            r'\b[A-Z]{2}\s+\d{5}\b',   # State + ZIP
            r'\b\d+\s+\w+\s+\w+\s+\w+\b'  # Multi-word addresses
        ]
        
        # Generic/low-quality indicators
        self.low_quality_patterns = [
            r'\bunknown\b',
            r'\blocation\b',
            r'\bplace\b',
            r'\barea\b',
            r'\bcoordinates?\b',
            r'^\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*$'  # Just bare coordinates
        ]
        
        # Address completeness indicators
        self.completeness_patterns = {
            'street_number': r'\b\d+\b',
            'street_name': r'\b(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|boulevard|blvd)\b',
            'city': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            'postal_code': r'\b\d{5}(?:-\d{4})?\b',
            'country': r'\b(?:USA|US|United States|Spain|France|Germany|UK|Canada)\b'
        }
    
    def assess_extraction_quality(self, extraction: Union[StandardizedExtraction, Dict[str, Any]]) -> QualityAssessment:
        """
        Assess quality of a single extraction
        
        Args:
            extraction: StandardizedExtraction object or dictionary
            
        Returns:
            QualityAssessment object with detailed scoring
        """
        if hasattr(extraction, '__dict__'):
            data = extraction.__dict__
        else:
            data = extraction
        
        # Generate unique ID for this assessment
        extraction_id = self._generate_extraction_id(data)
        
        # Individual assessment factors
        factors = {}
        factors['confidence_score'] = self._assess_confidence(data)
        factors['address_quality'] = self._assess_address_quality(data)
        factors['coordinate_validity'] = self._assess_coordinate_validity(data)
        factors['text_coherence'] = self._assess_text_coherence(data)
        
        # Calculate weighted overall score
        overall_score = sum(
            factors[factor] * self.weights.get(factor.replace('_score', '').replace('_quality', '').replace('_validity', '').replace('_coherence', 'consistency'), 0.25)
            for factor in factors
        )
        
        # Determine quality label
        quality_label = self._get_quality_label(overall_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(factors, data)
        
        if SCHEMAS_AVAILABLE:
            return QualityAssessment(
                extraction_id=extraction_id,
                quality_score=round(overall_score, 3),
                confidence_level=data.get('confidence_level', data.get('confidence', 0.0)),
                assessment_factors=factors,
                quality_label=quality_label,
                recommendations=recommendations
            )
        else:
            # Fallback dictionary format
            return {
                'extraction_id': extraction_id,
                'quality_score': round(overall_score, 3),
                'confidence_level': data.get('confidence_level', data.get('confidence', 0.0)),
                'assessment_factors': factors,
                'quality_label': quality_label,
                'recommendations': recommendations
            }
    
    def assess_batch_quality(self, extractions: List[Union[StandardizedExtraction, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Assess quality of a batch of extractions
        
        Args:
            extractions: List of extractions to assess
            
        Returns:
            Comprehensive batch quality report
        """
        if not extractions:
            return {
                'overall_quality': 'No Data',
                'total_extractions': 0,
                'quality_distribution': {},
                'recommendations': ['No extractions provided for assessment']
            }
        
        individual_assessments = []
        quality_scores = []
        
        for extraction in extractions:
            assessment = self.assess_extraction_quality(extraction)
            individual_assessments.append(assessment)
            
            if hasattr(assessment, 'quality_score'):
                quality_scores.append(assessment.quality_score)
            else:
                quality_scores.append(assessment['quality_score'])
        
        # Calculate batch statistics
        avg_quality = sum(quality_scores) / len(quality_scores)
        quality_distribution = self._calculate_quality_distribution(individual_assessments)
        
        # Generate batch recommendations
        batch_recommendations = self._generate_batch_recommendations(individual_assessments, extractions)
        
        return {
            'overall_quality': self._get_quality_label(avg_quality),
            'average_score': round(avg_quality, 3),
            'total_extractions': len(extractions),
            'quality_distribution': quality_distribution,
            'individual_assessments': individual_assessments,
            'recommendations': batch_recommendations,
            'assessment_timestamp': datetime.now().isoformat()
        }
    
    def _assess_confidence(self, data: Dict[str, Any]) -> float:
        """Assess confidence score component"""
        confidence = data.get('confidence_level', data.get('confidence', 0.0))
        return min(1.0, max(0.0, float(confidence)))
    
    def _assess_address_quality(self, data: Dict[str, Any]) -> float:
        """Assess address quality based on specificity and completeness"""
        address = data.get('inferred_address', data.get('normalized_address', ''))
        
        if not address or address.strip() == '':
            return 0.0
        
        score = 0.0
        
        # Check for high-quality patterns
        for pattern in self.high_quality_patterns:
            if re.search(pattern, address, re.IGNORECASE):
                score += 0.2
        
        # Penalize low-quality patterns
        for pattern in self.low_quality_patterns:
            if re.search(pattern, address, re.IGNORECASE):
                score -= 0.3
        
        # Assess completeness
        completeness_score = 0.0
        for component, pattern in self.completeness_patterns.items():
            if re.search(pattern, address, re.IGNORECASE):
                completeness_score += 0.1
        
        # Combine scores
        final_score = score + completeness_score
        
        # Length bonus for detailed addresses
        if len(address) > 20:
            final_score += 0.1
        
        # Word count bonus
        word_count = len(address.split())
        if word_count >= 3:
            final_score += 0.1
        
        return min(1.0, max(0.0, final_score))
    
    def _assess_coordinate_validity(self, data: Dict[str, Any]) -> float:
        """Assess coordinate validity and precision"""
        lat = data.get('latitude_dd', data.get('latitude', 0.0))
        lon = data.get('longitude_dd', data.get('longitude', 0.0))
        
        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            return 0.0
        
        score = 0.0
        
        # Check basic validity
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            score += 0.5
        else:
            return 0.0  # Invalid coordinates
        
        # Check for non-zero coordinates
        if lat != 0.0 or lon != 0.0:
            score += 0.2
        
        # Check for reasonable precision (not obviously fake)
        lat_str = str(lat)
        lon_str = str(lon)
        
        # Precision assessment
        if '.' in lat_str:
            lat_decimals = len(lat_str.split('.')[1])
            if 2 <= lat_decimals <= 6:  # Reasonable precision
                score += 0.15
        
        if '.' in lon_str:
            lon_decimals = len(lon_str.split('.')[1])
            if 2 <= lon_decimals <= 6:  # Reasonable precision
                score += 0.15
        
        return min(1.0, score)
    
    def _assess_text_coherence(self, data: Dict[str, Any]) -> float:
        """Assess coherence between original text and inferred address"""
        original = data.get('original_text', '')
        inferred = data.get('inferred_address', data.get('normalized_address', ''))
        
        if not original or not inferred:
            return 0.5  # Neutral score if missing data
        
        # Simple coherence checks
        original_words = set(original.lower().split())
        inferred_words = set(inferred.lower().split())
        
        # Calculate word overlap
        if inferred_words:
            overlap = len(original_words & inferred_words) / len(inferred_words)
            return min(1.0, overlap + 0.3)  # Boost base score
        
        return 0.5
    
    def _generate_extraction_id(self, data: Dict[str, Any]) -> str:
        """Generate unique ID for extraction"""
        content = f"{data.get('original_text', '')}{data.get('latitude_dd', 0)}{data.get('longitude_dd', 0)}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _get_quality_label(self, score: float) -> str:
        """Convert numeric score to quality label"""
        if score >= self.excellent_threshold:
            return 'Excellent'
        elif score >= self.good_threshold:
            return 'Good'
        elif score >= self.fair_threshold:
            return 'Fair'
        else:
            return 'Poor'
    
    def _generate_recommendations(self, factors: Dict[str, float], data: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on assessment factors"""
        recommendations = []
        
        # Confidence recommendations
        if factors['confidence_score'] < 0.6:
            recommendations.append("Low confidence - consider manual verification")
        
        # Address quality recommendations
        if factors['address_quality'] < 0.5:
            recommendations.append("Address lacks specificity - try to include street numbers or postal codes")
        
        # Coordinate recommendations
        if factors['coordinate_validity'] < 0.7:
            recommendations.append("Coordinates may be imprecise - verify against known locations")
        
        # Text coherence recommendations
        if factors['text_coherence'] < 0.5:
            recommendations.append("Inferred address may not match original text context")
        
        # General recommendations
        if all(score < 0.6 for score in factors.values()):
            recommendations.append("Consider re-processing with higher context or different extraction mode")
        
        return recommendations[:3]  # Limit to top 3 recommendations
    
    def _calculate_quality_distribution(self, assessments: List[Union[QualityAssessment, Dict]]) -> Dict[str, int]:
        """Calculate distribution of quality labels"""
        distribution = {'Excellent': 0, 'Good': 0, 'Fair': 0, 'Poor': 0}
        
        for assessment in assessments:
            if hasattr(assessment, 'quality_label'):
                label = assessment.quality_label
            else:
                label = assessment['quality_label']
            
            if label in distribution:
                distribution[label] += 1
        
        return distribution
    
    def _generate_batch_recommendations(self, assessments: List[Union[QualityAssessment, Dict]], 
                                      extractions: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for the entire batch"""
        recommendations = []
        
        # Calculate overall statistics
        total = len(assessments)
        if total == 0:
            return ["No extractions to assess"]
        
        poor_count = sum(1 for a in assessments 
                        if (a.quality_label if hasattr(a, 'quality_label') else a['quality_label']) == 'Poor')
        excellent_count = sum(1 for a in assessments 
                            if (a.quality_label if hasattr(a, 'quality_label') else a['quality_label']) == 'Excellent')
        
        poor_ratio = poor_count / total
        excellent_ratio = excellent_count / total
        
        # Generate batch-level recommendations
        if poor_ratio > 0.5:
            recommendations.append("Over 50% of extractions are poor quality - consider adjusting extraction parameters")
        
        if excellent_ratio < 0.1:
            recommendations.append("Very few excellent extractions - try increasing context size or using more specific prompts")
        
        # Check for coordinate issues
        zero_coords = sum(1 for ext in extractions 
                         if ext.get('latitude_dd', 0) == 0 and ext.get('longitude_dd', 0) == 0)
        if zero_coords > total * 0.3:
            recommendations.append("Many extractions have zero coordinates - check geocoding configuration")
        
        # Check confidence distribution
        low_confidence = sum(1 for ext in extractions 
                           if ext.get('confidence_level', ext.get('confidence', 0)) < 0.5)
        if low_confidence > total * 0.4:
            recommendations.append("Many extractions have low confidence - consider stricter filtering")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def validate_export_readiness(self, extractions: List[Union[StandardizedExtraction, Dict]]) -> Dict[str, Any]:
        """
        Validate if data is ready for export based on quality assessment
        
        Args:
            extractions: List of extractions to validate
            
        Returns:
            Validation report with readiness status
        """
        if not extractions:
            return {
                'ready_for_export': False,
                'reason': 'No data to export',
                'recommendations': ['Process a document first']
            }
        
        # Run batch quality assessment
        quality_report = self.assess_batch_quality(extractions)
        
        # Define readiness criteria
        min_acceptable_ratio = 0.6  # 60% should be Fair or better
        distribution = quality_report['quality_distribution']
        total = quality_report['total_extractions']
        
        acceptable_count = distribution.get('Excellent', 0) + distribution.get('Good', 0) + distribution.get('Fair', 0)
        acceptable_ratio = acceptable_count / total if total > 0 else 0
        
        ready_for_export = acceptable_ratio >= min_acceptable_ratio
        
        validation_report = {
            'ready_for_export': ready_for_export,
            'quality_summary': quality_report,
            'acceptable_ratio': round(acceptable_ratio, 3),
            'total_records': total,
            'quality_threshold_met': ready_for_export
        }
        
        if not ready_for_export:
            validation_report['reason'] = f"Only {acceptable_ratio:.1%} of extractions meet quality standards (need {min_acceptable_ratio:.1%})"
            validation_report['recommendations'] = quality_report['recommendations']
        else:
            validation_report['reason'] = "Data quality meets export standards"
            validation_report['recommendations'] = ['Data is ready for export']
        
        return validation_report
