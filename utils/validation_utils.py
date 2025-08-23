"""
Validation utility functions for GIS Document Processing Application
"""

from typing import Dict, Any


class ValidationUtils:
	"""Simple validators for records and dataframes"""

	REQUIRED_FIELDS = ["original_text", "normalized_address"]

	@staticmethod
	def validate_record(record: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate a single extracted record and report issues."""
		issues = []
		for field in ValidationUtils.REQUIRED_FIELDS:
			if field not in record or not str(record.get(field, "")).strip():
				issues.append(f"Missing field: {field}")
		return {
			"valid": len(issues) == 0,
			"issues": issues,
		} 