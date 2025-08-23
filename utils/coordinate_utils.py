"""
Coordinate utility functions for GIS Document Processing Application
"""

from typing import Optional, Tuple

try:
	from pyproj import Transformer
	PYPROJ_AVAILABLE = True
except ImportError:
	PYPROJ_AVAILABLE = False


class CoordinateUtils:
	"""Basic coordinate helper utilities"""

	@staticmethod
	def is_valid_lat_lon(latitude: Optional[float], longitude: Optional[float]) -> bool:
		if latitude is None or longitude is None:
			return False
		try:
			return -90.0 <= float(latitude) <= 90.0 and -180.0 <= float(longitude) <= 180.0
		except Exception:
			return False

	@staticmethod
	def transform_wgs84_to(target_crs: str, lon: float, lat: float) -> Tuple[Optional[float], Optional[float]]:
		"""Transform WGS84 (EPSG:4326) lon/lat to target CRS. Returns (x, y) or (None, None) if unavailable."""
		if not PYPROJ_AVAILABLE:
			return None, None
		try:
			transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
			x, y = transformer.transform(lon, lat)
			return float(x), float(y)
		except Exception:
			return None, None 