"""Network tests for the masks module.

These tests hit the NVE MSML MapServer and are marked slow.
Run with: pytest tests/test_masks.py -m slow
"""

import geopandas as gpd
import pytest

from losneomrade import masks


@pytest.mark.slow
class TestMSMLMask:
    """Test MSML mask fetching from the new NVE MapServer."""

    def test_get_msml_romerike(self):
        """Romerike: large area known to have MSML coverage."""
        bounds = (267867, 6645040, 311812, 6677133)
        result = masks.get_msml_mask(bounds)

        assert isinstance(result, gpd.GeoDataFrame)
        assert not result.empty
        area = result.area.sum()
        # Romerike should have substantial MSML area (>500 km²)
        assert area > 500_000_000, f"Expected >500 km², got {area / 1e6:.1f} km²"

    def test_get_msml_empty_area(self):
        """Ocean area with no MSML polygons returns empty GeoDataFrame."""
        # Middle of the North Sea
        bounds = (100000, 6400000, 100500, 6400500)
        result = masks.get_msml_mask(bounds)

        assert isinstance(result, gpd.GeoDataFrame)
        assert result.empty

    def test_check_msml_service(self):
        """Service health check returns True when service is up."""
        assert masks.check_msml_service() is True
