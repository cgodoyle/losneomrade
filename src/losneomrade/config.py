"""Configuration for the losneomrade package."""

from dataclasses import dataclass, field


@dataclass
class HoydedataConfig:
    """Configuration for the Høydedata DEM fetching service.

    Attributes:
        base_url: Base URL for the Høydedata ImageServer REST API.
        layer: Default DEM layer name.
        resolution: Default DEM resolution in meters.
        nodata: Value used for nodata pixels.
        max_retries: Maximum number of retry attempts for API requests.
        retry_wait: Seconds to wait between retries.
    """

    base_url: str = "https://hoydedata.no/arcgis/rest/services"
    layer: str = "NHM_DTM_25833"
    resolution: int = 5
    nodata: int = -9999
    max_retries: int = 5
    retry_wait: int = 1


@dataclass
class MSMLConfig:
    """Configuration for the MSML (marin leire) MapServer service.

    Attributes:
        base_url: Base URL for the NVE MSML MapServer (2025 recipe).
        layer_id: Layer ID for the MSML feature layer.
        max_records: Maximum records per paginated request.
    """

    base_url: str = "https://gis4.nve.no/map/rest/services/Mapservices/MSML/MapServer"
    layer_id: int = 0
    max_records: int = 2000


@dataclass
class Settings:
    """Global settings for the losneomrade package.

    Attributes:
        hoydedata: Configuration for Høydedata API.
        msml: Configuration for MSML/mask services.
        valid_layers: List of valid Høydedata layer names.
    """

    hoydedata: HoydedataConfig = field(default_factory=HoydedataConfig)
    msml: MSMLConfig = field(default_factory=MSMLConfig)
    valid_layers: tuple[str, ...] = (
        "dtm1_32_wcs",
        "dtm1_33_wcs",
        "dtm10_32_wcs",
        "dtm10_33_wcs",
        "NHM_DTM_25833",
        "NHM_DTM_25832",
    )


# Default settings instance
settings = Settings()
