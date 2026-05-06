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
class NVEConfig:
    """Configuration for NVE/NGU WFS services.

    Attributes:
        maringrense_url: URL for the marine limit WFS service.
        results_offset: Default offset for paginated results.
    """

    maringrense_url: str = (
        "https://geo.ngu.no/mapserver/LosacGmlWMS3?"
        "service=WFS&version=2.0.0&request=GetFeature"
    )
    results_offset: int = 100


@dataclass
class Settings:
    """Global settings for the losneomrade package.

    Attributes:
        hoydedata: Configuration for Høydedata API.
        nve: Configuration for NVE services.
        valid_layers: List of valid Høydedata layer names.
    """

    hoydedata: HoydedataConfig = field(default_factory=HoydedataConfig)
    nve: NVEConfig = field(default_factory=NVEConfig)
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
