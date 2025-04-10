"""Chart types supported by the visualization system."""

from enum import Enum

class ChartType(Enum):
    """Enumeration of supported chart types."""
    HEATMAP = "HEATMAP"
    HEAT_MAP = "HEAT_MAP"
    HEXAGON_MAP = "HEXAGON_MAP"
    HEXAGONMAP = "HEXAGONMAP"
    GEOJSON_MAP = "GEOJSON_MAP"
    GEOJSONMAP = "GEOJSONMAP"
    CORRELATION_SCATTER = "CORRELATION_SCATTER"
    CORRELATION_SCATTER_PLOT = "CORRELATION_SCATTER_PLOT"
    YEARLY_OBSERVATIONS = "YEARLY_OBSERVATIONS"
    SPECIES_IMAGES = "SPECIES_IMAGES"
    OCCURRENCE_MAP = "OCCURRENCE_MAP"
    FOREST_CORRELATION = "FOREST_CORRELATION"
    SPECIES_HCI_CORRELATION = "SPECIES_HCI_CORRELATION"
    HUMANMOD_CORRELATION = "HUMANMOD_CORRELATION"
    SCATTER_PLOT_3D = "SCATTER_PLOT_3D"
    FORCE_DIRECTED_GRAPH = "FORCE_DIRECTED_GRAPH"
    TREE_CHART = "TREE_CHART"
    JSON = "JSON"
    SPECIES_SHARED_HABITAT = "SPECIES_SHARED_HABITAT"
    HABITAT_ANALYSIS = "HABITAT_ANALYSIS"
    TOPOGRAPHY_ANALYSIS = "TOPOGRAPHY_ANALYSIS"
    CLIMATE_ANALYSIS = "CLIMATE_ANALYSIS"
    @classmethod
    def from_string(cls, chart_type: str) -> 'ChartType':
        """Convert string to ChartType enum"""
        try:
            return cls[chart_type.upper()]
        except KeyError as exc:
            raise ValueError(f"Unknown chart type: {chart_type}") from exc
