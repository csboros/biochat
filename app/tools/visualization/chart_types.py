"""Chart types supported by the visualization system."""

from enum import Enum, auto

class ChartType(Enum):
    """Enumeration of supported chart types."""
    HEATMAP = "heatmap"
    HEXAGON_MAP = "hexagon_map"
    GEOJSON_MAP = "geojson_map"
    CORRELATION_SCATTER = "correlation_scatter"
    YEARLY_OBSERVATIONS = "yearly_observations"
    SPECIES_IMAGES = "species_images"
    OCCURRENCE_MAP = "occurrence_map"
    FOREST_CORRELATION = "forest_correlation"
    SPECIES_HCI_CORRELATION = "species_hci_correlation"
    HUMANMOD_CORRELATION = "humanmod_correlation"
    SCATTER_PLOT_3D = "scatter_plot_3d"
    FORCE_DIRECTED_GRAPH = "force_directed_graph"
    TREE_CHART = "tree_chart"
    JSON = "json"
    SPECIES_SHARED_HABITAT = "species_shared_habitat"
    HABITAT_ANALYSIS = "habitat_analysis"
    TOPOGRAPHY_ANALYSIS = "topography_analysis"
    CLIMATE_ANALYSIS = "climate_analysis"
    @classmethod
    def from_string(cls, chart_type: str) -> 'ChartType':
        """Convert string to ChartType enum"""
        try:
            return cls[chart_type.upper()]
        except KeyError as exc:
            raise ValueError(f"Unknown chart type: {chart_type}") from exc
