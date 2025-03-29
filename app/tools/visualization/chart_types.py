from enum import Enum, auto

class ChartType(Enum):
    HEATMAP = auto()
    HEXAGON_MAP = auto()
    GEOJSON_MAP = auto()
    CORRELATION_SCATTER = auto()
    YEARLY_OBSERVATIONS = auto()
    SPECIES_IMAGES = auto()
    OCCURRENCE_MAP = auto()
    FOREST_CORRELATION = auto()
    SPECIES_HCI_CORRELATION = auto()
    HUMANMOD_CORRELATION = auto()
    SCATTER_PLOT_3D = auto()
    FORCE_DIRECTED_GRAPH = auto()
    TREE_CHART = auto()
    JSON = auto()
    SPECIES_SHARED_HABITAT = auto()
    HABITAT_ANALYSIS = auto()

    @classmethod
    def from_string(cls, chart_type: str) -> 'ChartType':
        """Convert string to ChartType enum"""
        try:
            return cls[chart_type.upper()]
        except KeyError as exc:
            raise ValueError(f"Unknown chart type: {chart_type}") from exc