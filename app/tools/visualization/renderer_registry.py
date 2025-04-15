"""
Registry for chart renderers.
"""
from typing import Dict, Type
import logging
from .base import BaseChartRenderer
from .chart_types import ChartType

# Import renderer classes at the top level
from .renderers.heatmap import HeatmapRenderer
from .renderers.hexagonmap import HexagonmapRenderer
from .renderers.geojson_map import GeoJsonMapRenderer
from .renderers.correlation_scatter import CorrelationScatterRenderer
from .renderers.yearly_observations import YearlyObservationsRenderer
from .renderers.species_images import SpeciesImagesRenderer
from .renderers.occurencemap import OccurrenceMapRenderer
from .renderers.forest_viz import ForestRenderer
from .renderers.hci_viz import HCIRenderer
from .renderers.human_modification_viz import HumanModificationVizRenderer
from .renderers.shared_habitat import SharedHabitatRenderer
from .renderers.scatter_plot_3d import ScatterPlot3DRenderer
from .renderers.force_directed_graph import ForceDirectedGraphRenderer
from .renderers.tree_viz import TreeRenderer
from .renderers.json_viz import JSONRenderer
from .renderers.habitat_viz import HabitatViz
from .renderers.topography_viz import TopographyViz
from .renderers.climate_viz import ClimateViz

# pylint: disable=no-member
class RendererRegistry:
    """
    Registry for chart renderers.
    """
    _renderers: Dict[ChartType, Type[BaseChartRenderer]] = {}

    @classmethod
    def register(cls, renderer_class: Type[BaseChartRenderer]) -> None:
        """Register a chart renderer"""
        # Create an instance to access the property
        logging.debug("Registering renderer %s", renderer_class.__name__)
        renderer_instance = renderer_class()
        for chart_type in renderer_instance.supported_chart_types:
            logging.debug("Registering %s for chart type: %s", renderer_class.__name__, chart_type)
            cls._renderers[chart_type] = renderer_class

    @classmethod
    def get_renderer(cls, chart_type: ChartType) -> Type[BaseChartRenderer]:
        """Get renderer class for given chart type"""
        logging.debug("Looking for renderer for chart type: %s", chart_type)
        logging.debug("Available renderers: %s", cls._renderers)

        # Convert string to ChartType enum if needed
        if isinstance(chart_type, str):
            try:
                chart_type = ChartType(chart_type)
            except ValueError:
                # Try to find by string value
                for enum_type in ChartType:
                    if enum_type.value == chart_type:
                        chart_type = enum_type
                        break

        if chart_type not in cls._renderers:
            logging.info("Renderer not found for chart type: %s. Attempting to re-register all renderers.", chart_type)
            # Re-register all renderers
            cls._register_all_renderers()

            # Check again after re-registration
            if chart_type not in cls._renderers:
                # Try to find by string value if it's an enum
                if hasattr(chart_type, 'value'):
                    for key, renderer in cls._renderers.items():
                        if key.value == chart_type.value:
                            return renderer

                raise ValueError(f"No renderer registered for chart type: {chart_type} even after re-registration")

        return cls._renderers[chart_type]

    @classmethod
    def _register_all_renderers(cls):
        """Register all available renderers"""
        logging.info("Re-registering all chart renderers")

        # No imports here, just register the classes
        cls.register(HeatmapRenderer)
        cls.register(HexagonmapRenderer)
        cls.register(GeoJsonMapRenderer)
        cls.register(CorrelationScatterRenderer)
        cls.register(YearlyObservationsRenderer)
        cls.register(SpeciesImagesRenderer)
        cls.register(OccurrenceMapRenderer)
        cls.register(ForestRenderer)
        cls.register(HCIRenderer)
        cls.register(HumanModificationVizRenderer)
        cls.register(SharedHabitatRenderer)
        cls.register(ScatterPlot3DRenderer)
        cls.register(ForceDirectedGraphRenderer)
        cls.register(TreeRenderer)
        cls.register(JSONRenderer)
        cls.register(HabitatViz)
        cls.register(TopographyViz)
        cls.register(ClimateViz)

        logging.info("Renderer registration complete. Available renderers: %s", cls._renderers)
