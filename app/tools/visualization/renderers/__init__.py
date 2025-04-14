"""
This module initializes and registers all chart renderers.
"""
from .heatmap import HeatmapRenderer
from .hexagonmap import HexagonmapRenderer
from .geojson_map import GeoJsonMapRenderer
from .correlation_scatter import CorrelationScatterRenderer
from .yearly_observations import YearlyObservationsRenderer
from .species_images import SpeciesImagesRenderer
from .occurencemap import OccurrenceMapRenderer
from .forest_viz import ForestRenderer
from .hci_viz import HCIRenderer
from .human_modification_viz import HumanModificationVizRenderer
from .shared_habitat import SharedHabitatRenderer
from .scatter_plot_3d import ScatterPlot3DRenderer
from ..renderer_registry import RendererRegistry
from .force_directed_graph import ForceDirectedGraphRenderer
from .tree_viz import TreeRenderer
from .json_viz import JSONRenderer
from .habitat_viz import HabitatViz
from .topography_viz import TopographyViz
from .climate_viz import ClimateViz


# Register all renderers
RendererRegistry.register(HeatmapRenderer)
RendererRegistry.register(HexagonmapRenderer)
RendererRegistry.register(GeoJsonMapRenderer)
RendererRegistry.register(CorrelationScatterRenderer)
RendererRegistry.register(YearlyObservationsRenderer)
RendererRegistry.register(SpeciesImagesRenderer)
RendererRegistry.register(OccurrenceMapRenderer)
RendererRegistry.register(ForestRenderer)
RendererRegistry.register(HCIRenderer)
RendererRegistry.register(HumanModificationVizRenderer)
RendererRegistry.register(SharedHabitatRenderer)
RendererRegistry.register(ScatterPlot3DRenderer)
RendererRegistry.register(ForceDirectedGraphRenderer)
RendererRegistry.register(TreeRenderer)
RendererRegistry.register(JSONRenderer)
RendererRegistry.register(HabitatViz)
RendererRegistry.register(TopographyViz)
RendererRegistry.register(ClimateViz)

import logging
import os
logging.info(f"[PID: {os.getpid()}] Renderer registration in renderers/__init__.py completed.")
