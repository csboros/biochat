"""
Registry for chart renderers.
"""
from typing import Dict, Type
import logging
from .base import BaseChartRenderer
from .chart_types import ChartType

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
        if chart_type not in cls._renderers:
            raise ValueError(f"No renderer registered for chart type: {chart_type}")
        return cls._renderers[chart_type]
