"""
Registry for chart renderers.
"""
from typing import Dict, Type
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
        renderer_instance = renderer_class()
        for chart_type in renderer_instance.supported_chart_types:
            cls._renderers[chart_type] = renderer_class

    @classmethod
    def get_renderer(cls, chart_type: ChartType) -> Type[BaseChartRenderer]:
        """Get renderer class for given chart type"""
        if chart_type not in cls._renderers:
            raise ValueError(f"No renderer registered for chart type: {chart_type}")
        return cls._renderers[chart_type]
