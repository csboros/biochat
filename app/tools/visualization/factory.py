"""
This module provides a factory class for creating chart renderers based on chart type.

Classes:
    ChartFactory: Factory class for creating chart renderers.
"""
from .base import BaseChartRenderer
from .chart_types import ChartType
from .renderer_registry import RendererRegistry


class ChartFactory:
    """
    Factory class for creating chart renderers based on chart type.
    """
    @staticmethod
    def create_renderer(chart_type: str | ChartType) -> BaseChartRenderer:
        """Create appropriate renderer for given chart type"""
        if isinstance(chart_type, str):
            chart_type = ChartType.from_string(chart_type)

        renderer_class = RendererRegistry.get_renderer(chart_type)
        return renderer_class()
