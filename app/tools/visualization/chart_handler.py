"""
Handler for creating visualizations.
"""
import logging
from .factory import ChartFactory

# pylint: disable=no-member
class ChartHandler:
    """
    Handler for creating visualizations.
    """
    def __init__(self):
        self.logger = logging.getLogger("BioChat." + self.__class__.__name__)

    def draw_chart(self, df, chart_type, parameters, cache_buster=None):
        """Main entry point for creating visualizations"""
        try:
            renderer = ChartFactory.create_renderer(chart_type)
            return renderer.render(df, parameters, cache_buster)
        except Exception as e:
            self.logger.error("Error creating chart: %s", str(e), exc_info=True)
            raise
