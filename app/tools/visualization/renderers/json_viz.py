"""Module for rendering JSON data visualizations."""

import time
import json
import pandas as pd
import streamlit as st
from ..base import BaseChartRenderer
from ..chart_types import ChartType

class JSONRenderer(BaseChartRenderer):
    """Renderer for JSON data visualizations."""

    @property
    def supported_chart_types(self) -> list[ChartType]:
        """Return list of chart types this renderer supports."""
        return [ChartType.JSON]

    # pylint: disable=no-member
    def render(self, data: dict, parameters: dict = None, _cache_buster: str = None) -> None:
        """Render a JSON data visualization.

        Args:
            data (dict): JSON data to visualize
            parameters (dict, optional): Visualization parameters
            _cache_buster (str, optional): Cache buster string
        """
        try:
            message_index = _cache_buster if _cache_buster is not None else int(time.time())
            df = pd.DataFrame(data)
            st.dataframe(df, key=f"json_data_{message_index}")
        except (json.JSONDecodeError, pd.errors.EmptyDataError) as e:
            self.logger.error("Error parsing JSON data: %s", str(e))
            raise
        except Exception as e:
            self.logger.error("Error creating table: %s", str(e))
            raise