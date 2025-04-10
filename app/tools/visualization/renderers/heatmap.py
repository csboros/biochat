"""
This module provides a renderer for heatmap visualizations.

Classes:
    HeatmapRenderer: Renderer for heatmap visualizations.
"""
import time
from typing import Any, Dict, Optional
import pandas as pd
import pydeck as pdk
import streamlit as st
from ..base import BaseChartRenderer
from ..chart_types import ChartType

class HeatmapRenderer(BaseChartRenderer):
    """
    Renderer for heatmap visualizations.
    """
    @property
    def supported_chart_types(self) -> list[ChartType]:
        return [ChartType.HEATMAP, ChartType.HEAT_MAP]

    def render(self, data: Any, parameters: Optional[Dict] = None,
               cache_buster: Optional[str] = None) -> Any:
        """Creates a heatmap visualization using PyDeck's HeatmapLayer."""
        try:
            message_index = cache_buster if cache_buster is not None else int(time.time())
            df = pd.DataFrame(data["occurrences"])
            bounds = self.get_bounds_from_data(df)
            view_state = self.default_view_state
            if bounds:
                view_state = pdk.ViewState(
                    latitude=sum(coord[0] for coord in bounds) / 2,
                    longitude=sum(coord[1] for coord in bounds) / 2,
                    zoom=3,
                    pitch=30,
                )
            # pylint: disable=no-member
            col1, col2 = st.columns([3, 1])  # 3:1 ratio for map:legend
            with col1:
                st.pydeck_chart(
                    pdk.Deck(
                        initial_view_state=view_state,
                        layers=[
                            pdk.Layer(
                                "HeatmapLayer",
                                data=df,
                                get_position=["decimallongitude", "decimallatitude"],
                                pickable=False,
                                opacity=0.7,
                                get_weight=1,
                            ),
                            pdk.Layer(
                                "HexagonLayer",
                                data=df,
                                get_position=["decimallongitude", "decimallatitude"],
                                radius=10000,
                                elevation_scale=1,
                                pickable=True,
                                opacity=0,
                                auto_highlight=False,
                                highlight_color=[0, 0, 0, 0],
                                extruded=True,
                                get_fill_color=[0, 0, 0, 0],
                            ),
                        ],
                        tooltip={
                            "html": (
                                f"<b>{parameters.get('species_name', 'Unknown')}</b><br/>"
                                "Observations: {elevationValue}<br/>"
                                "Latitude: {decimallatitude}<br/>"
                                "Longitude: {decimallongitude}"
                            ),
                            "style": {"backgroundColor": "steelblue", "color": "white"},
                        },
                    ),
                    height=700,
                    key=f"heatmap_{message_index}"
                )
            # Add legend in the second column
            with col2:
                st.markdown("### Heatmap")
                st.markdown(f"**Species**: {parameters.get('species_name', 'Unknown')}")

                # Add explanation of visualization
                st.markdown(
                    """
                    This heatmap shows species distribution:

                    - Areas with more observations appear hotter (red)
                    - Areas with fewer observations appear cooler (yellow)
                    - Intensity indicates density of observations

                    ### Color Scale
                """
                )
                # Create a gradient legend for density
                st.markdown(
                    """
                    <div style="margin-top: 10px; display: flex; align-items: stretch;">
                        <div style="
                            height: 200px;
                            width: 40px;
                            background: linear-gradient(
                                to bottom,
                                rgb(139, 0, 0),    /* Dark red (very high density) */
                                rgb(255, 0, 0),     /* Red (high density) */
                                rgb(255, 128, 0),   /* Orange (medium density) */
                                rgb(255, 255, 0)    /* Yellow (low density) */
                            );
                            margin-right: 10px;
                            border: 1px solid black;
                        "></div>
                        <div style="display: flex; flex-direction: column;
                            justify-content: space-between;">
                            <span>High</span>
                            <span>Low</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except (ValueError, TypeError) as e:
            self.logger.error("Error creating heatmap: %s", str(e), exc_info=True)
            raise
