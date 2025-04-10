"""
This module provides a renderer for hexagon map visualizations.

Classes:
    HexagonmapRenderer: Renderer for hexagon map visualizations.
"""
import logging
import time
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from ..base import BaseChartRenderer
from ..chart_types import ChartType

class HexagonmapRenderer(BaseChartRenderer):
    """
    Renderer for hexagon map visualizations.
    """
    @property
    def supported_chart_types(self) -> list[ChartType]:
        return [ChartType.HEXAGON_MAP, ChartType.HEXAGONMAP]

    def render(self, data: Any, parameters: Optional[Dict] = None,
               _cache_buster: Optional[str] = None) -> Any:
        try:
            if not isinstance(parameters, dict):
                raise TypeError("Parameters must be a dictionary")
            logging.debug("Drawing hexagon map with parameters: %s", parameters)
            occurrences = data["occurrences"]
            message_index = _cache_buster if _cache_buster is not None else int(time.time())

            # Convert the JSON array to a DataFrame
            df = pd.DataFrame(occurrences)
            bounds = self.get_bounds_from_data(df)

            hull_geojson = self.create_alpha_shapes(df, parameters)

            # Force a new view state each time with a slight random variation to prevent caching
            if bounds is not None:
                view_state = pdk.ViewState(
                    latitude=sum(coord[0] for coord in bounds) / len(bounds),
                    longitude=sum(coord[1] for coord in bounds) / len(bounds),
                    zoom=3,
                    pitch=30,
                    bearing=0.001 * np.random.randn()  # Tiny random bearing to force redraw
                )
            else:
                view_state = self.default_view_state

            # Add CSS styling for the map
            # pylint: disable=no-member
            st.markdown(
                """
                <style>
                .stDeckGLChart {
                    width: 100% !important;
                    height: 800px !important;
                    max-width: none !important;
                    box-sizing: border-box !important;
                    margin: 0 !important;
                    padding: 0 !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            # pylint: disable=no-member
            col1, col2 = st.columns([3, 1])
            with col1:
                st.pydeck_chart(
                    pdk.Deck(
                        initial_view_state=view_state,
                        layers=[
                            pdk.Layer(
                                "HexagonLayer",
                                data=df,
                                get_position=["decimallongitude", "decimallatitude"],
                                radius=10000,
                                elevation_scale=5000,
                                elevation_range=[0, 1000],
                                pickable=True,
                                extruded=True,
                            ),
                            pdk.Layer(
                                "GeoJsonLayer",
                                data=hull_geojson,
                                stroked=True,
                                filled=False,
                                line_width_min_pixels=2,
                                get_line_color=[255, 255, 0],
                                get_line_width=3,
                            ),
                        ],
                        tooltip={
                            "html": (
                                f"{parameters.get('species_name', '')}<br/>"
                                "Occurrences: {elevationValue}"
                            ),
                            "style": {"backgroundColor": "steelblue", "color": "white"},
                        },
                    ),
                    height=800,
                    use_container_width=True,  # Ensure proper sizing
                    key=f"hexagon_map_{message_index}"
                )

            # Add legend in the second column
            with col2:
                st.markdown("### Distribution Map")
                st.markdown(f"**Species**: {parameters.get('species_name', 'Unknown')}")

                # Add explanation of visualization
                st.markdown(
                    """
                    This map shows species distribution using:
                    **Hexagons**
                    - Each hexagon represents a geographic area
                    - Height indicates number of observations
                    - Darker color means more observations
                    - Hover over hexagons to see exact counts

                    **Yellow Outline**
                    - Shows the species' range boundary
                    - Calculated using alpha shape algorithm
                    - Connects outermost observation points

                    **Color Scale**
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

        except Exception as e:
            self.logger.error("Error creating hexagon map: %s", str(e), exc_info=True)
            raise
