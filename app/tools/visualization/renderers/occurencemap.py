"""
Renderer for occurrence map visualizations.
"""
from typing import Any, Dict, Optional
import time
import pandas as pd
import pydeck as pdk
import streamlit as st
from ..base import BaseChartRenderer
from ..chart_types import ChartType

# pylint: disable=no-member
class OccurrenceMapRenderer(BaseChartRenderer):
    """
    Renderer for occurrence map visualizations.
    """
    @property
    def supported_chart_types(self) -> list[ChartType]:
        return [ChartType.OCCURRENCE_MAP]

    def render(self, data: Any, parameters: Optional[Dict] = None,
               cache_buster: Optional[str] = None) -> Any:
        """
        Render an occurrence map visualization.

        Args:
            data: DataFrame or dictionary containing occurrence data
            parameters: Dictionary containing visualization parameters
            cache_buster: Optional cache buster string

        Returns:
            PyDeck chart object
        """
        try:
            message_index = cache_buster if cache_buster is not None else int(time.time())

            # Handle both DataFrame and raw data formats
            if isinstance(data, pd.DataFrame):
                df = data
                country_code = parameters.get('country_code', 'Unknown')
                total_occurrences = len(df)
            else:
                # Create DataFrame from occurrences
                df = pd.DataFrame(data['occurrences'])
                country_code = data.get('country_code', 'Unknown')
                total_occurrences = data.get('total_occurrences', len(df))

            if df.empty:
                st.markdown("No occurrence data to display")
                return

            # Define color scheme for conservation status
            color_scheme = {
                'Extinct': [139, 0, 0],                # Dark Red
                'Critically Endangered': [214, 39, 40], # Red
                'Endangered': [255, 127, 14],          # Orange
                'Vulnerable': [255, 215, 0],           # Gold
                'Near Threatened': [44, 160, 44],      # Green
                'Least Concern': [31, 119, 180],       # Blue
                'Data Deficient': [127, 127, 127]      # Gray
            }

            # Add color column to DataFrame with RGB arrays
            df['color'] = df['conservation_status'].map(color_scheme)

            # Create two columns for map and legend
            col1, col2 = st.columns([3, 1])

            with col1:
                # Create scatter plot layer
                scatter_layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=df,
                    get_position=['decimallongitude', 'decimallatitude'],
                    get_color='color',  # Now points to RGB array
                    get_radius=10,
                    radius_min_pixels=2,
                    radius_max_pixels=100,
                    opacity=0.8,
                    stroked=True,
                    filled=True,
                    pickable=True,
                    auto_highlight=True
                )

                # Set initial view state
                view_state = pdk.ViewState(
                    latitude=df['decimallatitude'].mean(),
                    longitude=df['decimallongitude'].mean(),
                    zoom=7,
                    pitch=45,
                    bearing=0
                )

                # Define tooltip content
                tooltip_content = """
                    <div style="background-color: rgba(0, 0, 0, 0.8); color: white; padding: 10px; border-radius: 5px;">
                        <b>Species:</b> {species}<br/>
                        <b>Conservation Status:</b> {conservation_status}
                    </div>
                """

                # Check if column exists and has non-null value
                if 'species_name_en' in df.columns and not df['species_name_en'].isna().all():
                    tooltip_content = """
                        <div style="background-color: rgba(0, 0, 0, 0.8); color: white; padding: 10px; border-radius: 5px;">
                            <b>Species:</b> {species}<br/>
                            <b>Common name:</b> {species_name_en}<br/>
                            <b>Conservation Status:</b> {conservation_status}
                        </div>
                    """

                # Enhanced tooltip
                tooltip = {
                    "html": tooltip_content,
                    "style": {
                        "backgroundColor": "transparent",
                        "color": "white"
                    }
                }

                # Create and display the map
                deck = pdk.Deck(
                    layers=[scatter_layer],
                    initial_view_state=view_state,
                    tooltip=tooltip,
                    map_style='mapbox://styles/mapbox/dark-v10',
                )

                # Add CSS styling for the map
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
                st.pydeck_chart(deck, height=800, use_container_width=True, key=f"occurrence_map_{message_index}")

            # Add legend in the second column
            with col2:
                st.markdown("### Conservation Status")
                st.markdown(f"**Country**: {country_code}")
                st.markdown(f"**Total Occurrences**: {total_occurrences:,}")

                # Create legend with colored boxes
                for status, color in color_scheme.items():
                    status_count = len(df[df['conservation_status'] == status])
                    if status_count > 0:  # Only show statuses that are present in the data
                        st.markdown(
                            f"""
                            <div style="display: flex; align-items: center; margin: 5px 0;">
                                <div style="width: 20px; height: 20px;
                                            background: rgb({color[0]},{color[1]},{color[2]});
                                            margin-right: 5px; border: 1px solid black;">
                                </div>
                                <span><b>{status}</b> ({status_count:,} occurrences)</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
        except Exception as e:
            self.logger.error("Error creating occurrence map: %s", str(e), exc_info=True)
            raise
