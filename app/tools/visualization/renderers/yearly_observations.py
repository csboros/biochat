"""
Renderer for yearly observations visualizations.
"""
from typing import Optional, List, Dict, Any
import time
import colorsys
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ..base import BaseChartRenderer
from ..chart_types import ChartType

# pylint: disable=no-member
class YearlyObservationsRenderer(BaseChartRenderer):
    """
    Renderer for yearly observations visualization.
    """
    @property
    def supported_chart_types(self) -> list[ChartType]:
        return [ChartType.YEARLY_OBSERVATIONS]

    def render(self, data: Any, parameters: Optional[Dict] = None,
               cache_buster: Optional[str] = None) -> Any:
        """
        Render a yearly observations visualization.

        Args:
            data: Dictionary containing yearly observation data
            parameters: Additional visualization parameters
            cache_buster: Optional cache buster string

        Returns:
            Plotly figure object
        """
        try:
            message_index = cache_buster if cache_buster is not None else int(time.time())
            if "error" in data:
                raise ValueError(data["error"])

            names = {
                "common": data.get("common_name", "Unknown"),
                "scientific": data.get("scientific_name", "Unknown"),
            }
            yearly_data = data.get("yearly_data", {})

            col1, col2 = st.columns([3, 1])

            with col1:
                fig = self._draw_observation_chart(yearly_data, names["common"],
                                                 key=f"yearly_observations_{message_index}")
            with col2:
                self._display_observation_summary(yearly_data, names,
                                                key=f"yearly_observations_summary_{message_index}")

            return fig

        except Exception as e:
            self.logger.error("Error drawing yearly observations: %s", str(e), exc_info=True)
            raise

    def _draw_observation_chart(self, yearly_data: Dict, title: str, key: str) -> go.Figure:
        """Creates and displays the observation chart."""
        if isinstance(yearly_data, dict):
            return self._draw_country_chart(yearly_data, title,
                                          key=f"yearly_observations_country_chart_{key}")
        else:
            return self._draw_global_chart(yearly_data,
                                         key=f"yearly_observations_global_chart_{key}")

    def _draw_country_chart(self, yearly_data: Dict, title: str, key: str) -> go.Figure:
        """Creates and displays the country-specific observation chart."""
        try:
            # Create DataFrame for plotting
            colors = self._get_distinct_colors(len(yearly_data))

            fig = go.Figure()

            # Add traces for each country
            for country, data in yearly_data.items():
                df = pd.DataFrame(data)
                color = next(colors)
                fig.add_trace(go.Scatter(
                    x=df['year'],
                    y=df['count'],
                    name=country,
                    line={"color": color},
                    mode='lines+markers'
                ))

            # Add global observations trace
            df = pd.DataFrame(yearly_data)
            fig.add_trace(go.Scatter(
                x=df['year'],
                y=df['count'],
                mode='lines+markers',
                name='Global Observations',
                line={"color": '#1f77b4'}
            ))

            fig.update_layout(
                title=f"Yearly Observations: {title}",
                xaxis_title="Year",
                yaxis_title="Number of Observations",
                height=700,
                hovermode='x unified',
                showlegend=True,
                legend={
                    "yanchor": "top",
                    "y": 0.99,
                    "xanchor": "left",
                    "x": 0.01
                }
            )
            st.plotly_chart(fig, use_container_width=True, key=key)
            return fig

        except Exception as e:
            self.logger.error("Error creating country chart: %s", str(e))
            raise

    def _draw_global_chart(self, yearly_data: List[Dict], key: str) -> go.Figure:
        """Creates and displays the global observation chart."""
        try:
            df = pd.DataFrame(yearly_data)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df['year'],
                y=df['count'],
                mode='lines+markers',
                name='Global Observations',
                line={"color": '#1f77b4'}
            ))

            fig.update_layout(
                title="Global Yearly Observations",
                xaxis_title="Year",
                yaxis_title="Number of Observations",
                height=700,
                hovermode='x',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True, key=key)
            return fig

        except Exception as e:
            self.logger.error("Error creating global chart: %s", str(e))
            raise

    def _display_observation_summary(self, yearly_data: Dict, names: Dict, key: str) -> None:
        """Displays the observation summary sidebar."""
        st.markdown("### Species Information")
        st.markdown(f"**Common Name**: {names['common']}")
        st.markdown(f"**Scientific Name**: {names['scientific']}")
        st.markdown("### Observation Summary")

        if isinstance(yearly_data, dict):
            self._display_country_summary(yearly_data, key=f"yearly_observations_country_summary_{key}")
        else:
            self._display_global_summary(pd.DataFrame(yearly_data),
                                      key=f"yearly_observations_global_summary_{key}")

    def _display_country_summary(self, yearly_data: Dict, key: str) -> None:
        """Displays the country-specific observation summary."""
        total_observations = 0
        for country, data in yearly_data.items():
            country_total = sum(item["count"] for item in data)
            total_observations += country_total
            st.markdown(f"**{country}**")
            st.markdown(f"- Total observations: {country_total:,}")
            if data:
                years = [item["year"] for item in data]
                st.markdown(f"- Year range: {min(years)} - {max(years)}")
        st.markdown(f"**Total Observations**: {total_observations:,}")

    def _display_global_summary(self, df, key=None):
        """Display global summary statistics."""
        total_observations = df['count'].sum()
        st.markdown(f"**Total Observations**: {total_observations:,}")
        st.markdown(f"**Number of Years**: {len(df['year'].unique())}")
        st.markdown(f"**Average Observations per Year**: {total_observations / len(df['year'].unique()):.1f}")

    def _get_distinct_colors(self, n: int) -> List[str]:
        """
        Generate n visually distinct colors using HSV color space.
        Args:
            n (int): Number of distinct colors needed
        Returns:
            list: List of hex color codes
        """
        colors = []
        for i in range(n):
            hue = i / n
            # High saturation and value for vivid, distinct colors
            saturation = 0.8
            value = 0.9
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            # Convert RGB to hex
            hex_color = f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"
            colors.append(hex_color)
        return colors
