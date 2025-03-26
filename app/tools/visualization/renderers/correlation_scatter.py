"""
Renderer for correlation scatter plot visualizations.
"""
from typing import Any, Dict, Optional
import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ..base import BaseChartRenderer
from ..chart_types import ChartType

# pylint: disable=no-member
class CorrelationScatterRenderer(BaseChartRenderer):
    """
    Renderer for correlation scatter plot visualization.
    """
    @property
    def supported_chart_types(self) -> list[ChartType]:
        return [ChartType.CORRELATION_SCATTER]

    def render(self, data: Any, parameters: Optional[Dict] = None,
               cache_buster: Optional[str] = None) -> Any:
        """
        Render a correlation scatter plot visualization.

        Args:
            data: Dictionary containing correlation data
            parameters: Additional visualization parameters
            cache_buster: Optional cache buster string

        Returns:
            Plotly figure object
        """
        try:
            message_index = cache_buster if cache_buster is not None else int(time.time())
            col1, col2 = st.columns([3, 1])  # 3:1 ratio for plot:legend

            with col1:
                fig = self._draw_correlation_plot(data, parameters, message_index)
            with col2:
                self._display_correlation_summary(parameters)

            return fig

        except Exception as e:
            self.logger.error("Error creating correlation scatter plot: %s", str(e), exc_info=True)
            raise

    def _draw_correlation_plot(self, data: Dict, parameters: Dict, message_index: str) -> go.Figure:
        """Creates and displays the correlation scatter plot."""
        countries = data.get("countries", [])
        if not countries:
            raise ValueError("No correlation data available")

        df = pd.DataFrame([{
            'Country': country['iso3'],
            'HCI': country['hci'],
            'Species': country['species_count']
        } for country in countries])

        fig = go.Figure(data=go.Scatter(
            x=df['HCI'],
            y=df['Species'],
            mode='markers',
            marker=dict(
                size=15,
                color='#FF4B4B'
            ),
            text=df['Country'],
            hovertemplate=(
                "Country: %{text}<br>"
                "HCI: %{x:.2f}<br>"
                "Species: %{y}<br>"
                "<extra></extra>"
            )
        ))

        fig.update_layout(
            height=700,
            showlegend=False,
            xaxis_title="Human Coexistence Index (HCI)",
            yaxis_title="Number of Endangered Species",
            yaxis_type="log",
            paper_bgcolor='rgb(50, 50, 50)',
            plot_bgcolor='rgb(50, 50, 50)',
            font=dict(
                color='white',
                size=16  # Increased base font size
            ),
            xaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.2)',
                zerolinecolor='rgba(255, 255, 255, 0.2)',
                title_font=dict(size=20),  # Larger axis title
                tickfont=dict(size=14)     # Larger tick labels
            ),
            yaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.2)',
                zerolinecolor='rgba(255, 255, 255, 0.2)',
                title_font=dict(size=22),  # Larger axis title
                tickfont=dict(size=16)     # Larger tick labels
            ),
            hoverlabel=dict(font_size=16)  # Larger hover text
        )

        st.plotly_chart(fig, use_container_width=True, key=f"correlation_scatter_{message_index}")
        return fig

    def _display_correlation_summary(self, parameters: Dict) -> None:
        """Displays the correlation analysis summary."""
        st.markdown(f"### Correlation Analysis for {parameters.get('continent', 'Africa')}")
        st.markdown("""
            This scatter plot shows the relationship between:
            - Human Coexistence Index (HCI)
            - Number of Endangered Species (all categories)

            **How to Read:**
            - Each point represents a country
            - X-axis: HCI value
            - Y-axis: Number of Endangered species (log scale)

            **Interpretation:**
            - Upward trend: Positive correlation
            - Downward trend: Negative correlation
            - Scattered points: Weak/no correlation
            """)