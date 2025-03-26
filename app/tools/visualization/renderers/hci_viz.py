"""
Renderer for species-HCI correlation visualizations.
"""
from typing import Any, Dict, Optional
import time
import plotly.graph_objects as go
import streamlit as st
from ..base import BaseChartRenderer
from ..chart_types import ChartType

# pylint: disable=no-member
class HCIRenderer(BaseChartRenderer):
    """
    Renderer for species-PCI visualization.
    """
    @property
    def supported_chart_types(self) -> list[ChartType]:
        return [ChartType.SPECIES_HCI_CORRELATION]

    def render(self, data: Any, parameters: Optional[Dict] = None,
               cache_buster: Optional[str] = None) -> Any:
        """
        Render a scatter plot showing species-HCI correlation.

        Args:
            data: Dictionary containing correlation results
            parameters: Parameters for visualization
            cache_buster: Optional cache buster string

        Returns:
            Plotly figure object
        """
        try:
            message_index = cache_buster if cache_buster is not None else int(time.time())
            correlations = data["correlations"]
            if not correlations:
                st.warning("No correlation data found.")
                return None

            col1, col2 = st.columns([7, 3])
            with col1:
                fig = go.Figure()

                # Check if we're dealing with a single conservation status
                statuses = {item["conservation_status"] for item in correlations}
                is_single_status = len(statuses) == 1

                if is_single_status:
                    # For single status, color by correlation coefficient
                    status = list(statuses)[0]
                    correlations_array = [item["correlation_coefficient"] for item in correlations]
                    min_corr = min(correlations_array)
                    max_corr = max(correlations_array)

                    fig.add_trace(go.Scatter(
                        x=[item["avg_hci"] for item in correlations],
                        y=[item["correlation_coefficient"] for item in correlations],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=[item["correlation_coefficient"] for item in correlations],
                            colorscale='RdYlBu',  # Red for negative, Blue for positive correlations
                            colorbar=dict(
                                title=dict(
                                    text="Correlation<br>Coefficient",
                                    side='right'
                                )
                            ),
                            line=dict(width=1, color='black')
                        ),
                        text=[f"{item['species_name']}<br>"
                              f"English Name: {item['species_name_en'] or 'N/A'}<br>"
                              f"Correlation: {item['correlation_coefficient']:.3f}<br>"
                              f"Grid Cells: {item['number_of_grid_cells']}<br>"
                              f"Total Individuals: {item['total_individuals']}"
                              for item in correlations],
                        hoverinfo='text'
                    ))

                    title = f"Species-HCI Correlation for {status} Species"

                else:
                    # Original behavior for multiple conservation statuses
                    color_scheme = {
                        'Extinct': '#8B0000',                # Dark Red
                        'Critically Endangered': '#d62728',   # Red
                        'Endangered': '#ff7f0e',             # Orange
                        'Vulnerable': '#ffd700',             # Gold
                        'Near Threatened': '#2ca02c',         # Green
                        'Least Concern': '#1f77b4',          # Blue
                        'Data Deficient': '#7f7f7f'          # Gray
                    }

                    for status, color in color_scheme.items():
                        status_data = [item for item in correlations
                                     if item["conservation_status"] == status]
                        if status_data:
                            fig.add_trace(go.Scatter(
                                x=[item["avg_hci"] for item in status_data],
                                y=[item["correlation_coefficient"] for item in status_data],
                                mode='markers',
                                name=status,
                                text=[f"{item['species_name']}<br>"
                                      f"English Name: {item['species_name_en'] or 'N/A'}<br>"
                                      f"Status: {item['conservation_status']}<br>"
                                      f"Grid Cells: {item['number_of_grid_cells']}<br>"
                                      f"Total Individuals: {item['total_individuals']}"
                                      for item in status_data],
                                hoverinfo='text',
                                marker=dict(
                                    size=10,
                                    color=color,
                                    line=dict(width=1, color='black')
                                )
                            ))

                    title = f"Species-HCI Correlation in {parameters.get('country_code', 'KEN')}"

                # Update layout
                fig.update_layout(
                    title=title,
                    xaxis_title="Average HCI",
                    yaxis_title="Correlation Coefficient",
                    hovermode='closest',
                    height=700,
                    showlegend=not is_single_status  # Hide legend for single status
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### Correlation Analysis")
                if is_single_status:
                    status = list(statuses)[0]
                    st.markdown(f"### {status} Species")
                    st.markdown("""
                        This scatter plot shows the relationship between:
                        - Species occurrence
                        - Human Coexistence Index (HCI)

                        **How to Read:**
                        - Each point represents a species
                        - X-axis: Average HCI where species is found
                        - Y-axis: Correlation coefficient
                        - Color: Correlation strength and direction
                          - Blue: Positive correlation
                          - Red: Negative correlation

                        **Interpretation:**
                        - Blue points: Species more common in high HCI areas
                        - Red points: Species more common in low HCI areas
                        - White/pale points: No clear relationship with HCI
                    """)

                    # Add summary statistics
                    pos_threshold = 0.3
                    pos_corr = len([c for c in correlations
                                  if c["correlation_coefficient"] > pos_threshold])
                    neg_corr = len([c for c in correlations
                                  if c["correlation_coefficient"] < -pos_threshold])
                    neut_corr = len([c for c in correlations
                                   if abs(c["correlation_coefficient"]) <= pos_threshold])

                    st.markdown(f"""
                        **Summary Statistics:**
                        - Total species analyzed: {len(correlations)}
                        - Strong positive correlation (>0.3): {pos_corr}
                        - Strong negative correlation (<-0.3): {neg_corr}
                        - Weak/no correlation: {neut_corr}
                        - Correlation range: {min_corr:.3f} to {max_corr:.3f}
                    """)

                else:
                    st.markdown("""
                        This scatter plot shows the relationship between:
                        - Species occurrence
                        - Human Coexistence Index (HCI)

                        **How to Read:**
                        - Each point represents a species
                        - X-axis: Average HCI where species is found
                        - Y-axis: Correlation coefficient
                        - Color: Conservation status

                        **Interpretation:**
                        - Positive correlation: Species more common in high HCI areas
                        - Negative correlation: Species more common in low HCI areas
                        - Near zero: No clear relationship with HCI
                    """)

                    # Add status counts
                    st.markdown("### Status Distribution")
                    color_scheme = {
                        'Extinct': '#8B0000',
                        'Critically Endangered': '#d62728',
                        'Endangered': '#ff7f0e',
                        'Vulnerable': '#ffd700',
                        'Near Threatened': '#2ca02c',
                        'Least Concern': '#1f77b4',
                        'Data Deficient': '#7f7f7f'
                    }

                    for status, color in color_scheme.items():
                        count = len([item for item in correlations
                                   if item["conservation_status"] == status])
                        if count > 0:
                            st.markdown(
                                f"""
                                <div style="display: flex; align-items: center; margin: 5px 0;">
                                    <div style="width: 20px; height: 20px;
                                              background-color: {color};
                                              margin-right: 5px; border: 1px solid black;">
                                    </div>
                                    <span>{status}: {count} species</span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

            return fig

        except Exception as e:
            self.logger.error("Error creating species-HCI correlation plot: %s",
                            str(e), exc_info=True)
            raise
