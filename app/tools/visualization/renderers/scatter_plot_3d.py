"""
Renderer for 3D scatter plot visualizations.
"""
from typing import Any, Dict, Optional, Tuple, List
import time
import pydeck as pdk
import streamlit as st
from ..base import BaseChartRenderer
from ..chart_types import ChartType

# pylint: disable=no-member
class ScatterPlot3DRenderer(BaseChartRenderer):
    """
    Renderer for 3D scatter plot visualization.
    """
    @property
    def supported_chart_types(self) -> list[ChartType]:
        return [ChartType.SCATTER_PLOT_3D]

    def render(self, data: Any, parameters: Optional[Dict] = None,
               cache_buster: Optional[str] = None) -> Any:
        """
        Render a 3D scatter plot visualization.

        Args:
            data: Dictionary containing property_name and countries data
            parameters: Additional visualization parameters
            cache_buster: Optional cache buster string

        Returns:
            PyDeck deck object
        """
        try:
            message_index = cache_buster if cache_buster is not None else int(time.time())
            property_name = data.get("property_name", "")
            countries_data = data.get("countries", {})

            if not countries_data:
                raise ValueError("No country data provided")

            col1, col2 = st.columns([3, 1])  # 3:1 ratio for map:legend
            all_points, country_stats = self._process_country_data(countries_data, property_name)

            if not all_points:
                raise ValueError("No valid data points found")

            global_min, global_max = self._get_global_bounds(all_points, property_name)

            for point in all_points:
                point["formatted_long"] = f"{point['decimallongitude']:.2f}"
                point["formatted_lat"] = f"{point['decimallatitude']:.2f}"
                point["formatted_value"] = f"{point[property_name]:.2f}"

            layer = self._create_column_layer(all_points, property_name, global_min, global_max)
            view_state = self._calculate_view_state(all_points)

            viz_config = {
                "property_name": property_name,
                "country_stats": country_stats,
                "global_min": global_min,
                "global_max": global_max,
            }

            deck = self._render_visualization((col1, col2), layer, view_state, viz_config,
                                           key=f"3d_scatterplot_{message_index}")
            return deck

        except Exception as e:
            self.logger.error("Error creating 3D visualization: %s", str(e), exc_info=True)
            raise

    def _process_country_data(self, countries_data: Dict, property_name: str) -> Tuple[List[Dict], Dict]:
        """Process country data and return points and stats."""
        all_points = []
        country_stats = {}
        for country_name, country_info in countries_data.items():
            if "error" in country_info or not country_info.get("data"):
                continue
            country_data = country_info["data"]
            values = [point[property_name] for point in country_data]
            country_stats[country_name] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "count": len(values),
            }
            all_points.extend(country_data)
        return all_points, country_stats

    def _get_global_bounds(self, points: List[Dict], property_name: str) -> Tuple[float, float]:
        """Calculate global min/max values."""
        values = [point[property_name] for point in points]
        return min(values), max(values)

    def _create_column_layer(self, points: List[Dict], property_name: str,
                           global_min: float, global_max: float) -> pdk.Layer:
        """Create PyDeck column layer."""
        # Ensure we have a valid range
        value_range = global_max - global_min
        return pdk.Layer(
            "ColumnLayer",
            data=points,
            get_position=["decimallongitude", "decimallatitude"],
            get_elevation=f"[{property_name}]",
            elevation_scale=5000,
            radius=8000,
            get_fill_color=f"[255, 255 * ({global_max} - {property_name})/({value_range}), 0, 150]",
            pickable=True,
            auto_highlight=True,
            extruded=True,
        )

    def _calculate_view_state(self, points: List[Dict]) -> pdk.ViewState:
        """Calculate view state from points."""
        lats = [p["decimallatitude"] for p in points]
        lons = [p["decimallongitude"] for p in points]
        return pdk.ViewState(
            latitude=sum(lats) / len(lats),
            longitude=sum(lons) / len(lons),
            zoom=4,
            pitch=45,
        )

    def _render_visualization(self, columns: Tuple[Any, Any], layer: pdk.Layer,
                            view_state: pdk.ViewState, viz_config: Dict, key: str) -> pdk.Deck:
        """Render the visualization."""
        col1, col2 = columns
        with col1:
            deck = pdk.Deck(
                initial_view_state=view_state,
                layers=[layer],
                tooltip={
                    "html": (
                        "<b>Value:</b> {formatted_value}<br/>"
                        "<b>Location:</b> {formatted_long}, {formatted_lat}"
                    ),
                    "style": {"backgroundColor": "steelblue", "color": "white"},
                },
            )
            st.pydeck_chart(deck, height=700, key=key)

        with col2:
            st.markdown(f"### {viz_config['property_name'].replace('_', ' ').title()}")
            st.markdown("### Country Statistics")

            for country_name, stats in viz_config["country_stats"].items():
                st.markdown(
                    f"""
                    <div style="margin-bottom: 20px;">
                        <div style="margin-bottom: 10px;">
                            <strong>{country_name}</strong>
                        </div>
                        <div style="margin-left: 10px;">
                            <p style="margin: 0;">Points: {stats['count']:,}</p>
                            <p style="margin: 0;">Mean: {stats['mean']:.2f}</p>
                            <p style="margin: 0;">Range: {stats['min']:.2f}
                                - {stats['max']:.2f}</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Add color gradient legend
            st.markdown("### Color Scale")
            st.markdown(
                f"""
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
                    <div style="display: flex; flex-direction: column; justify-content: space-between;">
                        <span>{viz_config['global_max']:.2f}</span>
                        <span>{(viz_config['global_max'] * 2/3 +
                                    viz_config['global_min'] * 1/3):.2f}</span>
                        <span>{(viz_config['global_max'] * 1/3 +
                                    viz_config['global_min'] * 2/3):.2f}</span>
                        <span>{viz_config['global_min']:.2f}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                ### How to Read
                - Each column represents a data point
                - Height indicates value
                - Color indicates value (red=high, yellow=low)
                - Hover over points for exact values
            """
            )

        return deck