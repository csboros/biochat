"""Renderer for topography analysis visualization."""

from typing import Dict, Any, Optional
import folium
import plotly.graph_objects as go
from streamlit_folium import folium_static
from folium.plugins import Fullscreen
import streamlit as st
from ..base import BaseChartRenderer
from ..chart_types import ChartType

# pylint: disable=broad-except
# pylint: disable=no-member
class TopographyViz(BaseChartRenderer):
    """Renders topography analysis results on a map and as charts."""

    def __init__(self):
        super().__init__()
        self.species_points_style = {
            'color': 'red',
            'fillColor': 'red',
            'fillOpacity': 0.7,
            'radius': 5
        }

    @property
    def supported_chart_types(self) -> list[ChartType]:
        """Return list of chart types this renderer supports."""
        return [ChartType.TOPOGRAPHY_ANALYSIS]

    def render(
        self,
        data: Dict[str, Any],
        parameters: Optional[Dict] = None,
        cache_buster: Optional[str] = None
    ) -> None:
        """
        Render topography analysis results on a map and as charts.
        """
        try:
            # Create two columns for the main layout
            left_col, right_col = st.columns([3, 1])

            with left_col:
                # Create three columns for the charts
                chart_col1, chart_col2, chart_col3 = st.columns(3)

                # Get the data from the correct structure
                analysis_results = data.get('data', {})
                visualizations = data.get('visualizations', {})
                if not visualizations:
                    st.error("No visualization data available")
                    return

                # Create charts
                with chart_col1:
                    elevation_chart = self._create_elevation_chart(
                        analysis_results.get('elevation', {})
                    )
                    st.plotly_chart(elevation_chart, use_container_width=True)

                with chart_col2:
                    slope_chart = self._create_slope_chart(
                        analysis_results.get('slope', {})
                    )
                    st.plotly_chart(slope_chart, use_container_width=True)

                with chart_col3:
                    aspect_chart = self._create_aspect_chart(
                        analysis_results.get('aspect', {})
                    )
                    st.plotly_chart(aspect_chart, use_container_width=True)

                # Create and display map below charts
                map_viz = self._create_map(
                    visualizations.get('elevation'),
                    visualizations.get('slope'),
                    visualizations.get('aspect'),
                    visualizations.get('species_points'),
                    visualizations.get('center')
                )

                # Map styling
                st.markdown(
                    """
                    <style>
                    .folium-map {
                        width: 100% !important;
                        height: 800px !important;
                        max-width: none !important;
                        box-sizing: border-box !important;
                        margin: 0 !important;
                        padding: 0 !important;
                    }
                    iframe {
                        width: 100% !important;
                        height: 900px !important;
                        border: none !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                folium_static(map_viz, width=1200)

            with right_col:
                # Display detailed statistics and descriptions
                st.markdown("### Topography Analysis Results")

                # Elevation Statistics
                elevation_data = analysis_results.get('elevation', {})
                st.markdown("#### Elevation")
                mean_elev = elevation_data.get('mean')
                min_elev = elevation_data.get('min')
                max_elev = elevation_data.get('max')
                std_elev = elevation_data.get('std')
                st.markdown(f"""
                    - **Mean:** {f"{mean_elev:.0f}m" if mean_elev is not None else "N/A"}
                    - **Range:** {f"{min_elev:.0f}m - {max_elev:.0f}m"
                                if min_elev is not None and max_elev is not None
                                else "N/A"}
                    - **Standard Deviation:** {f"{std_elev:.1f}m"
                                            if std_elev is not None else "N/A"}
                """)

                # Slope Statistics
                slope_data = analysis_results.get('slope', {})
                st.markdown("#### Slope")
                mean_slope = slope_data.get('mean')
                min_slope = slope_data.get('min')
                max_slope = slope_data.get('max')
                std_slope = slope_data.get('std')
                st.markdown(f"""
                    - **Mean:** {f"{mean_slope:.1f}°" if mean_slope is not None else "N/A"}
                    - **Range:** {f"{min_slope:.1f}° - {max_slope:.1f}°"
                                if min_slope is not None and max_slope is not None
                                else "N/A"}
                    - **Standard Deviation:** {f"{std_slope:.1f}°" if std_slope is not None else "N/A"}
                """)

                # Slope Categories
                slope_categories = slope_data.get('categories', {})
                st.markdown("##### Slope Categories")
                for category, count in slope_categories.items():
                    st.markdown(f"- **{category.title()}:** {count} observations")

                # Aspect Statistics
                aspect_data = analysis_results.get('aspect', {})
                st.markdown("#### Aspect")
                mean_aspect = aspect_data.get('mean')
                min_aspect = aspect_data.get('min')
                max_aspect = aspect_data.get('max')
                std_aspect = aspect_data.get('std')
                st.markdown(f"""
                    - **Mean:** {f"{mean_aspect:.1f}°" if mean_aspect is not None else "N/A"}
                    - **Range:** {f"{min_aspect:.1f}° - {max_aspect:.1f}°"
                                if min_aspect is not None and max_aspect is not None
                                else "N/A"}
                    - **Standard Deviation:** {f"{std_aspect:.1f}°"
                                            if std_aspect is not None else "N/A"}
                """)

                # Aspect Categories
                aspect_categories = aspect_data.get('categories', {})
                st.markdown("##### Aspect Categories")
                for category, count in aspect_categories.items():
                    st.markdown(f"- **{category.title()}:** {count} observations")

                # Map Layer Controls
                st.markdown("### Map Layers")
                st.markdown("""
                    <p style="margin: 5px 0;"><span style="color: #00ff00">■</span> Elevation</p>
                    <p style="margin: 5px 0;"><span style="color: #ffff00">■</span> Slope</p>
                    <p style="margin: 5px 0;"><span style="color: #0000ff">■</span> Aspect</p>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error rendering topography visualization: {str(e)}")

    def _create_elevation_chart(self, elevation_data: Dict[str, Any]) -> go.Figure:
        """Create histogram showing elevation distribution."""
        # Create histogram data
        elevation_values = []
        counts = []
        for value, count in elevation_data.get('distribution', {}).items():
            elevation_values.append(float(value))
            counts.append(count)

        # Create figure
        fig = go.Figure(data=[go.Histogram(
            x=elevation_values,
            y=counts,
            nbinsx=20,
            marker_color='#00ff00',
            hovertemplate=(
                "Elevation: %{x:.0f}m<br>" +
                "Count: %{y}<br>" +
                "<extra></extra>"
            )
        )])

        # Add mean line if available
        mean_elevation = elevation_data.get('mean')
        if mean_elevation is not None:
            fig.add_vline(
                x=mean_elevation,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_elevation:.0f}m",
                annotation_position="top right"
            )

        fig.update_layout(
            title={
                'text': 'Elevation Distribution',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Elevation (m)",
            yaxis_title="Number of Observations",
            height=500,
            showlegend=False
        )
        return fig

    def _create_slope_chart(self, slope_data: Dict[str, Any]) -> go.Figure:
        """Create bar chart showing slope distribution."""
        # Create figure
        fig = go.Figure()

        # Add slope distribution bars
        categories = slope_data.get('categories', {})
        colors = ['#ffff00', '#ffcc00', '#ff9900', '#ff6600']

        for (category, count), color in zip(categories.items(), colors):
            fig.add_trace(
                go.Bar(
                    name=category.title(),
                    x=['Slope Categories'],
                    y=[count],
                    marker_color=color,
                    hovertemplate=(
                        "<b>%{data.name}</b><br>" +
                        "Count: %{y}<br>" +
                        "<extra></extra>"
                    )
                )
            )

        fig.update_layout(
            title={
                'text': 'Slope Distribution',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            barmode='group',
            height=500,
            showlegend=True,
            legend={
                'orientation': "h",
                'yanchor': "bottom",
                'y': -0.5,
                'xanchor': "center",
                'x': 0.5
            },
            yaxis_title="Number of Observations"
        )
        return fig

    def _create_aspect_chart(self, aspect_data: Dict[str, Any]) -> go.Figure:
        """Create pie chart showing aspect distribution."""
        # Create figure
        fig = go.Figure()

        # Get aspect categories
        categories = aspect_data.get('categories', {})
        colors = ['#0000ff', '#00ff00', '#ffff00', '#ff0000']  # N, E, S, W

        # Create pie chart
        fig.add_trace(go.Pie(
            labels=[cat.title() for cat in categories.keys()],
            values=list(categories.values()),
            hole=0.4,
            marker={'colors': colors},
            hovertemplate=(
                "<b>%{label}</b><br>" +
                "Count: %{value}<br>" +
                "Percentage: %{percent}<br>" +
                "<extra></extra>"
            )
        ))

        fig.update_layout(
            title={
                'text': 'Aspect Distribution',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            height=500,
            showlegend=True,
            legend={
                'orientation': "h",
                'yanchor': "bottom",
                'y': -0.5,
                'xanchor': "center",
                'x': 0.5
            }
        )
        return fig

    def _create_map(
        self,
        elevation: Dict[str, Any],
        slope: Dict[str, Any],
        aspect: Dict[str, Any],
        species_points: Dict[str, Any],
        center: Dict[str, float]
    ) -> folium.Map:
        """Create the map visualization."""
        try:
            m = folium.Map(
                location=[center['lat'], center['lon']],
                zoom_start=6
            )

            # Add elevation layer
            if elevation and elevation.get('tiles'):
                folium.raster_layers.TileLayer(
                    tiles=elevation['tiles'][0],
                    attr=elevation['attribution'],
                    name='Elevation'
                ).add_to(m)

            # Add slope layer
            if slope and slope.get('tiles'):
                folium.raster_layers.TileLayer(
                    tiles=slope['tiles'][0],
                    attr=slope['attribution'],
                    name='Slope'
                ).add_to(m)

            # Add aspect layer
            if aspect and aspect.get('tiles'):
                folium.raster_layers.TileLayer(
                    tiles=aspect['tiles'][0],
                    attr=aspect['attribution'],
                    name='Aspect'
                ).add_to(m)

            # Add species points
            if species_points and species_points.get('features'):
                points_group = folium.FeatureGroup(name="Species Observations")

                for feature in species_points['features']:
                    try:
                        coords = feature['geometry']['coordinates']
                        if not coords:
                            continue

                        # Create tooltip
                        tooltip_content = f"""
                            <div style="font-family: Arial, sans-serif;
                                        background-color: white;
                                        padding: 5px;
                                        border-radius: 3px;
                                        border: 1px solid #ccc;">
                                <b>Location:</b><br>
                                Lat: {coords[1]:.4f}<br>
                                Lon: {coords[0]:.4f}
                            </div>
                        """

                        # Add point to the feature group
                        folium.CircleMarker(
                            location=[coords[1], coords[0]],
                            radius=5,
                            color='white',
                            weight=2,
                            fill=True,
                            fillColor='red',
                            fillOpacity=0.7,
                            tooltip=folium.Tooltip(
                                tooltip_content,
                                permanent=False,
                                sticky=True
                            )
                        ).add_to(points_group)

                    except Exception as e:
                        st.error(f"Error processing point: {str(e)}")
                        continue

                # Add the feature group to the map
                points_group.add_to(m)

            # Add layer control and fullscreen
            folium.LayerControl().add_to(m)
            Fullscreen().add_to(m)

            return m

        except Exception as e:
            st.error(f"Error creating map: {str(e)}")
            return folium.Map(location=[center['lat'], center['lon']], zoom_start=6)