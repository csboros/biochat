"""
Module for rendering climate analysis visualizations.

This module provides visualization capabilities for climate data analysis,
including temperature trends, precipitation patterns, and species-climate
relationships.
"""

from typing import Any, Dict, Optional
import folium
from folium.plugins import Fullscreen
import pandas as pd
import plotly.graph_objects as go
from streamlit_folium import folium_static
import streamlit as st
from ..base import BaseChartRenderer
from ..chart_types import ChartType

# pylint: disable=no-member
# pylint: disable=broad-except
class ClimateViz(BaseChartRenderer):
    """
    Visualization handler for climate analysis results.
    Inherits from BaseRenderer to maintain consistent visualization patterns.
    """
    @property
    def supported_chart_types(self) -> list[ChartType]:
        return [ChartType.CLIMATE_ANALYSIS]

    def render(
        self,
        data: Dict[str, Any],
        parameters: Optional[Dict] = None,
        cache_buster: Optional[str] = None
    ) -> None:
        """
        Renders climate analysis visualizations following the BaseRenderer pattern.

        Args:
            data: Dictionary containing climate analysis results
            parameters: Additional parameters for visualization
            cache_buster: Cache busting parameter
        """
        try:
            # Create two columns for the main layout
            left_col, right_col = st.columns([3, 1])

            with left_col:
                # Create three columns for the charts
                chart_col1, chart_col2, chart_col3 = st.columns(3)

                # Get the data from the correct structure
                climate_trends = data.get('climate_trends', {})
                species_climate_data = data.get('species_climate_data', {})
                visualizations = data.get('visualizations', {})

                if not visualizations:
                    st.error("No visualization data available")
                    return

                # Create charts
                with chart_col1:
                    if "temperature" in climate_trends:
                        temp_chart = self._create_temperature_chart(climate_trends["temperature"])
                        st.plotly_chart(temp_chart, use_container_width=True)

                with chart_col2:
                    if "precipitation" in climate_trends:
                        precip_chart = self._create_precipitation_chart(
                                climate_trends["precipitation"]
                                )
                        st.plotly_chart(precip_chart, use_container_width=True)

                with chart_col3:
                    if species_climate_data:
                        species_chart = self._create_species_climate_chart(species_climate_data)
                        if species_chart:
                            st.plotly_chart(species_chart, use_container_width=True)

                # Create the map
                st.markdown("### Climate Analysis Map")

                # Create a Folium map with multiple base layers
                m = folium.Map(
                    location=[visualizations['center']['lat'], visualizations['center']['lon']],
                    zoom_start=5,
                    tiles=None  # Start with no default tiles
                )

                # Add base layers
                folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
                folium.TileLayer(
                    'https://server.arcgisonline.com/ArcGIS/rest/services/'
                    'World_Imagery/MapServer/tile/{z}/{y}/{x}',
                    attr='Esri',
                    name='Satellite'
                ).add_to(m)
                folium.TileLayer('https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
                                attr='Google',
                                name='Google Maps').add_to(m)
                folium.TileLayer('https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                                attr='Google',
                                name='Google Satellite').add_to(m)

                # Set satellite as the default layer
                m.options['preferCanvas'] = True

                # Add fullscreen control
                Fullscreen().add_to(m)

                # Add temperature layer if available
                if 'temperature' in visualizations:
                    temp_layers = visualizations['temperature']
                    if 'coldest_month' in temp_layers:
                        coldest_layer = temp_layers['coldest_month']
                        folium.TileLayer(
                            tiles=coldest_layer['tiles'][0],
                            attr=coldest_layer['attribution'],
                            name='Temperature (Coldest Month)',
                            overlay=True
                        ).add_to(m)
                    if 'warmest_month' in temp_layers:
                        warmest_layer = temp_layers['warmest_month']
                        folium.TileLayer(
                            tiles=warmest_layer['tiles'][0],
                            attr=warmest_layer['attribution'],
                            name='Temperature (Warmest Month)',
                            overlay=True
                        ).add_to(m)

                # Add precipitation layer if available
                if 'precipitation' in visualizations:
                    precip_layers = visualizations['precipitation']
                    if 'driest_month' in precip_layers:
                        driest_layer = precip_layers['driest_month']
                        folium.TileLayer(
                            tiles=driest_layer['tiles'][0],
                            attr=driest_layer['attribution'],
                            name='Precipitation (Driest Month)',
                            overlay=True
                        ).add_to(m)
                    if 'wettest_month' in precip_layers:
                        wettest_layer = precip_layers['wettest_month']
                        folium.TileLayer(
                            tiles=wettest_layer['tiles'][0],
                            attr=wettest_layer['attribution'],
                            name='Precipitation (Wettest Month)',
                            overlay=True
                        ).add_to(m)

                # Add species points if available
                if 'species_points' in visualizations:
                    points_group = folium.FeatureGroup(name="Species Observations")

                    for point in visualizations['species_points']['features']:
                        coords = point['geometry']['coordinates']
                        props = point['properties']
                        popup_text = f"""
                        <div style="font-family: Arial, sans-serif;
                                    background-color: white;
                                    padding: 5px;
                                    border-radius: 3px;
                                    border: 1px solid #ccc;">
                            <b>Location:</b><br>
                            Lat: {coords[1]:.4f}<br>
                            Lon: {coords[0]:.4f}<br>
                            <b>Count:</b> {props.get('individual_count', 'N/A')}
                        </div>
                        """

                        folium.CircleMarker(
                            location=[coords[1], coords[0]],
                            radius=5,
                            color='white',  # Border color (white)
                            weight=2,       # Border width
                            fill=True,
                            fillColor='red',  # Fill color (red)
                            fillOpacity=0.7,
                            tooltip=folium.Tooltip(
                                popup_text,
                                permanent=False,
                                sticky=True
                            )
                        ).add_to(points_group)

                    # Add the feature group to the map
                    points_group.add_to(m)

                # Add layer control
                folium.LayerControl().add_to(m)

                # Display the map with CSS
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
                folium_static(m, width=3000)

            with right_col:
                # Display climate statistics
                if climate_trends:
                    self._display_climate_stats(climate_trends)

                # Add interpretation guide
                st.markdown("### Interpretation Guide")
                st.markdown("""
                **Temperature:**
                - Shows mean temperature distribution across the species range
                - Red areas indicate higher temperatures
                - Blue areas indicate lower temperatures

                **Precipitation:**
                - Shows mean precipitation distribution across the species range
                - Blue areas indicate higher precipitation
                - White areas indicate lower precipitation

                **Species Climate Space:**
                - Shows the relationship between temperature and precipitation at observation points
                - Circle size indicates number of observations
                - Helps identify the species' climate preferences
                """)

        except Exception as e:
            st.error(f"Error generating visualization: {str(e)}")

    def _create_temperature_chart(self, temp_data: Dict) -> go.Figure:
        """Creates temperature statistics visualization using Plotly."""
        # Create a DataFrame with the statistics
        df = pd.DataFrame([{
            'metric': 'Mean',
            'value': temp_data['mean'],
            'unit': '°C'
        }, {
            'metric': 'Coldest Month',
            'value': temp_data['min'],
            'unit': '°C'
        }, {
            'metric': 'Warmest Month',
            'value': temp_data['max'],
            'unit': '°C'
        }])

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df['metric'],
            y=df['value'],
            marker_color='red',
            text=df['value'].round(2),
            textposition='auto',
            hovertemplate="<b>%{x}</b><br>" +
                         "Value: %{y:.2f}°C<br>" +
                         "<extra></extra>"
        ))

        fig.update_layout(
            title='Temperature Statistics',
            xaxis_title='Metric',
            yaxis_title='Temperature (°C)',
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=False,
            modebar_remove=[
                'zoom', 'pan', 'select', 'zoomIn', 'zoomOut',
                'autoScale', 'resetScale', 'toImage'
            ]
        )

        return fig

    def _create_precipitation_chart(self, precip_data: Dict) -> go.Figure:
        """Creates precipitation statistics visualization using Plotly."""
        # Create a DataFrame with the statistics
        df = pd.DataFrame([{
            'metric': 'Mean',
            'value': precip_data['mean'],
            'unit': 'mm'
        }, {
            'metric': 'Driest Month',
            'value': precip_data['min'],
            'unit': 'mm'
        }, {
            'metric': 'Wettest Month',
            'value': precip_data['max'],
            'unit': 'mm'
        }])

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df['metric'],
            y=df['value'],
            marker_color='blue',
            text=df['value'].round(2),
            textposition='auto',
            hovertemplate="<b>%{x}</b><br>" +
                         "Value: %{y:.2f}mm<br>" +
                         "<extra></extra>"
        ))

        fig.update_layout(
            title='Precipitation Statistics',
            xaxis_title='Metric',
            yaxis_title='Precipitation (mm)',
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=False,
            modebar_remove=[
                'zoom', 'pan', 'select', 'zoomIn', 'zoomOut',
                'autoScale', 'resetScale', 'toImage'
            ]
        )

        return fig

    def _create_species_climate_chart(self, species_data: Dict) -> Optional[go.Figure]:
        """Creates species-climate relationship visualization using Plotly."""
        if not species_data.get("climate_conditions"):
            return None

        # Convert climate conditions to DataFrame and apply temperature scale factor
        df = pd.DataFrame(species_data["climate_conditions"])
        df['temperature'] = df['temperature'] * 0.1  # Apply WorldClim scale factor

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['temperature'],
            y=df['precipitation'],
            mode='markers',
            marker=dict(
                size=5,
                color='rgba(255, 0, 0, 0.6)',
                line=dict(
                    color='white',
                    width=1
                )
            ),
            text=df['count'],
            hovertemplate=(
                "<b>Annual Mean Temperature:</b> %{x:.2f}°C<br>"
                "<b>Annual Mean Precipitation:</b> %{y:.2f}mm<br>"
                "<b>Observations:</b> %{text}<br>"
                "<extra></extra>"
            )
        ))

        fig.update_layout(
            title='Species Climate Space (Annual Means)',
            xaxis_title='Annual Mean Temperature (°C)',
            yaxis_title='Annual Mean Precipitation (mm)',
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
            modebar_remove=[
                'zoom', 'pan', 'select', 'zoomIn', 'zoomOut',
                'autoScale', 'resetScale', 'toImage'
            ]
        )

        return fig

    def _display_climate_stats(self, climate_trends: Dict):
        """Display climate statistics in the sidebar."""
        st.markdown("### Climate Statistics")

        if "temperature" in climate_trends:
            temp_data = climate_trends["temperature"]
            st.markdown("#### Temperature")
            st.markdown(f"""
                - Mean: {temp_data['mean']:.2f}°C
                - Coldest Month: {temp_data['min']:.2f}°C
                - Warmest Month: {temp_data['max']:.2f}°C
            """)

        if "precipitation" in climate_trends:
            precip_data = climate_trends["precipitation"]
            st.markdown("#### Precipitation")
            st.markdown(f"""
                - Mean: {precip_data['mean']:.2f}mm
                - Driest Month: {precip_data['min']:.2f}mm
                - Wettest Month: {precip_data['max']:.2f}mm
            """)
