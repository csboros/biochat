"""Renderer for habitat analysis visualization."""

from typing import Dict, Any, Optional
import folium
import plotly.graph_objects as go
from streamlit_folium import folium_static
from folium.plugins import Fullscreen
import streamlit as st
from ..base import BaseChartRenderer
from ..chart_types import ChartType
from ..config.land_cover_config import LandCoverConfig

 # pylint: disable=no-member
 # pylint: disable=broad-except
class HabitatViz(BaseChartRenderer):
    """Renders habitat analysis results on a map."""

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
        return [ChartType.HABITAT_ANALYSIS]

    def render(
        self,
        data: Dict[str, Any],
        parameters: Optional[Dict] = None,
        cache_buster: Optional[str] = None
    ) -> None:
        """
        Render habitat analysis results on a map and as charts.
        """
        try:
            # Create two columns for the main layout
            left_col, right_col = st.columns([3, 1])

            with left_col:
                # Create two columns for the charts
                chart_col1, chart_col2 = st.columns(2)

                # Get the data from the correct structure
                analysis_results = data.get('data', {})
                visualizations = data.get('visualizations', {})
                if not visualizations:
                    st.error("No visualization data available")
                    return

                # Create charts
                with chart_col1:
                    habitat_usage_chart = self._create_habitat_usage_chart(
                        analysis_results.get('habitat_usage', {})
                    )
                    st.plotly_chart(
                        habitat_usage_chart,
                        use_container_width=True,
                        config={'displayModeBar': False}
                    )

                with chart_col2:
                    # Only show forest dependency chart if forest analysis was performed
                    if analysis_results.get('forest_analysis') is not None:
                        forest_dependency_chart = self._create_forest_dependency_chart(
                            analysis_results.get('forest_analysis', {})
                        )
                        st.plotly_chart(
                            forest_dependency_chart,
                            use_container_width=True,
                            config={'displayModeBar': False}
                        )
                    else:
                        st.markdown("#### Forest Dependency Analysis")
                        st.info(
                            "Forest dependency analysis was not performed as forest "
                            "is not the primary habitat type."
                        )

                # Create second row of charts
                chart_col3, chart_col4 = st.columns(2)

                with chart_col3:
                    fragmentation_chart = self._create_fragmentation_chart(
                        analysis_results.get('habitat_fragmentation', {})
                    )
                    st.plotly_chart(
                        fragmentation_chart,
                        use_container_width=True,
                        config={'displayModeBar': False}
                    )

                with chart_col4:
                    connectivity_chart = self._create_connectivity_chart(
                        analysis_results.get('connectivity', {})
                    )
                    if connectivity_chart:
                        st.plotly_chart(
                            connectivity_chart,
                            use_container_width=True,
                            config={'displayModeBar': False}
                        )
                    else:
                        st.info("Connectivity analysis data is not available.")

                # Create and display map below charts
                map_viz = self._create_map(
                    visualizations.get('landcover'),
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
                # Display main results at the top
                st.markdown("### Key Results")

                # Habitat Usage Summary
                habitat_usage = analysis_results.get('habitat_usage', {})
                if habitat_usage:
                    st.markdown("#### Habitat Usage")
                    for habitat, percentage in habitat_usage.items():
                        st.markdown(f"- {habitat}: {percentage:.1f}%")

                # Forest Analysis Summary
                forest_analysis = analysis_results.get('forest_analysis', {})
                if forest_analysis:
                    st.markdown("#### Forest Analysis")
                    st.markdown(
                        f"- Forest Dependency: "
                        f"{forest_analysis.get('forest_dependency_ratio', 0):.1f}%"
                    )

                # Fragmentation Summary
                fragmentation = analysis_results.get('habitat_fragmentation', {})
                if fragmentation and 'patch_statistics' in fragmentation:
                    stats = fragmentation['patch_statistics']
                    st.markdown("#### Fragmentation Analysis")
                    st.markdown(f"- Total Patches: {stats.get('total_patches', 0)}")
                    st.markdown(f"- Mean Patch Size: {stats.get('mean_patch_size', 0):.1f} ha")
                    st.markdown(f"- Habitat Coverage: {stats.get('habitat_coverage', 0):.1f}%")

                # Connectivity Summary
                connectivity = analysis_results.get('connectivity', {})
                if connectivity:
                    st.markdown("#### Connectivity Analysis")
                    st.markdown(
                        f"- Connectivity Score: "
                        f"{connectivity.get('connectivity_score', 0):.2f}"
                    )

                # Add a divider
                st.markdown("---")

                # Display land cover types legend below the results
                st.markdown("### Land Cover Types")
                # Sort land cover types by their codes
                sorted_codes = sorted(LandCoverConfig.LAND_COVER_CLASSES.keys())
                for code in sorted_codes:
                    name = LandCoverConfig.LAND_COVER_CLASSES[code]
                    st.markdown(
                        f'<p style="margin: 5px 0;">'
                        f'<span style="color: {LandCoverConfig.get_color_for_code(code)}">■</span> '
                        f'{name}</p>',
                        unsafe_allow_html=True
                    )

        except Exception as e:
            st.error(f"Error rendering habitat visualization: {str(e)}")

    def _style_species_points(self, feature):
        """Style function for species points on the map."""
        # Get the land cover type from the feature properties and convert to int
        land_cover_type = feature['properties'].get('discrete_classification')
        try:
            land_cover_code = int(land_cover_type) if land_cover_type is not None else 0
            color = LandCoverConfig.get_color_for_code(land_cover_code)
        except (ValueError, TypeError):
            color = LandCoverConfig.COLOR_PALETTE[0]  # Default to unknown color if conversion fails

        return {
            'color': color,
            'fillColor': color,
            'fillOpacity': 0.8,
            'radius': 3,  # Smaller radius for dot-like appearance
            'weight': 0,  # No border
            'opacity': 0.8
        }

    def _create_map(
        self,
        landcover: Dict[str, Any],
        species_points: Dict[str, Any],
        center: Dict[str, float]
    ) -> folium.Map:
        """Create the map visualization."""
        try:

            m = folium.Map(
                location=[center['lat'], center['lon']],
                zoom_start=6
            )

            # Add land cover layer
            if landcover and landcover.get('tiles'):
                folium.raster_layers.TileLayer(
                    tiles=landcover['tiles'][0],
                    attr=landcover['attribution'],
                    name='Land Cover Types'
                ).add_to(m)
            else:
                st.warning("No landcover layer available")

            # Add species points
            if species_points and species_points.get('features'):
                # Create a feature group for all points
                points_group = folium.FeatureGroup(name="Species Observations")

                for feature in species_points['features']:
                    try:
                        coords = feature['geometry']['coordinates']
                        if not coords:
                            continue

                        # Get land cover type for coloring
                        properties = feature.get('properties', {})
                        land_cover_type = properties.get('discrete_classification')

                        # Get color based on land cover
                        try:
                            land_cover_code = int(float(land_cover_type)) if land_cover_type is not None else 0
                            color = LandCoverConfig.get_color_for_code(land_cover_code)
                            land_cover_name = LandCoverConfig.LAND_COVER_CLASSES.get(
                                land_cover_code,
                                f"Unknown (Code: {land_cover_code})"
                            )
                        except (ValueError, TypeError):
                            color = '#808080'  # Default gray for unknown
                            land_cover_name = "Unknown"

                        # Create tooltip
                        tooltip_content = f"""
                            <div style="font-family: Arial, sans-serif;
                                        background-color: white;
                                        padding: 5px;
                                        border-radius: 3px;
                                        border: 1px solid #ccc;">
                                <b>Location:</b><br>
                                Lat: {coords[1]:.4f}<br>
                                Lon: {coords[0]:.4f}<br>
                                <b>Vegetation:</b><br>
                                {land_cover_name}
                            </div>
                        """

                        # Add point to the feature group
                        folium.CircleMarker(
                            location=[coords[1], coords[0]],
                            radius=5,
                            color='white',  # Border color (white)
                            weight=2,       # Border width
                            fill=True,
                            fillColor=color,  # Fill color based on land cover type
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
            else:
                st.warning("No species points available")

            # Add layer control and legend
            folium.LayerControl().add_to(m)
            Fullscreen().add_to(m)
            m.get_root().html.add_child(folium.Element(self._create_legend()))

            return m

        except Exception as e:
            st.error(f"Error creating map: {str(e)}")
            return folium.Map(location=[center['lat'], center['lon']], zoom_start=6)

    def _create_habitat_usage_chart(self, habitat_usage: Dict[str, float]) -> go.Figure:
        """Create pie chart showing habitat usage distribution."""
        # Create reverse mapping of names to codes
        name_to_code = {name: code for code, name in LandCoverConfig.LAND_COVER_CLASSES.items()}

        # Prepare data for the pie chart
        names = list(habitat_usage.keys())
        values = list(habitat_usage.values())
        colors = []

        # Get colors for each segment
        for name in names:
            if name.startswith('Unknown ('):
                try:
                    code = int(name.split('(')[1].split(')')[0])
                    colors.append(LandCoverConfig.get_color_for_code(code))
                except (IndexError, ValueError):
                    colors.append(LandCoverConfig.COLOR_PALETTE[0])
            else:
                code = name_to_code.get(name)
                if code is not None:
                    colors.append(LandCoverConfig.get_color_for_code(code))
                else:
                    colors.append(LandCoverConfig.COLOR_PALETTE[0])

        # Create figure
        fig = go.Figure(data=[go.Pie(
            labels=names,
            values=values,
            hole=0.4,
            marker={'colors': colors}
        )])

        fig.update_layout(
            height=500,
            showlegend=False,
            title={
                'text': 'Habitat Usage Distribution',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )
        return fig

    def _create_forest_dependency_chart(self, forest_analysis: Dict[str, Any]) -> go.Figure:
        """Create bar chart showing forest dependency and type distribution."""
        # Create figure
        fig = go.Figure()

        # Add forest type distribution bars
        for forest_type, percentage in forest_analysis['forest_type_distribution'].items():
            # Get the code from LandCoverConfig
            code = None
            for code_val, name in LandCoverConfig.LAND_COVER_CLASSES.items():
                if name == forest_type:
                    code = code_val
                    break

            color = (LandCoverConfig.get_color_for_code(code)
                    if code else LandCoverConfig.COLOR_PALETTE[0])

            fig.add_trace(
                go.Bar(
                    name=forest_type,
                    x=['Forest Types'],
                    y=[percentage],
                    marker_color=color,
                    hovertemplate=(
                        "<b>%{data.name}</b><br>" +
                        "Percentage: %{y:.1f}%<br>" +
                        "<extra></extra>"
                    ),
                    text=f"{percentage:.1f}%",
                    textposition='auto',
                )
            )

        fig.update_layout(
            title={
                'text': 'Forest Dependency Analysis',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            barmode='group',
            height=500,
            showlegend=False,
            margin={'l': 20, 'r': 20, 't': 40, 'b': 20},
            hoverlabel={
                'bgcolor': "white",
                'font_size': 14,
                'font_family': "Arial"
            },
            yaxis_title="Percentage (%)",
            yaxis=dict(
                range=[0, max(forest_analysis['forest_type_distribution'].values()) * 1.1]
            )
        )
        return fig

    def _create_fragmentation_chart(self, fragmentation: Dict[str, Any]) -> go.Figure:
        """Create bar chart showing habitat fragmentation metrics."""
        fig = go.Figure()

        stats = fragmentation['patch_statistics']
        habitat_type = stats['habitat_type'].lower()

        # Add patch statistics with dynamic habitat type labels
        metrics = [
            ('Total Patches', stats['total_patches'], '#d9ef8b'),
            ('Mean Patch Size (ha)', stats['mean_patch_size'], '#91cf60'),
            (f'{habitat_type.title()} Coverage (%)', stats['habitat_coverage'], '#1a9850')
        ]

        for name, value, color in metrics:
            fig.add_trace(
                go.Bar(
                    name=name,
                    x=['Habitat Metrics'],
                    y=[value],
                    marker_color=color,
                    hovertemplate=(
                        "<b>%{data.name}</b><br>" +
                        "Value: %{y:.1f}<br>" +
                        "<extra></extra>"
                    )
                )
            )

        fig.update_layout(
            title=dict(
                text=f'{habitat_type.title()} Fragmentation Analysis',
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
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
            yaxis_title="Value"
        )
        return fig

    def _create_connectivity_chart(self, connectivity: Dict[str, Any]) -> go.Figure:
        """Create gauge chart showing habitat connectivity metrics."""
        if not connectivity:
            return None

        # Create figure with secondary y-axis
        fig = go.Figure()

        # Add connectivity score gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=connectivity['connectivity_score'] * 100,  # Convert to percentage
                title={'text': "Connectivity Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 33], 'color': "#d73027"},    # Red for low connectivity
                        {'range': [33, 66], 'color': "#f46d43"},   # Orange for medium connectivity
                        {'range': [66, 100], 'color': "#1a9850"}   # Green for high connectivity
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': connectivity['connectivity_score'] * 100
                    }
                }
            )
        )

        # Update layout
        fig.update_layout(
            title={
                'text': 'Habitat Connectivity Analysis',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            height=500,
            showlegend=False,
            margin={'l': 20, 'r': 20, 't': 40, 'b': 20},
            hoverlabel={
                'bgcolor': "white",
                'font_size': 14,
                'font_family': "Arial"
            },
            yaxis_title="Value",
        )

        return fig

    def _create_legend(self) -> str:
        """Create HTML legend for land cover types."""
        legend_html = '''
            <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
                <h4>Land Cover Types</h4>
        '''

        # Sort land cover types by their codes
        sorted_codes = sorted(LandCoverConfig.LAND_COVER_CLASSES.keys())
        for code in sorted_codes:
            name = LandCoverConfig.LAND_COVER_CLASSES[code]
            legend_html += (
                f'<p style="margin: 5px 0;">'
                f'<span style="color: {LandCoverConfig.get_color_for_code(code)}">■</span> '
                f'{name}</p>'
            )

        legend_html += '</div>'
        return legend_html
