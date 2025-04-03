"""
Renderer for species-forest correlation visualizations.
"""
import time
from typing import Any, Dict, Optional
import folium
from folium.plugins import MarkerCluster, Fullscreen
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_folium import folium_static
import streamlit as st
from ..base import BaseChartRenderer
from ..chart_types import ChartType

# pylint: disable=broad-except
# pylint: disable=no-member
class ForestRenderer(BaseChartRenderer):
    """
    Renderer for species-forest correlation visualization.
    """
    @property
    def supported_chart_types(self) -> list[ChartType]:
        return [ChartType.FOREST_CORRELATION]

    def render(self, data: Any, parameters: Optional[Dict] = None,
               _cache_buster: Optional[str] = None) -> Any:
        """Draw a map showing species observations, alpha shapes, and forest layers using Folium.

        Args:
            data (dict): Dictionary containing:
                - correlation_data: Dictionary of correlation statistics
                - analysis: Analysis results
                - species_name: Name of the species
                - observations: List of species observations
                - forest_layers: Dictionary of Earth Engine layer URLs
                - alpha_shapes: List of alpha shape polygons (optional)
            parameters (dict): Visualization parameters
        """
        try:
            message_index = (_cache_buster if _cache_buster is not None
                           else int(time.time()))

            # Create DataFrame from observations
            df = pd.DataFrame(data['observations'])

            if df.empty:
                st.warning("No observation data to display")
                return

            # Create two columns for map and analysis
            col1, col2 = st.columns([3, 1])

            with col1:
                species_name = parameters.get('species_name', 'Species')
                st.markdown(f"### {species_name} - Forest Correlation Map")
                st.markdown(f"**Total Observations**: {len(df):,}")

                # Create a Folium map with all layers
                m = self._initialize_map(df, data.get('forest_layers', {}))

                # Check if we have processed results with forest cover data
                if 'all_results' in data and len(data['all_results']) > 0:
                    self._add_forest_cover_groups(m, data['all_results'])
                else:
                    # Use the original cluster approach if no forest metrics
                    self._add_observation_cluster(m, df)

                # Add layer control
                folium.LayerControl().add_to(m)

                # Check if we have the necessary data for histograms
                has_results = ('all_results' in data and
                             len(data.get('all_results', [])) > 0)
                has_corr = ('correlation_data' in data and
                          'forest_cover' in data['correlation_data'])

                if has_results and has_corr:
                    col11, col12 = st.columns(2)
                    with col11:
                        bins = (data['correlation_data']['forest_metrics_distribution']
                               ['forest_cover_bins'])
                        mean = data['correlation_data']['forest_cover']['mean']
                        self._create_forest_cover_histogram(
                            bins, mean, key=f"forest_dist_{message_index}"
                        )
                    with col12:
                        self._create_forest_distribution_piechart(
                            data, f"forest_dist_{message_index}"
                        )

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
            with col2:
                # Display correlation statistics
                if 'correlation_data' in data:
                    self.display_forest_correlation_stats(data['correlation_data'])

                    # Show color legend for forest cover categories
                    if 'all_results' in data and len(data['all_results']) > 0:
                        st.markdown("### Forest Cover Categories")
                        st.markdown("""
                        Points on the map are colored by forest cover:
                        - <span style="color:yellow">⬤</span> **Yellow**: Low cover (0-10%)
                        - <span style="color:orange">⬤</span> **Orange**: Medium cover (10-90%)
                        - <span style="color:purple">⬤</span> **Purple**: High cover (90-100%)
                        """, unsafe_allow_html=True)

                    # Add forest loss legend
                    st.markdown("### Forest Loss Years")
                    st.markdown("""
                        Forest loss is shown in different shades of red:
                        - <span style="color: #ff0000">⬤</span> **Dark Red**: 2020-2023
                        - <span style="color: #ff3333">⬤</span> **Red**: 2015-2019
                        - <span style="color: #ff6666">⬤</span> **Medium Red**: 2010-2014
                        - <span style="color: #ff9999">⬤</span> **Light Red**: 2005-2009
                        - <span style="color: #ffcccc">⬤</span> **Very Light Red**: 2001-2004
                        """, unsafe_allow_html=True)

        except Exception as e:
            self.logger.error("Error drawing forest correlation: %s", str(e), exc_info=True)
            st.error(f"Error generating visualization: {str(e)}")

    def _initialize_map(self, df, forest_layers):
        """Initialize Folium map with base layers and forest overlays."""
        # Create base map
        m = folium.Map(
            location=[df['decimallatitude'].mean(),
                     df['decimallongitude'].mean()],
            zoom_start=5,
            tiles=None
        )

        # Add base layers
        folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)

        satellite_url = ('https://server.arcgisonline.com/ArcGIS/rest/services/'
                        'World_Imagery/MapServer/tile/{z}/{y}/{x}')
        folium.TileLayer(
            satellite_url,
            attr='Esri',
            name='Satellite'
        ).add_to(m)

        gmaps_url = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}'
        folium.TileLayer(
            gmaps_url,
            attr='Google',
            name='Google Maps'
        ).add_to(m)

        gsat_url = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'
        folium.TileLayer(
            gsat_url,
            attr='Google',
            name='Google Satellite'
        ).add_to(m)

        # Set satellite as the default layer
        m.options['preferCanvas'] = True

        # Add fullscreen control
        Fullscreen().add_to(m)

        # Add Earth Engine forest layers if available
        for layer_name in ['forest_cover', 'forest_loss', 'forest_gain', 'alpha_shapes']:
            if layer_name in forest_layers:
                folium.TileLayer(
                    tiles=forest_layers[layer_name]['tiles'][0],
                    attr=forest_layers[layer_name]['attribution'],
                    name=layer_name.replace('_', ' ').title(),
                    overlay=True
                ).add_to(m)

        return m

    def _add_forest_cover_groups(self, m, results):
        """Add forest cover feature groups to the map."""
        # Create feature groups
        low_cover_group = folium.FeatureGroup(name="Low Forest Cover (0-10%)")
        mid_cover_group = folium.FeatureGroup(name="Medium Forest Cover (10-90%)")
        high_cover_group = folium.FeatureGroup(name="High Forest Cover (90-100%)")

        # Keep track of counts
        low_count = mid_count = high_count = 0

        # Process each point
        for point in results:
            if 'geometry' not in point or 'coordinates' not in point['geometry']:
                continue

            coords = point['geometry']['coordinates']
            if not coords or len(coords) < 2:
                continue

            lon, lat = coords[0], coords[1]
            forest_cover = point.get('remaining_cover', 0)

            popup_content = f"""
                <b>Location:</b> {lat:.4f}, {lon:.4f}<br>
                <b>Forest Cover:</b> {forest_cover:.1f}%<br>
                <b>Forest Loss:</b> {'Yes' if point.get('forest_loss', 0) > 0 else 'No'}<br>
                <b>Year:</b> {point.get('year', 'Unknown')}
            """

            marker = folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color='black',
                fill=True,
                fill_opacity=0.7,
                weight=1,
                popup=folium.Popup(popup_content, max_width=300)
            )

            # Add to appropriate group based on forest cover
            if forest_cover <= 10:
                marker.fill_color = 'yellow'
                marker.add_to(low_cover_group)
                low_count += 1
            elif forest_cover >= 90:
                marker.fill_color = 'purple'
                marker.add_to(high_cover_group)
                high_count += 1
            else:
                marker.fill_color = 'orange'
                marker.add_to(mid_cover_group)
                mid_count += 1

        # Update names with counts
        low_cover_group.layer_name = f"Low Forest Cover (0-10%): {low_count}"
        mid_cover_group.layer_name = f"Medium Forest Cover (10-90%): {mid_count}"
        high_cover_group.layer_name = f"High Forest Cover (90-100%): {high_count}"

        # Add all groups to map
        low_cover_group.add_to(m)
        mid_cover_group.add_to(m)
        high_cover_group.add_to(m)

    def _add_observation_cluster(self, m, df):
        """Add clustered observation markers to the map."""
        cluster = MarkerCluster(
            name="Observations",
            options={
                'disableClusteringAtZoom': 6,
                'maxClusterRadius': 80,
                'spiderfyOnMaxZoom': True
            }
        ).add_to(m)

        for _, row in df.iterrows():
            popup_text = f"""
            <b>Location:</b> {row['decimallatitude']:.5f}, {row['decimallongitude']:.5f}<br>
            <b>Year:</b> {row['observation_year']}<br>
            <b>Count:</b> {row['individual_count']}
            """

            folium.Marker(
                location=[row['decimallatitude'], row['decimallongitude']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(cluster)

    def _create_forest_cover_histogram(self, cover_bins, mean_cover, key):
        """Create and display a histogram for forest cover distribution."""
        # Convert bin labels to numeric values for proper ordering
        bin_numeric_values = []
        for bin_label in cover_bins.keys():
            values = bin_label.replace("%", "").split("-")
            bin_start = float(values[0])
            bin_end = float(values[1])
            bin_center = (bin_start + bin_end) / 2
            bin_numeric_values.append(bin_center)

        # Create plotly figure
        fig = go.Figure()

        # Add bar chart with numeric x-values for proper positioning
        fig.add_trace(go.Bar(
            x=bin_numeric_values,
            y=list(cover_bins.values()),
            marker_color='rgb(0, 128, 0)',
            marker_line_color='rgb(255, 255, 255)',
            marker_line_width=1,
            opacity=0.7,
            name='Forest Cover Distribution',
            text=list(cover_bins.keys()),
            hovertemplate="Bin: %{text}<br>Count: %{y}<extra></extra>"
        ))

        # Update layout with fixed x-axis range from 0-100%
        fig.update_layout(
            title='Forest Cover Distribution',
            xaxis_title='Remaining Forest Cover (%)',
            yaxis_title='Number of Individuals',
            bargap=0.1,
            height=300,
            margin={"l": 40, "r": 40, "t": 40, "b": 40},
            xaxis={
                "range": [0, 100],
                "dtick": 10,
                "title_standoff": 25
            }
        )

        # Add vertical line for mean
        fig.add_vline(
            x=mean_cover,
            line_dash="dash",
            line_color="red",
            line_width=2
        )

        # Add annotation for the mean
        fig.add_annotation(
            x=mean_cover,
            y=max(cover_bins.values()) * 0.95,
            text=f"Mean: {mean_cover:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red",
            arrowsize=1,
            arrowwidth=2,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="red",
            borderwidth=1,
            font=dict(color="red", size=12)
        )

        # Display the plot
        st.plotly_chart(fig, use_container_width=True, key=f"{key}_cover")

    def _create_forest_cover_histogram_raw(self, cover_values, mean_cover, key):
        """Create and display a histogram from raw forest cover values."""
        fig = go.Figure()

        # Add histogram
        fig.add_trace(go.Histogram(
            x=cover_values,
            nbinsx=10,  # Use 10 bins to match the binned version
            marker_color='rgb(0, 128, 0)',
            marker_line_color='rgb(255, 255, 255)',
            marker_line_width=1,
            opacity=0.7,
            name='Forest Cover Distribution'
        ))

        # Update layout
        fig.update_layout(
            title='Forest Cover Distribution',
            xaxis_title='Remaining Forest Cover (%)',
            yaxis_title='Frequency',
            bargap=0.1,
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis=dict(
                range=[0, 100],
                dtick=10
            )
        )

        # Add vertical line for mean
        fig.add_vline(
            x=mean_cover,
            line_dash="dash",
            line_color="red",
            line_width=2
        )

        # Add annotation for the mean
        fig.add_annotation(
            x=mean_cover,
            y=max([fig.data[0].y[i] for i in range(len(fig.data[0].y))]) * 0.95,
            text=f"Mean: {mean_cover:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red",
            arrowsize=1,
            arrowwidth=2,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="red",
            borderwidth=1,
            font=dict(color="red", size=12)
        )

        # Display the plot
        st.plotly_chart(fig, use_container_width=True, key=key)

    def display_forest_correlation_stats(self, correlation_data):
        """Display correlation statistics for forest metrics.

        Args:
            correlation_data (dict): Dictionary of correlation statistics
        """
         # pylint: disable=no-member
        # Display correlation statistics
        st.markdown("### Forest Correlation Analysis")

        # Forest Cover Statistics
        st.markdown("#### Forest Cover")
        st.markdown(f"""
            - Mean: {correlation_data['forest_cover']['mean']:.2f}%
            - Standard Deviation: {correlation_data['forest_cover']['std']:.2f}%
            - Correlation: {correlation_data['forest_cover']['correlation']:.3f}
            - P-value: {correlation_data['forest_cover']['p_value']:.3f}
        """)

        # Forest Loss Statistics
        st.markdown("#### Forest Loss")
        st.markdown(f"""
            - Mean: {correlation_data['forest_loss']['mean'] * 100:.2f}% (2001-2023)
            - Standard Deviation: {correlation_data['forest_loss']['std'] * 100:.2f}%
            - Correlation: {correlation_data['forest_loss']['correlation']:.3f}
            - P-value: {correlation_data['forest_loss']['p_value']:.3f}
        """)

        # Add clarification note about forest statistics
        st.markdown("""
        **Note about statistics:**
        - Forest cover is measured as percentage of tree canopy closure (0-100%) and includes forest gain where detected
        - Forest loss is calculated as percentage of the area that experienced loss (0-100%)
        - Forest gain (2000-2012) is incorporated into the final forest cover calculations
        - Forest metrics are sampled at exact observation points
        - Forest loss for observations takes into account the year of the observation with respect to the lossyear laye
        """)

        # Add interpretation hints
        st.markdown("""
        ### Forest Cover
        - **Green areas:** Forest cover (as of 2000) - Tree canopy closure for vegetation taller than 5m

        Toggle layers using the controls in the upper right corner.
        """)

    def _create_forest_cover_comparison(self, names, means, stds, correlations, message_index):
        """Create and display a comparison chart for forest cover across species."""
        # Sort by mean forest cover
        sorted_indices = np.argsort(means)[::-1]  # Descending order
        sorted_names = [names[i] for i in sorted_indices]
        sorted_means = [means[i] for i in sorted_indices]
        sorted_stds = [stds[i] for i in sorted_indices]
        sorted_corrs = [correlations[i] for i in sorted_indices]

        fig = go.Figure()

        # Add mean value bars
        fig.add_trace(go.Bar(
            x=sorted_names,
            y=sorted_means,
            error_y=dict(type='data', array=sorted_stds, visible=True),
            marker_color='rgb(0, 128, 0)',
            name='Mean Forest Cover (%)'
        ))

        # Add correlation line on secondary y-axis
        fig.add_trace(go.Scatter(
            x=sorted_names,
            y=sorted_corrs,
            mode='lines+markers',
            marker=dict(color='blue', size=8, symbol='circle'),
            line=dict(color='blue', width=2, dash='dot'),
            name='Correlation',
            yaxis='y2'
        ))

        # Update layout with dual y-axes
        fig.update_layout(
            title='Forest Cover by Species',
            xaxis_title='Species',
            yaxis_title='Forest Cover (%)',
            yaxis2={
                "title": 'Correlation',
                "titlefont": {"color": 'blue'},
                "tickfont": {"color": 'blue'},
                "overlaying": 'y',
                "side": 'right',
                "range": [-1, 1]
            },
            height=500,
            margin={"l": 50, "r": 50, "t": 50, "b": 100},
            showlegend=True,
            xaxis={"tickangle": 45}
        )

        # Add zero line for correlation
        fig.add_hline(
            y=0,
            line_dash="solid",
            line_color="gray",
            line_width=1,
            layer="below",
            yref="y2"
        )

        st.plotly_chart(fig, use_container_width=True, key=f"cover_comparison_{message_index}")

    def _create_forest_loss_comparison(self, names, means, stds, correlations, message_index):
        """Create and display a comparison chart for forest loss across species."""
        # Sort by mean forest loss
        sorted_indices = np.argsort(means)[::-1]  # Descending order
        sorted_names = [names[i] for i in sorted_indices]
        sorted_means = [means[i] for i in sorted_indices]
        sorted_stds = [stds[i] for i in sorted_indices]
        sorted_corrs = [correlations[i] for i in sorted_indices]

        fig = go.Figure()

        # Add mean value bars
        fig.add_trace(go.Bar(
            x=sorted_names,
            y=sorted_means,
            error_y=dict(type='data', array=sorted_stds, visible=True),
            marker_color='rgb(220, 20, 60)',
            name='Mean Forest Loss (%)'
        ))

        # Add correlation line on secondary y-axis
        fig.add_trace(go.Scatter(
            x=sorted_names,
            y=sorted_corrs,
            mode='lines+markers',
            marker=dict(color='blue', size=8, symbol='circle'),
            line=dict(color='blue', width=2, dash='dot'),
            name='Correlation',
            yaxis='y2'
        ))

        # Update layout with dual y-axes
        fig.update_layout(
            title='Forest Loss by Species',
            xaxis_title='Species',
            yaxis_title='Forest Loss (%)',
            yaxis2=dict(
                title='Correlation',
                titlefont=dict(color='blue'),
                tickfont=dict(color='blue'),
                overlaying='y',
                side='right',
                range=[-1, 1]
            ),
            height=500,
            margin=dict(l=50, r=50, t=50, b=100),
            showlegend=True,
            xaxis=dict(tickangle=45)
        )

        # Add zero line for correlation
        fig.add_hline(
            y=0,
            line_dash="solid",
            line_color="gray",
            line_width=1,
            layer="below",
            yref="y2"
        )

        st.plotly_chart(fig, use_container_width=True, key=f"loss_comparison_{message_index}")

    def draw_forest_comparison_by_species(self, data, parameters, _cache_buster=None):
        """Draw a comparison chart for multiple species and their forest metrics.

        Args:
            data (dict): Dictionary containing multiple species and their forest data
            parameters (dict): Visualization parameters
        """
        # pylint: disable=no-member
        try:
            message_index = _cache_buster if _cache_buster is not None else int(time.time())
            if not data or 'species_data' not in data:
                st.warning("No data available for visualization")
                return

            species_data = data['species_data']

            # Prepare data for plotting
            species_cover_means = []
            species_cover_stds = []
            species_loss_means = []
            species_loss_stds = []
            species_names = []
            species_cover_corrs = []
            species_loss_corrs = []

            for species_name, species_info in species_data.items():
                if 'correlation_data' in species_info:
                    corr_data = species_info['correlation_data']
                    if 'forest_cover' in corr_data and 'forest_loss' in corr_data:
                        species_names.append(species_name)
                        species_cover_means.append(corr_data['forest_cover']['mean'])
                        species_cover_stds.append(corr_data['forest_cover']['std'])
                        species_cover_corrs.append(corr_data['forest_cover']['correlation'])
                        species_loss_means.append(corr_data['forest_loss']['mean'] * 100)
                        species_loss_stds.append(corr_data['forest_loss']['std'] * 100)
                        species_loss_corrs.append(corr_data['forest_loss']['correlation'])

            if not species_names:
                st.warning("No valid data for comparison")
                return

            # Create tabs for different visualizations
            tab1, tab2 = st.tabs(["Forest Cover", "Forest Loss"])

            with tab1:
                # Sort and create comparison chart
                self._create_forest_cover_comparison(
                    species_names, species_cover_means, species_cover_stds,
                    species_cover_corrs, message_index
                )

            with tab2:
                # Sort and create loss comparison chart
                self._create_forest_loss_comparison(
                    species_names, species_loss_means, species_loss_stds,
                    species_loss_corrs, message_index
                )

            # Add interpretation guide
            st.markdown("### Interpretation Guide")
            st.markdown("""
            **Forest Cover:**
            - High values indicate species found in more forested landscapes
            - Positive correlation: Species prefers areas with more forest cover
            - Negative correlation: Species avoids densely forested areas

            **Forest Loss:**
            - High values indicate species found in areas with significant forest loss
            - Positive correlation: Species is more abundant in disturbed/deforested areas
            - Negative correlation: Species avoids areas with high forest loss

            **Conservation Implications:**
            - Species with high forest cover requirements and negative correlations with forest loss are most sensitive to deforestation
            - Species with positive correlations to forest loss may be more adaptable to certain types of disturbance
            """)

        except Exception as e:
            self.logger.error("Error drawing forest comparison: %s", str(e), exc_info=True)
            st.error("Error generating comparison visualization: %s", str(e))

    def _create_forest_distribution_piechart(self, data, key):
        """Create pie charts showing distribution of forest cover and loss."""
        # Create loss data counts
        results = data['all_results']
        loss_data = {
            'Recent (2020-2023)': len([
                r for r in results if r.get('lossyear', 0) >= 2020
            ]),
            '2015-2019': len([
                r for r in results if 2015 <= r.get('lossyear', 0) < 2020
            ]),
            '2010-2014': len([
                r for r in results if 2010 <= r.get('lossyear', 0) < 2015
            ]),
            '2005-2009': len([
                r for r in results if 2005 <= r.get('lossyear', 0) < 2010
            ]),
            '2001-2004': len([
                r for r in results if 2001 <= r.get('lossyear', 0) < 2005
            ]),
            'No Loss': len([
                r for r in results if r.get('lossyear', 0) == 0
            ])
        }

        # Create pie chart if there's any loss data
        if any(v > 0 for k, v in loss_data.items() if k != 'No Loss'):
            colors = ['#ff0000', '#ff3333', '#ff6666', '#ff9999', '#ffcccc', '#006400']
            fig2 = go.Figure(data=[go.Pie(
                labels=list(loss_data.keys()),
                values=list(loss_data.values()),
                hole=.3,
                marker_colors=colors  # Dark green for last color
            )])
            fig2.update_layout(
                title='Forest Loss Distribution',
                height=300,
                margin={"l": 20, "r": 20, "t": 40, "b": 20}
            )
            st.plotly_chart(fig2, use_container_width=True, key=f"{key}_loss_pie")
        else:
            st.write("No forest loss detected in the data")