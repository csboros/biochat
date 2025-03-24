"""Module for visualizing forest data and species correlations."""

import logging
import time
import folium
from folium.plugins import MarkerCluster, Fullscreen
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_folium import folium_static
import streamlit as st

class ForestViz:
    """Visualization handler for forest data and species correlations."""

    def __init__(self):
        """Initialize the ForestViz class."""
        self.logger = logging.getLogger("BioChat." + self.__class__.__name__)

    def draw_species_forest_correlation(self, data, parameters, _cache_buster=None):
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
            # pylint: disable=no-member
            message_index = _cache_buster if _cache_buster is not None else int(time.time())

            # Create DataFrame from observations
            df = pd.DataFrame(data['observations'])

            if df.empty:
                st.warning("No observation data to display")
                return

            # Create two columns for map and analysis
            col1, col2 = st.columns([3, 1])

            with col1:
                # First display the map using the full width
                st.markdown(
                    f"### {parameters.get('species_name', 'Species')} - "
                    f"Forest Correlation Map"
                )
                st.markdown(f"**Total Observations**: {len(df):,}")

                # Create a Folium map with multiple base layers
                m = folium.Map(
                    location=[df['decimallatitude'].mean(), df['decimallongitude'].mean()],
                    zoom_start=5,
                    tiles=None  # Start with no default tiles
                )

                # Add base layers
                folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
                folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                                attr='Esri',
                                name='Satellite').add_to(m)
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

                # Add Earth Engine forest layers if available
                forest_layers = data.get('forest_layers', {})

                # Forest Cover Layer
                if 'forest_cover' in forest_layers:
                    forest_cover_url = forest_layers['forest_cover']['tiles'][0]
                    folium.TileLayer(
                        tiles=forest_cover_url,
                        attr=forest_layers['forest_cover']['attribution'],
                        name='Forest Cover',
                        overlay=True
                    ).add_to(m)

                # Forest Loss Layer
                if 'forest_loss' in forest_layers:
                    forest_loss_url = forest_layers['forest_loss']['tiles'][0]
                    folium.TileLayer(
                        tiles=forest_loss_url,
                        attr=forest_layers['forest_loss']['attribution'],
                        name='Forest Loss',
                        overlay=True
                    ).add_to(m)

                # Forest Gain Layer
                if 'forest_gain' in forest_layers:
                    forest_gain_url = forest_layers['forest_gain']['tiles'][0]
                    folium.TileLayer(
                        tiles=forest_gain_url,
                        attr=forest_layers['forest_gain']['attribution'],
                        name='Forest Gain',
                        overlay=True
                    ).add_to(m)

                # Add alpha shapes if available
                if 'alpha_shapes' in forest_layers:
                    alpha_url = forest_layers['alpha_shapes']['tiles'][0]
                    folium.TileLayer(
                        tiles=alpha_url,
                        attr=forest_layers['alpha_shapes']['attribution'],
                        name='Species Range',
                        overlay=True
                    ).add_to(m)

                # Check if we have processed results with forest cover data
                if 'all_results' in data and len(data['all_results']) > 0:
                    # Create separate feature groups for different forest cover categories
                    low_cover_group = folium.FeatureGroup(name="Low Forest Cover (0-10%)")
                    mid_cover_group = folium.FeatureGroup(name="Medium Forest Cover (10-90%)")
                    high_cover_group = folium.FeatureGroup(name="High Forest Cover (90-100%)")

                    # Keep track of counts
                    low_count = 0
                    mid_count = 0
                    high_count = 0

                    # Process each point
                    for point in data['all_results']:
                        if 'geometry' not in point or 'coordinates' not in point['geometry']:
                            continue

                        coords = point['geometry']['coordinates']
                        if not coords or len(coords) < 2:
                            continue

                        lon, lat = coords[0], coords[1]
                        forest_cover = point.get('remaining_cover', 0)

                        # Create popup content
                        popup_content = f"""
                            <b>Location:</b> {lat:.4f}, {lon:.4f}<br>
                            <b>Forest Cover:</b> {forest_cover:.1f}%<br>
                            <b>Forest Loss:</b> {'Yes' if point.get('forest_loss', 0) > 0 else 'No'}<br>
                            <b>Year:</b> {point.get('year', 'Unknown')}
                        """

                        # Add to appropriate group based on forest cover
                        if forest_cover <= 10:
                            folium.CircleMarker(
                                location=[lat, lon],
                                radius=5,
                                color='magenta',
                                fill=True,
                                fill_color='magenta',
                                fill_opacity=0.7,
                                popup=folium.Popup(popup_content, max_width=300)
                            ).add_to(low_cover_group)
                            low_count += 1
                        elif forest_cover >= 90:
                            folium.CircleMarker(
                                location=[lat, lon],
                                radius=5,
                                color='darkgreen',
                                fill=True,
                                fill_color='darkgreen',
                                fill_opacity=0.7,
                                popup=folium.Popup(popup_content, max_width=300)
                            ).add_to(high_cover_group)
                            high_count += 1
                        else:
                            folium.CircleMarker(
                                location=[lat, lon],
                                radius=5,
                                color='blue',
                                fill=True,
                                fill_color='blue',
                                fill_opacity=0.7,
                                popup=folium.Popup(popup_content, max_width=300)
                            ).add_to(mid_cover_group)
                            mid_count += 1

                    # Update names with counts
                    low_cover_group.layer_name = f"Low Forest Cover (0-10%): {low_count}"
                    mid_cover_group.layer_name = f"Medium Forest Cover (10-90%): {mid_count}"
                    high_cover_group.layer_name = f"High Forest Cover (90-100%): {high_count}"

                    # Add all groups to map
                    low_cover_group.add_to(m)
                    mid_cover_group.add_to(m)
                    high_cover_group.add_to(m)
                else:
                    # Use the original cluster approach if no forest metrics
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

                # Add layer control
                folium.LayerControl().add_to(m)

                # Check if we have the necessary data for histograms
                if ('all_results' in data and len(data.get('all_results', [])) > 0 and
                    'correlation_data' in data and 'forest_cover' in data['correlation_data']):
                    self.plot_distribution_histograms(data, key=f"forest_dist_{message_index}")

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
                        - <span style="color:magenta">⬤</span> **Magenta**: Low cover (0-10%)
                        - <span style="color:blue">⬤</span> **Blue**: Medium cover (10-90%)
                        - <span style="color:darkgreen">⬤</span> **Green**: High cover (90-100%)
                        """, unsafe_allow_html=True)

        except Exception as e:
            self.logger.error("Error drawing forest correlation: %s", str(e), exc_info=True)
            st.error(f"Error generating visualization: {str(e)}")

    def plot_distribution_histograms(self, data, key=None):
        """Plot the distribution of species observations across forest metrics."""
        # First check if we have required data
        if ('correlation_data' not in data or
            'forest_cover' not in data.get('correlation_data', {}) or
            'all_results' not in data or
            len(data.get('all_results', [])) == 0):
            return

        # pylint: disable=no-member
        st.markdown("### Forest Metrics Distribution")

        # Create two columns for the histograms
        col1, col2 = st.columns(2)

        with col1:
            # Check if we have binned distribution data
            if ('forest_metrics_distribution' in data['correlation_data'] and
                'forest_cover_bins' in data['correlation_data']['forest_metrics_distribution']):

                # Use precomputed bins
                cover_bins = data['correlation_data']['forest_metrics_distribution']['forest_cover_bins']

                # Convert bin labels to numeric values for proper ordering
                # Extract the middle value of each bin for positioning
                bin_numeric_values = []
                for bin_label in cover_bins.keys():
                    # Extract the values from bin labels like "0-10%"
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
                    # Use the bin labels for hover text
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
                    margin=dict(l=40, r=40, t=40, b=40),
                    xaxis=dict(
                        range=[0, 100],  # Ensure the full range is displayed
                        dtick=10,        # Tick every 10%
                        title_standoff=25
                    )
                )

                # Add vertical line for mean with proper positioning
                mean_cover = data['correlation_data']['forest_cover']['mean']
                fig.add_vline(
                    x=mean_cover,
                    line_dash="dash",
                    line_color="red",
                    line_width=2
                )

                # Add annotation for the mean
                fig.add_annotation(
                    x=mean_cover,
                    y=max(cover_bins.values()) * 0.95,  # Position near the top
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

            # Fallback to old method if binned data not available
            elif 'all_results' in data and len(data['all_results']) > 0:
                cover_values = [r['remaining_cover'] for r in data['all_results']]

                # Create plotly figure
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
                        range=[0, 100],  # Ensure the full range is displayed
                        dtick=10         # Tick every 10%
                    )
                )

                # Add vertical line for mean
                mean_cover = data['correlation_data']['forest_cover']['mean']
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
                st.plotly_chart(fig, use_container_width=True, key=f"{key}_cover")

        with col2:
            # Check if we have binned distribution data
            if ('forest_metrics_distribution' in data['correlation_data'] and
                'forest_loss_bins' in data['correlation_data']['forest_metrics_distribution']):

                # Use precomputed bins
                loss_bins = data['correlation_data']['forest_metrics_distribution']['forest_loss_bins']

                # Create a pie chart for binary data
                fig = go.Figure()

                # Add pie chart
                fig.add_trace(go.Pie(
                    labels=list(loss_bins.keys()),
                    values=list(loss_bins.values()),
                    marker_colors=['rgb(0, 128, 0)', 'rgb(220, 20, 60)'],
                    hole=0.4,
                    textinfo='label+percent+value',
                    insidetextorientation='radial'
                ))

                # Update layout
                fig.update_layout(
                    title='Forest Loss Distribution',
                    height=300,
                    margin=dict(l=40, r=40, t=40, b=40),
                    annotations=[
                        dict(
                            text=f"Mean: {data['correlation_data']['forest_loss']['mean'] * 100:.1f}%",
                            x=0.5, y=0.5,
                            font_size=12,
                            showarrow=False
                        )
                    ]
                )

                # Display the plot
                st.plotly_chart(fig, use_container_width=True, key=f"{key}_loss")

            # Fallback to old method if binned data not available
            elif 'all_results' in data and len(data['all_results']) > 0:
                # Count observations with and without forest loss
                loss_values = [r['forest_loss'] for r in data['all_results']]
                loss_count = sum(1 for x in loss_values if x > 0)
                no_loss_count = len(loss_values) - loss_count

                # Create a pie chart for binary data
                fig = go.Figure()

                # Add pie chart
                fig.add_trace(go.Pie(
                    labels=['No Loss', 'Loss'],
                    values=[no_loss_count, loss_count],
                    marker_colors=['rgb(0, 128, 0)', 'rgb(220, 20, 60)'],
                    hole=0.4,
                    textinfo='label+percent',
                    insidetextorientation='radial'
                ))

                # Update layout
                fig.update_layout(
                    title='Forest Loss Distribution',
                    height=300,
                    margin=dict(l=40, r=40, t=40, b=40),
                    annotations=[
                        dict(
                            text=f"Mean: {data['correlation_data']['forest_loss']['mean'] * 100:.1f}%",
                            x=0.5, y=0.5,
                            font_size=12,
                            showarrow=False
                        )
                    ]
                )

                # Display the plot
                st.plotly_chart(fig, use_container_width=True, key=f"{key}_loss")

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
        - All values are averaged across all analysis regions
        """)

        # Add explanation of analysis methodology
        st.markdown("### Analysis Methodology")

        st.markdown("""
        **Point-based Analysis:**
        - Forest metrics are sampled at exact observation points
        - Surrounding habitat conditions are also considered
        - Each point represents one species observation
        """)

        # Add interpretation hints
        st.markdown("""
        ### Interpreting the Map
        - **Blue dots:** Species observations
        - **Green areas:** Forest cover (as of 2000) - Tree canopy closure for vegetation taller than 5m
        - **Red areas:** Forest loss (2001-2023) - Stand-replacement disturbance or change from forest to non-forest

        Toggle layers using the controls in the upper right corner.
        """)


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
                # Sort by mean forest cover
                sorted_indices = np.argsort(species_cover_means)[::-1]  # Descending order
                sorted_names = [species_names[i] for i in sorted_indices]
                sorted_cover_means = [species_cover_means[i] for i in sorted_indices]
                sorted_cover_stds = [species_cover_stds[i] for i in sorted_indices]
                sorted_cover_corrs = [species_cover_corrs[i] for i in sorted_indices]

                fig = go.Figure()

                # Add mean value bars
                fig.add_trace(go.Bar(
                    x=sorted_names,
                    y=sorted_cover_means,
                    error_y=dict(
                        type='data',
                        array=sorted_cover_stds,
                        visible=True
                    ),
                    marker_color='rgb(0, 128, 0)',
                    name='Mean Forest Cover (%)'
                ))

                # Add correlation line on secondary y-axis
                fig.add_trace(go.Scatter(
                    x=sorted_names,
                    y=sorted_cover_corrs,
                    mode='lines+markers',
                    marker=dict(
                        color='blue',
                        size=8,
                        symbol='circle'
                    ),
                    line=dict(
                        color='blue',
                        width=2,
                        dash='dot'
                    ),
                    name='Correlation',
                    yaxis='y2'
                ))

                # Update layout with dual y-axes
                fig.update_layout(
                    title='Forest Cover by Species',
                    xaxis_title='Species',
                    yaxis_title='Forest Cover (%)',
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
                    xaxis=dict(
                        tickangle=45
                    )
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

                st.plotly_chart(fig, use_container_width=True,
                                key=f"cover_comparison_{message_index}")

            with tab2:
                # Sort by mean forest loss
                sorted_indices = np.argsort(species_loss_means)[::-1]  # Descending order
                sorted_names = [species_names[i] for i in sorted_indices]
                sorted_loss_means = [species_loss_means[i] for i in sorted_indices]
                sorted_loss_stds = [species_loss_stds[i] for i in sorted_indices]
                sorted_loss_corrs = [species_loss_corrs[i] for i in sorted_indices]

                fig = go.Figure()

                # Add mean value bars
                fig.add_trace(go.Bar(
                    x=sorted_names,
                    y=sorted_loss_means,
                    error_y=dict(
                        type='data',
                        array=sorted_loss_stds,
                        visible=True
                    ),
                    marker_color='rgb(220, 20, 60)',
                    name='Mean Forest Loss (%)'
                ))

                # Add correlation line on secondary y-axis
                fig.add_trace(go.Scatter(
                    x=sorted_names,
                    y=sorted_loss_corrs,
                    mode='lines+markers',
                    marker=dict(
                        color='blue',
                        size=8,
                        symbol='circle'
                    ),
                    line=dict(
                        color='blue',
                        width=2,
                        dash='dot'
                    ),
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
                    xaxis=dict(
                        tickangle=45
                    )
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