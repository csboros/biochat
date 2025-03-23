"""Module for visualizing human modification data and species correlations."""

import logging
import time
import folium
import pandas as pd
import plotly.graph_objects as go
from folium.plugins import MarkerCluster, Fullscreen
from streamlit_folium import folium_static
import numpy as np
import streamlit as st

class HumanModificationViz:
    """Visualization handler for human modification data and species correlations."""

    def __init__(self):
        """Initialize the HumanModificationViz class."""
        self.logger = logging.getLogger("BioChat." + self.__class__.__name__)

    def draw_species_humanmod_correlation(self, data, parameters, _cache_buster=None):
        """Draw a map showing species observations and human modification using Folium.
        
        Args:
            data (dict): Dictionary containing:
                - correlation_data: Dictionary of correlation statistics
                - analysis: Analysis results
                - species_name: Name of the species
                - observations: List of species observations 
                - ghm_layers: Dictionary of Earth Engine layer URLs
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
                    f"Human Modification Correlation Map"
                )
                st.markdown(f"**Total Observations**: {len(df):,}")

                # Create a Folium map with full width
                m = folium.Map(
                    location=[df['decimallatitude'].mean(), df['decimallongitude'].mean()],
                    zoom_start=5,
                    tiles="CartoDB dark_matter"
                )
                Fullscreen().add_to(m)
                
                # Add Earth Engine human modification layer if available
                ghm_layers = data.get('ghm_layers', {})

                # Human Modification Layer
                if 'human_modification' in ghm_layers:
                    ghm_url = ghm_layers['human_modification']['tiles'][0]
                    folium.TileLayer(
                        tiles=ghm_url,
                        attr=ghm_layers['human_modification']['attribution'],
                        name='Human Modification',
                        overlay=True
                    ).add_to(m)

                # Add alpha shapes if available
                if 'alpha_shapes' in ghm_layers:
                    alpha_url = ghm_layers['alpha_shapes']['tiles'][0]
                    folium.TileLayer(
                        tiles=alpha_url,
                        attr=ghm_layers['alpha_shapes']['attribution'],
                        name='Species Range',
                        overlay=True
                    ).add_to(m)

                # Add observation points
                cluster = MarkerCluster(name="Observations").add_to(m)
                
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
                
                # Display the map
                folium_static(m, width=700, height=500)

                # Display distribution histogram if we have gHM values
                if 'all_results' in data and len(data['all_results']) > 0:
                    self.plot_distribution_histogram(data['all_results'], key=f"hm_dist_{message_index}")
            
            with col2:
                # Display correlation statistics
                if 'correlation_data' in data:
                    corr_data = data['correlation_data']['human_modification']
                    
                    st.markdown("### Human Modification Statistics")
                    st.markdown(f"**Mean value:** {corr_data['mean']:.2f} (0-1 scale)")
                    st.markdown(f"**Standard deviation:** {corr_data['std']:.2f}")
                    
                    # Display correlation value with color based on value
                    corr_value = corr_data['correlation']
                    p_value = corr_data['p_value']
                    
                    corr_color = "gray"
                    interpretation = ""
                    if p_value <= 0.05:  # Statistically significant
                        if corr_value > 0.3:
                            corr_color = "red"  # Strong positive (prefers human-modified areas)
                            interpretation = "Species appears to tolerate or prefer human-modified areas"
                        elif corr_value < -0.3:
                            corr_color = "green"  # Strong negative (avoids human-modified areas)
                            interpretation = "Species appears to avoid human-modified areas"
                        elif corr_value >= -0.3 and corr_value <= 0.3:
                            corr_color = "orange"  # Weak correlation
                            interpretation = "Species shows weak relationship with human modification"
                    else:
                        interpretation = "No significant relationship with human modification"
                    
                    st.markdown(
                        f"**Correlation:** <span style='color:{corr_color}'>{corr_value:.3f}</span> "
                        f"(p={p_value:.3f})",
                        unsafe_allow_html=True
                    )
                    
                    st.markdown(f"**Interpretation:** {interpretation}")
                    
                    # Add reference scale for human modification
                    self.display_ghm_reference_scale()
                
                # Display analysis from LLM
                if 'analysis' in data:
                    st.markdown("### Ecological Analysis")
                    st.markdown(data['analysis'])
                
        except Exception as e:
            self.logger.error("Error drawing human modification correlation: %s", str(e), exc_info=True)
            st.error(f"Error generating visualization: {str(e)}")

    def display_ghm_reference_scale(self):
        """Display a reference scale for human modification index values."""
        # pylint: disable=no-member
        st.markdown("### Human Modification Scale")
        
        # Create reference data for the color scale
        scale_data = {
            'Level': ['Very Low', 'Low', 'Moderate', 'High', 'Very High'],
            'Range': ['0.0-0.1', '0.1-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'],
            'Description': [
                'Pristine or nearly pristine areas', 
                'Limited human impact (e.g., sparse rural)',
                'Mixed natural and developed areas',
                'Predominantly modified (e.g., agriculture, suburban)',
                'Intensively modified (e.g., urban, industrial)'
            ]
        }
        
        # Create color blocks for each level
        colors = ['#071aff', '#4287f5', '#ffbd03', '#ff6b03', '#ff0000']
        html_blocks = ""
        
        for i, (level, range_val, desc) in enumerate(zip(
            scale_data['Level'], scale_data['Range'], scale_data['Description']
        )):
            html_blocks += f"""
            <div style="display: flex; margin-bottom: 8px; align-items: center;">
                <div style="background-color: {colors[i]}; width: 20px; height: 20px; margin-right: 10px;"></div>
                <div>
                    <strong>{level}</strong> ({range_val}): {desc}
                </div>
            </div>
            """
        
        st.markdown(html_blocks, unsafe_allow_html=True)
        
        st.markdown("""
        <small>Data source: Global Human Modification dataset (gHM), Conservation Science Partners.
        Citation: Kennedy et al. 2019, Global Change Biology.</small>
        """, unsafe_allow_html=True)

    def draw_human_modification_by_species(self, data, parameters, _cache_buster=None):
        """Draw a comparison chart for multiple species and their gHM values.
        
        Args:
            data (dict): Dictionary containing multiple species and their gHM data
            parameters (dict): Visualization parameters
        """         
        try:    
            # pylint: disable=no-member 
            message_index = _cache_buster if _cache_buster is not None else int(time.time())
            
            if not data or 'species_data' not in data:
                st.warning("No data available for visualization")
                return
            
            species_data = data['species_data']
            
            # Prepare data for plotting
            species_means = []
            species_stds = []
            species_names = []
            species_correlations = []
            
            for species_name, species_info in species_data.items():
                if 'correlation_data' in species_info and 'human_modification' in species_info['correlation_data']:
                    hm_data = species_info['correlation_data']['human_modification']
                    species_names.append(species_name)
                    species_means.append(hm_data['mean'])
                    species_stds.append(hm_data['std'])
                    species_correlations.append(hm_data['correlation'])
            
            if not species_means:
                st.warning("No valid data for comparison")
                return
            
            # Create two columns
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Create figure for mean values with error bars
                fig = go.Figure()
                
                # Sort by mean gHM value
                sorted_indices = np.argsort(species_means)
                sorted_names = [species_names[i] for i in sorted_indices]
                sorted_means = [species_means[i] for i in sorted_indices]
                sorted_stds = [species_stds[i] for i in sorted_indices]
                sorted_correlations = [species_correlations[i] for i in sorted_indices]
                
                # Add mean value bars
                fig.add_trace(go.Bar(
                    x=sorted_names,
                    y=sorted_means,
                    error_y=dict(
                        type='data',
                        array=sorted_stds,
                        visible=True
                    ),
                    marker_color='rgb(55, 83, 109)',
                    name='Mean Human Modification'
                ))
                
                # Add correlation line on secondary y-axis
                fig.add_trace(go.Scatter(
                    x=sorted_names,
                    y=sorted_correlations,
                    mode='lines+markers',
                    marker=dict(
                        color='red',
                        size=8,
                        symbol='circle'
                    ),
                    line=dict(
                        color='red',
                        width=2,
                        dash='dot'
                    ),
                    name='Correlation',
                    yaxis='y2'
                ))
                
                # Update layout with dual y-axes
                fig.update_layout(
                    title='Human Modification Index by Species',
                    xaxis_title='Species',
                    yaxis_title='gHM Value (0-1)',
                    yaxis2=dict(
                        title='Correlation',
                        titlefont=dict(color='red'),
                        tickfont=dict(color='red'),
                        overlaying='y',
                        side='right',
                        range=[-1, 1]
                    ),
                    height=500,
                    margin=dict(l=50, r=50, t=50, b=100),
                    hovermode='x unified',
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    ),
                    xaxis=dict(
                        tickangle=45
                    )
                )
                
                # Add horizontal reference lines
                fig.add_hline(
                    y=0.4, 
                    line_dash="dot", 
                    line_color="orange",
                    annotation_text="Moderate Human Modification",
                    annotation_position="top right"
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
                
                # Display the plot
                st.plotly_chart(fig, use_container_width=True, key=f"species_comparison_{message_index}")
            
            with col2:
                # Show color scale reference
                self.display_ghm_reference_scale()
                
                # Add interpretation guide
                st.markdown("### Interpretation Guide")
                st.markdown("""
                **Mean gHM Value**  
                Higher values indicate species found in more human-modified landscapes
                
                **Correlation**  
                - Positive: Species occurs more frequently in modified areas
                - Negative: Species avoids human-modified areas
                - Near zero: No clear relationship with human modification
                
                **Conservation Implications**  
                - Species with high gHM values may be more adaptable to human disturbance
                - Species with low gHM values and negative correlation may be more vulnerable to habitat modification
                """)
                
        except Exception as e:
            self.logger.error("Error drawing human modification comparison: %s", str(e), exc_info=True)
            st.error(f"Error generating comparison visualization: {str(e)}")
