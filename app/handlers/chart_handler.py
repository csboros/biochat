"""
Chart Handler Module for Biodiversity Application

This module provides visualization capabilities for biodiversity data using PyDeck.
It supports various chart types including heatmaps and hexagon maps, with automatic
view state adjustment based on data bounds.
"""

import json
import logging
import colorsys
import numpy as np
import pandas as pd
import pydeck as pdk
import folium
from streamlit_folium import folium_static
from folium.plugins import Fullscreen
import time


try:
    from scipy.spatial import Delaunay
except ImportError:
    Delaunay = None
from shapely.geometry import Polygon, MultiPoint
import streamlit as st

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None
try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from .d3js_visualization import display_force_visualization, display_tree, display_shared_habitat
from ..utils.alpha_shape_utils import AlphaShapeUtils


class ChartHandler:
    """
    Handles the creation and rendering of geographic visualizations.

    This class manages different types of map visualizations for biodiversity data,
    including heatmaps and hexagon maps. It automatically calculates appropriate
    view states based on data distribution and handles both DataFrame inputs.

    Attributes:
        default_view_state (pdk.ViewState): Default map view configuration used when
            bounds cannot be determined from data
    """

    def __init__(self):
        """
        Initializes the ChartHandler with default view state settings.

        The default view provides a global perspective centered slightly east of
        Greenwich (30° longitude) to show most landmasses.
        """
        self.logger = logging.getLogger("BioChat." + self.__class__.__name__)
        self.default_view_state = pdk.ViewState(
            latitude=0.0, longitude=30, zoom=2, pitch=30
        )
        self.alpha_shape_utils = AlphaShapeUtils()

    def draw_chart(self, df, chart_type, parameters, _cache_buster=None):
        """
        Main entry point for creating visualizations.
        Args:
            data (pd.DataFrame): DataFrame containing coordinate data
            chart_type (str): Type of visualization to create
            params (dict): Additional parameters for visualization
                - For distribution maps: Can include 'alpha', 'eps', and 'min_samples' 
                  to control alpha shape generation (defaults: 0.5, 1.0, 3)
                - For distribution maps: Can include 'avoid_overlaps' (bool) to control
                  whether overlapping clusters should be merged (default: True)
        Raises:
            ValueError: If chart_type is invalid or data format is incorrect
            TypeError: If arguments are of wrong type
            pd.errors.EmptyDataError: If DataFrame is empty
        """
        try:
            # pylint: disable=no-member
            if chart_type == "occurrence_map":
                with st.spinner("Rendering occurrence map..."):
                    self.draw_occurrence_map(df, parameters, _cache_buster)
                    return
            elif chart_type == "species_hci_correlation":
                with st.spinner("Rendering correlation plot..."):
                    self.draw_species_hci_correlation(df, parameters, _cache_buster)
                    return
            elif chart_type == "species_forest_correlation":
                with st.spinner("Rendering forest correlation plot..."):
                    # For forest correlation, df contains the complete data structure
                    self.draw_species_forest_correlation(df, parameters, _cache_buster)
                    return
            elif chart_type == "heatmap":
                self.draw_heatmap(parameters, df, _cache_buster)
                return
            elif chart_type.lower() == "geojson":
                with st.spinner(
                    "Rendering geojson map..."
                ):
                    self.draw_geojson_map(df)
                    return
            elif chart_type.lower() == "json":
                with st.spinner("Rendering json data..."):
                    self.draw_json_data(df)
                    return
            elif chart_type.lower() == "3d_scatterplot":
                with st.spinner(
                    "Rendering 3d scatterplot..."
                ):
                    self._draw_3d_scatterplot(df)
                    return
            elif chart_type.lower() == "yearly_observations":
                with st.spinner(
                    "Rendering yearly observations..."
                ):
                    self.draw_yearly_observations(df)
                    return
            elif chart_type.lower() == "correlation_scatter":
                with st.spinner(
                    "Rendering correlation scatterplot..."
                ):
                    self.draw_correlation_scatter(df, parameters, _cache_buster)
                    return
            elif chart_type.lower() == "images":
                with st.spinner(
                    "Rendering images..."
                ):
                    self.display_species_images(df, _cache_buster)
                    return
            elif chart_type.lower() == "tree":
                with st.spinner("Rendering tree visualization..."):
                    display_tree(df, _cache_buster=None)
                    return
            elif chart_type.lower() == "force_directed_graph":
                with st.spinner("Rendering circle packing visualization..."):
                    display_force_visualization(df, _cache_buster=None)
                    return
            elif chart_type.lower() == "species_shared_habitat":
                with st.spinner("Rendering shared habitat visualization..."):
                    display_shared_habitat(df, _cache_buster=None)
                    return
            if isinstance(df, pd.DataFrame):
                if df.empty:
                    raise pd.errors.EmptyDataError("Empty DataFrame provided")
                # Downsample data for large datasets
                if len(df) > 1000:
                    df = df.sample(n=1000, random_state=42)
            # hexagon is default chart type
            else:
                with st.spinner(  # pylint: disable=no-member
                    "Rendering distribution map..."
                ):  # pylint: disable=no-member
                    self.draw_hexagon_map(df, parameters, _cache_buster)
        except (TypeError, ValueError, pd.errors.EmptyDataError) as e:
            self.logger.error("Error creating visualization: %s", str(e), exc_info=True)
            raise

    def draw_heatmap(self, parameters, data, _cache_buster=None):
        """Creates a heatmap visualization using PyDeck's HeatmapLayer."""
        try:
            message_index = _cache_buster if _cache_buster is not None else int(time.time())
            df = pd.DataFrame(data["occurrences"])
            bounds = self._get_bounds_from_data(df)
            view_state = self.default_view_state
            if bounds:
                view_state = pdk.ViewState(
                    latitude=sum(coord[0] for coord in bounds) / 2,
                    longitude=sum(coord[1] for coord in bounds) / 2,
                    zoom=3,
                    pitch=30,
                )
            # pylint: disable=no-member
            col1, col2 = st.columns([3, 1])  # 3:1 ratio for map:legend
            with col1:
                st.pydeck_chart(
                    pdk.Deck(
                        initial_view_state=view_state,
                        layers=[
                            pdk.Layer(
                                "HeatmapLayer",
                                data=df,
                                get_position=["decimallongitude", "decimallatitude"],
                                pickable=False,
                                opacity=0.7,
                                get_weight=1,
                            ),
                            pdk.Layer(
                                "HexagonLayer",
                                data=df,
                                get_position=["decimallongitude", "decimallatitude"],
                                radius=10000,
                                elevation_scale=1,
                                pickable=True,
                                opacity=0,
                                auto_highlight=False,
                                highlight_color=[0, 0, 0, 0],
                                extruded=True,
                                get_fill_color=[0, 0, 0, 0],
                            ),
                        ],
                        tooltip={
                            "html": (
                                f"<b>{parameters.get('species_name', 'Unknown')}</b><br/>"
                                "Observations: {elevationValue}<br/>"
                                "Latitude: {decimallatitude}<br/>"
                                "Longitude: {decimallongitude}"
                            ),
                            "style": {"backgroundColor": "steelblue", "color": "white"},
                        },
                    ),
                    height=700,
                    key=f"heatmap_{message_index}"
                )
            # Add legend in the second column
            with col2:
                st.markdown("### Heatmap")
                st.markdown(f"**Species**: {parameters.get('species_name', 'Unknown')}")

                # Add explanation of visualization
                st.markdown(
                    """
                    This heatmap shows species distribution:
                    
                    - Areas with more observations appear hotter (red)
                    - Areas with fewer observations appear cooler (yellow)
                    - Intensity indicates density of observations
                    
                    ### Color Scale
                """
                )
                # Create a gradient legend for density
                st.markdown(
                    """
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
                        <div style="display: flex; flex-direction: column;
                            justify-content: space-between;">
                            <span>High</span>
                            <span>Low</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except (ValueError, TypeError) as e:
            self.logger.error("Error creating heatmap: %s", str(e), exc_info=True)
            raise

    def draw_hexagon_map(self, data, parameters, _cache_buster=None):
        """
        Creates a 3D hexagon bin visualization.
        Raises:
            ValueError: If coordinate data is invalid
            TypeError: If parameters are of wrong type
            st.StreamlitAPIException: If chart rendering fails
        """
        try:
            if not isinstance(parameters, dict):
                raise TypeError("Parameters must be a dictionary")
            logging.debug("Drawing hexagon map with parameters: %s", parameters)
            occurrences = data["occurrences"]
            message_index = _cache_buster if _cache_buster is not None else int(time.time())

            # Convert the JSON array to a DataFrame
            df = pd.DataFrame(occurrences)
            bounds = self._get_bounds_from_data(df)

            # Get alpha parameters from parameters or use defaults
            alpha = parameters.get('alpha', 0.5)
            eps = parameters.get('eps', 1.0)
            min_samples = parameters.get('min_samples', 3)
            avoid_overlaps = parameters.get('avoid_overlaps', True)

            # Calculate concave hull using the same parameters as ForestHandlerEE
            points = df[["decimallongitude", "decimallatitude"]].values
            hull_geojson = self.alpha_shape_utils.calculate_alpha_shape(
                points,
                alpha=alpha,
                eps=eps,
                min_samples=min_samples,
                avoid_overlaps=avoid_overlaps
            )

            # Force a new view state each time with a slight random variation to prevent caching
            if bounds is not None:
                view_state = pdk.ViewState(
                    latitude=sum(coord[0] for coord in bounds) / len(bounds),
                    longitude=sum(coord[1] for coord in bounds) / len(bounds),
                    zoom=3,
                    pitch=30,
                    bearing=0.001 * np.random.randn()  # Tiny random bearing to force redraw
                )
            else:
                view_state = self.default_view_state
            # pylint: disable=no-member
            col1, col2 = st.columns([3, 1])  # 3:1 ratio for map:legend
            with col1:
                st.pydeck_chart(
                    pdk.Deck(
                        initial_view_state=view_state,
                        layers=[
                            pdk.Layer(
                                "HexagonLayer",
                                data=df,
                                get_position=["decimallongitude", "decimallatitude"],
                                radius=10000,
                                elevation_scale=5000,
                                elevation_range=[0, 1000],
                                pickable=True,
                                extruded=True,
                            ),
                            pdk.Layer(
                                "GeoJsonLayer",
                                data=hull_geojson,
                                stroked=True,
                                filled=False,
                                line_width_min_pixels=2,
                                get_line_color=[255, 255, 0],
                                get_line_width=3,
                            ),
                        ],
                        tooltip={
                            "html": (
                                f"{parameters.get('species_name', '')}<br/>"
                                "Occurrences: {elevationValue}"
                            ),
                            "style": {"backgroundColor": "steelblue", "color": "white"},
                        },
                    ),
                    height=700,
                    use_container_width=True,  # Ensure proper sizing
                    key=f"hexagon_map_{message_index}"
                )

            # Add legend in the second column
            with col2:
                st.markdown("### Distribution Map")
                st.markdown(f"**Species**: {parameters.get('species_name', 'Unknown')}")

                # Add explanation of visualization
                st.markdown(
                    """
                    This map shows species distribution using:                    
                    **Hexagons**
                    - Each hexagon represents a geographic area
                    - Height indicates number of observations
                    - Darker color means more observations
                    - Hover over hexagons to see exact counts
                    
                    **Yellow Outline**
                    - Shows the species' range boundary
                    - Calculated using alpha shape algorithm
                    - Connects outermost observation points
                    
                    **Color Scale**
                """
                )

                # Create a gradient legend for density
                st.markdown(
                    """
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
                        <div style="display: flex; flex-direction: column;
                            justify-content: space-between;">
                            <span>High</span>
                            <span>Low</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        except Exception as e:
            self.logger.error("Error creating hexagon map: %s", str(e), exc_info=True)
            raise

    def _get_bounds_from_data(self, df, percentile_cutoff=2.5, min_points=10):
        """Calculates appropriate map bounds from coordinate data with outlier filtering."""
        try:
            if not self._has_valid_coordinates(df):
                return None
            filtered_df = self._apply_percentile_filtering(df, percentile_cutoff)
            filtered_df = self._apply_density_filtering(filtered_df, min_points)
            return self._calculate_bounds(filtered_df)
        except (ValueError, TypeError, pd.errors.EmptyDataError) as e:
            logging.error("Error getting bounds from data: %s", e)
            return None

    def _has_valid_coordinates(self, df):
        """Check if DataFrame has valid coordinate data."""
        if (
            df.empty
            or df["decimallatitude"].isna().all()
            or df["decimallongitude"].isna().all()
        ):
            logging.warning("No valid coordinates in dataset")
            return False
        return True

    def _apply_percentile_filtering(self, df, percentile_cutoff):
        """Apply percentile-based filtering to remove extreme outliers."""
        if not 0 <= percentile_cutoff <= 50:
            raise ValueError("Percentile cutoff must be between 0 and 50")
        lat_lower = np.percentile(df["decimallatitude"], percentile_cutoff)
        lat_upper = np.percentile(df["decimallatitude"], 100 - percentile_cutoff)
        lon_lower = np.percentile(df["decimallongitude"], percentile_cutoff)
        lon_upper = np.percentile(df["decimallongitude"], 100 - percentile_cutoff)
        return df[
            (df["decimallatitude"] >= lat_lower)
            & (df["decimallatitude"] <= lat_upper)
            & (df["decimallongitude"] >= lon_lower)
            & (df["decimallongitude"] <= lon_upper)
        ]

    def _apply_density_filtering(self, df, min_points):
        """Apply density-based filtering to focus on areas of interest."""
        try:
            lat_bins = pd.qcut(df["decimallatitude"], q=15, duplicates="drop")
            lon_bins = pd.qcut(df["decimallongitude"], q=15, duplicates="drop")
            grid_counts = df.groupby([lat_bins, lon_bins]).size()

            density_threshold = max(min_points, np.percentile(grid_counts, 25))
            dense_cells = grid_counts[grid_counts >= density_threshold].index

            if not dense_cells.empty:
                filtered_df = df[
                    df.apply(
                        lambda x: (
                            pd.qcut([x["decimallatitude"]], q=15, duplicates="drop")[0],
                            pd.qcut([x["decimallongitude"]], q=15, duplicates="drop")[
                                0
                            ],
                        )
                        in dense_cells,
                        axis=1,
                    )
                ]
                if (
                    len(filtered_df) < len(df) * 0.1
                ):  # If we've removed more than 90% of points
                    logging.warning(
                        "Density filtering too aggressive. "
                        "Using percentile-filtered dataset."
                    )
                    return df
                return filtered_df

            return df
        except ValueError as e:
            logging.warning(
                "Density-based filtering failed: %s. Using original dataset.", e
            )
            return df

    def _calculate_bounds(self, df):
        """Calculate the final bounds from the filtered dataset."""
        min_lat = df["decimallatitude"].min()
        max_lat = df["decimallatitude"].max()
        min_lon = df["decimallongitude"].min()
        max_lon = df["decimallongitude"].max()

        if (
            np.isnan(min_lat)
            or np.isnan(min_lon)
            or np.isnan(max_lat)
            or np.isnan(max_lon)
        ):
            logging.warning("Invalid bounds calculated")
            return None

        return [[min_lat, min_lon], [max_lat, max_lon]]

    def _get_iucn_color(self, category):
        """Return color for IUCN category."""
        # IUCN color scheme based on standard Protected Area colors
        iucn_colors = {
            "Ia": [0, 68, 27],  # Dark Green - Strict Nature Reserve
            "Ib": [0, 109, 44],  # Forest Green - Wilderness Area
            "II": [35, 139, 69],  # Green - National Park
            "III": [65, 171, 93],  # Light Green - Natural Monument
            "IV": [116, 196, 118],  # Pale Green - Habitat/Species Management
            "V": [161, 217, 155],  # Very Light Green - Protected Landscape
            "VI": [199, 233, 192],  # Lightest Green - Sustainable Use
            "Not Reported": [189, 189, 189],  # Gray
            "Not Applicable": [224, 224, 224],  # Light Gray
            "Not Assigned": [242, 242, 242],  # Very Light Gray
        }
        return iucn_colors.get(category, [150, 150, 150])  # Default gray for unknown

    def draw_geojson_map(self, data, _cache_buster=None):
        """
        Draws a geojson map.

        Args:
            data (str): JSON string containing array of GeoJSON features
            parameters (dict): Additional visualization parameters
        """
        try:
            message_index = _cache_buster if _cache_buster is not None else int(time.time())
            geojson_data = json.loads(data)
            bounds = self._get_bounds_from_geojson(geojson_data)
            if bounds is not None:
                view_state = pdk.ViewState(
                    latitude=sum(coord[0] for coord in bounds) / len(bounds),
                    longitude=sum(coord[1] for coord in bounds) / len(bounds),
                    zoom=5,
                    pitch=30,
                )
            else:
                view_state = self.default_view_state

            # Extract features with IUCN category
            features = [
                {
                    "type": "Feature",
                    "geometry": item["geojson"],
                    "properties": {
                        "name": item["name"],
                        "category": item["category"],
                        "color": self._get_iucn_color(item["category"]),
                    },
                }
                for item in geojson_data
            ]
            geojson_layer = {"type": "FeatureCollection", "features": features}
            # pylint: disable=no-member
            col1, col2 = st.columns([3, 1])  # 3:1 ratio for map:legend

            with col1:
                st.pydeck_chart(
                    pdk.Deck(
                        initial_view_state=view_state,
                        layers=[
                            pdk.Layer(
                                "GeoJsonLayer",
                                data=geojson_layer,
                                get_fill_color="properties.color",
                                stroked=True,
                                filled=True,
                                pickable=True,
                                line_width_min_pixels=1,
                                get_line_color=[0, 0, 0],
                                opacity=0.8,
                                get_tooltip=["properties.name", "properties.category"],
                            )
                        ],
                        tooltip={"text": "Name: {name}\nIUCN Category: {category}"},
                    ),  
                    height=700,
                    key=f"geojson_map_{message_index}"
                )

            # Add legend in the second column
            with col2:
                st.markdown("### IUCN Categories")
                st.markdown("Only areas with an IUCN category are shown.")
                categories = {
                    "Ia": "Strict Nature Reserve",
                    "Ib": "Wilderness Area",
                    "II": "National Park",
                    "III": "Natural Monument",
                    "IV": "Habitat/Species Management",
                    "V": "Protected Landscape",
                    "VI": "Sustainable Use",
                }
                for cat, desc in categories.items():
                    color = self._get_iucn_color(cat)
                    st.markdown(
                        f"""
                        <div style="display: flex; align-items: center; margin: 5px 0;">
                            <div style="width: 20px; height: 20px; background:
                              rgb({color[0]},{color[1]},{color[2]});
                                      margin-right: 5px; border: 1px solid black;"></div>
                            <span><b>{cat}</b> - {desc}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        except Exception as e:
            self.logger.error("Error creating geojson map: %s", str(e), exc_info=True)
            raise

    def _get_bounds_from_geojson(self, geojson_data):
        """Calculate bounds from GeoJSON features.
        Args:
            geojson_data (list): List of dictionaries containing GeoJSON features
        Returns:
            list: [[min_lat, min_lon], [max_lat, max_lon]] or None if invalid
        """
        try:
            all_coords = []
            for feature in geojson_data:
                geometry = feature["geojson"]
                if geometry["type"] == "Polygon":
                    coords = geometry["coordinates"][0]  # First ring of polygon
                    all_coords.extend(coords)
                elif geometry["type"] == "MultiPolygon":
                    for polygon in geometry["coordinates"]:
                        all_coords.extend(polygon[0])  # First ring of each polygon
            if not all_coords:
                return None
            # Convert to lat/lon pairs and find min/max
            lons, lats = zip(*all_coords)
            return [[min(lats), min(lons)], [max(lats), max(lons)]]
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error("Error calculating GeoJSON bounds: %s", str(e))
            return None

    def draw_json_data(self, data: str, _cache_buster=None) -> None:
        """Draw JSON data as a table.
        Args:
            data (str): JSON string containing array of dictionaries
            parameters (dict, optional): Additional visualization parameters
        """
        try:
            message_index = _cache_buster if _cache_buster is not None else int(time.time())
            df = pd.DataFrame(data)
            # pylint: disable=no-member
            st.dataframe(df, key=f"json_data_{message_index}")
        except (json.JSONDecodeError, pd.errors.EmptyDataError) as e:
            self.logger.error("Error parsing JSON data: %s", str(e))
            raise
        except Exception as e:
            self.logger.error("Error creating table: %s", str(e))
            raise

    def _draw_3d_scatterplot(self, data, _cache_buster=None):
        """Draws a 3D scatter visualization using PyDeck's ColumnLayer."""
        try:
            message_index = _cache_buster if _cache_buster is not None else int(time.time())
            property_name = data.get("property_name", "")
            countries_data = data.get("countries", {})
            if not countries_data:
                raise ValueError("No country data provided")
            # pylint: disable=no-member
            col1, col2 = st.columns([3, 1])  # 3:1 ratio for map:legend
            all_points, country_stats = self._process_country_data(
                countries_data, property_name
            )
            if not all_points:
                raise ValueError("No valid data points found")
            global_min, global_max = self._get_global_bounds(all_points, property_name)
            for point in all_points:
                point["formatted_long"] = f"{point['decimallongitude']:.2f}"
                point["formatted_lat"] = f"{point['decimallatitude']:.2f}"
                point["formatted_value"] = f"{point[property_name]:.2f}"
            layer = self._create_column_layer(
                all_points, property_name, global_min, global_max
            )
            view_state = self._calculate_view_state(all_points)

            viz_config = {
                "property_name": property_name,
                "country_stats": country_stats,
                "global_min": global_min,
                "global_max": global_max,
            }
            self._render_visualization((col1, col2), layer, view_state, viz_config, key=f"3d_scatterplot_{message_index}")

        except Exception as e:
            self.logger.error(
                "Error creating 3D visualization: %s", str(e), exc_info=True
            )
            raise

    def _process_country_data(self, countries_data, property_name):
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

    def _get_global_bounds(self, points, property_name):
        """Calculate global min/max values."""
        values = [point[property_name] for point in points]
        return min(values), max(values)

    def _create_column_layer(self, points, property_name, global_min, global_max):
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

    def _calculate_view_state(self, points):
        """Calculate view state from points."""
        lats = [p["decimallatitude"] for p in points]
        lons = [p["decimallongitude"] for p in points]
        return pdk.ViewState(
            latitude=sum(lats) / len(lats),
            longitude=sum(lons) / len(lons),
            zoom=4,
            pitch=45,
        )

    def _render_visualization(self, columns, layer, view_state, viz_config, key=None):
        """Render the visualization."""
        col1, col2 = columns
        with col1:
            # pylint: disable=no-member
            st.pydeck_chart(
                pdk.Deck(
                    initial_view_state=view_state,
                    layers=[layer],
                    tooltip={
                        "html": (
                            "<b>Value:</b> {formatted_value}<br/>"
                            "<b>Location:</b> {formatted_long}, {formatted_lat}"
                        ),
                        "style": {"backgroundColor": "steelblue", "color": "white"},
                    },
                ),
                height=700,
                key=key
            )

        with col2:
            # pylint: disable=no-member
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

    def _get_distinct_colors(self, n: int) -> list:
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

    def draw_yearly_observations(self, data, _cache_buster=None):
        """Draws a visualization of yearly observation data with distinct colors per country."""
        try:
            message_index = _cache_buster if _cache_buster is not None else int(time.time())
            if "error" in data:
                raise ValueError(data["error"])

            names = {
                "common": data.get("common_name", "Unknown"),
                "scientific": data.get("scientific_name", "Unknown"),
            }
            yearly_data = data.get("yearly_data", {})

            # pylint: disable=no-member
            col1, col2 = st.columns([3, 1])

            with col1:
                self._draw_observation_chart(yearly_data, names["common"], key=f"yearly_observations_{message_index}")
            with col2:
                self._display_observation_summary(yearly_data, names, key=f"yearly_observations_summary_{message_index}")

        except Exception as e:
            self.logger.error(
                "Error drawing yearly observations: %s", str(e), exc_info=True
            )
            raise

    def _draw_observation_chart(self, yearly_data, title, key=None):
        """Creates and displays the observation chart."""
        print(yearly_data)
        if isinstance(yearly_data, dict):
            self._draw_country_chart(yearly_data, title, key=f"yearly_observations_country_chart_{key}")
        else:
            self._draw_global_chart(yearly_data, key=f"yearly_observations_global_chart_{key}")

    def _display_observation_summary(self, yearly_data, names, key=None):
        """Displays the observation summary sidebar."""
        # pylint: disable=no-member
        st.markdown("### Species Information")
        st.markdown(f"**Common Name**: {names['common']}")
        st.markdown(f"**Scientific Name**: {names['scientific']}")
        st.markdown("### Observation Summary")

        if isinstance(yearly_data, dict):
            self._display_country_summary(yearly_data, key=f"yearly_observations_country_summary_{key}")
        else:
            self._display_global_summary(pd.DataFrame(yearly_data), key=f"yearly_observations_global_summary_{key}")

    def _draw_country_chart(self, yearly_data, title, key=None):
        """Creates and displays the country-specific observation chart."""
        try:
            # Create DataFrame for plotting
            colors = self._get_distinct_colors(len(yearly_data))

            fig = go.Figure()

            for (country, data), color in zip(yearly_data.items(), colors):
                df = pd.DataFrame(data)
                fig.add_trace(go.Scatter(
                    x=df['year'],
                    y=df['count'],
                    name=country,
                    line=dict(color=color),
                    mode='lines+markers'
                ))

            fig.update_layout(
                title=f"Yearly Observations: {title}",
                xaxis_title="Year",
                yaxis_title="Number of Observations",
                height=700,
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            # pylint: disable=no-member
            st.plotly_chart(fig, use_container_width=True, key=key)

        except Exception as e:
            self.logger.error("Error creating country chart: %s", str(e))
            raise

    def _draw_global_chart(self, yearly_data, key=None):
        """Creates and displays the global observation chart."""
        try:
            df = pd.DataFrame(yearly_data)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df['year'],
                y=df['count'],
                mode='lines+markers',
                name='Global Observations',
                line=dict(color='#1f77b4')
            ))

            fig.update_layout(
                title="Global Yearly Observations",
                xaxis_title="Year",
                yaxis_title="Number of Observations",
                height=700,
                hovermode='x',
                showlegend=False
            )
            # pylint: disable=no-member
            st.plotly_chart(fig, use_container_width=True, key=key)

        except Exception as e:
            self.logger.error("Error creating global chart: %s", str(e))
            raise

    def _display_country_summary(self, yearly_data, key=None):
        """Displays the country-specific observation summary."""
        total_observations = 0
        # pylint: disable=no-member
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
        """Displays the global observation summary."""
        total_observations = df["count"].sum()
        # pylint: disable=no-member
        st.markdown(f"**Total Observations**: {total_observations:,}", key=key)
        st.markdown(f"**Year Range**: {df['year'].min()} - {df['year'].max()}", key=key)

    def draw_correlation_scatter(self, data, parameters, _cache_buster=None):
        """Draw a scatter plot showing correlation between HCI and species counts."""
        try:
            message_index = _cache_buster if _cache_buster is not None else int(time.time())
            # pylint: disable=no-member
            col1, col2 = st.columns([3, 1])  # 3:1 ratio for plot:legend
            with col1:
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

                st.plotly_chart(fig, use_container_width=True)

            with col2:
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
        except Exception as e:
            self.logger.error(
                "Error creating correlation scatter plot: %s", str(e), exc_info=True
            )
            raise

    def display_species_images(self, images_data, _cache_buster=None):
        """Display species images in the Streamlit interface."""
        message_index = _cache_buster if _cache_buster is not None else int(time.time())
        # pylint: disable=no-member
        if images_data["image_count"] > 0:
            st.subheader(f"Images of {images_data['species']}", key=f"species_images_{message_index}")
            cols = st.columns(min(images_data["image_count"], 3))  # Up to 3 columns

            for idx, img in enumerate(images_data["images"]):
                with cols[idx % 3]:
                    try:
                        st.image(
                            img["url"],
                            use_column_width=True,
                            caption=f"Source: {img['publisher']}\nBy: {img['creator']}\n"
                            f"License: {img['license']}"
                        )
                    except Exception:  # pylint: disable=broad-except
                        st.warning("Could not load image")
        else:
            st.info(f"No images found for {images_data['species']}")

    def  draw_occurrence_map(self, data, parameters, _cache_buster=None):
        """
        Creates an interactive map of endangered species occurrences colored by conservation status.
        
        Args:
            data (dict): Dictionary containing:
                - occurrences: List of species occurrences with location and status
                - country_code: Two-letter country code
            parameters (dict): Visualization parameters
        """
        try:
            message_index = _cache_buster if _cache_buster is not None else int(time.time())
            # Create DataFrame from occurrences
            # pylint: disable=no-member
            df = pd.DataFrame(data['occurrences'])

            if df.empty:
                st.markdown("No occurrence data to display")
                return

            # Define color scheme for conservation status
            color_scheme = {
                'Extinct': [139, 0, 0],                # Dark Red
                'Critically Endangered': [214, 39, 40], # Red
                'Endangered': [255, 127, 14],          # Orange
                'Vulnerable': [255, 215, 0],           # Gold
                'Near Threatened': [44, 160, 44],      # Green
                'Least Concern': [31, 119, 180],       # Blue
                'Data Deficient': [127, 127, 127]      # Gray
            }

            # Add color column to DataFrame with RGB arrays
            df['color'] = df['conservation_status'].map(color_scheme)

            # Create two columns for map and legend
            col1, col2 = st.columns([3, 1])

            with col1:
                # Create scatter plot layer
                scatter_layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=df,
                    get_position=['decimallongitude', 'decimallatitude'],
                    get_color='color',  # Now points to RGB array
                    get_radius=10,
                    radius_min_pixels=2,
                    radius_max_pixels=100,
                    opacity=0.8,
                    stroked=True,
                    filled=True,
                    pickable=True,
                    auto_highlight=True
                )

                # Set initial view state
                view_state = pdk.ViewState(
                    latitude=df['decimallatitude'].mean(),
                    longitude=df['decimallongitude'].mean(),
                    zoom=7,
                    pitch=45,
                    bearing=0
                )

                # Define tooltip content
                tooltip_content = """
                    <div style="background-color: rgba(0, 0, 0, 0.8); color: white; padding: 10px; border-radius: 5px;">
                        <b>Species:</b> {species}<br/>
                        <b>Conservation Status:</b> {conservation_status}
                    </div>
                """

                # Check if column exists and has non-null value
                if 'species_name_en' in df.columns and not df['species_name_en'].isna().all():
                    tooltip_content = """
                        <div style="background-color: rgba(0, 0, 0, 0.8); color: white; padding: 10px; border-radius: 5px;">
                            <b>Species:</b> {species}<br/>
                            <b>Common name:</b> {species_name_en}<br/>
                            <b>Conservation Status:</b> {conservation_status}
                        </div>
                    """

                # Enhanced tooltip
                tooltip = {
                    "html": tooltip_content,
                    "style": {
                        "backgroundColor": "transparent",
                        "color": "white"
                    }
                }

                # Create and display the map
                deck = pdk.Deck(
                    layers=[scatter_layer],
                    initial_view_state=view_state,
                    tooltip=tooltip,
                    map_style='mapbox://styles/mapbox/dark-v10',
                )

                st.pydeck_chart(deck, height=700, key=f"occurrence_map_{message_index}")

            # Add legend in the second column
            with col2:
                st.markdown("### Conservation Status")
                st.markdown(f"**Country**: {data['country_code']}")
                st.markdown(f"**Total Occurrences**: {data['total_occurrences']:,}")

                # Create legend with colored boxes
                for status, color in color_scheme.items():
                    status_count = len(df[df['conservation_status'] == status])
                    if status_count > 0:  # Only show statuses that are present in the data
                        st.markdown(
                            f"""
                            <div style="display: flex; align-items: center; margin: 5px 0;">
                                <div style="width: 20px; height: 20px; 
                                          background: rgb({color[0]},{color[1]},{color[2]});
                                          margin-right: 5px; border: 1px solid black;">
                                </div>
                                <span><b>{status}</b> ({status_count:,} occurrences)</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

        except Exception as e:
            self.logger.error("Error creating occurrence map: %s", str(e), exc_info=True)
            raise

    def draw_species_hci_correlation(self, correlation_data, parameters, _cache_buster=None):
        """
        Draw a scatter plot showing species-HCI correlation.
        
        Args:
            correlation_data (dict): Dictionary containing correlation results
            parameters (dict): Parameters for visualization
        """
        try:
            message_index = _cache_buster if _cache_buster is not None else int(time.time())
            # pylint: disable=no-member
            correlations = correlation_data["correlations"]
            if not correlations:
                st.warning("No correlation data found.")
                return

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

        except Exception as e:
            self.logger.error("Error creating species-HCI correlation plot: %s",
                            str(e), exc_info=True)
            raise

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
            # Create DataFrame from observations
            df = pd.DataFrame(data['observations'])

            if df.empty:
                st.warning("No observation data to display")
                return

            # Create two columns for map and legend
            col1, col2 = st.columns([3, 1])

            with col1:
                # First display the map using the full width
                st.markdown(f"### {data.get('species_name', 'Species')} - Forest Correlation Map")
                st.markdown(f"**Total Observations**: {len(df):,}")

                # Create a Folium map with full width
                m = folium.Map(
                    location=[df['decimallatitude'].mean(), df['decimallongitude'].mean()],
                    zoom_start=5,
                    tiles="CartoDB dark_matter"
                )
                Fullscreen().add_to(m)
                # Add Earth Engine forest layers if available
                forest_layers = data.get('forest_layers', {})

                # Forest Cover Layer
                if 'forest_cover' in forest_layers:
                    forest_cover_url = forest_layers['forest_cover']['tiles'][0]
                    folium.TileLayer(
                        tiles=forest_cover_url,
                        attr=forest_layers['forest_cover']['attribution'],
                        name='Forest Cover 2000',
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
                        overlay=True,
                        show=False  # Hidden by default
                    ).add_to(m)

                # Add Alpha Shapes Layer if available
                if 'alpha_shapes' in forest_layers:
                    alpha_shapes_url = forest_layers['alpha_shapes']['tiles'][0]
                    folium.TileLayer(
                        tiles=alpha_shapes_url,
                        attr=forest_layers['alpha_shapes']['attribution'],
                        name='Analysis Areas',
                        overlay=True,
                        show=True  # Show by default
                    ).add_to(m)
                # Create a feature group for species observations
                observations = folium.FeatureGroup(name="Species Observations")
                # Add species observations
                for _, row in df.iterrows():
                    folium.CircleMarker(
                        location=[row['decimallatitude'], row['decimallongitude']],
                        radius=4,  # Slightly larger for better visibility
                        color='4285F4',
                        fill=True,
                        fill_color='4285F4',
                        fill_opacity=0.9,  # More opaque red dots
                        popup=f"Year: {row['observation_year']}<br>Count: {row['individual_count']}"
                    ).add_to(observations)
                # Add the observations layer to the map
                observations.add_to(m)

                # Add alpha shapes as GeoJSON if available (for more interactivity)
                if 'alpha_shapes' in data and data['alpha_shapes']:
                    # Convert alpha shapes to GeoJSON format
                    alpha_shapes_geojson = {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": shape['coordinates']
                                },
                                "properties": shape.get('properties', {})
                            }
                            for shape in data['alpha_shapes']
                        ]
                    }
                    # Add alpha shapes as an interactive layer
                    folium.GeoJson(
                        alpha_shapes_geojson,
                        name="Analysis Areas (Interactive)",
                        style_function=lambda x: {
                            'fillColor': '#4285F4',
                            'color': '#4285F4',
                            'weight': 2,
                            'fillOpacity': 0.2
                        },
                        tooltip=folium.GeoJsonTooltip(
                            fields=['year', 'num_observations', 'total_individuals'],
                            aliases=['Year:', 'Observations:', 'Individuals:'],
                            style=("background-color: white; color: #333333; font-family: arial; "
                                  "font-size: 12px; padding: 10px;")
                        )
                    ).add_to(m)
                # Add layer control
                folium.LayerControl().add_to(m)
                # Display the map at full width with stronger CSS
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
                folium_static(m, width=3000 )  # Force wide display

            with col2:
                # Display correlation statistics
                corr_data = data['correlation_data']
                st.markdown("### Forest Correlation Analysis")

                # Forest Cover Statistics
                st.markdown("#### Forest Cover")
                st.markdown(f"""
                    - Mean: {corr_data['forest_cover']['mean']:.2f}%
                    - Standard Deviation: {corr_data['forest_cover']['std']:.2f}%
                    - Correlation: {corr_data['forest_cover']['correlation']:.3f}
                    - P-value: {corr_data['forest_cover']['p_value']:.3f}
                """)

                # Forest Loss Statistics
                st.markdown("#### Forest Loss")
                st.markdown(f"""
                    - Mean: {corr_data['forest_loss']['mean'] * 100:.2f}% (2001-2023)
                    - Standard Deviation: {corr_data['forest_loss']['std'] * 100:.2f}%
                    - Correlation: {corr_data['forest_loss']['correlation']:.3f}
                    - P-value: {corr_data['forest_loss']['p_value']:.3f}
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

                # Check if we're using alpha shapes
                if 'alpha_shapes' in data and data['alpha_shapes']:
                    # Check if overlaps are avoided in alpha shapes
                    overlaps_avoided = "Unknown"
                    if 'forest_layers' in data and 'alpha_shapes' in data['forest_layers']:
                        overlaps_avoided = data['forest_layers']['alpha_shapes'].get('non_overlapping', True)

                    st.markdown("""
                    **Range-based Analysis:**
                    - Blue polygons show analysis areas created using alpha shapes
                    - Forest metrics are calculated across entire species ranges
                    """)                    
                    # Show alpha shape count if available
                    if 'alpha_shapes' in forest_layers and 'count' in forest_layers['alpha_shapes']:
                        st.markdown(f"**Number of Analysis Areas:** {forest_layers['alpha_shapes']['count']}")
                else:
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
                - **Blue areas:** Forest gain (2000-2012) - New forest growth, included in forest cover calculations
                
                Toggle layers using the controls in the upper right corner.
                """)

        except ImportError:
            # pylint: disable=no-member
            st.error("""
            This visualization requires additional packages. 
            Please install them with:
            
            pip install folium streamlit-folium
            """)
        except Exception as e:
            self.logger.error("Error creating forest correlation map: %s", str(e), exc_info=True)
            raise
