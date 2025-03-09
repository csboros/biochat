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

from .d3js_visualization import display_force_visualization, display_tree

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

    def draw_chart(self, df, chart_type, parameters):
        """
        Main entry point for creating visualizations.
        Args:
            data (pd.DataFrame): DataFrame containing coordinate data
            chart_type (str): Type of visualization to create
            params (dict): Additional parameters for visualization
        Raises:
            ValueError: If chart_type is invalid or data format is incorrect
            TypeError: If arguments are of wrong type
            pd.errors.EmptyDataError: If DataFrame is empty
        """
        try:
            # pylint: disable=no-member
            if chart_type.lower() == "geojson":
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
                    self.draw_correlation_scatter(df, parameters)
                    return
            elif chart_type.lower() == "images":
                with st.spinner(
                    "Rendering images..."
                ):
                    self.display_species_images(df)
                    return
            elif chart_type.lower() == "circle_packing":
                with st.spinner("Rendering circle packing visualization..."):
                    display_force_visualization(df)
                    return
            if isinstance(df, pd.DataFrame):
                if df.empty:
                    raise pd.errors.EmptyDataError("Empty DataFrame provided")
                # Downsample data for large datasets
                if len(df) > 1000:
                    df = df.sample(n=1000, random_state=42)
            if chart_type.lower() == "heatmap":
                with st.spinner("Rendering heatmap..."):  # pylint: disable=no-member
                    self.draw_heatmap(parameters, df)
            # hexagon is default chart type
            else:
                with st.spinner(  # pylint: disable=no-member
                    "Rendering distribution map..."
                ):  # pylint: disable=no-member
                    self.draw_hexagon_map(df, parameters)
        except (TypeError, ValueError, pd.errors.EmptyDataError) as e:
            self.logger.error("Error creating visualization: %s", str(e), exc_info=True)
            raise

    def draw_heatmap(self, parameters, data):
        """Creates a heatmap visualization using PyDeck's HeatmapLayer."""
        try:
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

    def draw_hexagon_map(self, data, parameters):
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
            # Convert the JSON array to a DataFrame
            df = pd.DataFrame(occurrences)
            bounds = self._get_bounds_from_data(df)

            # Calculate concave hull
            points = df[["decimallongitude", "decimallatitude"]].values
            hull_geojson = self._calculate_alpha_shape(points, alpha=0.5)
            if bounds is not None:
                view_state = pdk.ViewState(
                    latitude=sum(coord[0] for coord in bounds) / len(bounds),
                    longitude=sum(coord[1] for coord in bounds) / len(bounds),
                    zoom=3,
                    pitch=30,
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

    def draw_geojson_map(self, data):
        """
        Draws a geojson map.

        Args:
            data (str): JSON string containing array of GeoJSON features
            parameters (dict): Additional visualization parameters
        """
        try:
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

    def draw_json_data(self, data: str) -> None:
        """Draw JSON data as a table.
        Args:
            data (str): JSON string containing array of dictionaries
            parameters (dict, optional): Additional visualization parameters
        """
        try:
            df = pd.DataFrame(data)
            # pylint: disable=no-member
            st.dataframe(df)
        except (json.JSONDecodeError, pd.errors.EmptyDataError) as e:
            self.logger.error("Error parsing JSON data: %s", str(e))
            raise
        except Exception as e:
            self.logger.error("Error creating table: %s", str(e))
            raise

    def _calculate_alpha_shape(self, points, alpha):
        """Calculate clustered alpha shapes for point distributions."""
        if len(points) < 4:
            return MultiPoint(points).convex_hull
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=2, min_samples=3).fit(points)
        labels = clustering.labels_
        # Process each cluster
        hulls = self._process_clusters(points, labels, alpha)
        # Convert hulls to GeoJSON format
        return self._convert_hulls_to_geojson(hulls)

    def _process_clusters(self, points, labels, alpha):
        """Process each cluster to create hulls."""
        hulls = []
        for label in set(labels):
            if label == -1:  # Skip noise points
                continue
            cluster_points = points[labels == label]
            hull = self._process_single_cluster(cluster_points, alpha)
            if hull is not None:
                hulls.append(hull)
        return hulls

    def _process_single_cluster(self, cluster_points, alpha):
        """Process a single cluster to create a hull."""
        if len(cluster_points) < 4:
            hull = MultiPoint(cluster_points).convex_hull
            return hull.buffer(0.5, resolution=16)
        try:
            return self._create_alpha_shape(cluster_points, alpha)
        except Exception:  # pylint: disable=broad-except
            m = MultiPoint(cluster_points)
            return m.convex_hull.buffer(0.5, resolution=16)

    def _create_alpha_shape(self, points, alpha):
        """Create alpha shape from points."""
        tri = Delaunay(points)
        edges = set()
        edge_points = []

        for ia, ib, ic in tri.simplices:
            pa = points[ia]
            pb = points[ib]
            pc = points[ic]

            circum_r = self._calculate_circumradius(pa, pb, pc)
            if circum_r < 1.0 / alpha:
                self._add_edge(edges, edge_points, (pa, ia, ib))
                self._add_edge(edges, edge_points, (pb, ib, ic))
                self._add_edge(edges, edge_points, (pc, ic, ia))

        m = MultiPoint(points)
        polygon = Polygon(m.convex_hull)
        if polygon.is_valid:
            return polygon.buffer(0.5, resolution=16)
        return m.convex_hull.buffer(0.5, resolution=16)

    def _calculate_circumradius(self, pa, pb, pc):
        """Calculate circumradius of triangle."""
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        return a * b * c / (4.0 * area) if area > 0 else float("inf")

    def _convert_hulls_to_geojson(self, hulls):
        """Convert hulls to GeoJSON format."""
        return {
            "type": "Feature",
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [[list(p) for p in hull.exterior.coords]]
                    for hull in hulls
                    if hasattr(hull, "exterior")
                ],
            },
        }

    def _add_edge(self, edges: set, edge_points: list, point_data: tuple) -> None:
        """Helper method to add edges.
        Args:
            edges (set): Set of existing edges
            edge_points (list): List of edge point coordinates
            point_data (tuple): Tuple containing (coords, i, j) where:
                - coords: Array of coordinates
                - i: First vertex index
                - j: Second vertex index
        """
        coords, i, j = point_data
        if (i, j) in edges or (j, i) in edges:
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])

    def _draw_3d_scatterplot(self, data):
        """Draws a 3D scatter visualization using PyDeck's ColumnLayer."""
        try:
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
            self._render_visualization((col1, col2), layer, view_state, viz_config)

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

    def _render_visualization(self, columns, layer, view_state, viz_config):
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

    def draw_yearly_observations(self, data):
        """Draws a visualization of yearly observation data with distinct colors per country."""
        try:
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
                self._draw_observation_chart(yearly_data, names["common"])
            with col2:
                self._display_observation_summary(yearly_data, names)

        except Exception as e:
            self.logger.error(
                "Error drawing yearly observations: %s", str(e), exc_info=True
            )
            raise

    def _draw_observation_chart(self, yearly_data, title):
        """Creates and displays the observation chart."""
        print(yearly_data)
        if isinstance(yearly_data, dict):
            self._draw_country_chart(yearly_data, title)
        else:
            self._draw_global_chart(yearly_data)

    def _display_observation_summary(self, yearly_data, names):
        """Displays the observation summary sidebar."""
        # pylint: disable=no-member
        st.markdown("### Species Information")
        st.markdown(f"**Common Name**: {names['common']}")
        st.markdown(f"**Scientific Name**: {names['scientific']}")
        st.markdown("### Observation Summary")

        if isinstance(yearly_data, dict):
            self._display_country_summary(yearly_data)
        else:
            self._display_global_summary(pd.DataFrame(yearly_data))

    def _draw_country_chart(self, yearly_data, title):
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
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            self.logger.error("Error creating country chart: %s", str(e))
            raise

    def _draw_global_chart(self, yearly_data):
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
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            self.logger.error("Error creating global chart: %s", str(e))
            raise

    def _display_country_summary(self, yearly_data):
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

    def _display_global_summary(self, df):
        """Displays the global observation summary."""
        total_observations = df["count"].sum()
        # pylint: disable=no-member
        st.markdown(f"**Total Observations**: {total_observations:,}")
        st.markdown(f"**Year Range**: {df['year'].min()} - {df['year'].max()}")

    def draw_correlation_scatter(self, data, parameters):
        """Draw a scatter plot showing correlation between HCI and species counts."""
        try:
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

    def display_species_images(self, images_data):
        """Display species images in the Streamlit interface."""
        # pylint: disable=no-member
        if images_data["image_count"] > 0:
            st.subheader(f"Images of {images_data['species']}")
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
