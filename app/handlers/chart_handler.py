"""
Chart Handler Module for Biodiversity Application

This module provides visualization capabilities for biodiversity data using PyDeck.
It supports various chart types including heatmaps and hexagon maps, with automatic
view state adjustment based on data bounds.

"""

import json
import logging

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
            latitude=0.0,
            longitude=30,
            zoom=2,
            pitch=30
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
            if chart_type.lower() == "geojson":
                 with st.spinner("Rendering geojson map..."): #pylint: disable=no-member
                    self.draw_geojson_map(df, parameters)
                    return
            elif chart_type.lower() == "json":
                with st.spinner("Rendering json data..."):  #pylint: disable=no-member
                    self.draw_json_data(df, parameters)
                    return
            if not isinstance(df, pd.DataFrame):
                raise TypeError("Data must be a pandas DataFrame")
            if df.empty:
                raise pd.errors.EmptyDataError("Empty DataFrame provided")
            # Downsample data for large datasets
            if len(df) > 1000:
                df = df.sample(n=1000, random_state=42)
            if chart_type.lower() == "heatmap":
                with st.spinner("Rendering heatmap..."): #pylint: disable=no-member
                    self.draw_heatmap(df)
            # hexagon is default chart type
            else:
                with st.spinner("Rendering distribution map..."):  #pylint: disable=no-member
                    self.draw_hexagon_map(df, parameters)
        except (TypeError, ValueError, pd.errors.EmptyDataError) as e:
            self.logger.error("Error creating visualization: %s", str(e), exc_info=True)
            raise

    def draw_heatmap(self, df):
        """
        Creates a heatmap visualization using PyDeck's HeatmapLayer.

        Raises:
            ValueError: If coordinate data is invalid
            TypeError: If DataFrame columns are of wrong type
        """
        try:
            bounds = self._get_bounds_from_data(df)
            view_state = self.default_view_state
            if bounds:
                view_state = pdk.ViewState(
                    latitude=sum(coord[0] for coord in bounds)/2,
                    longitude=sum(coord[1] for coord in bounds)/2,
                    zoom=3,
                    pitch=30
                )
    # pylint: disable=no-member
            st.pydeck_chart(
                pdk.Deck(
                    initial_view_state=view_state,
                    layers=[
                        pdk.Layer(
                            "HeatmapLayer",
                            data=df,
                            get_position=["decimallongitude", "decimallatitude"],
                            pickable=True,
                            auto_highlight=True,
                        )
                    ],
                    tooltip={
                        "html": "Species: {parameters['species_name']}",
                        "style": {
                            "backgroundColor": "steelblue",
                            "color": "white"
                        }
                    },
                ),
                height=700  # Set the height in pixels
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
            bounds = self._get_bounds_from_data(data)

            # Calculate concave hull
            points = data[['decimallongitude', 'decimallatitude']].values
            hull_geojson = self._calculate_alpha_shape(points, alpha=0.5)  # Already in GeoJSON format

            if bounds is not None:
                view_state = pdk.ViewState(
                    latitude=sum(coord[0] for coord in bounds)/len(bounds),
                    longitude=sum(coord[1] for coord in bounds)/len(bounds),
                    zoom=3,
                    pitch=30
                )
            else:
                view_state = self.default_view_state
            # pylint: disable=no-member
            st.pydeck_chart(
                pdk.Deck(
                    initial_view_state=view_state,
                    layers=[
                        pdk.Layer(
                            "HexagonLayer",
                            data=data,
                            get_position=["decimallongitude", "decimallatitude"],
                            radius=10000,
                            elevation_scale=5000,
                            elevation_range=[0, 1000],
                            pickable=True,
                            extruded=True,
                        ),
                        pdk.Layer(
                            "GeoJsonLayer",
                            data=hull_geojson,  # Use the GeoJSON directly
                            stroked=True,
                            filled=False,
                            line_width_min_pixels=2,
                            get_line_color=[255, 255, 0],
                            get_line_width=3
                        )
                    ],
                    tooltip={
                        "html": (
                            f"{parameters.get('species_name', '')}"
                            "<br/>Occurrences: {elevationValue}"
                        ) if parameters.get('species_name') else None,
                        "style": {
                            "backgroundColor": "steelblue",
                            "color": "white"
                        } if parameters.get('species_name') else None
                    },
                ),
                height=700  # Set the height in pixels
            )
        except (ValueError, TypeError) as e:
            self.logger.error("Error creating hexagon map: %s", str(e), exc_info=True)
            raise
        except Exception as e:
            self.logger.error("Streamlit chart error: %s", str(e), exc_info=True)
            raise

    def draw_chart_client_side_rendering(self, df):
        """
        Draws a chart using client-side rendering. Experimental and not yet working.
        """
        mapbox_token = "YOUR_MAXPOX_TOKEN"  # Replace with your token
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Deck.gl Map</title>
            <script defer src="https://unpkg.com/deck.gl@8.9.33/dist.min.js"></script>
            <script defer src="https://unpkg.com/@deck.gl/core@8.9.33/dist.min.js"></script>
            <script defer src="https://unpkg.com/@deck.gl/layers@8.9.33/dist.min.js"></script>
            <script defer src="https://unpkg.com/@deck.gl/mapbox@8.9.33/dist.min.js"></script>
            <script defer src='https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.js'></script>
            <link href='https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.css' rel='stylesheet' />
        </head>
        <body>
            <div id="deck-container" style="width: 100%; height: 600px; position: relative;"></div>
            <script>
                document.addEventListener('DOMContentLoaded', function() {{
                    console.log('DOM loaded, checking for libraries...');
                    const checkLibraries = setInterval(() => {{
                        if (window.deck && window.mapboxgl) {{
                            console.log('Libraries loaded, initializing...');
                            clearInterval(checkLibraries);
                            initMap();
                        }}
                    }}, 100);

                    function initMap() {{
                        const MAPBOX_ACCESS_TOKEN = '{mapbox_token}';
                        mapboxgl.accessToken = MAPBOX_ACCESS_TOKEN;
                        const data = {df.to_json(orient='records')};
                        
                        try {{
                            const map = new mapboxgl.Map({{
                                container: 'deck-container',
                                style: 'mapbox://styles/mapbox/dark-v10',
                                interactive: false,
                                center: [0, 0],
                                zoom: 2,
                                pitch: 30
                            }});
                            map.on('load', () => {{
                                console.log('Map loaded, initializing deck.gl...');
                                const deck = new deck.DeckGL({{
                                    container: 'deck-container',
                                    mapboxApiAccessToken: MAPBOX_ACCESS_TOKEN,
                                    mapStyle: 'mapbox://styles/mapbox/dark-v10',
                                    initialViewState: {{
                                        longitude: 0,
                                        latitude: 0,
                                        zoom: 2,
                                        pitch: 30
                                    }},
                                    controller: true,
                                    layers: [
                                        new deck.HexagonLayer({{
                                            id: 'hexagon',
                                            data: data,
                                            getPosition:
                                                d => [d.decimallongitude, d.decimallatitude],
                                            radius: 10000,
                                            elevationScale: 5000,
                                            pickable: true,
                                            extruded: true,
                                        }})
                                    ]
                                }});
                            }});
                        }} catch (error) {{
                            console.error('Error initializing map:', error);
                        }}
                    }}
                }});
            </script>
        </body>
        </html>
        """
        print(html_content)
#        html(html_content, height=600)

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
        if df.empty or df['decimallatitude'].isna().all() or df['decimallongitude'].isna().all():
            logging.warning("No valid coordinates in dataset")
            return False
        return True

    def _apply_percentile_filtering(self, df, percentile_cutoff):
        """Apply percentile-based filtering to remove extreme outliers."""
        if not 0 <= percentile_cutoff <= 50:
            raise ValueError("Percentile cutoff must be between 0 and 50")
        lat_lower = np.percentile(df['decimallatitude'], percentile_cutoff)
        lat_upper = np.percentile(df['decimallatitude'], 100 - percentile_cutoff)
        lon_lower = np.percentile(df['decimallongitude'], percentile_cutoff)
        lon_upper = np.percentile(df['decimallongitude'], 100 - percentile_cutoff)
        return df[
            (df['decimallatitude'] >= lat_lower) &
            (df['decimallatitude'] <= lat_upper) &
            (df['decimallongitude'] >= lon_lower) &
            (df['decimallongitude'] <= lon_upper)
        ]

    def _apply_density_filtering(self, df, min_points):
        """Apply density-based filtering to focus on areas of interest."""
        try:
            lat_bins = pd.qcut(df['decimallatitude'], q=15, duplicates='drop')
            lon_bins = pd.qcut(df['decimallongitude'], q=15, duplicates='drop')
            grid_counts = df.groupby([lat_bins, lon_bins]).size()

            density_threshold = max(min_points, np.percentile(grid_counts, 25))
            dense_cells = grid_counts[grid_counts >= density_threshold].index

            if not dense_cells.empty:
                filtered_df = df[
                    df.apply(lambda x:
                        (pd.qcut([x['decimallatitude']], q=15, duplicates='drop')[0],
                         pd.qcut([x['decimallongitude']], q=15, duplicates='drop')[0])
                        in dense_cells, axis=1)
                ]
                if len(filtered_df) < len(df) * 0.1:  # If we've removed more than 90% of points
                    logging.warning("Density filtering too aggressive. "
                                        "Using percentile-filtered dataset.")
                    return df
                return filtered_df

            return df
        except ValueError as e:
            logging.warning("Density-based filtering failed: %s. Using original dataset.", e)
            return df

    def _calculate_bounds(self, df):
        """Calculate the final bounds from the filtered dataset."""
        min_lat = df['decimallatitude'].min()
        max_lat = df['decimallatitude'].max()
        min_lon = df['decimallongitude'].min()
        max_lon = df['decimallongitude'].max()

        if np.isnan(min_lat) or np.isnan(min_lon) or np.isnan(max_lat) or np.isnan(max_lon):
            logging.warning("Invalid bounds calculated")
            return None

        return [[min_lat, min_lon], [max_lat, max_lon]]

    def draw_geojson_map(self, data, parameters):
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
                        latitude=sum(coord[0] for coord in bounds)/len(bounds),
                        longitude=sum(coord[1] for coord in bounds)/len(bounds),
                        zoom=5,
                        pitch=30
                )
            else:
                view_state = self.default_view_state
            # Extract just the GeoJSON features
            features = [
                {
                    "type": "Feature",
                    "geometry": item["geojson"],
                    "properties": {
                        "name": item["name"],
                        "category": item["category"]
                    }
                }
                for item in geojson_data
            ]
            geojson_layer = {
                "type": "FeatureCollection",
                "features": features
            }
            # pylint: disable=no-member
            st.pydeck_chart(
                pdk.Deck(
                    initial_view_state=view_state,
                    layers=[
                        pdk.Layer(
                            "GeoJsonLayer",
                            data=geojson_layer,
                            get_fill_color=[0, 256, 0],
                            stroked=True,
                            filled=True,
                            pickable=True,
                            line_width_min_pixels=1,
                            get_line_color=[0, 0, 0],
                            get_tooltip=["properties.name", "properties.category"]
                        )
                    ],
                    tooltip={"text": "Name: {name}\nIUCN Category: {category}"}
                ),
                height=700  # Set the height in pixels
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

    def draw_json_data(self, data: str, parameters: dict = None) -> None:
        """Draw JSON data as a table.
        
        Args:
            data (str): JSON string containing array of dictionaries
            parameters (dict, optional): Additional visualization parameters
        """
        try:
            # Parse JSON string to DataFrame
            df = pd.DataFrame(json.loads(data))

            # Display the table with formatting
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

        # Create hulls for each cluster
        hulls = []
        for label in set(labels):
            if label == -1:  # Skip noise points
                continue
                
            cluster_points = points[labels == label]
            if len(cluster_points) < 4:
                hull = MultiPoint(cluster_points).convex_hull
                # Add buffer to smooth edges (0.5 degrees ≈ 55km at equator)
                hull = hull.buffer(0.5, resolution=16)
                hulls.append(hull)
                continue

            try:
                # Try to calculate alpha shape
                tri = Delaunay(cluster_points)
                edges = set()
                edge_points = []

                for ia, ib, ic in tri.simplices:
                    pa = cluster_points[ia]
                    pb = cluster_points[ib]
                    pc = cluster_points[ic]

                    a = np.sqrt((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2)
                    b = np.sqrt((pb[0] - pc[0])**2 + (pb[1] - pc[1])**2)
                    c = np.sqrt((pc[0] - pa[0])**2 + (pc[1] - pa[1])**2)
                    s = (a + b + c) / 2.0
                    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                    circum_r = a * b * c / (4.0 * area) if area > 0 else float('inf')

                    if circum_r < 1.0 / alpha:
                        self._add_edge(edges, edge_points, cluster_points, ia, ib)
                        self._add_edge(edges, edge_points, cluster_points, ib, ic)
                        self._add_edge(edges, edge_points, cluster_points, ic, ia)

                m = MultiPoint(cluster_points)
                polygon = Polygon(m.convex_hull)
                if polygon.is_valid:
                    # Add buffer to smooth edges
                    polygon = polygon.buffer(0.5, resolution=16)
                    hulls.append(polygon)
                else:
                    hull = m.convex_hull.buffer(0.5, resolution=16)
                    hulls.append(hull)
            except Exception:  # Removed 'as e'
                # Fallback to convex hull if alpha shape fails
                m = MultiPoint(cluster_points)
                hull = m.convex_hull.buffer(0.5, resolution=16)
                hulls.append(hull)

        # Convert hulls to GeoJSON format
        hull_geojson = {
            "type": "Feature",
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [[[list(p) for p in hull.exterior.coords]] for hull in hulls if hasattr(hull, 'exterior')]
            }
        }
        
        return hull_geojson

    def _add_edge(self, edges, edge_points, coords, i, j):
        """Helper method to add edges."""
        if (i, j) in edges or (j, i) in edges:
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])
