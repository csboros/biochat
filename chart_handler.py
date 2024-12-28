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
import streamlit as st

class ChartHandler:
    """
    Handles the creation and rendering of geographic visualizations.

    This class manages different types of map visualizations for biodiversity data,
    including heatmaps and hexagon maps. It automatically calculates appropriate
    view states based on data distribution and handles both DataFrame and GeoJSON inputs.

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
        self.default_view_state = pdk.ViewState(
            latitude=0.0,
            longitude=30,
            zoom=2,
            pitch=30
        )

    def draw_chart(self, data, chart_type, params):
        """
        Main entry point for creating visualizations.

        Args:
            data (pd.DataFrame): DataFrame containing at minimum 'decimallatitude'
                and 'decimallongitude' columns
            chart_type (str): Type of visualization to create. Supported types:
                - "heatmap": Density-based heatmap
                - "hexagon"/"hex"/"hexagons"/"hexbin": Hexagonal binning
            params (dict): Additional parameters for the visualization including:
                - species_name (str): Name of species being visualized
                - geojson (str, optional): GeoJSON string for boundary overlay

        Raises:
            ValueError: If an unsupported chart type is specified
        """
        if chart_type.lower() == "heatmap":
            self._draw_heatmap(data)
        elif chart_type.lower() == "hexagon" or  chart_type.lower() == "hex" \
            or chart_type.lower() == "hexagons" or chart_type.lower() == "hexbin":
            self._draw_hexagon_map(data, params)
        else:
            st.write(data)

    def _draw_heatmap(self, df):
        """
        Creates a heatmap visualization using PyDeck's HeatmapLayer.

        Args:
            df (pd.DataFrame): DataFrame containing coordinate data
            parameters (dict): Visualization parameters including:
                - species_name (str): Species name for tooltip display

        Note:
            The heatmap intensity is automatically calculated based on point density.
            View state is adjusted to fit all data points with outlier filtering.
        """
        # Get bounds from data
        bounds = self._get_bounds_from_data(df)
        view_state = self.default_view_state
        if bounds:
            view_state = pdk.ViewState(
                latitude=sum(coord[0] for coord in bounds)/2,
                longitude=sum(coord[1] for coord in bounds)/2,
                zoom=3,
                pitch=30
            )

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
                    "html": "Species<br/>Total Load: {parameters['species_name']}",
                    "style": {
                        "backgroundColor": "steelblue",
                        "color": "white"
                    }
                },
            )
        )
    def _draw_hexagon_map(self, data, parameters):
        """
        Creates a 3D hexagon bin visualization using PyDeck's HexagonLayer.

        Args:
            data (pd.DataFrame): DataFrame containing coordinate data
            parameters (dict): Visualization parameters including:
                - species_name (str): Species name for tooltip display
                - geojson (str, optional): GeoJSON string for boundary overlay

        Note:
            Hexagon height represents point density in each bin.
            The map includes interactive tooltips showing occurrence counts.
        """
        logging.debug("Drawing hexagon map with parameters: %s", parameters)
        # Handle GeoJSON data
        geojson_data = parameters.get("geojson")
        bounds = None

        if geojson_data and isinstance(geojson_data, str):
            try:
                geojson_data = json.loads(geojson_data)
                bounds = self._get_bounds_from_geojson(geojson_data)
                logging.debug("Calculated bounds: %s", bounds)
                logging.debug("Successfully parsed GeoJSON string")
            except json.JSONDecodeError as e:
                logging.error("Failed to parse GeoJSON: %s", e)
        else:
            bounds = self._get_bounds_from_data(data)

        print(bounds)

        if bounds is not None:
            view_state = pdk.ViewState(
                    latitude=sum(coord[0] for coord in bounds)/len(bounds),
                    longitude=sum(coord[1] for coord in bounds)/len(bounds),
                    zoom=3,
                    pitch=30
            )
        else:
            view_state = self.default_view_state

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
            )
        )

    def _get_bounds_from_geojson(self, geojson):
        """
        Extracts coordinate bounds from GeoJSON data.

        Args:
            geojson (dict): Parsed GeoJSON object containing either:
                - FeatureCollection
                - Single Feature with Polygon or MultiPolygon

        Returns:
            list: List of two coordinates [[min_lat, min_lon], [max_lat, max_lon]]
                defining the bounding box

        Note:
            Supports both Polygon and MultiPolygon geometries.
        """
        coordinates = []
        if geojson['type'] == 'FeatureCollection':
            for feature in geojson['features']:
                if feature['geometry']['type'] == 'Polygon':
                    coordinates.extend(feature['geometry']['coordinates'][0])
                elif feature['geometry']['type'] == 'MultiPolygon':
                    for polygon in feature['geometry']['coordinates']:
                        coordinates.extend(polygon[0])
        elif geojson['type'] == 'Feature':
            if geojson['geometry']['type'] == 'Polygon':
                coordinates = geojson['geometry']['coordinates'][0]
            elif geojson['geometry']['type'] == 'MultiPolygon':
                for polygon in geojson['geometry']['coordinates']:
                    coordinates.extend(polygon[0])
        lats = []
        lons = []
        if coordinates:
            lats = [coord[1] for coord in coordinates]
            lons = [coord[0] for coord in coordinates]
        return [[min(lats), min(lons)], [max(lats), max(lons)]] if lats else None

    def _get_bounds_from_data(self, df, percentile_cutoff=2.5, min_points=10):
        """
        Calculates appropriate map bounds from coordinate data with outlier filtering.

        Args:
            df (pd.DataFrame): DataFrame containing 'decimallatitude' and 
                'decimallongitude' columns
            percentile_cutoff (float): Percentile threshold for initial outlier removal
                (default: 2.5)
            min_points (int): Minimum number of points required in a grid cell
                (default: 10)

        Returns:
            list: List of two coordinates [[min_lat, min_lon], [max_lat, max_lon]]
                defining the bounding box, or None if bounds cannot be determined

        Note:
            Uses a two-step filtering process:
            1. Removes extreme outliers using percentile cutoff
            2. Applies density-based filtering to focus on areas of interest
        """
        try:
            # Create a copy to avoid modifying original data
            filtered_df = df.copy()
            # First check if we have any valid data
            if filtered_df.empty or filtered_df['decimallatitude'].isna().all() \
                or filtered_df['decimallongitude'].isna().all():
                logging.warning("No valid coordinates in dataset")
                return None

            # Apply percentile filtering first and store the bounds
            lat_lower = np.percentile(filtered_df['decimallatitude'], percentile_cutoff)
            lat_upper = np.percentile(filtered_df['decimallatitude'], 100 - percentile_cutoff)
            lon_lower = np.percentile(filtered_df['decimallongitude'], percentile_cutoff)
            lon_upper = np.percentile(filtered_df['decimallongitude'], 100 - percentile_cutoff)
            filtered_df = filtered_df[
                (filtered_df['decimallatitude'] >= lat_lower) &
                (filtered_df['decimallatitude'] <= lat_upper) &
                (filtered_df['decimallongitude'] >= lon_lower) &
                (filtered_df['decimallongitude'] <= lon_upper)
            ]

            # Create a rough grid and count points in each cell
            try:
                lat_bins = pd.qcut(filtered_df['decimallatitude'], q=15, duplicates='drop')
                lon_bins = pd.qcut(filtered_df['decimallongitude'], q=15, duplicates='drop')
                # Count points in each grid cell
                grid_counts = filtered_df.groupby([lat_bins, lon_bins]).size()

                # Keep only cells with point counts above the 25th percentile
                density_threshold = max(min_points, np.percentile(grid_counts, 25))
                dense_cells = grid_counts[grid_counts >= density_threshold].index
                if not dense_cells.empty:
                    filtered_df = filtered_df[
                        filtered_df.apply(lambda x:
                            (pd.qcut([x['decimallatitude']], q=15, duplicates='drop')[0],
                             pd.qcut([x['decimallongitude']], q=15, duplicates='drop')[0])
                            in dense_cells, axis=1)
                    ]
            except ValueError as e:
                logging.warning("Density-based filtering failed: %s. "
                                "Using percentile-filtered dataset.", e)
            # If filtering removed too many points, use percentile-filtered dataset
            if len(filtered_df) < len(df) * 0.1:  # If we've removed more than 90% of points
                logging.warning("Density filtering too aggressive. "
                                "Using percentile-filtered dataset.")
                filtered_df = df[
                    (df['decimallatitude'] >= lat_lower) &
                    (df['decimallatitude'] <= lat_upper) &
                    (df['decimallongitude'] >= lon_lower) &
                    (df['decimallongitude'] <= lon_upper)
                ]
            # Calculate bounds from filtered data
            min_lat = filtered_df['decimallatitude'].min()
            max_lat = filtered_df['decimallatitude'].max()
            min_lon = filtered_df['decimallongitude'].min()
            max_lon = filtered_df['decimallongitude'].max()
            # Verify we have valid bounds
            if np.isnan(min_lat) or np.isnan(min_lon) or np.isnan(max_lat) or np.isnan(max_lon):
                logging.warning("Invalid bounds calculated")
                return None
            return [[min_lat, min_lon], [max_lat, max_lon]]
        except (ValueError, TypeError, pd.errors.EmptyDataError) as e:
            logging.error("Error getting bounds from data: %s", e)
            return None
