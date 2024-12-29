"""
Chart Handler Module for Biodiversity Application

This module provides visualization capabilities for biodiversity data using PyDeck.
It supports various chart types including heatmaps and hexagon maps, with automatic
view state adjustment based on data bounds.

"""

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

        Raises:
            ValueError: If an unsupported chart type is specified
        """
        if chart_type.lower() == "heatmap":
            self.draw_heatmap(data)
        elif chart_type.lower() == "hexagon" or  chart_type.lower() == "hex" \
            or chart_type.lower() == "hexagons" or chart_type.lower() == "hexbin":
            self.draw_hexagon_map(data, params)
        else:
            st.write(data)

    def draw_heatmap(self, df):
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
    def draw_hexagon_map(self, data, parameters):
        """
        Creates a 3D hexagon bin visualization using PyDeck's HexagonLayer.

        Args:
            data (pd.DataFrame): DataFrame containing coordinate data
            parameters (dict): Visualization parameters including:
                - species_name (str): Species name for tooltip display

        Note:
            Hexagon height represents point density in each bin.
            The map includes interactive tooltips showing occurrence counts.
        """
        logging.debug("Drawing hexagon map with parameters: %s", parameters)
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
