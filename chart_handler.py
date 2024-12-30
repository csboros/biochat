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
from streamlit.errors import StreamlitAPIException

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
        self.logger = logging.getLogger(self.__class__.__name__)
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
            data (pd.DataFrame): DataFrame containing coordinate data
            chart_type (str): Type of visualization to create
            params (dict): Additional parameters for visualization

        Raises:
            ValueError: If chart_type is invalid or data format is incorrect
            TypeError: If arguments are of wrong type
            pd.errors.EmptyDataError: If DataFrame is empty
        """
        try:
            if not isinstance(data, pd.DataFrame):
                raise TypeError("Data must be a pandas DataFrame")
            if data.empty:
                raise pd.errors.EmptyDataError("Empty DataFrame provided")
            if chart_type.lower() == "heatmap":
                self.draw_heatmap(data)
            # hexagon is default chart type
            else:
                self.draw_hexagon_map(data, params)
        except (TypeError, ValueError, pd.errors.EmptyDataError) as e:
            self.logger.error("Error creating visualization: %s", str(e), exc_info=True)
            raise

    def draw_heatmap(self, df):
        """
        Creates a heatmap visualization using PyDeck's HeatmapLayer.

        Raises:
            ValueError: If coordinate data is invalid
            TypeError: If DataFrame columns are of wrong type
            st.StreamlitAPIException: If chart rendering fails
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
                )
            )
        except (ValueError, TypeError) as e:
            self.logger.error("Error creating heatmap: %s", str(e), exc_info=True)
            raise
        except (StreamlitAPIException, RuntimeError) as e:
            self.logger.error("Streamlit chart error: %s", str(e), exc_info=True)
            raise ValueError("Failed to render chart") from e

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
        except (ValueError, TypeError) as e:
            self.logger.error("Error creating hexagon map: %s", str(e), exc_info=True)
            raise
        except Exception as e:
            self.logger.error("Streamlit chart error: %s", str(e), exc_info=True)
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
