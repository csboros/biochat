"""
This module provides the base classes and types for chart rendering in the visualization tool.

Classes:
    BaseChartRenderer: Abstract base class for chart renderers.

Enums:
    ChartType: Enum for different types of charts supported.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
import pandas as pd
import numpy as np
import pydeck as pdk
from app.utils.alpha_shape_utils import AlphaShapeUtils
from .chart_types import ChartType

class BaseChartRenderer(ABC):
    """
    Abstract base class for chart renderers.
    """
    def __init__(self):
        self.default_view_state = pdk.ViewState(
            latitude=0.0, longitude=30, zoom=2, pitch=30
        )
        self.logger = logging.getLogger("BioChat." + self.__class__.__name__)
        self.alpha_shape_utils = AlphaShapeUtils()
    @abstractmethod
    def render(self, data: Any, parameters: Optional[Dict] = None,
               cache_buster: Optional[str] = None) -> Any:
        """Render the chart with given data and parameters"""

    @property
    @abstractmethod
    def supported_chart_types(self) -> list[ChartType]:
        """Return list of chart types this renderer supports"""


    def get_bounds_from_data(self, df, percentile_cutoff=2.5, min_points=10):
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


    def create_alpha_shapes(self, df, parameters):
        """
        Create alpha shapes from a dataframe of coordinates.

        Args:
            df (pd.DataFrame): DataFrame containing coordinate data with decimallongitude and decimallatitude columns
            parameters (dict): Dictionary containing alpha shape parameters
                - alpha (float): Alpha value for concave hull algorithm (default: 0.5)
                - eps (float): Epsilon value for DBSCAN clustering (default: 1.0)
                - min_samples (int): Minimum samples for DBSCAN clustering (default: 3)
                - avoid_overlaps (bool): Whether to merge overlapping clusters (default: True)

        Returns:
            dict: GeoJSON representation of the alpha shapes
        """
        # Get alpha parameters from parameters or use defaults
        alpha = parameters.get('alpha', 0.5)
        eps = parameters.get('eps', 1.0)
        min_samples = parameters.get('min_samples', 3)
        avoid_overlaps = parameters.get('avoid_overlaps', True)

        # Calculate concave hull
        points = df[["decimallongitude", "decimallatitude"]].values
        hull_geojson = self.alpha_shape_utils.calculate_alpha_shape(
            points,
            alpha=alpha,
            eps=eps,
            min_samples=min_samples,
            avoid_overlaps=avoid_overlaps
        )

        return hull_geojson

