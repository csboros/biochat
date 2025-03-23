"""Module for base Earth Engine data processing functionality."""

import os
from typing import Optional, List
import logging
import numpy as np
import ee
from google.cloud import bigquery
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
from app.utils.alpha_shape_utils import AlphaShapeUtils

class EarthEngineHandler:
    """Base class for Earth Engine data processing and analysis."""

    def __init__(self):
        """Initialize the EarthEngineHandler."""
        self.logger = logging.getLogger("BioChat." + self.__class__.__name__)
        self.alpha_shape_utils = AlphaShapeUtils()

        # Initialize Earth Engine
        try:
            ee.Initialize()
        except ee.EEException as e:
            self.logger.error("Failed to initialize Earth Engine: %s", str(e))
            raise

    def get_species_observations(self, species_name: str, min_observations: int = 10) -> list:
        """Retrieve species observations from BigQuery.

        Args:
            species_name (str): Scientific name of the species
            min_observations (int): Minimum number of observations required

        Returns:
            list: List of observation dictionaries with coordinates and metadata

        Raises:
            ValueError: If insufficient observations are found for the species
        """
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        client = bigquery.Client(project=project_id)

        query = f"""
        SELECT
            decimallongitude,
            decimallatitude,
            EXTRACT(YEAR FROM eventdate) as observation_year,
            COALESCE(individualcount, 1) as individual_count
        FROM `{project_id}.biodiversity.occurances_endangered_species_mammals`
        WHERE species = @species_name
            AND decimallongitude IS NOT NULL
            AND decimallatitude IS NOT NULL
            AND eventdate IS NOT NULL
            AND EXTRACT(YEAR FROM eventdate) > 2000
        """

        if isinstance(species_name, dict) and 'species_name' in species_name:
            species_name = species_name['species_name']

        params = [bigquery.ScalarQueryParameter("species_name", "STRING", species_name)]
        job_config = bigquery.QueryJobConfig(query_parameters=params)

        # Convert query results to a list of dictionaries
        observations_raw = list(client.query(query, job_config=job_config).result())
        observations = [
            {
                "decimallongitude": obs.decimallongitude,
                "decimallatitude": obs.decimallatitude,
                "observation_year": obs.observation_year,
                "individual_count": obs.individual_count
            }
            for obs in observations_raw
        ]

        if len(observations) < min_observations:
            raise ValueError(
                f"Insufficient observations ({len(observations)}) "
                f"for species {species_name}"
            )

        return observations

    def create_ee_point_features(self, observations: list, buffer_radius: int = None) -> list:
        """Create Earth Engine features from individual observation points.

        Args:
            observations (list): List of observation dictionaries from the database
            buffer_radius (int, optional): Buffer radius in meters

        Returns:
            list: List of Earth Engine features for individual observation points
        """
        ee_point_features = []

        for obs in observations:
            try:
                # Create point geometry
                ee_geom = ee.Geometry.Point([obs["decimallongitude"], obs["decimallatitude"]])

                # Apply buffer if specified (in meters)
                if buffer_radius:
                    ee_geom = ee_geom.buffer(buffer_radius)

                # Create Earth Engine feature with properties
                ee_feature = ee.Feature(ee_geom, {
                    'year': obs["observation_year"],
                    'individual_count': obs["individual_count"]
                })

                ee_point_features.append(ee_feature)
            except Exception as e: # pylint: disable=broad-except
                self.logger.warning("Error creating Earth Engine feature for point: %s", str(e))

        return ee_point_features

    def generate_alpha_shapes_for_visualization(
        self,
        observations: list,
        alpha: float = 0.5,
        eps: float = 1.0,
        min_samples: int = 3,
        avoid_overlaps: bool = True
    ) -> list:
        """Generate alpha shapes from species observations for visualization purposes only.

        Args:
            observations (list): List of observation dictionaries
            alpha (float): Alpha parameter for alpha shapes
            eps (float): DBSCAN epsilon parameter for clustering
            min_samples (int): Minimum samples for DBSCAN clustering
            avoid_overlaps (bool): Whether to merge overlapping alpha shapes

        Returns:
            list: List of alpha shapes in GeoJSON format with properties for visualization
        """
        # Convert observations to numpy array of coordinates
        points = np.array([[obs["decimallongitude"], obs["decimallatitude"]]
                          for obs in observations])

        # Calculate alpha shapes using utility class
        alpha_shapes_geojson = self.alpha_shape_utils.calculate_alpha_shape(
            points,
            alpha=alpha,
            eps=eps,
            min_samples=min_samples,
            avoid_overlaps=avoid_overlaps
        )

        # Prepare alpha shapes for visualization
        all_alpha_shapes = []

        if alpha_shapes_geojson and 'geometry' in alpha_shapes_geojson:
            # Calculate common properties
            years = [obs["observation_year"] for obs in observations]
            representative_year = max(years) if years else 2000
            total_individuals = sum(obs["individual_count"] for obs in observations)

            # Process the GeoJSON to extract alpha shapes for visualization
            if alpha_shapes_geojson['geometry']['type'] == 'MultiPolygon':
                for polygon_coords in alpha_shapes_geojson['geometry']['coordinates']:
                    all_alpha_shapes.append({
                        'type': 'Polygon',
                        'coordinates': polygon_coords,
                        'properties': {
                            'year': representative_year,
                            'num_observations': len(observations),
                            'total_individuals': total_individuals
                        }
                    })
            else:
                # For single polygon (not MultiPolygon)
                polygon_coords = alpha_shapes_geojson['geometry']['coordinates']
                all_alpha_shapes.append({
                    'type': 'Polygon',
                    'coordinates': polygon_coords,
                    'properties': {
                        'year': representative_year,
                        'num_observations': len(observations),
                        'total_individuals': total_individuals
                    }
                })

        # If no valid alpha shapes were created, log a warning
        if not all_alpha_shapes:
            self.logger.warning("No valid alpha shapes created for visualization")

        return all_alpha_shapes

    def send_to_llm(self, prompt: str) -> str:
        """Sends a prompt to Google's Gemini LLM and returns the response."""
        try:
            self.logger.info("Sending prompt to Gemini")

            # Lower temperature for more analytical responses
            model = GenerativeModel(
                "gemini-2.0-flash-lite",
                generation_config=GenerationConfig(
                    temperature=0.5,  # Lower temperature for more precise analysis
                    max_output_tokens=8192,  # Ensure we get detailed response
                    top_p=0.95,  # More focused token selection
                )
            )

            # Add explicit instruction for quantitative analysis
            response = model.generate_content(prompt)

            self.logger.info("Received response from Gemini")
            return response.text

        except Exception as e: # pylint: disable=broad-except
            self.logger.error("Error getting response from Gemini: %s", str(e), exc_info=True)
            raise

    def create_visualization_features_from_alpha_shapes(
        self,
        alpha_shapes: List[dict]
    ) -> List[ee.Feature]:
        """Create Earth Engine features from alpha shapes for visualization.

        Args:
            alpha_shapes (list): List of alpha shape GeoJSON polygons

        Returns:
            list: List of Earth Engine features for visualization
        """
        ee_features = []

        if not alpha_shapes:
            return ee_features

        for shape in alpha_shapes:
            try:
                # Create a geometry from the coordinates
                geom = ee.Geometry.Polygon(shape['coordinates'])

                # Create a feature with properties
                props = shape.get('properties', {})
                feature = ee.Feature(geom, props)

                ee_features.append(feature)
            except Exception as e: # pylint: disable=broad-except
                self.logger.error("Error converting alpha shape: %s", str(e))

        return ee_features

    def create_styled_alpha_visualization(self, alpha_shapes: List[dict]) -> Optional[List[str]]:
        """Create styled visualization layers for alpha shapes.

        Args:
            alpha_shapes (list): List of alpha shape GeoJSON polygons

        Returns:
            list or None: List of tile URLs for visualization or None if creation failed
        """
        if not alpha_shapes or len(alpha_shapes) == 0:
            return None

        try:
            # Get Earth Engine features from alpha shapes
            ee_features = self.create_visualization_features_from_alpha_shapes(alpha_shapes)

            if not ee_features:
                return None

            # Create a feature collection from the features
            alpha_fc = ee.FeatureCollection(ee_features)

            # Style the alpha shapes for visualization
            styled_alpha = alpha_fc.style(
                color='4285F4',         # Blue outline
                fillColor='4285F433',   # Semi-transparent blue fill (33 = 20% opacity)
                width=2                 # Line width
            )

            # Get the map ID for the styled alpha shapes
            alpha_viz = styled_alpha.getMapId()
            return [alpha_viz['tile_fetcher'].url_format]

        except Exception as e: # pylint: disable=broad-except
            self.logger.error("Failed to create alpha shape visualization: %s", str(e))
            return None