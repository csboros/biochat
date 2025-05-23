"""Module for base Earth Engine data processing functionality."""

import os
from typing import Optional, List, Dict, Any
import logging
import json
from scipy import stats
import numpy as np
import ee
from google.cloud import bigquery
try:
    from google.cloud import secretmanager
except ImportError:
    secretmanager = None
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
import streamlit as st
from app.utils.alpha_shape_utils import AlphaShapeUtils
from app.tools.message_bus import message_bus
from ...base_handler import BaseHandler

# pylint: disable=no-member
# pylint: disable=broad-except
class EarthEngineHandler(BaseHandler):
    """Base class for Earth Engine data processing and analysis."""

    def __init__(self):
        """Initialize the EarthEngineHandler."""
        super().__init__()  # Call the parent class's __init__ method
        self.logger = logging.getLogger("BioChat." + self.__class__.__name__)
        self.alpha_shape_utils = AlphaShapeUtils()

        # Initialize Earth Engine
        try:
            # Check if running locally via environment variable
            is_local = os.getenv('LOCAL_DEVELOPMENT') == 'true'

            if not is_local:
                # Cloud initialization with Secret Manager
                if secretmanager:
                    client = secretmanager.SecretManagerServiceClient()
                    name = (f"projects/{os.getenv('GOOGLE_CLOUD_PROJECT')}/secrets/"
                            "EARTH_ENGINE_CREDENTIALS/versions/latest")
                    response = client.access_secret_version(request={"name": name})
                    credentials_str = response.payload.data.decode("UTF-8")
                    credentials_dict = json.loads(credentials_str)
                    credentials = ee.ServiceAccountCredentials(
                        email=credentials_dict.get('client_email'),
                        key_data=credentials_str
                    )
                    ee.Initialize(credentials, project=os.getenv("GOOGLE_CLOUD_PROJECT"))
                else:
                    raise RuntimeError("Secret Manager not available")
            else:
                # Local initialization
                ee.Initialize()

        except Exception as e:
            self.logger.error("Failed to initialize Earth Engine: %s", str(e))
            raise

    def check_cancellation(self) -> None:
        """Check if the operation has been cancelled by the user.

        Raises:
            RuntimeError: If the operation has been cancelled
        """
        if st.session_state.get("is_cancelled", False):
            message_bus.publish("status_update", {
                "message": "Analysis cancelled by user",
                "state": "error",
                "progress": 0
            })
            raise RuntimeError("Analysis cancelled by user")

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
        try:
            # Check for cancellation before starting
            self.check_cancellation()

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
            ORDER BY eventdate DESC
            LIMIT 10000
            """


            if isinstance(species_name, dict) and 'species_name' in species_name:
                species_name = species_name['species_name']

            scientific_name = self.translate_to_scientific_name_from_api({"name": species_name})

            if ("error" in scientific_name or
                    "scientific_name" not in scientific_name or
                    not scientific_name["scientific_name"]):
                message_bus.publish("status_update", {
                    "message": f"Could not translate species name: {species_name}",
                    "state": "error",
                    "progress": 0
                })
                return {
                    "species": species_name,
                    "occurrence_count": 0,
                    "occurrences": [],
                }

            species_name = scientific_name["scientific_name"]
            self.logger.info(f"Translated species name: {species_name}")

            params = [bigquery.ScalarQueryParameter("species_name", "STRING", species_name)]
            job_config = bigquery.QueryJobConfig(query_parameters=params)

            # Execute query
            query_job = client.query(query, job_config=job_config)

            # Check for cancellation after query execution
            self.check_cancellation()

            # Convert query results to a list of dictionaries
            observations_raw = list(query_job.result())
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
                    f"Insufficient observations ({len(observations)}) for species {species_name}"
                )

            return observations

        except Exception as e:
            self.logger.error("Error getting species observations: %s", str(e))
            raise

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

    def create_error_response(
        self,
        species_name: str,
        observations: list,
        all_alpha_shapes: list,
        scale: int,
        error_message: str = "No valid results were obtained from Earth Engine analysis.",
        analysis_message: str = "Analysis failed. Please try with "
                 "different parameters or a smaller area.",
        correlation_data_structure: dict = None
    ) -> Dict[str, Any]:
        """Create a standardized error response for analysis failures.

        Args:
            species_name (str): Scientific name of the species
            observations (list): List of observation dictionaries
            all_alpha_shapes (list): List of alpha shapes for visualization
            scale (int): Resolution used for analysis
            avoid_overlaps (bool): Whether alpha shapes avoid overlaps
            error_message (str): Specific error message to include
            analysis_message (str): User-friendly analysis message
            correlation_data_structure (dict): Structure of correlation data specific to the handler

        Returns:
            dict: Structured error response with consistent format
        """
        return {
            'correlation_data': correlation_data_structure or {
                'mean': 0,
                'std': 0,
                'correlation': 0,
                'p_value': 1.0,
                'total_observations': len(observations) if observations else 0,
                'spatial_resolution': scale,
                'error': error_message
            },
            'analysis': analysis_message,
            'species_name': species_name,
            'observations': observations,
            'alpha_shapes': all_alpha_shapes,
            'error': True
        }

    def create_alpha_shape_visualization(self, alpha_shapes: list) -> Optional[list]:
        """Create visualization tiles for alpha shapes.

        Args:
            alpha_shapes (list): List of alpha shape GeoJSON polygons

        Returns:
            list or None: List of tile URLs for visualization or None if creation failed
        """
        if not alpha_shapes or len(alpha_shapes) == 0:
            return None

        try:
            # Convert the GeoJSON alpha shapes to Earth Engine features
            ee_features = []
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

            if ee_features:
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
                alpha_shape_tiles = [alpha_viz['tile_fetcher'].url_format]

                self.logger.info("Alpha shape visualization created successfully")
                return alpha_shape_tiles

            return None

        except Exception as e: # pylint: disable=broad-except
            self.logger.error("Failed to create alpha shape visualization: %s", str(e))
            return None

    def calculate_correlations(
        self,
        all_results: list,
        scale: int,
        metric_name: str,
        metric_values: list,
        bin_size: float = 5.0,
        bin_range: tuple = (0, 100)
    ) -> Dict[str, Any]:
        """Calculate correlations between species observations and metric values.

        Args:
            all_results (list): Processed results with metric values for each point
            scale (int): Resolution in meters used for Earth Engine analysis
            metric_name (str): Name of the metric being analyzed
            metric_values (list): List of metric values to analyze
            bin_size (float): Size of bins for distribution analysis
            bin_range (tuple): Range for binning (min, max)

        Returns:
            dict: Correlation data including statistics and distribution
        """
        # Create observation counts by location
        obs_bins = {}
        for r in all_results:
            loc_key = round(r[metric_name], 2)  # Round to 2 decimal places for binning
            obs_bins[loc_key] = obs_bins.get(loc_key, 0) + r['individual_count']

        # Create arrays for correlation calculation
        bin_counts = list(obs_bins.values())
        bin_metrics = list(obs_bins.keys())

        # Calculate correlation if sufficient data
        if len(bin_counts) > 5 and np.std(bin_counts) > 0.01 and np.std(bin_metrics) > 0.01:
            correlation, p_value = stats.pearsonr(bin_counts, bin_metrics)
        else:
            correlation, p_value = 0.0, 1.0

        # Calculate mean and standard deviation
        mean_value = np.mean(metric_values)
        std_value = np.std(metric_values)

        # Create distribution bins
        distribution_bins = {}
        for i in range(int((bin_range[1] - bin_range[0]) / bin_size)):
            bin_start = bin_range[0] + (i * bin_size)
            bin_end = bin_start + bin_size
            bin_label = f"{bin_start:.2f}-{bin_end:.2f}"
            distribution_bins[bin_label] = 0

        # Bin the observations
        for r in all_results:
            metric_value = r[metric_name]
            bin_index = min(
                len(distribution_bins) - 1,
                int((metric_value - bin_range[0]) / bin_size)
            )
            bin_start = bin_range[0] + (bin_index * bin_size)
            bin_end = bin_start + bin_size
            bin_label = f"{bin_start:.2f}-{bin_end:.2f}"
            distribution_bins[bin_label] += r['individual_count']

        return {
            metric_name: {
                'mean': float(mean_value),
                'std': float(std_value),
                'correlation': float(correlation),
                'p_value': float(p_value)
            },
            'total_observations': len(all_results),
            'spatial_resolution': scale,
            'notes': "Correlations calculated based on observation frequency in habitat bins",
            'distribution': distribution_bins
        }

    def process_sample_results(
        self,
        point_sample_results: list,
        metric_names: list,
        default_year: int = 2000
    ) -> tuple:
        """Process sample results from Earth Engine point features.

        Args:
            point_sample_results (list): List of feature results from Earth Engine
            metric_names (list): List of metric names to extract from properties
            default_year (int): Default year to use if not specified

        Returns:
            tuple: (all_results, filtering_stats)
                - all_results: List of processed results with calculated metrics
                - filtering_stats: Dictionary with statistics about filtered points
        """
        all_results = []
        total_points = len(point_sample_results)
        filtered_points = 0
        batch_size = 1000  # Process points in batches to avoid collection size limit

        # Process results in batches
        for i in range(0, len(point_sample_results), batch_size):
            batch = point_sample_results[i:i + batch_size]

            for sample in batch:
                props = sample['properties']

                # Skip points with no data
                if not all(props.get(metric, 0) is not None for metric in metric_names):
                    filtered_points += 1
                    continue

                year = props.get('year', default_year)
                individual_count = props.get('individual_count', 1)

                # Create result dictionary with all metrics
                result = {
                    'year': year,
                    'individual_count': individual_count,
                    'geometry': sample.get('geometry', {})
                }

                # Add all metrics to the result
                for metric in metric_names:
                    result[metric] = props.get(metric, 0)

                all_results.append(result)

        # Create filtering statistics
        filtering_stats = {
            'total_filtered': filtered_points,
            'valid_observations': len(all_results),
            'total_observations': total_points,
            'filtered_at_earth_engine': True
        }

        self.logger.info("Processed %d observations (%d points filtered)",
                        len(all_results),
                        filtered_points)

        return all_results, filtering_stats

    def filter_marine_observations(self, observations: list) -> list:
        """Filter out observations that are over water bodies (oceans, seas).

        Args:
            observations (list): List of observation dictionaries with coordinates

        Returns:
            list: Filtered list of observations on land only
        """
        try:
            # Check for cancellation before starting
            self.check_cancellation()

            # Get the land mask from ESA WorldCover
            worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
            filtered_observations = []
            batch_size = 1000  # Process points in batches to avoid collection size limit

            # Process observations in batches
            for i in range(0, len(observations), batch_size):
                self.check_cancellation()
                batch = observations[i:i + batch_size]

                # Create EE points from batch
                points = [ee.Geometry.Point([obs["decimallongitude"], obs["decimallatitude"]])
                         for obs in batch]

                # Create a FeatureCollection from the points
                points_fc = ee.FeatureCollection([ee.Feature(point) for point in points])

                # Sample the worldcover values at each point
                points_with_landcover = worldcover.sampleRegions(
                    collection=points_fc,
                    properties=['system:index'],
                    scale=10
                )

                # Get the results
                results = points_with_landcover.getInfo()['features']

                # Filter observations based on the land cover value
                # ESA WorldCover class 80 is permanent water bodies
                for obs, result in zip(batch, results):
                    landcover_value = result['properties'].get('Map', None)
                    if landcover_value is not None and landcover_value != 80:  # Not water
                        filtered_observations.append(obs)

            self.logger.info(
                "Filtered %d marine observations out of %d total observations",
                len(observations) - len(filtered_observations),
                len(observations)
            )

            return filtered_observations

        except Exception as e:
            self.logger.error("Error filtering marine observations: %s", str(e))
            return observations  # Return original observations if filtering fails

    def filter_outlier_points(
        self,
        observations: list,
        std_dev_threshold: float = 2.0
    ) -> list:
        """Filter out points that are far from the main species range."""
        try:
            # Calculate centroid
            coords = [(obs["decimallongitude"], obs["decimallatitude"]) for obs in observations]
            centroid = (sum(x[0] for x in coords) / len(coords),
                       sum(x[1] for x in coords) / len(coords))

            # Calculate distances using Haversine formula
            distances = []
            for lon2, lat2 in coords:
                # Convert to radians
                points = [x * 3.14159 / 180 for x in [centroid[0], centroid[1], lon2, lat2]]

                # Haversine calculation
                diffs = (points[3] - points[1], points[2] - points[0])
                a = (diffs[0]/2)**2 + (1 - (diffs[0]/2)**2 - (diffs[1]/2)**2) * (diffs[1]/2)**2
                distances.append(6371 * 2 * (abs(a) ** 0.5))  # Earth's radius * arc

            # Filter points within threshold
            mean_dist = sum(distances) / len(distances)
            std_dist = (sum((d - mean_dist) ** 2 for d in distances) / len(distances)) ** 0.5
            threshold = mean_dist + (std_dev_threshold * std_dist)

            filtered = [obs for obs, dist in zip(observations, distances) if dist <= threshold]

            self.logger.info(
                "Filtered %d outlier observations out of %d total observations",
                len(observations) - len(filtered),
                len(observations)
            )
            return filtered

        except Exception as e:
            self.logger.error("Error filtering outlier points: %s", str(e))
            return observations
