"""Module for handling human modification data processing using Earth Engine."""

import os
import logging
from typing import Dict, Any, Optional
import numpy as np
import ee
from google.cloud import bigquery
from scipy import stats
from app.utils.alpha_shape_utils import AlphaShapeUtils
from app.handlers.earth_engine_handler import EarthEngineHandler

class HumanModificationHandlerEE(EarthEngineHandler):
    """Handles human modification data processing and analysis using Google Earth Engine."""

    def __init__(self):
        """Initialize the HumanModificationHandlerEE."""
        super().__init__()
        self.logger = logging.getLogger("BioChat." + self.__class__.__name__)
        self.alpha_shape_utils = AlphaShapeUtils()

    def calculate_species_humanmod_correlation_ee(
        self,
        species_name: str,
        min_observations: int = 10,
        alpha: float = 0.5,     # Alpha parameter for the alpha shape
        eps: float = 1.0,
        min_samples: int = 3,
        avoid_overlaps: bool = True,  # New parameter to control overlap avoidance
        scale: int = 1000       # Resolution in meters for Earth Engine analysis (gHM is 1km)
    ) -> Dict[str, Any]:
        """Calculate correlation between species observations and human modification using Earth Engine.
        
        This method samples human modification metrics directly at each species observation point,
        providing an analysis of species distribution across human-modified landscapes.
        The correlation is calculated based on observation frequency in different
        habitat types, accounting for individual counts at each location.
        
        Human modification is defined using the global Human Modification dataset (gHM),
        which provides a cumulative measure of human impact on terrestrial lands at a 1km resolution.
        Values range from 0.0 (no modification) to 1.0 (complete modification) and incorporate:
          - human settlement (population density, built-up areas)
          - agriculture (cropland, livestock)
          - transportation (major, minor, and two-track roads; railroads)
          - mining and energy production
          - electrical infrastructure (power lines, nighttime lights)
        
        Args:
            species_name (str): Scientific name of the species
            min_observations (int): Minimum number of observations required
            alpha (float): Alpha parameter for alpha shapes (used only for visualization)
            eps (float): DBSCAN epsilon parameter for clustering (visualization only)
            min_samples: Minimum samples for DBSCAN clustering (visualization only)
            avoid_overlaps (bool): Whether to merge overlapping alpha shapes (visualization only)
            scale: Resolution in meters for Earth Engine analysis (gHM is 1km resolution)
            
        Returns:
            dict: Dictionary containing correlation results and statistics for the relationship
                  between species occurrence and human modification index.
        """
        try:
            # Get species observations from BigQuery
            observations = self.get_species_observations(species_name, min_observations)

            # Create point features from individual observations
            ee_point_features = self.create_ee_point_features(observations)

            # Generate alpha shapes for visualization only
            all_alpha_shapes = self.generate_alpha_shapes_for_visualization(
                observations,
                alpha=alpha,
                eps=eps,
                min_samples=min_samples,
                avoid_overlaps=avoid_overlaps
            )

            # Sample human modification values at observation points
            all_results = self.process_ghm_sample_results(
                self.sample_ghm_at_points(ee_point_features, scale)
            )

            # Calculate correlations using all results
            if not all_results:
                self.logger.warning("No valid results were obtained from Earth Engine. "
                                    "Cannot calculate correlations.")
                return self.create_error_response(
                    species_name=species_name,
                    observations=observations,
                    all_alpha_shapes=all_alpha_shapes,
                    scale=scale,
                    avoid_overlaps=avoid_overlaps
                )

            # Calculate correlation data from the results
            correlation_data = self.calculate_ghm_correlations(all_results, scale)
            print(correlation_data)

            # Create prompt and get analysis from LLM
            analysis = self.send_to_llm(
                self.create_ghm_analysis_prompt(species_name, correlation_data)
            )

            # Return with alpha shapes included for visualization
            return {
                'correlation_data': correlation_data,
                'analysis': analysis,
                'species_name': species_name,
                'observations': observations, 
                'alpha_shapes': all_alpha_shapes,
                'ghm_layers': self.get_ghm_layers(
                    alpha_shapes=all_alpha_shapes,
                    avoid_overlaps=avoid_overlaps
                )
            }
        except (ValueError, ee.EEException) as e:
            self.logger.error(lambda: f"Error calculating human modification correlations: {str(e)}")
            return self.create_error_response(
                species_name=species_name,
                observations=observations,
                all_alpha_shapes=all_alpha_shapes if 'all_alpha_shapes' in locals() else [],
                scale=scale,
                avoid_overlaps=avoid_overlaps,
                error_message=str(e),
                analysis_message="Analysis failed. Please try with different parameters "
                "or a smaller area."
            )

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

    def sample_ghm_at_points(self, ee_point_features: list, scale: int) -> list:
        """Sample human modification values at each observation point.
        
        Args:
            ee_point_features (list): List of Earth Engine point features
            scale (int): Resolution in meters for Earth Engine analysis
            
        Returns:
            list: Sample results containing human modification values for each point
            
        Raises:
            ee.EEException: If Earth Engine encounters an error during processing
        """
        # Load Global Human Modification dataset
        ghm = ee.ImageCollection("CSP/HM/GlobalHumanModification").first()
        
        # Sample the dataset at each observation point
        try:
            # Create a feature collection from individual point features
            point_collection = ee.FeatureCollection(ee_point_features)
            self.logger.info("Sampling human modification at %d individual observation points",
                            len(ee_point_features))

            # Sample human modification at each point
            point_ghm_stats = ghm.reduceRegions(
                collection=point_collection,
                reducer=ee.Reducer.first(),
                scale=scale,
                tileScale=4
            )
            point_sample_results = point_ghm_stats.getInfo()['features']
            self.logger.info("Received %d sample results for individual points",
                            len(point_sample_results))

            return point_sample_results

        except ee.EEException as e:
            self.logger.error("Earth Engine error: %s", str(e))
            raise

    def process_ghm_sample_results(self, point_sample_results: list) -> list:
        """Process human modification sample results from Earth Engine point features.
        
        Args:
            point_sample_results (list): List of feature results from Earth Engine
            
        Returns:
            list: Processed results with human modification values for each point
        """
        all_results = []
        for sample in point_sample_results:
            props = sample['properties']
            year = props.get('year', 2016)  # Default year for ghm dataset
            individual_count = props.get('individual_count', 1)

            # Extract human modification value
            ghm_value = props.get('gHM', 0)

            # Check inputs for validity
            if ghm_value is None:
                ghm_value = 0

            # Store each point result separately
            all_results.append({
                'year': year,
                'individual_count': individual_count,
                'ghm_value': ghm_value,
                'geometry': sample.get('geometry', {})  # Store point geometry for debugging
            })
        return all_results

    def calculate_ghm_correlations(self, all_results: list, scale: int) -> Dict[str, Any]:
        """Calculate correlations between species observations and human modification.
        
        Args:
            all_results (list): Processed results with human modification values for each point
            scale (int): Resolution in meters used for Earth Engine analysis
            
        Returns:
            dict: Correlation data including statistics for human modification
        """
        individual_counts = [r['individual_count'] for r in all_results]
        ghm_values = [r['ghm_value'] for r in all_results]

        # Log the values for debugging
        self.logger.info("Calculating correlations with %s data points", len(individual_counts))
        self.logger.info("Human modification values range: %s to %s",
                        min(ghm_values) if ghm_values else 0, 
                        max(ghm_values) if ghm_values else 0)

        # Handle cases where there's no variation in the data
        try:
            # Use standard deviation to check for meaningful variation
            ghm_values_std = np.std(ghm_values) if ghm_values else 0

            # Log the standard deviation for debugging
            self.logger.info("Standard deviation - human modification: %.5f",
                            ghm_values_std)

            # Create observation counts by location and habitat characteristics
            obs_bins = {}
            # Include precision for human modification to capture ecological patterns
            for r in all_results:
                # Create a location key (rounded to 2 decimal places for binning)
                loc_key = round(r.get('ghm_value', 0), 2)
                # Add the individual_count instead of just incrementing by 1
                obs_bins[loc_key] = obs_bins.get(loc_key, 0) + r.get('individual_count', 1)

            # Create arrays for correlation calculation using observation frequency
            bin_counts = list(obs_bins.values())
            bin_ghm = list(obs_bins.keys())

            # Log information about the bins
            self.logger.info("Created %d distinct habitat bins "
                            "for correlation analysis", len(bin_counts))
            self.logger.info("Observation frequency range: %s to %s",
                            min(bin_counts) if bin_counts else 0,
                            max(bin_counts) if bin_counts else 0)

            # Make sure we have sufficient data for correlation
            if len(bin_counts) > 5 and np.std(bin_counts) > 0.01 and np.std(bin_ghm) > 0.01:
                ghm_corr, ghm_p = stats.pearsonr(bin_counts, bin_ghm)
                self.logger.info("Successfully calculated human modification "
                            "correlation using observation frequency: %.3f (p=%.3f)",
                            ghm_corr, ghm_p)
            else:
                ghm_corr, ghm_p = 0.0, 1.0
                self.logger.warning("Insufficient variation "
                    "in binned data. Setting human modification correlation to 0.")

            # Set the calculation method note
            correlation_data_notes = "Correlations calculated based on observation frequency in habitat bins"

        except Exception as e: # pylint: disable=broad-except
            self.logger.error("Error calculating correlations: %s", str(e))
            ghm_corr, ghm_p = 0.0, 1.0
            correlation_data_notes = "Error calculating correlations."

        correlation_data = {
            'human_modification': {
                'mean': 
                    sum(ghm_values) / len(ghm_values) if ghm_values else 0.0,
                'std': 
                    stats.tstd(ghm_values) if len(set(ghm_values)) > 1 else 0.0,
                'correlation': ghm_corr,
                'p_value': ghm_p
            },
            'total_observations': len(all_results),
            'spatial_resolution': scale,
            'notes': correlation_data_notes
        }
        return correlation_data

    def get_ghm_layers(
        self,
        alpha_shapes: Optional[list] = None,
        avoid_overlaps: bool = True
    ) -> Dict[str, Any]:
        """Get human modification and alpha shape visualization layers.
        
        Args:
            alpha_shapes (list, optional): List of alpha shape GeoJSON polygons to visualize
            avoid_overlaps (bool): Whether to merge overlapping alpha shapes (default: True)
            
        Returns:
            dict: Dictionary containing map visualization URLs for different layers
        """
        try:
            # Load Global Human Modification dataset
            ghm = ee.ImageCollection("CSP/HM/GlobalHumanModification").first()

            # Human modification visualization
            ghm_vis = ghm.visualize(
                min=0.0,
                max=1.0,
                palette=['0c0c0c', '071aff', 'ff0000', 'ffbd03', 'fbff05', 'fffdfd']
            )

            # Get map ID and token
            ghm_layer = ghm_vis.getMapId()

            # Alpha shape visualization (same as in forest handler)
            alpha_shape_tiles = None
            if alpha_shapes and len(alpha_shapes) > 0:
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
                except Exception as e: # pylint: disable=broad-except
                    self.logger.error("Failed to create alpha shape visualization: %s", str(e))

            # Construct the result dictionary with all layers
            result = {
                'human_modification': {
                    'tiles': [ghm_layer['tile_fetcher'].url_format],
                    'attribution': 'Conservation Science Partners'
                }
            }

            # Add alpha shapes layer if available
            if alpha_shape_tiles:
                result['alpha_shapes'] = {
                    'tiles': alpha_shape_tiles,
                    'attribution': 'Species Range Analysis',
                    'count': len(alpha_shapes),
                    'non_overlapping': avoid_overlaps
                }
            return result

        except ee.EEException as e:
            self.logger.error(lambda: f"Earth Engine error in get_ghm_layers: {str(e)}")
            raise
        except (ValueError, TypeError, KeyError, IndexError) as e:
            self.logger.error(lambda: f"Error creating GHM layers: {str(e)}")
            raise

    def create_ghm_analysis_prompt(self, species_name: str, correlation_data: Dict[str, Any]) -> str:
        """Generate a prompt for LLM analysis of species-human modification correlations.
        
        Args:
            species_name (str): Scientific name of the species being analyzed
            correlation_data (dict): Dictionary containing correlation results and statistics
            
        Returns:
            str: Formatted prompt text for the LLM
        """
        prompt = """You are a conservation biology expert. Analyze this species
        correlation data between species occurrence and human modification.

        Context:
        - Correlation shows if species occur more (+ve) or less (-ve) in human-modified areas
        - Human modification index (gHM) ranges from 0.0 (no modification) to 1.0 (complete modification)
        - This index measures cumulative human impacts from multiple sources, including:
          * human settlement (population density, built-up areas)
          * agriculture (cropland, livestock)
          * transportation (roads, railroads)
          * mining and energy production
          * electrical infrastructure (power lines, nighttime lights)
        """

        if species_name:
            prompt += f"\nAnalyzing correlations for species: {species_name}\n"

        prompt += """
        Please analyze:
        1. Please show average human modification value for the species' habitat
        2. Overall correlation patterns between species occurrence and human modification
        3. Conservation implications of these findings
        4. Data limitations and caveats
        5. Please summarize the results in a few sentences and show the summary at the end of your response.

        Data:
        """
        prompt += f"\n{correlation_data}"

        # Create a version with explicit percentages
        percentage_data = {
            'human_modification': {
                'mean_percent': correlation_data['human_modification']['mean'] * 100,
                'std_percent': correlation_data['human_modification']['std'] * 100,
                'correlation': correlation_data['human_modification']['correlation'],
                'p_value': correlation_data['human_modification']['p_value']
            },
            'total_observations': correlation_data['total_observations'],
            'spatial_resolution': correlation_data['spatial_resolution']
        }

        prompt += """
        
        Here's the same data with values expressed as percentages:
        """
        prompt += f"\n{percentage_data}"

        # Add a clarification about the percentage values
        prompt += """

        Note on interpretation:
        - Human modification mean_percent represents the percentage of modification (0-100%)
        - The original human_modification mean value is in decimal form (0-1)
        - Positive correlation indicates species is more prevalent in human-modified landscapes
        - Negative correlation indicates species avoids human-modified landscapes
        """
        return prompt


    def create_error_response(
        self,
        species_name: str,
        observations: list,
        all_alpha_shapes: list,
        scale: int,
        avoid_overlaps: bool,
        error_message: str = "No valid results were obtained from Earth Engine analysis.",
        analysis_message: str = "Analysis failed. Please try with different parameters or a smaller area."
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
            
        Returns:
            dict: Structured error response with consistent format
        """
        return {
            'correlation_data': {
                'human_modification': {'mean': 0, 'std': 0, 'correlation': 0, 'p_value': 1.0},
                'total_observations': len(observations) if observations else 0,
                'spatial_resolution': scale,
                'error': error_message
            },
            'analysis': analysis_message,
            'species_name': species_name,
            'observations': observations,
            'alpha_shapes': all_alpha_shapes,
            'ghm_layers': self.get_ghm_layers(
                alpha_shapes=all_alpha_shapes,
                avoid_overlaps=avoid_overlaps
            ),
            'error': True
        }