"""Module for handling human modification data processing using Earth Engine."""

import logging
from typing import Dict, Any, Optional
import ee
from app.utils.alpha_shape_utils import AlphaShapeUtils
from .earth_engine_handler import EarthEngineHandler

class HumanModificationHandlerEE(EarthEngineHandler):
    """Handles human modification data processing and analysis using Google Earth Engine."""

    def __init__(self):
        """Initialize the HumanModificationHandlerEE."""
        super().__init__()
        self.logger = logging.getLogger("BioChat." + self.__class__.__name__)
        self.alpha_shape_utils = AlphaShapeUtils()

    def calculate_species_humanmod_correlation(
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

            all_alpha_shapes = []
            # Generate alpha shapes for visualization only
            # all_alpha_shapes = self.generate_alpha_shapes_for_visualization(
            #     observations,
            #     alpha=alpha,
            #     eps=eps,
            #     min_samples=min_samples,
            #     avoid_overlaps=avoid_overlaps
            # )

            # Sample human modification values at observation points
            all_results = self.process_ghm_sample_results(
                self.sample_ghm_at_points(ee_point_features, scale)
            )

            # Calculate correlations using all results
            if not all_results:
                self.logger.warning("No valid results were obtained from Earth Engine. "
                                    "Cannot calculate correlations.")
                return self.create_humanmod_error_response(
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
                'all_results': all_results,
                'ghm_layers': self.get_ghm_layers(
                    alpha_shapes=all_alpha_shapes,
                    avoid_overlaps=avoid_overlaps
                )
            }
        except (ValueError, ee.EEException) as e:
            self.logger.error(lambda: f"Error calculating human modification correlations: {str(e)}")
            return self.create_humanmod_error_response(
                species_name=species_name,
                observations=observations,
                all_alpha_shapes=all_alpha_shapes if 'all_alpha_shapes' in locals() else [],
                scale=scale,
                avoid_overlaps=avoid_overlaps,
                error_message=str(e),
                analysis_message="Analysis failed. Please try with different parameters "
                "or a smaller area."
            )

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

        # Select the gHM band explicitly
        ghm = ghm.select('gHM')

        # Process points in batches to avoid Earth Engine limit
        batch_size = 4000  # Keep under 5000 limit
        all_results = []

        try:
            # Process points in batches
            for i in range(0, len(ee_point_features), batch_size):
                batch_features = ee_point_features[i:i + batch_size]
                point_collection = ee.FeatureCollection(batch_features)

                self.logger.info("Sampling human modification for batch %d-%d of %d points",
                                i + 1, min(i + batch_size, len(ee_point_features)), len(ee_point_features))

                # Sample human modification at each point
                point_ghm_stats = ghm.reduceRegions(
                    collection=point_collection,
                    reducer=ee.Reducer.first(),
                    scale=scale,
                    tileScale=4
                )
                batch_results = point_ghm_stats.getInfo()['features']
                all_results.extend(batch_results)

                # Log the first result to check the structure
                if i == 0 and batch_results:
                    self.logger.info("Sample result structure: %s", batch_results[0])

                self.logger.info("Received %d sample results for batch", len(batch_results))

            self.logger.info("Total sample results: %d", len(all_results))
            return all_results

        except ee.EEException as e:
            self.logger.error("Earth Engine error: %s", str(e))
            raise

    def process_ghm_sample_results(self, point_sample_results: list) -> list:
        """Process human modification sample results from Earth Engine point features."""
        all_results, _ = self.process_sample_results(
            point_sample_results,
            metric_names=['first'],  # 'first' is the property name for gHM values
            default_year=2016
        )
        return all_results

    def calculate_ghm_correlations(self, all_results: list, scale: int) -> Dict[str, Any]:
        """Calculate correlations between species observations and human modification.

        Args:
            all_results (list): Processed results with human modification values for each point
            scale (int): Resolution in meters used for Earth Engine analysis

        Returns:
            dict: Correlation data including statistics for human modification
        """
        # Use base class method to calculate correlations
        correlation_data = self.calculate_correlations(
            all_results=all_results,
            scale=scale,
            metric_name='first',  # 'first' is the property name for gHM values
            metric_values=[r['first'] for r in all_results],
            bin_size=0.05,  # 5% bins for human modification (0-1 range)
            bin_range=(0, 1)  # Human modification ranges from 0 to 1
        )

        # Restructure the data to match the expected format
        return {
            'human_modification': correlation_data['first'],  # Rename 'first' to 'human_modification'
            'total_observations': correlation_data['total_observations'],
            'spatial_resolution': correlation_data['spatial_resolution'],
            'notes': correlation_data['notes'],
            'distribution': correlation_data['distribution']
        }

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

            # Alpha shape visualization
            alpha_shape_tiles = self.create_alpha_shape_visualization(alpha_shapes)

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

    def create_humanmod_error_response(
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
        # Define the human modification-specific correlation data structure
        correlation_data_structure = {
            'human_modification': {'mean': 0, 'std': 0, 'correlation': 0, 'p_value': 1.0},
            'total_observations': len(observations) if observations else 0,
            'spatial_resolution': scale,
            'error': error_message
        }

        # Get base error response
        error_response = super().create_error_response(
            species_name=species_name,
            observations=observations,
            all_alpha_shapes=all_alpha_shapes,
            scale=scale,
            avoid_overlaps=avoid_overlaps,
            error_message=error_message,
            analysis_message=analysis_message,
            correlation_data_structure=correlation_data_structure
        )

        # Add human modification-specific layers
        error_response['ghm_layers'] = self.get_ghm_layers(
            alpha_shapes=all_alpha_shapes,
            avoid_overlaps=avoid_overlaps
        )

        return error_response