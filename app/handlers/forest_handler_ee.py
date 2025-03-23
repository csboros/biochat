"""Module for handling forest cover and forest change data processing using Earth Engine."""

import logging
from typing import Dict, Any, Optional
import numpy as np
import ee
from scipy import stats
from app.handlers.earth_engine_handler import EarthEngineHandler

class ForestHandlerEE(EarthEngineHandler):
    """Handles forest data processing and analysis using Google Earth Engine."""

    def __init__(self):
        """Initialize the ForestHandlerEE."""
        super().__init__()
        self.logger = logging.getLogger("BioChat." + self.__class__.__name__)

    def calculate_species_forest_correlation_ee(
        self,
        species_name: str,
        min_observations: int = 10,
        alpha: float = 0.5,     # Alpha parameter for the alpha shape
        eps: float = 1.0,
        min_samples: int = 3,
        avoid_overlaps: bool = True,  # New parameter to control overlap avoidance
        scale: int = 200            # Resolution in meters for Earth Engine analysis
    ) -> Dict[str, Any]:
        """Calculate correlation between species observations and forest metrics using Earth Engine.

        This method samples forest metrics directly at each species observation point,
        providing an ecologically precise analysis of species-habitat relationships.
        The correlation is calculated based on observation frequency in different
        habitat types rather than using alpha shapes for analysis.

        Forest cover is defined using the Hansen dataset as tree canopy cover for year 2000,
        representing percentage of canopy closure for all vegetation taller than 5m in height.
        Forest cover is enhanced by incorporating forest gain data (2000-2012)
        into the calculations,
        which accounts for new forest growth that occurred after the baseline year.

        Forest gain is defined as the establishment of tree canopy from a non-forest state.
        The 'gain' band represents areas where tree canopy cover was established between 2000-2012,
        encoded as either 0 (no gain) or 1 (gain detected). In our analysis, we estimate that areas
        with gain represent an approximately 20% increase in tree cover where it occurs.

        Forest loss is defined as a stand-replacement disturbance, or a change from forest to
        non-forest state. The 'lossyear' band in the Hansen dataset is encoded as either 0 (no loss)
        or a value in the range 1-23, representing loss detected primarily in the year 2001-2023.

        Args:
            species_name (str): Scientific name of the species
            min_observations (int): Minimum number of observations required
            alpha (float): Alpha parameter for alpha shapes (used only for visualization)
            eps (float): DBSCAN epsilon parameter for clustering (visualization only)
            min_samples: Minimum samples for DBSCAN clustering (visualization only)
            avoid_overlaps (bool): Whether to merge overlapping alpha shapes (visualization only)
            scale: Resolution in meters for Earth Engine analysis (higher = faster)

        Returns:
            dict: Dictionary containing correlation results and statistics, including enhanced
                  forest cover values that incorporate both loss and gain data. Correlation
                  is based on observation frequency in habitat bins, not individual counts.
        """
        try:
            # Get species observations from BigQuery
            observations = self.get_species_observations(species_name, min_observations)

            # Instead of using alpha shapes, create point features from individual observations
            ee_point_features = self.create_ee_point_features(observations)

            # Generate alpha shapes for visualization only (completely separate process)
            all_alpha_shapes = []
            # all_alpha_shapes = self.generate_alpha_shapes_for_visualization(
            #     observations,
            #     alpha=alpha,
            #     eps=eps,
            #     min_samples=min_samples,
            #     avoid_overlaps=avoid_overlaps
            # )

            # Sample forest metrics at observation points
            all_results = self.process_sample_results(
                self.sample_forest_metrics_at_points(ee_point_features, scale)
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
            correlation_data = self.calculate_correlations(all_results, scale)
            print(correlation_data)

            # Create prompt and get analysis from LLM
            analysis = self.send_to_llm(
                self.create_analysis_prompt(species_name, correlation_data)
            )

            # Return with alpha shapes included, but note these are just for visualization
            return {
                'correlation_data': correlation_data,
                'analysis': analysis,
                'all_results': all_results,
                'species_name': species_name,
                'observations': observations,
 #               'alpha_shapes': all_alpha_shapes,
                'forest_layers': self.get_forest_layers(
 #                   alpha_shapes=all_alpha_shapes,
                    avoid_overlaps=avoid_overlaps
                )
            }
        except (ValueError, ee.EEException) as e:
            self.logger.error(lambda: f"Error calculating correlations with Earth Engine: {str(e)}")
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

    def calculate_correlations(self, all_results: list, scale: int) -> Dict[str, Any]:
        """Calculate correlations between species observations and forest metrics.

        Args:
            all_results (list): Processed results with forest metrics for each observation point
            scale (int): Resolution in meters used for Earth Engine analysis

        Returns:
            dict: Correlation data including statistics for forest cover and forest loss
        """
        individual_counts = [r['individual_count'] for r in all_results]
        remaining_covers = [r['remaining_cover'] for r in all_results]
        forest_losses = [r['forest_loss'] for r in all_results]

        # Log the values for debugging
        self.logger.info("Calculating correlations with %s data points", len(individual_counts))
        self.logger.info("Remaining covers range: %s to %s",
                         min(remaining_covers) if remaining_covers else 0,
                         max(remaining_covers) if remaining_covers else 0)
        self.logger.info("Forest losses range: %s to %s",
                         min(forest_losses) if forest_losses else 0,
                         max(forest_losses) if forest_losses else 0)

        # Create distribution bins for forest cover (10 equal bins from 0 to 100%)
        forest_cover_bins = {}
        for i in range(10):  # 10 bins: 0-10%, 10-20%, ..., 90-100%
            bin_start = i * 10
            bin_end = (i + 1) * 10
            bin_label = f"{bin_start}-{bin_end}%"
            forest_cover_bins[bin_label] = 0

        # Create distribution bins for forest loss (2 bins: 0=no loss, 1=loss)
        forest_loss_bins = {
            "No Loss": 0,
            "Loss": 0
        }

        # Bin the observations
        for r in all_results:
            # Bin forest cover (0-100%)
            cover_value = r['remaining_cover']
            bin_index = min(9, int(cover_value / 10))  # Ensure max index is 9
            bin_start = bin_index * 10
            bin_end = (bin_index + 1) * 10
            bin_label = f"{bin_start}-{bin_end}%"
            forest_cover_bins[bin_label] += r['individual_count']

            # Bin forest loss (binary)
            if r['forest_loss'] > 0:
                forest_loss_bins["Loss"] += r['individual_count']
            else:
                forest_loss_bins["No Loss"] += r['individual_count']

        # Handle cases where there's no variation in the data
        try:
            # Use standard deviation to check for meaningful variation
            remaining_covers_std = np.std(remaining_covers) if remaining_covers else 0
            forest_losses_std = np.std(forest_losses) if forest_losses else 0

            # Log the standard deviations for debugging
            self.logger.info("Standard deviations - remaining_covers: %.5f, "
                            "forest_losses: %.5f",
                            remaining_covers_std,
                            forest_losses_std)

            # Create observation counts by location and habitat characteristics
            obs_bins = {}
            # Include more precision for forest metrics to capture ecological patterns
            for r in all_results:
                # Create a location key (rounded to 2 decimal places for binning)
                loc_key = (
                    round(r.get('remaining_cover', 0), 2),
                    round(r.get('forest_loss', 0), 2)
                )
                # Add the individual_count instead of just incrementing by 1
                obs_bins[loc_key] = obs_bins.get(loc_key, 0) + r.get('individual_count', 1)

            # Create arrays for correlation calculation using observation frequency
            bin_counts = list(obs_bins.values())
            bin_covers = [k[0] for k in obs_bins]
            bin_losses = [k[1] for k in obs_bins]

            # Log information about the bins
            self.logger.info("Created %d distinct habitat bins "
                            "for correlation analysis", len(bin_counts))
            self.logger.info("Observation frequency range: %s to %s",
                            min(bin_counts) if bin_counts else 0,
                            max(bin_counts) if bin_counts else 0)

            # Make sure we have sufficient data for correlation
            if len(bin_counts) > 5 and np.std(bin_counts) > 0.01 and np.std(bin_covers) > 0.01:
                tree_cover_corr, tree_cover_p = stats.pearsonr(bin_counts, bin_covers)
                self.logger.info("Successfully calculated tree cover "
                            "correlation using observation frequency: %.3f (p=%.3f)",
                            tree_cover_corr, tree_cover_p)
            else:
                tree_cover_corr, tree_cover_p = 0.0, 1.0
                self.logger.warning("Insufficient variation "
                    "in binned data. Setting tree cover correlation to 0.")

            # Same for forest loss
            if len(bin_counts) > 5 and np.std(bin_counts) > 0.01 and np.std(bin_losses) > 0.01:
                forest_loss_corr, forest_loss_p = stats.pearsonr(bin_counts, bin_losses)
                self.logger.info("Successfully calculated forest "
                                "loss correlation using observation "
                                "frequency: %.3f (p=%.3f)",
                                forest_loss_corr, forest_loss_p)
            else:
                forest_loss_corr, forest_loss_p = 0.0, 1.0
                self.logger.warning("Insufficient variation in binned data. "
                                    "Setting forest loss correlation to 0.")

            # Set the calculation method note
            correlation_data_notes = "Correlations calculated based on observation frequency in habitat bins"

        except Exception as e: # pylint: disable=broad-except
            self.logger.error("Error calculating correlations: %s", str(e))
            tree_cover_corr, tree_cover_p = 0.0, 1.0
            forest_loss_corr, forest_loss_p = 0.0, 1.0
            correlation_data_notes = "Error calculating correlations."

        correlation_data = {
            'forest_cover': {
                'mean':
                    sum(remaining_covers) / len(remaining_covers) if remaining_covers else 0.0,
                'std':
                    stats.tstd(remaining_covers) if len(set(remaining_covers)) > 1 else 0.0,
                'correlation': tree_cover_corr,
                'p_value': tree_cover_p
            },
            'forest_loss': {
                'mean': sum(forest_losses) / len(forest_losses) if forest_losses else 0.0,
                'std': stats.tstd(forest_losses) if len(set(forest_losses)) > 1 else 0.0,
                'correlation': forest_loss_corr,
                'p_value': forest_loss_p
            },
            'forest_gain_applied': True,  # Indicate that gain is now included in calculations
            'total_observations': len(all_results),
            'spatial_resolution': scale,
            'notes': correlation_data_notes,  # Add notes about calculation method

            # Add the binned distribution data
            'forest_metrics_distribution': {
                'forest_cover_bins': forest_cover_bins,
                'forest_loss_bins': forest_loss_bins
            }
        }
        return correlation_data

    def process_sample_results(self, point_sample_results: list) -> list:
        """Process sample results from Earth Engine point features with temporal awareness."""
        all_results = []
        for sample in point_sample_results:
            props = sample['properties']
            year = props.get('year', 2000)  # Observation year
            individual_count = props.get('individual_count', 1)

            # Extract tree cover and forest-related properties
            tree_cover = props.get('treecover2000', 0)
            lossyear = props.get('lossyear', 0)  # Year of loss (1-23 for 2001-2023)
            forest_mask = props.get('forest_mask', 0)
            gain_mask = props.get('gain_mask', 0)

            # Check inputs for validity
            if tree_cover is None or tree_cover < 0:
                tree_cover = 0
            if lossyear is None:
                lossyear = 0
            if forest_mask is None:
                forest_mask = 0
            if gain_mask is None:
                gain_mask = 0

            # Critical temporal comparison
            # Convert lossyear to actual calendar year (0 = no loss, 1-23 = 2001-2023)
            actual_loss_year = 2000 + lossyear if lossyear > 0 else 0

            # Was the forest lost before or after the observation?
            if actual_loss_year == 0 or year < actual_loss_year:
                # No loss occurred or observation was before loss
                # Calculate forest cover at time of observation
                forest_loss = 0.0  # No loss at time of observation

                # Calculate remaining cover with gain if applicable
                if gain_mask > 0 and year >= 2012:  # Observation after potential gain period
                    remaining_cover = min(100, tree_cover + (gain_mask * 20.0))
                else:
                    remaining_cover = tree_cover
            else:
                # Loss occurred before observation - area was already deforested
                forest_loss = 1.0  # Complete loss at time of observation

                # Remaining cover would be 0 or modified by gain
                if gain_mask > 0 and year >= 2012:
                    remaining_cover = min(100, gain_mask * 20.0)  # Only gain contributes
                else:
                    remaining_cover = 0  # No forest remained at observation time

            # Store temporally-adjusted results
            all_results.append({
                'year': year,
                'individual_count': individual_count,
                'tree_cover_2000': tree_cover,
                'remaining_cover': remaining_cover,
                'forest_loss': forest_loss,
                'lossyear': actual_loss_year,  # Store actual calendar year of loss
                'geometry': sample.get('geometry', {})
            })
        return all_results

    def get_forest_layers(
        self,
        alpha_shapes: Optional[list] = None,  # Parameter for alpha shapes
        avoid_overlaps: bool = True  # New parameter to control overlap avoidance
    ) -> Dict[str, Any]:
        """Get forest cover, loss, and alpha shape visualization layers.

        Args:
            alpha_shapes (list, optional): List of alpha shape GeoJSON polygons to visualize
            avoid_overlaps (bool): Whether to merge overlapping alpha shapes (default: True)

        Returns:
            dict: Dictionary containing map visualization URLs for different layers
        """
        try:
            # Load Hansen dataset
            hansen = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')

            # Forest cover 2000 visualization - Make white areas transparent
            forest_cover = hansen.select(['treecover2000'])
            # Create a mask where treecover is greater than 0
            forest_mask = forest_cover.gt(0)
            # Apply the mask to make areas with no forest (value = 0) transparent
            forest_cover_masked = forest_cover.updateMask(forest_mask)
            forest_cover_vis = forest_cover_masked.visualize(
                min=0,
                max=100,
                palette=['00ff00']  # Only green for areas with forest
            )

            # Forest loss visualization - using the same approach as
            # in calculate_species_forest_correlation_ee
            # Check if 'loss' band exists in the Hansen dataset
            hansen_bands = hansen.bandNames().getInfo()
            if 'loss' in hansen_bands:
                # If 'loss' band exists, use it directly
                self.logger.info("Using 'loss' band directly for visualization")
                loss_mask = hansen.select('loss')
            else:
                # If 'loss' band doesn't exist, create it from 'lossyear'
                self.logger.info("Creating binary loss from 'lossyear' for visualization")
                loss_year = hansen.select(['lossyear'])
                # Show all loss (2001-2023)
                loss_mask = loss_year.gt(0)

            # Apply the mask to make areas with no loss transparent
            loss_vis = loss_mask.selfMask().visualize(
                min=0,
                max=1,
                palette=['ff0000']  # Red for forest loss
            )

            # Forest gain visualization (2000-2012)
            gain = hansen.select(['gain'])
            # Apply mask to make areas with no gain transparent
            gain_vis = gain.selfMask().visualize(
                min=0,
                max=1,
                palette=['0000ff']  # Blue for forest gain
            )

            # Get map IDs and tokens
            forest_cover_layer = forest_cover_vis.getMapId()
            loss_layer = loss_vis.getMapId()
            gain_layer = gain_vis.getMapId()

            # Alpha shape visualization (new code)
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
                'forest_cover': {
                    'tiles': [forest_cover_layer['tile_fetcher'].url_format],
                    'attribution': 'Hansen/UMD/Google/USGS/NASA'
                },
                'forest_loss': {
                    'tiles': [loss_layer['tile_fetcher'].url_format],
                    'attribution': 'Hansen/UMD/Google/USGS/NASA'
                },
                'forest_gain': {
                    'tiles': [gain_layer['tile_fetcher'].url_format],
                    'attribution': 'Hansen/UMD/Google/USGS/NASA'
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
            self.logger.error(lambda: f"Earth Engine error in get_forest_layers: {str(e)}")
            raise
        except (ValueError, TypeError, KeyError, IndexError) as e:
            self.logger.error(lambda: f"Error creating forest layers: {str(e)}")
            raise

    def create_analysis_prompt(self, species_name: str, correlation_data: Dict[str, Any]) -> str:
        """Generate a prompt for LLM analysis of species-forest correlations.

        Args:
            species_name (str): Scientific name of the species being analyzed
            correlation_data (dict): Dictionary containing correlation results and statistics

        Returns:
            str: Formatted prompt text for the LLM
        """
        prompt = """You are a conservation biology expert. Analyze this species
            correlation data between species occurrence and forest cover and forest loss.

        Context:
        - Correlation shows if species occur more (+ve) or less (-ve) in human-impacted areas
        - Forest cover is defined as tree canopy closure for all vegetation taller than 5m in height
          (measured as percentage 0-100 in the year 2000)
        - Remaining forest cover is calculated by applying detected forest loss to the initial tree cover percentage
          and then adding estimated forest gain where detected
        - Forest gain represents areas where tree cover was established between 2000-2012, and we estimate
          this adds approximately 20% tree cover in areas where gain is detected
        - Forest loss represents areas where tree cover was removed, defined as a
          stand-replacement disturbance or change from forest to non-forest state between 2001-2023
        - IMPORTANT: Forest loss values are expressed as a decimal between 0-1, where 1.0 equals 100% loss
          and 0.0 equals 0% loss. For example, a forest_loss mean of 0.7168 means 71.68% of forest was lost.
        """

        if species_name:
            prompt += f"\nAnalyzing correlations for species: {species_name}\n"

        # Add instructions to analyze the binned distribution data
        prompt += """
        Please analyze:
        1. Please show average forest cover and forest loss for the species
        2. Overall correlation patterns
        3. Conservation implications of both average forest cover and average forest loss on one hand side and
           correlation between forest cover, forest loss and species occurrence on the other hand side
        4. Data limitations and caveats
        5. Additionally, analyze the distribution pattern of observations across forest metrics:
           - Examine the forest_metrics_distribution data showing frequency of observations in different forest cover bins (0-10%, 10-20%, etc.)
           - Look for peaks, gaps, or thresholds in the forest cover distribution that might indicate habitat preferences
           - Consider the proportion of observations in areas with and without forest loss (Loss vs No Loss bins)
           - Interpret what these distributions reveal about the species' habitat requirements beyond simple averages
        6. Please summarize the results in a few sentences and show the summary at the end of your response.

        Data:
        """
        prompt += f"\n{correlation_data}"

        # Create a version with explicit percentages
        percentage_data = {
            'forest_cover': {
                'mean_percent': correlation_data['forest_cover']['mean'],
                'std_percent': correlation_data['forest_cover']['std'],
                'correlation': correlation_data['forest_cover']['correlation'],
                'p_value': correlation_data['forest_cover']['p_value']
            },
            'forest_loss': {
                'mean_percent': correlation_data['forest_loss']['mean'] * 100,
                'std_percent': correlation_data['forest_loss']['std'] * 100,
                'correlation': correlation_data['forest_loss']['correlation'],
                'p_value': correlation_data['forest_loss']['p_value']
            },
            'total_observations': correlation_data['total_observations'],
            'spatial_resolution': correlation_data['spatial_resolution']
        }

        prompt += """

        Here's the same data with all values expressed as percentages:
        """
        prompt += f"\n{percentage_data}"

        # Add a clarification about the percentage values and forest gain
        prompt += """

        Note on interpretation:
        - Forest cover mean_percent is already a percentage (0-100%)
        - Forest loss mean_percent is the percentage of forest lost (0-100%)
        - The original forest_loss mean value is in decimal form (0-1)
        - Forest gain (2000-2012) has been incorporated into the forest cover calculations

        When analyzing the forest_metrics_distribution:
        - The forest_cover_bins show observation counts in each 10% interval from 0-100% cover
        - The forest_loss_bins show counts of observations in areas with loss vs. no loss
        - These distributions may reveal important ecological patterns that aren't captured by simple averages
        - Look for multiple peaks that might indicate the species uses different habitat types
        - Consider whether the species shows threshold responses to forest cover
        """
        return prompt

    def sample_forest_metrics_at_points(self, ee_point_features: list, scale: int) -> list:
        """Prepare Earth Engine data and sample forest metrics at observation points."""
        # Load Hansen dataset
        hansen = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')

        # Extract the lossyear band directly (contains years 1-23 for 2001-2023)
        lossyear = hansen.select('lossyear')

        # Create a mask for meaningful tree cover (areas with at least some forest)
        forest_mask = hansen.select('treecover2000').gte(10)

        # Binary gain mask
        binary_gain = hansen.select('gain')

        # Combine all bands
        combined = hansen.select(['treecover2000']).addBands(lossyear.rename('lossyear'))
        combined = combined.addBands(binary_gain.rename('gain_mask'))
        combined = combined.addBands(forest_mask.rename('forest_mask'))

        # Create feature collection and sample
        try:
            point_collection = ee.FeatureCollection(ee_point_features)
            self.logger.info("Sampling forest metrics at %d individual observation points",
                             len(ee_point_features))

            # Sample forest metrics at each point
            point_forest_stats = combined.reduceRegions(
                collection=point_collection,
                reducer=ee.Reducer.first(),
                scale=scale,
                tileScale=4
            )
            point_sample_results = point_forest_stats.getInfo()['features']
            self.logger.info("Received %d sample results for individual points",
                             len(point_sample_results))

            return point_sample_results

        except ee.EEException as e:
            self.logger.error("Earth Engine error: %s", str(e))
            raise

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
                'forest_cover': {'mean': 0, 'std': 0, 'correlation': 0, 'p_value': 1.0},
                'forest_loss': {'mean': 0, 'std': 0, 'correlation': 0, 'p_value': 1.0},
                'forest_gain_applied': False,
                'total_observations': len(observations) if observations else 0,
                'spatial_resolution': scale,
                'error': error_message
            },
            'analysis': analysis_message,
            'species_name': species_name,
            'observations': observations,
            'alpha_shapes': all_alpha_shapes,
            'forest_layers': self.get_forest_layers(
                alpha_shapes=all_alpha_shapes,
                avoid_overlaps=avoid_overlaps
            ),
            'error': True
        }
