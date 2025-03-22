"""Module for handling forest cover and forest change data processing using Earth Engine."""

import os
import logging
from typing import Dict, Any, Optional
import numpy as np
import ee
from google.cloud import bigquery
from scipy import stats
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
from app.utils.alpha_shape_utils import AlphaShapeUtils

class ForestHandlerEE:
    """Handles forest data processing and analysis using Google Earth Engine."""

    def __init__(self):
        """Initialize the ForestHandlerEE."""
        self.logger = logging.getLogger("BioChat." + self.__class__.__name__)
        self.alpha_shape_utils = AlphaShapeUtils()

        # Initialize Earth Engine
        try:
            ee.Initialize()
        except ee.EEException as e:
            self.logger.error("Failed to initialize Earth Engine: %s", str(e))
            raise

    def calculate_species_forest_correlation_ee(
        self,
        species_name: str,
        min_observations: int = 10,
        alpha: float = 0.5,     # Alpha parameter for the alpha shape
        max_alpha_shapes: int = 5,  # Reduced from 10 to 5 for better performance
        eps: float = 1.0,            # New parameter
        min_samples: int = 3,        # New parameter
        avoid_overlaps: bool = True,  # New parameter to control overlap avoidance
        scale: int = 100            # Resolution in meters for Earth Engine analysis
    ) -> Dict[str, Any]:
        """Calculate correlation between species observations and forest metrics using Earth Engine.
        
        This method samples forest metrics across the species range defined by an alpha shape,
        providing an ecologically relevant analysis of species-forest relationships.
        
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
            batch_size (int): Number of observations to process in each batch
            alpha (float): Alpha parameter for the alpha shape (lower values create tighter shapes)
            max_alpha_shapes (int): Maximum number of alpha shapes to create
            eps (float): DBSCAN epsilon parameter - smaller values create smaller clusters
            min_samples: Minimum samples for DBSCAN - smaller values allow smaller clusters
            avoid_overlaps (bool): Whether to merge overlapping alpha shapes (default: True)
            scale: Resolution in meters for Earth Engine analysis (higher = faster)
            
        Returns:
            dict: Dictionary containing correlation results and statistics, including enhanced
                  forest cover values that incorporate both loss and gain data
        """
        try:
            # Get species observations from BigQuery
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

            # Load Hansen dataset
            hansen = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')


            # Get the observation count to decide on strategy
            obs_count = len(observations)

            # For species with large ranges (like orangutans), adjust parameters
            if obs_count > 500:
                self.logger.info("Large dataset detected (%s observations). "
                                 "Adjusting parameters.", obs_count)
                max_alpha_shapes = min(3, max_alpha_shapes)

            # Create alpha shapes with the same parameters as the chart handler
            # First convert observations to numpy array of coordinates
            points = np.array([[obs["decimallongitude"], obs["decimallatitude"]]
                              for obs in observations])

            # Use the same calculation as in chart_handler
            alpha_shapes_geojson = self.alpha_shape_utils.calculate_alpha_shape(
                points,
                alpha=alpha,
                eps=eps,
                min_samples=min_samples,
                avoid_overlaps=avoid_overlaps
            )

            # Convert the GeoJSON to Earth Engine features
            ee_alpha_features = []
            all_alpha_shapes = []

            if alpha_shapes_geojson and 'geometry' in alpha_shapes_geojson:
                # Process the GeoJSON to extract alpha shapes and create EE features
                if alpha_shapes_geojson['geometry']['type'] == 'MultiPolygon':
                    for polygon_coords in alpha_shapes_geojson['geometry']['coordinates']:
                        try:
                            # Create Earth Engine geometry
                            ee_geom = ee.Geometry.Polygon(polygon_coords[0])

                            # Calculate properties (use most recent year from observations)
                            years = [obs["observation_year"] for obs in observations]
                            representative_year = max(years) if years else 2000
                            total_individuals = sum(
                                obs["individual_count"]
                                for obs in observations
                            )

                            # Create Earth Engine feature
                            ee_feature = ee.Feature(ee_geom, {
                                'year': representative_year,
                                'individual_count': total_individuals,
                                'num_observations': len(observations)
                            })
                            ee_alpha_features.append(ee_feature)
                            # Add to alpha shapes for visualization
                            all_alpha_shapes.append({
                                'type': 'Polygon',
                                'coordinates': polygon_coords,
                                'properties': {
                                    'year': representative_year,
                                    'num_observations': len(observations),
                                    'total_individuals': total_individuals
                                }
                            })
                        except Exception as e: # pylint: disable=broad-except
                            self.logger.warning("Error creating Earth Engine feature: %s", str(e))
                else:
                    # For single polygon (not MultiPolygon)
                    try:
                        # Create Earth Engine geometry
                        polygon_coords = alpha_shapes_geojson['geometry']['coordinates']
                        ee_geom = ee.Geometry.Polygon(polygon_coords[0])

                        # Calculate properties
                        years = [obs["observation_year"] for obs in observations]
                        representative_year = max(years) if years else 2000
                        total_individuals = sum(
                            obs["individual_count"]
                            for obs in observations
                        )

                        # Create Earth Engine feature
                        ee_feature = ee.Feature(ee_geom, {
                            'year': representative_year,
                            'individual_count': total_individuals,
                            'num_observations': len(observations)
                        })

                        ee_alpha_features.append(ee_feature)

                        # Add to alpha shapes for visualization
                        all_alpha_shapes.append({
                            'type': 'Polygon',
                            'coordinates': polygon_coords,
                            'properties': {
                                'year': representative_year,
                                'num_observations': len(observations),
                                'total_individuals': total_individuals
                            }
                        })
                    except Exception as e: # pylint: disable=broad-except
                        self.logger.warning("Error creating Earth Engine feature: %s", str(e))
            # If no valid alpha shapes were created, fall back to a simple convex hull
            if not ee_alpha_features:
                self.logger.warning("No valid alpha shapes created. Falling back to convex hull.")
            # Initialize all_results before using it
            all_results = []
            # Hansen dataset bands:
            # - treecover2000: Tree canopy cover for year 2000, in the range 0-100
            # - lossyear: Forest loss during the period 2001-2023, encoded as either 0 (no loss)
            # or year of loss (1-23)
            # - gain: Forest gain during the period 2000-2012, encoded as either 0 (no gain)
            # or 1 (gain)

            # Check if 'loss' band exists in the Hansen dataset
            hansen_bands = hansen.bandNames().getInfo()
            if 'loss' in hansen_bands:
                # If 'loss' band exists, use it directly
                self.logger.info("Using 'loss' band directly")
                binary_loss = hansen.select('loss')
            else:
                # If 'loss' band doesn't exist, create it from 'lossyear'
                self.logger.info("Creating binary loss from 'lossyear'")
                # Convert lossyear > 0 to binary (1 where loss occurred, 0 otherwise)
                binary_loss = hansen.select('lossyear').gt(0)

            # 2. Create a mask for meaningful tree cover (areas with at least some forest)
             # Consider areas with at least 10% cover as forest
            forest_mask = hansen.select('treecover2000').gte(10)

            # 3. Create a binary gain mask (1 where gain occurred, 0 otherwise)
            binary_gain = hansen.select('gain').gt(0)

            # 4. Combine to get tree cover, loss, and gain information
            treecover = hansen.select(['treecover2000'])
            treecover_with_loss = treecover.addBands(binary_loss.rename('loss_mask'))
            treecover_with_loss_gain = treecover_with_loss.addBands(binary_gain.rename('gain_mask'))
            combined = treecover_with_loss_gain.addBands(forest_mask.rename('forest_mask'))

            # Sample the dataset directly without retry logic
            try:
                # Using tileScale instead of maxPixels to handle large computations
                forest_stats = combined.reduceRegions(
                    collection=ee.FeatureCollection(ee_alpha_features),
                    reducer=ee.Reducer.mean(),
                    scale=scale,
                    tileScale=4
                )
                sample_results = forest_stats.getInfo()['features']
            except ee.EEException as e:
                self.logger.error("Earth Engine error: %s", str(e))
                raise # Re-raise the error to be caught by the outer exception handler

            # Process sample results
            for sample in sample_results:
                props = sample['properties']
                year = props['year']

                # Extract tree cover and forest-related properties
                tree_cover = props.get('treecover2000', 0)
                loss_mask = props.get('loss_mask', 0)
                forest_mask = props.get('forest_mask', 0)
                gain_mask = props.get('gain_mask', 0)

                # Check inputs for validity
                if tree_cover is None or tree_cover < 0:
                    tree_cover = 0
                if loss_mask is None:
                    loss_mask = 0
                if forest_mask is None:
                    forest_mask = 0
                if gain_mask is None:
                    gain_mask = 0

                # Original working forest loss calculation
                if loss_mask <= 0 or forest_mask <= 0:
                    # No detectable loss or no forest to begin with
                    forest_loss = 0.0
                else:
                    # Loss occurred - use the loss_mask value directly
                    # This represents the proportion of the area that experienced loss
                    forest_loss = loss_mask

                # Calculate remaining forest cover as a percentage (0-100)
                if tree_cover <= 0:
                    # No forest to begin with
                    remaining_cover = 0
                elif forest_loss <= 0:
                    # No loss detected
                    if gain_mask > 0:
                        # If there's forest gain but no loss, add gain to initial cover
                        remaining_cover = min(100, tree_cover + (gain_mask * 20.0))
                    else:
                        remaining_cover = tree_cover
                else:
                    # Calculate remaining cover after loss
                    remaining_cover = max(0, tree_cover * (1 - forest_loss))

                    # Add gain if any
                    if gain_mask > 0:
                        remaining_cover = min(100, remaining_cover + (gain_mask * 20.0))

                # Debug: Print calculated values
                print(f"DEBUG - Calculated: forest_loss={forest_loss}, "
                      f"remaining_cover={remaining_cover}")

                all_results.append({
                    'year': year,
                    'individual_count': props['individual_count'],
                    'tree_cover_2000': tree_cover,
                    'remaining_cover': remaining_cover,
                    'forest_loss': forest_loss
                })

            # Calculate correlations using all results
            if not all_results:
                self.logger.warning("No valid results were obtained from Earth Engine. "
                                    "Cannot calculate correlations.")
                # Return early with a valid response structure
                return {
                    'correlation_data': {
                        'forest_cover': {'mean': 0, 'std': 0, 'correlation': 0, 'p_value': 1.0},
                        'forest_loss': {'mean': 0, 'std': 0, 'correlation': 0, 'p_value': 1.0},
                        'forest_gain_applied': False,
                        'total_observations': len(observations),
                        'spatial_resolution': scale,
                        'error': "No valid results were obtained from Earth Engine analysis."
                    },
                    'analysis': "Analysis could not be completed. Earth Engine "
                    "did not return valid data for the selected areas. Try "
                    "using different parameters or reducing the area size.",
                    'species_name': species_name,
                    'observations': observations,
                    'alpha_shapes': all_alpha_shapes,
                    'forest_layers': self.get_forest_layers(
                        alpha_shapes=all_alpha_shapes,
                        avoid_overlaps=avoid_overlaps
                    ),
                    'error': True
                }

            individual_counts = [r['individual_count'] for r in all_results]
            remaining_covers = [r['remaining_cover'] for r in all_results]
            forest_losses = [r['forest_loss'] for r in all_results]

            # Log the values for debugging
            self.logger.info("Calculating correlations with %s data points", len(individual_counts))
            self.logger.info("Individual counts range: %s to %s",
                             min(individual_counts), max(individual_counts))
            self.logger.info("Remaining covers range: %s to %s",
                             min(remaining_covers), max(remaining_covers))
            self.logger.info("Forest losses range: %s to %s",
                             min(forest_losses), max(forest_losses))

            # Handle cases where there's no variation in the data
            try:
                if len(set(remaining_covers)) <= 1 or len(set(individual_counts)) <= 1:
                    tree_cover_corr, tree_cover_p = 0.0, 1.0
                    self.logger.warning("No variation in tree cover or "
                                        "individual counts data. Setting correlation to 0.")
                else:
                    tree_cover_corr, tree_cover_p = stats.pearsonr(
                        individual_counts,
                        remaining_covers
                    )

                if len(set(forest_losses)) <= 1 or len(set(individual_counts)) <= 1:
                    forest_loss_corr, forest_loss_p = 0.0, 1.0
                    self.logger.warning("No variation in forest loss or "
                                        "individual counts data. Setting correlation to 0.")
                else:
                    forest_loss_corr, forest_loss_p = stats.pearsonr(
                        individual_counts,
                        forest_losses
                    )
            except Exception as e: # pylint: disable=broad-except
                self.logger.error("Error calculating correlations: %s", str(e))
                tree_cover_corr, tree_cover_p = 0.0, 1.0
                forest_loss_corr, forest_loss_p = 0.0, 1.0

            correlation_data =  {
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
                'spatial_resolution': scale
            }
            print(correlation_data)
            # Send to LLM for analysis
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

            prompt += """
            Please analyze:
            1. Please show average forest cover and forest loss for the species
            2. Overall correlation patterns
            3. Conservation implications of both average forest cover and average forest loss  on one hand side and 
            correlation between forest cover, forest loss and species occurrence on the other hand side
            4. Data limitations and caveats
            5. Please summarize the results in a few sentences and show the summary at the end of your response.

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
            """

            analysis = self.send_to_llm(prompt)

            # Return with alpha shapes included
            return {
                'correlation_data': correlation_data,
                'analysis': analysis,
                'species_name': species_name,
                'observations': observations, 
                'alpha_shapes': all_alpha_shapes,  # Limit to at most max_alpha_shapes
                'forest_layers': self.get_forest_layers(
                    alpha_shapes=all_alpha_shapes,
                    avoid_overlaps=avoid_overlaps
                )
            }
        except (ValueError, ee.EEException) as e:
            self.logger.error(lambda: f"Error calculating correlations with Earth Engine: {str(e)}")
            # Return a valid error response without any retry attempts
            return {
                'correlation_data': {
                    'forest_cover': {'mean': 0, 'std': 0, 'correlation': 0, 'p_value': 1.0},
                    'forest_loss': {'mean': 0, 'std': 0, 'correlation': 0, 'p_value': 1.0},
                    'forest_gain_applied': False,
                    'total_observations': len(observations) if observations else 0,
                    'spatial_resolution': scale,
                    'error': str(e)
                },
                'analysis': "Analysis failed. Please try with different"
                    " parameters or a smaller area.",
                'species_name': species_name,
                'observations': observations,
                'alpha_shapes': all_alpha_shapes if 'all_alpha_shapes' in locals() else [], 
                'forest_layers': self.get_forest_layers(
                    alpha_shapes=all_alpha_shapes if 'all_alpha_shapes' in locals() else [],
                    avoid_overlaps=avoid_overlaps
                ),
                'error': True
            }

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
