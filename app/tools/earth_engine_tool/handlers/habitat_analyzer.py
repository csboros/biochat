"""Handler for habitat analysis using Earth Engine."""

from typing import Dict, Any
import ee
from app.tools.earth_engine_tool.handlers.earth_engine_handler import EarthEngineHandler
from app.tools.message_bus import message_bus
from ...visualization.config.land_cover_config import LandCoverConfig


# pylint: disable=no-member
# pylint: disable=broad-except
class HabitatAnalyzer(EarthEngineHandler):
    """Analyzes habitat types and their distribution."""

    def __init__(self):
        """Initialize the habitat analyzer."""
        super().__init__()
        self.landcover_vis = LandCoverConfig.get_vis_params()

    def analyze_habitat_distribution(
        self,
        species_name: str
    ) -> Dict[str, Any]:
        """Analyze species habitat distribution using Copernicus land cover data."""
        try:
            message_bus.publish("status_update", {
                "message": "Starting habitat analysis...",
                "state": "running",
                "progress": 0,
                "expanded": True
            })

            # Get species occurrence points
            message_bus.publish("status_update", {
                "message": "📍 Fetching species observations...",
                "state": "running",
                "progress": 10
            })
            observations = self.filter_marine_observations(
                self.get_species_observations(species_name)
            )

            # Convert observations to Earth Engine features
            message_bus.publish("status_update", {
                "message": "🌍 Converting to Earth Engine features...",
                "state": "running",
                "progress": 15
            })
            species_points = ee.FeatureCollection([
                ee.Feature(
                    ee.Geometry.Point([
                        obs["decimallongitude"],
                        obs["decimallatitude"]
                    ]),
                    {'individual_count': obs["individual_count"]}
                ) for obs in observations
            ])

            # Get land cover data
            message_bus.publish("status_update", {
                "message": "🗺️ Loading land cover data...",
                "state": "running",
                "progress": 20
            })
            landcover = ee.Image('COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019')

            # Perform analysis
            message_bus.publish("status_update", {
                "message": "📊 Calculating habitat usage...",
                "state": "running",
                "progress": 25
            })
            results = self.analyze(landcover, species_points, species_points.geometry())

            # Get analysis from Gemini
            message_bus.publish("status_update", {
                "message": "🤖 Generating expert analysis...",
                "state": "running",
                "progress": 75
            })
            analysis = self.send_to_llm(
                self.create_analysis_prompt(species_name, results)
            )

            # Generate visualizations
            message_bus.publish("status_update", {
                "message": "🎨 Creating visualizations...",
                "state": "running",
                "progress": 90
            })
            visualizations = self._generate_visualizations(
                    landcover,
                    species_points,
                    results['connectivity']
            )

            message_bus.publish("status_update", {
                "message": "✅ Analysis complete!",
                "state": "complete",
                "progress": 100,
                "expanded": False
            })

            return {
                'success': True,
                'data': results,
                'message': f"Habitat analysis completed for {species_name}",
                'analysis': analysis,
                'observations': observations,
                'visualizations': visualizations
            }

        except Exception as e:
            message_bus.publish("status_update", {
                "message": f"❌ Error: {str(e)}",
                "state": "error",
                "progress": 0,
                "expanded": True
            })
            return {
                'success': False,
                'error': str(e),
                'message': f"Error analyzing habitat for {species_name}",
                'visualizations': None
            }

    def create_analysis_prompt(self, species_name: str, results: Dict[str, Any]) -> str:
        """Generate a prompt for LLM analysis of habitat distribution."""
        prompt = """You are a conservation biology expert. Analyze this habitat distribution data
        for a species to understand its habitat preferences and conservation implications.

        Context:
        - The data shows the species' distribution across different land cover types
        - Habitat fragmentation metrics indicate the quality and connectivity of preferred habitats
        - The analysis uses Copernicus land cover data at 100m resolution from 2019
        """

        if species_name:
            prompt += f"\nAnalyzing habitat distribution for species: {species_name}\n"

        # Add forest dependency context only if forest is the primary habitat
        if results.get('is_forest_primary', False):
            prompt += """
            - Forest dependency analysis shows how reliant the species is on forest habitats
            """

        prompt += """
        Please analyze:
        1. Primary habitat preferences and their ecological significance
        """

        # Add forest dependency analysis point only if forest is the primary habitat
        if results.get('is_forest_primary', False):
            prompt += """
        2. Forest dependency and its implications for conservation
            """

        prompt += """
        3. Habitat fragmentation patterns and their impact on species viability
        4. Habitat connectivity analysis:
           - Interpret the connectivity score (0-1) and its implications for species movement
           - Analyze the distribution of points across different habitat types
           - Assess potential barriers to species movement based on land cover types
           - Provide recommendations for improving habitat connectivity
        5. Data limitations and caveats
        6. Conservation recommendations based on the analysis
        7. Please summarize the key findings in a few sentences at the end.

        Data:
        """
        prompt += f"\n{results}"

        return prompt

    def analyze(
        self,
        landcover: ee.Image,
        species_points: ee.FeatureCollection,
        region: ee.Geometry
    ) -> Dict[str, Any]:
        """Analyze habitat types and their distribution."""
        # Sample land cover values at species points
        points_with_landcover = landcover.select('discrete_classification')\
            .sampleRegions(
                collection=species_points,
                scale=100,
                geometries=True
            )

        # Calculate habitat usage first
        message_bus.publish("status_update", {
            "message": "📊 Analyzing habitat usage",
            "state": "running",
            "progress": 30
        })
        habitat_usage = self._calculate_habitat_usage(points_with_landcover)

        # Log habitat usage for debugging
        self.logger.info("Habitat usage: %s", habitat_usage)

        # Determine if forest is the primary habitat
        forest_percentage = sum(
            percentage for name, percentage in habitat_usage.items()
            if 'forest' in name.lower()
        )

        # Get the maximum percentage among all habitats
        max_percentage = max(habitat_usage.values())

        # Log percentages for debugging
        self.logger.info("Forest percentage: %.2f, Max percentage: %.2f",
                        forest_percentage, max_percentage)

        # Forest is primary if it has the highest percentage (allowing for ties)
        is_primary_habitat = forest_percentage > 0 and forest_percentage >= max_percentage

        # Log the decision for debugging
        self.logger.info("Is forest primary habitat? %s", is_primary_habitat)

        # Only analyze forest dependency if forest is the primary habitat
        forest_analysis = None
        if is_primary_habitat:
            message_bus.publish("status_update", {
                "message": "📊 Analyzing forest dependency",
                "state": "running",
                "progress": 40
            })
            forest_analysis = self._analyze_forest_dependency(points_with_landcover)
        else:
            message_bus.publish("status_update", {
                "message": "📊 Skipping forest dependency analysis (not primary habitat)",
                "state": "running",
                "progress": 50
            })

        # Analyze habitat fragmentation with habitat usage information
        message_bus.publish("status_update", {
            "message": "📊 Analyzing habitat fragmentation",
            "state": "running",
            "progress": 60
        })
        fragmentation = self._analyze_habitat_fragmentation(
            landcover,
            region,
            habitat_usage
        )

        # Analyze habitat connectivity
        message_bus.publish("status_update", {
            "message": "📊 Analyzing habitat connectivity",
            "state": "running",
            "progress": 70
        })
        connectivity = self._analyze_habitat_connectivity(points_with_landcover, habitat_usage)

        return {
            'habitat_usage': habitat_usage,
            'forest_analysis': forest_analysis,
            'habitat_fragmentation': fragmentation,
            'connectivity': connectivity,
            'is_forest_primary': is_primary_habitat
        }

    def _calculate_habitat_usage(
            self, points_with_habitat: ee.FeatureCollection) -> Dict[str, float]:
        """Calculate the distribution of habitat types at species observation points."""
        # Get the histogram of land cover values at species points
        histogram = points_with_habitat.aggregate_histogram('discrete_classification').getInfo()

        # Convert histogram to percentages
        total_points = sum(histogram.values())
        habitat_usage = {}
        for code, count in histogram.items():
            percentage = (count / total_points) * 100
            habitat_name = LandCoverConfig.LAND_COVER_CLASSES.get(int(code), f"Unknown ({code})")
            habitat_usage[habitat_name] = percentage

        return habitat_usage

    def _analyze_forest_dependency(
            self, points_with_habitat: ee.FeatureCollection) -> Dict[str, Any]:
        """Analyze forest dependency of species observations."""
        # Define forest codes (both closed and open forests)
        forest_codes = [111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126]

        # Get the histogram of land cover values at species points
        histogram = points_with_habitat.aggregate_histogram('discrete_classification').getInfo()

        # Calculate forest dependency
        total_points = sum(histogram.values())
        forest_points = sum(histogram.get(str(code), 0) for code in forest_codes)
        forest_dependency_ratio = (forest_points / total_points) * 100 if total_points > 0 else 0

        # Calculate forest type distribution
        forest_type_distribution = {}
        for code in forest_codes:
            count = histogram.get(str(code), 0)
            if count > 0:
                percentage = (count / forest_points) * 100
                habitat_name = LandCoverConfig.LAND_COVER_CLASSES.get(code, f"Unknown ({code})")
                forest_type_distribution[habitat_name] = percentage

        return {
            'forest_dependency_ratio': forest_dependency_ratio,
            'forest_type_distribution': forest_type_distribution
        }

    def _analyze_habitat_fragmentation(
        self,
        landcover: ee.Image,
        region: ee.Geometry,
        habitat_usage: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Analyze habitat fragmentation metrics for the species' primary habitat type.
        """
        try:
            # Define habitat type groups with their corresponding codes and names
            habitat_groups = {
                'Forest': {
                    'codes': [111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126],
                    'keywords': ['forest', 'tree']
                },
                'Shrubland': {
                    'codes': [20],
                    'keywords': ['shrub', 'shrubland']
                },
                'Grassland': {
                    'codes': [30],
                    'keywords': ['grass', 'grassland']
                },
                'Wetland': {
                    'codes': [90],
                    'keywords': ['wetland', 'marsh']
                },
                'Cropland': {
                    'codes': [40],
                    'keywords': ['crop', 'cropland']
                }
            }

            # Determine primary habitat type and its usage percentage
            primary_habitat_group = 'Forest'
            primary_habitat_percentage = 0
            if habitat_usage:
                group_percentages = {}
                for group_name, group_info in habitat_groups.items():
                    total_percentage = sum(
                        percentage for name, percentage in habitat_usage.items()
                        if any(
                            keyword.lower() in name.lower()
                            for keyword in group_info['keywords']
                        )
                    )
                    group_percentages[group_name] = total_percentage

                # Find the habitat group with highest percentage
                primary_habitat_group, primary_habitat_percentage = max(
                    group_percentages.items(),
                    key=lambda x: x[1]
                )

            # Get the codes for the primary habitat group
            habitat_codes = habitat_groups[primary_habitat_group]['codes']

            # Create habitat mask
            habitat_mask = landcover.select('discrete_classification').remap(
                habitat_codes,
                ee.List.repeat(1, len(habitat_codes)),
                0
            )

            # Identify patches
            patches = habitat_mask.connectedComponents(
                connectedness=ee.Kernel.square(1),
                maxSize=1024
            )

            # Calculate statistics
            scale = 500
            pixel_stats = patches.select('labels').reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=region,
                scale=scale,
                maxPixels=1e9
            ).getInfo()

            patch_stats = patches.select('labels').reduceRegion(
                reducer=ee.Reducer.countDistinct(),
                geometry=region,
                scale=scale,
                maxPixels=1e9
            ).getInfo()

            # Calculate final metrics
            pixel_area = (scale * scale) / 10000
            total_pixels = pixel_stats.get('labels', 0)
            total_patches = patch_stats.get('labels', 0) - 1

            habitat_coverage = habitat_mask.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=scale,
                maxPixels=1e9
            ).get('remapped').getInfo() * 100

            mean_patch_size = (total_pixels * pixel_area / total_patches) if total_patches > 0 else 0

            return {
                'patch_statistics': {
                    'habitat_type': primary_habitat_group,
                    'total_patches': int(max(0, total_patches)),
                    'mean_patch_size': float(mean_patch_size),
                    'habitat_coverage': float(primary_habitat_percentage),
                    'area_unit': 'hectares',
                    'connectivity_type': '8-connected (including diagonals)',
                    'description': {
                        'habitat_type': (
                            f'Analysis performed for {primary_habitat_group} habitat'
                        ),
                        'connectivity': (
                            'Using 8-connectivity: patches are connected if they '
                            'share edges or corners'
                        ),
                        'total_patches': (
                            f'Number of separate {primary_habitat_group.lower()} '
                            'areas (8-connected)'
                        ),
                        'mean_patch_size': 'Average size of habitat patches in hectares',
                        'habitat_coverage': (
                            f'Percentage of observations in '
                            f'{primary_habitat_group.lower()} habitat'
                        )
                    }
                }
            }

        except Exception as e:
            message_bus.publish("status_update", {
                "message": f"❌ Error in fragmentation analysis: {str(e)}",
                "state": "error"
            })
            return {
                'patch_statistics': {
                    'habitat_type': 'Unknown',
                    'total_patches': 0,
                    'mean_patch_size': 0,
                    'habitat_coverage': 0,
                    'area_unit': 'hectares',
                    'connectivity_type': '8-connected (including diagonals)'
                }
            }

    def _analyze_habitat_connectivity(
            self, points_with_landcover, habitat_usage: Dict[str, float] = None):
        """
        Analyzes habitat connectivity between patches.

        Args:
            points_with_landcover: ee.FeatureCollection with land cover data
            habitat_usage: Dictionary of habitat usage percentages

        Returns:
            dict: Connectivity analysis results
        """
        try:
            # Define resistance values for different land cover types
            # Base resistance values (can be adjusted based on primary habitat)
            base_resistance_values = {
                111: 1,  # Closed forest, evergreen needle leaf
                112: 1,  # Closed forest, evergreen broad leaf
                113: 1,  # Closed forest, deciduous needle leaf
                114: 1,  # Closed forest, deciduous broad leaf
                115: 1,  # Closed forest, mixed
                116: 1,  # Closed forest, not matching other definitions
                121: 2,  # Open forest, evergreen needle leaf
                122: 2,  # Open forest, evergreen broad leaf
                123: 2,  # Open forest, deciduous needle leaf
                124: 2,  # Open forest, deciduous broad leaf
                125: 2,  # Open forest, mixed
                126: 2,  # Open forest, not matching other definitions
                20: 3,   # Shrubland
                30: 3,   # Herbaceous vegetation
                40: 4,   # Cropland
                50: 10,  # Urban/built up
                60: 5,   # Bare/sparse vegetation
                70: 8,   # Snow and ice
                80: 6,   # Permanent water bodies
                90: 3,   # Herbaceous wetland
                100: 3,  # Moss and lichen
                200: 7   # Oceans, seas
            }

            # Determine primary habitat type and adjust resistance values
            if habitat_usage:
                # Find the habitat type with highest percentage
                primary_habitat = max(habitat_usage.items(), key=lambda x: x[1])[0]

                # Adjust resistance values based on primary habitat
                resistance_values = base_resistance_values.copy()

                # If primary habitat is not forest, adjust resistance values
                if not any('forest' in name.lower() for name in habitat_usage.keys()):
                    # Find the code for the primary habitat
                    primary_code = None
                    for code, name in LandCoverConfig.LAND_COVER_CLASSES.items():
                        if name == primary_habitat:
                            primary_code = code
                            break

                    if primary_code:
                        # Set lowest resistance for primary habitat
                        resistance_values[primary_code] = 1

                        # Adjust other similar habitats to have lower resistance
                        for code, name in LandCoverConfig.LAND_COVER_CLASSES.items():
                            if code != primary_code:
                                if any(keyword in name.lower()
                                       for keyword in primary_habitat.lower().split()):
                                    resistance_values[code] = 2
                                elif 'urban' in name.lower() or 'built up' in name.lower():
                                    resistance_values[code] = 10
                                elif 'water' in name.lower() or 'ocean' in name.lower():
                                    resistance_values[code] = 8
                                else:
                                    resistance_values[code] = 3

            # Calculate connectivity metrics
            total_points = points_with_landcover.size()

            # Calculate resistance distribution
            resistance_histogram = (points_with_landcover
                .aggregate_histogram('discrete_classification')
                .getInfo())

            # Calculate weighted average resistance
            total_resistance = 0
            total_count = 0
            for code, count in resistance_histogram.items():
                code_int = int(code)
                resistance = resistance_values.get(code_int, 10)  # Default to 10 for unknown values
                total_resistance += resistance * count
                total_count += count

            avg_resistance = total_resistance / total_count if total_count > 0 else 10

            # Calculate connectivity score (0-1)
            # Higher score means better connectivity
            connectivity_score = 1 - (avg_resistance / 10)

            # Get the results
            results = {
                'connectivity_score': connectivity_score,
                'average_resistance': avg_resistance,
                'total_points': total_points.getInfo(),
                'primary_habitat': primary_habitat if habitat_usage else "Unknown",
                'habitat_distribution': habitat_usage if habitat_usage else {}
            }
            return results

        except Exception as e:
            self.logger.error("Error in habitat connectivity analysis: %s", str(e))
            return None

    def _generate_visualizations(
        self,
        landcover: ee.Image,
        species_points: ee.FeatureCollection,
        connectivity_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate visualization data for habitat analysis results."""
        try:
            # Sample land cover values at species points
            points_with_landcover = (
                landcover.select('discrete_classification')
                .sampleRegions(
                    collection=species_points,
                    properties=['individual_count'],
                    geometries=True
                )
            )

            # Get the visualization parameters for landcover directly from LandCoverConfig
            landcover_vis = landcover.select('discrete_classification')\
                .visualize()
#                .visualize(**LandCoverConfig.get_vis_params())

            print(LandCoverConfig.get_vis_params())

            # Get map ID for the landcover layer
            landcover_layer = landcover_vis.getMapId()

            # Convert to GeoJSON while preserving all properties
            species_points_geojson = points_with_landcover.getInfo()

            # Get the center of the points for map centering
            center = species_points.geometry().centroid().getInfo()['coordinates']

            return {
                'landcover': {
                    'tiles': [landcover_layer['tile_fetcher'].url_format],
                    'attribution': 'Copernicus Global Land Service'
                },
                'center': {
                    'lat': center[1],
                    'lon': center[0]
                },
                'species_points': species_points_geojson,
                'connectivity': connectivity_results
            }

        except Exception as e:
            message_bus.publish("status_update", {
                "message": f"❌ Error generating visualizations: {str(e)}",
                "state": "error"
            })
            self.logger.error("Visualization error: %s", str(e))
            # Return original points if sampling fails
            return {
                'landcover': None,
                'species_points': species_points.getInfo(),
                'center': {
                    'lat': species_points.geometry().centroid().getInfo()['coordinates'][1],
                    'lon': species_points.geometry().centroid().getInfo()['coordinates'][0]
                },
                'connectivity': None
            }
