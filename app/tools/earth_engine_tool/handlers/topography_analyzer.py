"""Module for handling topography analysis using Earth Engine."""

import logging
from typing import Dict, Any
import ee
from app.tools.message_bus import message_bus
from .earth_engine_handler import EarthEngineHandler

# pylint: disable=broad-except
class TopographyAnalyzer(EarthEngineHandler):
    """Analyzes topography characteristics of species habitats using Earth Engine."""

    def __init__(self):
        """Initialize the TopographyAnalyzer."""
        super().__init__()
        self.logger = logging.getLogger("BioChat." + self.__class__.__name__)

    def analyze_topography(
        self,
        species_name: str,
        min_observations: int = 10,
        scale: int = 30
    ) -> Dict[str, Any]:
        """Analyze topography characteristics of species habitat using SRTM data.

        This method analyzes elevation, slope, and aspect characteristics of species
        observation points to understand their topographic habitat preferences.

        Args:
            species_name (str): Scientific name of the species
            min_observations (int): Minimum number of observations required
            scale (int): Resolution in meters for Earth Engine analysis

        Returns:
            dict: Dictionary containing topography analysis results and statistics
        """
        try:
            message_bus.publish("status_update", {
                "message": "Starting topography analysis...",
                "state": "running",
                "progress": 0,
                "expanded": True
            })

            # Get species observations from BigQuery
            message_bus.publish("status_update", {
                "message": "📍 Fetching species observations...",
                "state": "running",
                "progress": 10
            })
            self.check_cancellation()
            observations = self.filter_marine_observations(
                self.get_species_observations(species_name, min_observations))

            # Check for cancellation
            self.check_cancellation()

            # Create point features from individual observations
            message_bus.publish("status_update", {
                "message": "🌍 Converting to Earth Engine features...",
                "state": "running",
                "progress": 20
            })
            self.check_cancellation()
            species_points = ee.FeatureCollection(
                [ee.Feature(
                    ee.Geometry.Point([obs["decimallongitude"], obs["decimallatitude"]]),
                    {'individual_count': obs["individual_count"]}
                ) for obs in observations]
            )

            # Check for cancellation again
            self.check_cancellation()

            # Load SRTM data
            message_bus.publish("status_update", {
                "message": "🗺️ Loading topographic data...",
                "state": "running",
                "progress": 30
            })
            srtm = ee.Image('USGS/SRTMGL1_003')

            # Calculate terrain metrics
            message_bus.publish("status_update", {
                "message": "📊 Calculating terrain metrics...",
                "state": "running",
                "progress": 40
            })
            self.check_cancellation()
            slope = ee.Terrain.slope(srtm)
            aspect = ee.Terrain.aspect(srtm)

            # Sample terrain metrics at species points
            message_bus.publish("status_update", {
                "message": "📈 Sampling terrain characteristics...",
                "state": "running",
                "progress": 50
            })
            self.check_cancellation()
            points_with_terrain = ee.Image.cat([srtm, slope, aspect]).sampleRegions(
                collection=species_points,
                scale=scale,
                geometries=True
            )

            # Check for cancellation again
            self.check_cancellation()

            # Process results
            message_bus.publish("status_update", {
                "message": "📊 Processing terrain analysis...",
                "state": "running",
                "progress": 60
            })
            self.check_cancellation()
            results = self._process_terrain_results(points_with_terrain)

            # Generate visualizations
            message_bus.publish("status_update", {
                "message": "🎨 Creating visualizations...",
                "state": "running",
                "progress": 75
            })
            self.check_cancellation()
            visualizations = self._generate_visualizations(
                srtm,
                slope,
                aspect,
                species_points
            )

            # Get analysis from Gemini
            message_bus.publish("status_update", {
                "message": "🤖 Generating expert analysis...",
                "state": "running",
                "progress": 90
            })
            analysis = self.send_to_llm(
                self.create_analysis_prompt(species_name, results)
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
                'message': f"Topography analysis completed for {species_name}",
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
                'message': f"Error analyzing topography for {species_name}",
                'visualizations': None
            }

    def _process_terrain_results(self, points_with_terrain: ee.FeatureCollection) -> Dict[str, Any]:
        """Process terrain analysis results from Earth Engine features."""
        # Get histograms for each metric
        elevation_hist = points_with_terrain.aggregate_histogram('elevation').getInfo()
        slope_hist = points_with_terrain.aggregate_histogram('slope').getInfo()
        aspect_hist = points_with_terrain.aggregate_histogram('aspect').getInfo()

        # Calculate statistics
        elevation_stats = points_with_terrain.aggregate_stats('elevation').getInfo()
        slope_stats = points_with_terrain.aggregate_stats('slope').getInfo()
        aspect_stats = points_with_terrain.aggregate_stats('aspect').getInfo()

        # Calculate aspect categories
        aspect_categories = {
            'north': self._count_aspect_range(points_with_terrain, 315, 45),
            'east': self._count_aspect_range(points_with_terrain, 45, 135),
            'south': self._count_aspect_range(points_with_terrain, 135, 225),
            'west': self._count_aspect_range(points_with_terrain, 225, 315)
        }

        # Calculate slope categories
        slope_categories = {
            'flat': self._count_slope_range(points_with_terrain, 0, 5),
            'gentle': self._count_slope_range(points_with_terrain, 5, 15),
            'moderate': self._count_slope_range(points_with_terrain, 15, 30),
            'steep': self._count_slope_range(points_with_terrain, 30, 90)
        }

        results = {
            'elevation': {
                'mean': elevation_stats.get('mean'),
                'min': elevation_stats.get('min'),
                'max': elevation_stats.get('max'),
                'std': elevation_stats.get('sample_sd'),
                'distribution': elevation_hist
            },
            'slope': {
                'mean': slope_stats.get('mean'),
                'min': slope_stats.get('min'),
                'max': slope_stats.get('max'),
                'std': slope_stats.get('sample_sd'),
                'distribution': slope_hist,
                'categories': slope_categories
            },
            'aspect': {
                'mean': aspect_stats.get('mean'),
                'min': aspect_stats.get('min'),
                'max': aspect_stats.get('max'),
                'std': aspect_stats.get('sample_sd'),
                'distribution': aspect_hist,
                'categories': aspect_categories
            }
        }
        return results

    def _count_aspect_range(self, points: ee.FeatureCollection, start: float, end: float) -> int:
        """Count points within a specific aspect range."""
        try:
            if start > end:
                # Handle ranges that cross 0/360 boundary
                mask = points.filter(ee.Filter.gte('aspect', start)).merge(
                    points.filter(ee.Filter.lte('aspect', end))
                )
            else:
                mask = points.filter(ee.Filter.gte('aspect', start)).merge(
                    points.filter(ee.Filter.lt('aspect', end))
                )

            return mask.size().getInfo()

        except Exception as e: # pylint: disable=broad-except
            self.logger.error("Error counting aspect range: %s", str(e))
            return 0

    def _count_slope_range(self, points: ee.FeatureCollection, start: float, end: float) -> int:
        """Count points within a specific slope range."""
        try:
            mask = points.filter(ee.Filter.gte('slope', start)).merge(
                points.filter(ee.Filter.lt('slope', end))
            )

            return mask.size().getInfo()

        except Exception as e: # pylint: disable=broad-except
            self.logger.error("Error counting slope range: %s", str(e))
            return 0

    def _generate_visualizations(
        self,
        srtm: ee.Image,
        slope: ee.Image,
        aspect: ee.Image,
        species_points: ee.FeatureCollection
    ) -> Dict[str, Any]:
        """Generate visualization data for topography analysis results."""
        try:
            # Create visualizations for each layer
            elevation_vis = srtm.visualize(
                min=0,
                max=4000,
                palette=['green', 'yellow', 'red', 'white']
            )

            slope_vis = slope.visualize(
                min=0,
                max=45,
                palette=['green', 'yellow', 'red']
            )

            aspect_vis = aspect.visualize(
                min=0,
                max=360,
                palette=['blue', 'green', 'yellow', 'red']
            )

            # Get map IDs
            elevation_layer = elevation_vis.getMapId()
            slope_layer = slope_vis.getMapId()
            aspect_layer = aspect_vis.getMapId()

            # Convert points to GeoJSON
            species_points_geojson = species_points.getInfo()

            # Get the center of the points for map centering
            center = species_points.geometry().centroid().getInfo()['coordinates']

            return {
                'elevation': {
                    'tiles': [elevation_layer['tile_fetcher'].url_format],
                    'attribution': 'SRTM/NASA'
                },
                'slope': {
                    'tiles': [slope_layer['tile_fetcher'].url_format],
                    'attribution': 'SRTM/NASA'
                },
                'aspect': {
                    'tiles': [aspect_layer['tile_fetcher'].url_format],
                    'attribution': 'SRTM/NASA'
                },
                'center': {
                    'lat': center[1],
                    'lon': center[0]
                },
                'species_points': species_points_geojson
            }

        except Exception as e:
            message_bus.publish("status_update", {
                "message": f"❌ Error generating visualizations: {str(e)}",
                "state": "error"
            })
            logging.error("Visualization error: %s", str(e))
            return {
                'elevation': None,
                'slope': None,
                'aspect': None,
                'species_points': species_points.getInfo(),
                'center': {
                    'lat': species_points.geometry().centroid().getInfo()['coordinates'][1],
                    'lon': species_points.geometry().centroid().getInfo()['coordinates'][0]
                }
            }

    def create_analysis_prompt(self, species_name: str, results: Dict[str, Any]) -> str:
        """Generate a prompt for LLM analysis of topography characteristics."""
        prompt = """You are a conservation biology expert. Analyze this topography data
        for a species to understand its habitat preferences and ecological requirements.

        Context:
        - The data shows the species' distribution across different topographic features
        - Elevation data comes from SRTM (Shuttle Radar Topography Mission)
        - Slope is calculated in degrees (0-90)
        - Aspect is measured in degrees (0-360, where 0/360 is North)
        """

        if species_name:
            prompt += f"\nAnalyzing topography for species: {species_name}\n"

        prompt += """
        Please analyze:
        1. Elevation preferences and range
        2. Slope preferences and terrain characteristics
        3. Aspect preferences and potential microclimate implications
        4. Ecological significance of the topographic patterns
        5. Conservation implications based on topographic requirements
        6. Data limitations and caveats
        7. Please summarize the key findings in a few sentences at the end.

        Data:
        """
        prompt += f"\n{results}"
        return prompt
