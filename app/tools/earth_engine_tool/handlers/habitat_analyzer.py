"""Handler for habitat analysis using Earth Engine."""

from typing import Dict, Any
import ee
import streamlit as st
from app.tools.earth_engine_tool.handlers.earth_engine_handler import EarthEngineHandler
from ...visualization.config.land_cover_config import LandCoverConfig
import logging


# pylint: disable=no-member
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
            # Create status messages
            st.write("🔍 Starting habitat analysis...")

            with st.status("Analyzing habitat distribution...", expanded=True) as status:
                # Get species occurrence points
                st.write("📍 Fetching species observations...")
                observations = self.filter_marine_observations(
                    self.get_species_observations(species_name)
                )
                status.update(label="Processing species data...", expanded=True)

                # Convert observations to Earth Engine features
                st.write("🌍 Converting to Earth Engine features...")
                species_points = ee.FeatureCollection(
                    [ee.Feature(
                        ee.Geometry.Point([obs["decimallongitude"], obs["decimallatitude"]]),
                        {'individual_count': obs["individual_count"]}
                    ) for obs in observations]
                )

                # Get land cover data
                st.write("🗺️ Loading land cover data...")
                landcover = ee.Image('COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019')
                status.update(label="Analyzing habitat patterns...", expanded=True)

                # Perform analysis
                st.write("📊 Calculating habitat usage...")
                results = self.analyze(landcover, species_points, species_points.geometry())

                # Get analysis from Gemini
                st.write("🤖 Generating expert analysis...")
                analysis = self.send_to_llm(
                    self.create_analysis_prompt(species_name, results)
                )

                # Generate visualizations if requested
                st.write("🎨 Creating visualizations...")
                visualizations = self._generate_visualizations(
                        landcover,
                        species_points
                )
                status.update(label="Analysis complete!", state="complete", expanded=False)

            return {
                'success': True,
                'data': results,
                'message': f"Habitat analysis completed for {species_name}",
                'analysis': analysis,
                'observations': observations,
                'visualizations': visualizations
            }

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
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
        - Forest dependency analysis shows how reliant the species is on forest habitats
        - Habitat fragmentation metrics indicate the quality and connectivity of preferred habitats
        - The analysis uses Copernicus land cover data at 100m resolution from 2019
        """

        if species_name:
            prompt += f"\nAnalyzing habitat distribution for species: {species_name}\n"

        prompt += """
        Please analyze:
        1. Primary habitat preferences and their ecological significance
        2. Forest dependency and its implications for conservation
        3. Habitat fragmentation patterns and their impact on species viability
        4. Data limitations and caveats
        5. Conservation recommendations based on the analysis
        6. Please summarize the key findings in a few sentences at the end.

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
        """
        Analyze habitat types and their distribution.

        Args:
            landcover: Land cover classification image
            species_points: Feature collection of species observation points
            region: Region of interest

        Returns:
            Dictionary containing analysis results
        """
        # Sample land cover values at species points
        points_with_landcover = landcover.select('discrete_classification')\
            .sampleRegions(
                collection=species_points,
                scale=100,
                geometries=True
            )

        # Calculate habitat usage
        habitat_usage = self._calculate_habitat_usage(points_with_landcover)

        # Analyze forest dependency
        forest_analysis = self._analyze_forest_dependency(points_with_landcover)

        # Analyze habitat fragmentation
        fragmentation = None
        fragmentation = self._analyze_habitat_fragmentation(landcover, region)

        return {
            'habitat_usage': habitat_usage,
            'forest_analysis': forest_analysis,
            'habitat_fragmentation': fragmentation
        }

    def _calculate_habitat_usage(self, points_with_habitat: ee.FeatureCollection) -> Dict[str, float]:
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

    def _analyze_forest_dependency(self, points_with_habitat: ee.FeatureCollection) -> Dict[str, Any]:
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
        region: ee.Geometry
    ) -> Dict[str, Any]:
        """Analyze habitat fragmentation metrics."""
        try:
            # Use simple progress messages instead of nested status
            st.write("🌲 Analyzing forest fragmentation...")

            # Define forest codes
            forest_codes = [111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126]

            # Create forest mask
            forest_mask = landcover.select('discrete_classification').remap(
                forest_codes,
                ee.List.repeat(1, len(forest_codes)),
                0
            )

            # Identify patches
            patches = forest_mask.connectedComponents(
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

            forest_coverage = forest_mask.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=scale,
                maxPixels=1e9
            ).get('remapped').getInfo() * 100

            mean_patch_size = (total_pixels * pixel_area / total_patches) if total_patches > 0 else 0

            st.write("✅ Fragmentation analysis complete")

            return {
                'patch_statistics': {
                    'total_patches': int(max(0, total_patches)),
                    'mean_patch_size': float(mean_patch_size),
                    'forest_coverage': float(forest_coverage or 0),
                    'area_unit': 'hectares',
                    'connectivity_type': '8-connected (including diagonals)',
                    'description': {
                        'connectivity': 'Using 8-connectivity: patches are connected if they share edges or corners',
                        'total_patches': 'Number of separate forest areas (8-connected)',
                        'mean_patch_size': 'Average size of forest patches in hectares',
                        'forest_coverage': 'Percentage of total area covered by forest'
                    }
                }
            }

        except Exception as e:
            st.error(f"❌ Error in fragmentation analysis: {str(e)}")
            return {
                'patch_statistics': {
                    'total_patches': 0,
                    'mean_patch_size': 0,
                    'forest_coverage': 0,
                    'area_unit': 'hectares',
                    'connectivity_type': '8-connected (including diagonals)'
                }
            }

    def _generate_visualizations(
        self,
        landcover: ee.Image,
        species_points: ee.FeatureCollection
    ) -> Dict[str, Any]:
        """Generate visualization data for habitat analysis results."""
        try:
            st.write("🎨 Preparing visualizations...")

            # Sample land cover values at species points
            points_with_landcover = (
                landcover.select('discrete_classification')
                .sampleRegions(
                    collection=species_points,
                    properties=['individual_count'],
                    geometries=True
                )
            )

            # Get the visualization parameters for landcover
            landcover_vis = landcover.select('discrete_classification').visualize(**self.landcover_vis)

            # Get map ID for the landcover layer
            landcover_layer = landcover_vis.getMapId()

            st.write("- Converting points to GeoJSON")
            # Convert to GeoJSON while preserving all properties
            species_points_geojson = points_with_landcover.getInfo()

            # Get the center of the points for map centering
            center = species_points.geometry().centroid().getInfo()['coordinates']

            st.write("✅ Visualization data prepared")

            return {
                'landcover': {
                    'tiles': [landcover_layer['tile_fetcher'].url_format],
                    'attribution': 'Copernicus Global Land Service'
                },
                'center': {
                    'lat': center[1],
                    'lon': center[0]
                },
                'species_points': species_points_geojson
            }

        except Exception as e:
            st.error(f"❌ Error generating visualizations: {str(e)}")
            logging.error(f"Visualization error: {str(e)}")
            # Return original points if sampling fails
            return {
                'landcover': None,
                'species_points': species_points.getInfo(),
                'center': {
                    'lat': species_points.geometry().centroid().getInfo()['coordinates'][1],
                    'lon': species_points.geometry().centroid().getInfo()['coordinates'][0]
                }
            }