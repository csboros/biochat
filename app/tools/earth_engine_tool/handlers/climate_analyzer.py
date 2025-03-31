"""
Module for analyzing climate data and its relationship with species distribution.

This module provides functionality to analyze climate trends, precipitation patterns,
and species-climate relationships using Earth Engine data.
"""

from typing import Dict, Optional, Any
import ee
from app.tools.earth_engine_tool.handlers.earth_engine_handler import EarthEngineHandler
from app.tools.message_bus import message_bus

# pylint: disable=broad-except
class ClimateAnalyzer(EarthEngineHandler):
    """
    Analyzes climate data and its relationship with species distribution.
    Inherits from EarthEngineHandler to utilize common Earth Engine functionality.
    """

    def __init__(self):
        super().__init__()
        # Initialize climate datasets
        self.climate_dataset = ee.ImageCollection('WORLDCLIM/V1/MONTHLY')
        self.normal_period = ('1991-01-01', '2020-12-31')

    def analyze(self, params: Dict) -> Dict:
        """
        Main analysis method following the pattern of other analyzers.

        Args:
            params: Dictionary containing:
                - geometry: GeoJSON geometry of the area of interest
                - species_name: Optional species name for species-specific analysis
                - start_year: Start year for analysis (default: 1990)
                - end_year: End year for analysis (default: 2023)

        Returns:
            Dictionary containing climate analysis results
        """
        try:
            species_name = params.get('species_name')

            message_bus.publish("status_update", {
                "message": "Starting climate analysis...",
                "state": "running",
                "progress": 0
            })

            # Get and filter observations
            observations = self.filter_marine_observations(
                self.get_species_observations(species_name)
            )

            # Filter out outlier points
            filtered_observations = self.filter_outlier_points(observations)

            # Convert to Earth Engine features
            ee_point_features = self.create_ee_point_features(filtered_observations)
            species_points = ee.FeatureCollection(ee_point_features)

            message_bus.publish("status_update", {
                "message": "Analyzing climate trends...",
                "state": "running",
                "progress": 20
            })

            # Get climate trends
            climate_trends = self._analyze_climate_trends(species_points)

            message_bus.publish("status_update", {
                "message": "Analyzing climate patterns...",
                "state": "running",
                "progress": 50
            })

            # If species name is provided, analyze species-climate relationship
            species_climate_data = None
            if species_name:
                species_climate_data = self._analyze_species_climate_relationship(
                    species_points
                )

            # Generate visualizations
            message_bus.publish("status_update", {
                "message": "🎨 Creating visualizations...",
                "state": "running",
                "progress": 90
            })
            visualizations = self._generate_visualizations(
                self.climate_dataset,
                species_points
            )

            # Prepare analysis results
            analysis_results = {
                "climate_trends": climate_trends,
                "species_climate_data": species_climate_data,
                "analysis": self._generate_analysis_text(climate_trends, species_climate_data),
                "visualizations": visualizations
            }

            message_bus.publish("status_update", {
                "message": "Climate analysis complete",
                "state": "complete",
                "progress": 100
            })

            return analysis_results

        except Exception as e:
            message_bus.publish("status_update", {
                "message": f"Error in climate analysis: {str(e)}",
                "state": "error"
            })
            raise

    def _analyze_climate_trends(self, species_points: ee.FeatureCollection) -> Dict:
        """Analyzes temperature and precipitation trends."""
        try:
            message_bus.publish("status_update", {
                "message": "🌡️ Processing climate data...",
                "state": "running",
                "progress": 25
            })

            # Filter out outlier points
            filtered_geometry = species_points.geometry()

            # Select all temperature and precipitation bands
            climate_data = self.climate_dataset.select(['tavg', 'tmin', 'tmax', 'prec'])

            message_bus.publish("status_update", {
                "message": "📊 Calculating temperature statistics...",
                "state": "running",
                "progress": 35
            })

            # Calculate mean temperature (average across months)
            temp_mean = climate_data.select('tavg').mean().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=filtered_geometry,
                scale=1000,
                maxPixels=1e9
            ).getInfo()

            # Get minimum temperature (coldest month)
            temp_min = climate_data.select('tmin').min().reduceRegion(
                reducer=ee.Reducer.min(),
                geometry=filtered_geometry,
                scale=1000,
                maxPixels=1e9
            ).getInfo()

            # Get maximum temperature (warmest month)
            temp_max = climate_data.select('tmax').max().reduceRegion(
                reducer=ee.Reducer.max(),
                geometry=filtered_geometry,
                scale=1000,
                maxPixels=1e9
            ).getInfo()

            message_bus.publish("status_update", {
                "message": "📊 Calculating precipitation statistics...",
                "state": "running",
                "progress": 40
            })

            # Calculate mean precipitation
            precip_mean = climate_data.select('prec').mean().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=filtered_geometry,
                scale=1000,
                maxPixels=1e9
            ).getInfo()

            # Get minimum precipitation (driest month)
            precip_min = climate_data.select('prec').min().reduceRegion(
                reducer=ee.Reducer.min(),
                geometry=filtered_geometry,
                scale=1000,
                maxPixels=1e9
            ).getInfo()

            # Get maximum precipitation (wettest month)
            precip_max = climate_data.select('prec').max().reduceRegion(
                reducer=ee.Reducer.max(),
                geometry=filtered_geometry,
                scale=1000,
                maxPixels=1e9
            ).getInfo()

            # Log the raw results for debugging
            self.logger.info("Temperature results: %s", {
                "mean": temp_mean,
                "min": temp_min,
                "max": temp_max
            })
            self.logger.info("Precipitation results: %s", {
                "mean": precip_mean,
                "min": precip_min,
                "max": precip_max
            })

            message_bus.publish("status_update", {
                "message": "✅ Climate trend analysis complete",
                "state": "running",
                "progress": 50
            })

            # Apply scale factor of 0.1 to temperature values
            return {
                "temperature": {
                    "mean": temp_mean.get('tavg', 0.0) * 0.1,
                    "min": temp_min.get('tmin', 0.0) * 0.1,
                    "max": temp_max.get('tmax', 0.0) * 0.1,
                },
                "precipitation": {
                    "mean": precip_mean.get('prec', 0.0),
                    "min": precip_min.get('prec', 0.0),
                    "max": precip_max.get('prec', 0.0),
                }
            }

        except Exception as e:
            message_bus.publish("status_update", {
                "message": f"❌ Error analyzing climate trends: {str(e)}",
                "state": "error",
                "progress": 0
            })
            self.logger.error("Error analyzing climate trends: %s", str(e))
            return {
                "temperature": {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0},
                "precipitation": {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
            }

    def _analyze_species_climate_relationship(
        self,
        occurrences: ee.FeatureCollection,
    ) -> Dict:
        """Analyzes relationship between species occurrences and climate variables."""

        if not occurrences:
            return None

        # Extract climate data at occurrence points
        climate_at_occurrences = self._extract_climate_at_points(
            occurrences
        )

        return {
            "occurrences": occurrences,
            "climate_conditions": climate_at_occurrences
        }

    def _extract_climate_at_points(
        self,
        occurrences: ee.FeatureCollection
    ) -> Dict:
        """Extracts climate data at occurrence points.

        Args:
            occurrences (ee.FeatureCollection): Collection of species occurrence points
            start_year (int): Start year for climate data
            end_year (int): End year for climate data

        Returns:
            Dict: Dictionary containing climate data at each occurrence point
        """
        try:
            # Select temperature and precipitation bands and get mean
            climate_data = (self.climate_dataset
                .select(['tavg', 'prec'])
                .mean())

            # Sample climate data at occurrence points
            samples = climate_data.sampleRegions(
                collection=occurrences,
                properties=['individual_count'],
                scale=1000
            )

            # Get the results
            results = samples.getInfo()['features']

            # Combine temperature and precipitation data
            climate_data = []
            for feat in results:
                climate_data.append({
                    'temperature': feat['properties']['tavg'] * 0.1,
                    'precipitation': feat['properties']['prec'],
                    'count': feat['properties']['individual_count']
                })

            return climate_data

        except Exception as e:
            self.logger.error("Error extracting climate data at points: %s", str(e))
            raise

    def _generate_analysis_text(
        self,
        climate_trends: Dict,
        species_climate_data: Optional[Dict]
    ) -> str:
        """Generates descriptive text of the analysis results using Gemini."""
        prompt = """You are a climate science and conservation biology expert.
        Analyze this climate data to understand climate patterns and their
        implications for species distribution.

        Context:
        - The data shows temperature and precipitation patterns in the species' range
        - Climate data is from WORLDCLIM v1 (temperature) and WORLDCLIM v1 (precipitation)
        - The analysis covers a specific time period with mean, min, max, and standard deviation values
        """

        # Add climate trends data
        prompt += "\nClimate Trends:\n"
        temp_trend = climate_trends["temperature"]
        precip_trend = climate_trends["precipitation"]

        prompt += f"""
        Temperature:
        - Mean: {temp_trend['mean']:.2f}°C
        - Range: {temp_trend['min']:.2f}°C to {temp_trend['max']:.2f}°C

        Precipitation:
        - Mean: {precip_trend['mean']:.2f}mm
        - Range: {precip_trend['min']:.2f}mm to {precip_trend['max']:.2f}mm
        """

        # Add species-specific climate data if available
        if species_climate_data:
            prompt += "\nSpecies-Climate Relationship:\n"
            climate_conditions = species_climate_data.get("climate_conditions", [])
            if climate_conditions:
                # Calculate temperature and precipitation ranges for species occurrences
                temps = [point['temperature'] for point in climate_conditions]
                precips = [point['precipitation'] for point in climate_conditions]

                prompt += f"""
                Species Climate Envelope:
                - Temperature Range: {min(temps):.2f}°C to {max(temps):.2f}°C
                - Precipitation Range: {min(precips):.2f}mm to {max(precips):.2f}mm
                - Number of Observations: {len(climate_conditions)}
                """

        prompt += """
        Please analyze:
        1. Climate patterns and their spatial distribution
        2. Temperature and precipitation variability
        3. Implications for species distribution and habitat suitability
        4. Potential climate change impacts
        5. Data limitations and caveats
        6. Please summarize the key findings in a few sentences at the end.

        Data:
        """
        prompt += f"\n{climate_trends}"
        if species_climate_data:
            prompt += f"\n{species_climate_data}"

        return self.send_to_llm(prompt)

    def _get_species_occurrences(
        self,
        species_name: str,
        geometry: ee.Geometry,
        min_observations: int = 10
    ) -> ee.FeatureCollection:
        """Get species occurrences as Earth Engine features within the given geometry.

        Args:
            species_name (str): Scientific name of the species
            geometry (ee.Geometry): Earth Engine geometry defining the area of interest
            min_observations (int): Minimum number of observations required

        Returns:
            ee.FeatureCollection: Collection of Earth Engine features
            representing species occurrences
        """
        try:
            # Get species observations using inherited method
            observations = self.get_species_observations(species_name, min_observations)

            # Filter marine observations using inherited method
            observations = self.filter_marine_observations(observations)

            # Create Earth Engine features from observations
            ee_features = self.create_ee_point_features(observations)

            # Create a feature collection
            feature_collection = ee.FeatureCollection(ee_features)

            # Filter features to only include those within the geometry
            filtered_features = feature_collection.filterBounds(geometry)

            return filtered_features

        except Exception as e:
            self.logger.error("Error getting species occurrences: %s", str(e))
            raise

    def _generate_visualizations(
        self,
        climate_dataset: ee.ImageCollection,
        species_points: ee.FeatureCollection
    ) -> Dict[str, Any]:
        """Generate visualization data for climate analysis results."""
        try:
            # Select temperature and precipitation bands
            climate_data = climate_dataset.select(['tavg', 'tmin', 'tmax', 'prec'])

            # Find the coldest month (minimum of tmin across all months)
            temp_min_vis = climate_data.select('tmin').min().multiply(0.1).visualize(
                min=-40,
                max=30,
                palette=['blue', 'purple', 'cyan', 'green', 'yellow', 'red']
            )

            # Find the warmest month (maximum of tmax across all months)
            temp_max_vis = climate_data.select('tmax').max().multiply(0.1).visualize(
                min=-40,
                max=30,
                palette=['blue', 'purple', 'cyan', 'green', 'yellow', 'red']
            )

            # Find the driest month (minimum precipitation across all months)
            precip_min_vis = climate_data.select('prec').min().visualize(
                min=0,
                max=800,
                palette=['#f7fbff','#deebf7','#c6dbef','#9ecae1','#6baed6',
                         '#4292c6','#2171b5','#08519c','#08306b']
            )

            # Find the wettest month (maximum precipitation across all months)
            precip_max_vis = climate_data.select('prec').max().visualize(
                min=0,
                max=800,
                palette=['#f7fbff','#deebf7','#c6dbef','#9ecae1','#6baed6','#4292c6',
                         '#2171b5','#08519c','#08306b']
            )

            # Get map IDs for the layers
            temp_min_layer = temp_min_vis.getMapId()
            temp_max_layer = temp_max_vis.getMapId()
            precip_min_layer = precip_min_vis.getMapId()
            precip_max_layer = precip_max_vis.getMapId()

            # Get the original species points GeoJSON
            species_points_geojson = species_points.getInfo()

            # Get the center of the points for map centering
            center = species_points.geometry().centroid().getInfo()['coordinates']

            # Log the number of points for debugging
            self.logger.info("Number of species points: %s",
                             len(species_points_geojson['features']))

            return {
                'temperature': {
                    'coldest_month': {
                        'tiles': [temp_min_layer['tile_fetcher'].url_format],
                        'attribution': 'WorldClim v1'
                    },
                    'warmest_month': {
                        'tiles': [temp_max_layer['tile_fetcher'].url_format],
                        'attribution': 'WorldClim v1'
                    }
                },
                'precipitation': {
                    'driest_month': {
                        'tiles': [precip_min_layer['tile_fetcher'].url_format],
                        'attribution': 'WorldClim v1'
                    },
                    'wettest_month': {
                        'tiles': [precip_max_layer['tile_fetcher'].url_format],
                        'attribution': 'WorldClim v1'
                    }
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
            self.logger.error("Visualization error: %s", str(e))
            # Return original points if sampling fails
            return {
                'temperature': None,
                'precipitation': None,
                'species_points': species_points.getInfo(),
                'center': {
                    'lat': species_points.geometry().centroid().getInfo()['coordinates'][1],
                    'lon': species_points.geometry().centroid().getInfo()['coordinates'][0]
                }
            }
