"""
Module for handling various API functions and data processing operations in the Biodiversity App.
Includes functionality for species data retrieval, GeoJSON processing, 
and occurrence data management.
"""

import json.scanner
import json
import os
import time
import logging
import requests
import google.api_core.exceptions
from google.cloud import bigquery
from pygbif import species

# from EcoNameTranslator import to_common
from langchain_google_community import GoogleSearchAPIWrapper
import streamlit as st
from app.models.function_declarations import FUNCTION_DECLARATIONS
from .endangered_species_handler import EndangeredSpeciesHandler
from .species_handler import SpeciesHandler
from .base_handler import BaseHandler


class FunctionHandler(BaseHandler):
    """
    Handles the declaration and implementation of biodiversity-related functions.

    This class manages various functions related to species information, geographical data,
    and endangered species queries. It provides caching mechanisms for performance optimization
    and handles error logging.

    Attributes:
        logger (Logger): Class-specific logger instance
        handlers (dict): Contains handler instances (endangered_handler, query_builder)
        declarations (dict): Dictionary of function declarations for Vertex AI
        function_handler (dict): Mapping of function names to their implementations
        search (GoogleSearchAPIWrapper): Instance of Google Search API wrapper
        world_data (dict): Contains world_gdf and world_geojson_url for geographical features
    """

    def __init__(self):
        """
        Initializes the FunctionHandler.

        Raises:
            FileNotFoundError: If required files cannot be loaded
            ValueError: If configuration is invalid
        """
        super().__init__()  # Call parent class's __init__
        try:
            self.logger = logging.getLogger("BioChat." + self.__class__.__name__)
            self.logger.info("Initializing FunctionHandler")

            # Combine handlers into a single dictionary
            self.handlers = {
                "endangered": EndangeredSpeciesHandler(),
                "species": SpeciesHandler(),
            }

            # Combine world data into a single dictionary
            self.world_data = {
                "gdf": None,
                "geojson_url": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
                "master/geojson/ne_110m_admin_0_countries.geojson",
            }

            self.setup_function_declarations()
            self.search = GoogleSearchAPIWrapper()

        except Exception as e:
            self.logger.error("Initialization error: %s", str(e), exc_info=True)
            raise

    def setup_function_declarations(self):
        """
        Sets up function declarations.

        Raises:
            ValueError: If declarations are invalid
            ImportError: If required modules cannot be imported
        """
        try:
            self.declarations = FUNCTION_DECLARATIONS
            self.function_handler = {
                "translate_to_scientific_name": self.translate_to_scientific_name_from_api,
                "get_occurences": self.get_occurrences,
                "get_species_info": self.get_species_info_from_api,
                "endangered_species_for_family": self.handlers[
                    "endangered"
                ].endangered_species_for_family,
                "endangered_classes_for_kingdom": self.handlers[
                    "endangered"
                ].endangered_classes_for_kingdom,
                "endangered_families_for_order": self.handlers[
                    "endangered"
                ].endangered_families_for_order,
                "endangered_orders_for_class": self.handlers[
                    "endangered"
                ].endangered_orders_for_class,
                "endangered_species_for_country": self.handlers[
                    "endangered"
                ].endangered_species_for_country,
                "get_protected_areas_geojson": self.get_protected_areas_geojson,
                "number_of_endangered_species_by_conservation_status": self.handlers[
                    "endangered"
                ].number_of_endangered_species_by_conservation_status,
                "google_search": self.google_search,
                "get_endangered_species_in_protected_area":
                    self.handlers["species"].get_endangered_species_in_protected_area,
                "get_species_occurrences_in_protected_area":
                    self.handlers["species"].get_species_occurrences_in_protected_area,
                "read_terrestrial_hci": self.read_terrestrial_hci,
                "get_yearly_occurrences": self.handlers["species"].get_yearly_occurrences,      
                "read_population_density": self.read_population_density,
            }
        except (ValueError, ImportError) as e:
            self.logger.error("Setup error: %s", str(e), exc_info=True)
            raise

    def google_search(self, content) -> str:
        """
        Performs a Google search focused on IUCN Red List results.

        Raises:
            ValueError: If query is invalid
            google.api_core.exceptions.GoogleAPIError: If search API fails
        """
        try:
            query_string = content.get("query")
            query = f"site:https://www.iucnredlist.org/ {query_string}"
            return self.search.run(query)
        except google.api_core.exceptions.GoogleAPIError as e:
            self.logger.error("Google Search API error: %s", str(e), exc_info=True)
            raise
        except ValueError as e:
            self.logger.error("Invalid query: %s", str(e), exc_info=True)
            raise

    # pylint: disable=no-member
    #    @st.cache_data(
    #        ttl=3600,  # Cache for 1 hour
    #        show_spinner="Fetching data...",
    #        max_entries=100
    #    )
    def get_occurrences(_self, content):  # pylint: disable=no-self-argument
        """
        Retrieves species occurrence data from BigQuery.
        If country_code is provided, the function will return the occurrences
        for the specified country.
        If country_code is not provided, the function will return the distribution for the species.

        Args:
            content (dict): Dictionary containing:
                - species_name (str): Name of the species to query
                - country_code (str, optional): Two-letter country code to filter results

        Returns:
            list: List of dictionaries containing occurrence data with latitude and longitude

        Raises:
            ValueError: If species_name is invalid or not found
            google.api_core.exceptions.GoogleAPIError: If BigQuery query fails
            TypeError: If content is not in expected format
            KeyError: If required fields are missing from the response
        """
        start_time = time.time()
        try:
            species_name = content["species_name"]
            scientific_name = _self.translate_to_scientific_name_from_api(
                {"name": species_name}
            )

            # Parse the JSON response and check for errors
            translated_result = json.loads(scientific_name)
            if "error" in translated_result:
                _self.logger.warning(
                    "Could not translate species name: %s - %s",
                    species_name,
                    translated_result["error"],
                )
                return []  # or return {"error": f"Species not found: {species_name}"}

            if "scientific_name" not in translated_result:
                _self.logger.warning(
                    "No scientific name in translation response for: %s", species_name
                )
                return (
                    []
                )  # or return {"error": f"Translation response invalid for: {species_name}"}

            species_name = translated_result["scientific_name"]
            if not species_name:  # Check if the value is empty string or None
                _self.logger.warning(
                    "Empty scientific name returned for: %s", species_name
                )
                return (
                    []
                )  # or return {"error": f"Empty scientific name returned for: {species_name}"}

            # Query setup timing
            if "country_code" in content:
                country_code = content["country_code"]
                _self.logger.info(
                    "Fetching occurrences for species: %s and country: %s",
                    species_name,
                    country_code,
                )
            else:
                country_code = None
                _self.logger.info("Fetching occurrences for species: %s", species_name)
            client = bigquery.Client(
                project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            )
            # Base query with parameterization
            base_query = """
                SELECT 
                    decimallatitude,
                    decimallongitude
                FROM `{project_id}.biodiversity.occurances_endangered_species_mammals`
                WHERE LOWER(species) = LOWER(@species_name)
                    AND decimallatitude IS NOT NULL
                    AND decimallongitude IS NOT NULL 
                    AND eventdate IS NOT NULL 
                    {where_clause}
            """

            where_clause = "AND countrycode = @country_code" if country_code else ""
            query = _self.build_query(
                base_query, where_clause=where_clause
            )

            # Query execution timing
            query_start = time.time()
            parameters = _self.get_parameters(
                species_name=species_name, country_code=country_code
            )

            job_config = bigquery.QueryJobConfig(query_parameters=parameters)
            query_job = client.query(query, job_config=job_config)
            total_occurrences = [
                {
                    "species": species_name,
                    "decimallatitude": row.decimallatitude,
                    "decimallongitude": row.decimallongitude,
                }
                for row in query_job
            ]
            _self.logger.info(
                "Query execution took %.2f seconds", time.time() - query_start
            )

            _self.logger.info(
                "Successfully fetched %d occurrences for species %s%s in %.2f seconds",
                len(total_occurrences),
                species_name,
                " and country " + country_code if country_code else "",
                time.time() - start_time,
            )
            return {
                "species": species_name,
                "occurrence_count": len(total_occurrences),
                "occurrences": total_occurrences,
            }

        except google.api_core.exceptions.GoogleAPIError as e:
            _self.logger.error(
                "BigQuery error (took %.2f seconds): %s",
                time.time() - start_time,
                str(e),
                exc_info=True,
            )
            raise
        except KeyError as e:
            _self.logger.error(
                "Missing required field (took %.2f seconds): %s",
                time.time() - start_time,
                str(e),
                exc_info=True,
            )
            raise
        except (TypeError, ValueError) as e:
            _self.logger.error(
                "Invalid input (took %.2f seconds): %s",
                time.time() - start_time,
                str(e),
                exc_info=True,
            )
            raise

    def get_species_info_from_api(self, content):
        """
        Retrieves species information from GBIF API.

        Raises:
            ValueError: If species name is invalid
            pygbif.gbif.GbifError: If GBIF API request fails
        """
        start_time = time.time()
        try:
            species_name = content["name"]
            self.logger.info("Fetching species info for: %s", species_name)
            # GBIF API call timing
            api_call_start = time.time()
            species_info = species.name_backbone(species_name)
            self.logger.info(
                "GBIF API call took %.2f seconds", time.time() - api_call_start
            )
            # JSON serialization timing
            json_start = time.time()
            result = json.dumps(species_info)
            self.logger.info(
                "JSON serialization took %.2f seconds", time.time() - json_start
            )
            self.logger.info(
                "Total get_species_info took %.2f seconds", time.time() - start_time
            )
            return result
        except KeyError as e:
            self.logger.error(
                "Missing species name (took %.2f seconds): %s",
                time.time() - start_time,
                str(e),
                exc_info=True,
            )
            raise ValueError("Species name is required") from e
        except Exception as e:  # GBIF doesn't expose specific exceptions
            self.logger.error(
                "GBIF API error (took %.2f seconds): %s",
                time.time() - start_time,
                str(e),
                exc_info=True,
            )
            raise

    def handle_get_country_geojson(self, content):
        """
        Retrieves GeoJSON data for a specific country from cached world data.

        Args:
            content (dict): Dictionary containing either:
                - country_name (str): Name of the country in English
                - country_code (str): Two-letter ISO country code

        Returns:
            str: JSON string containing GeoJSON data for the specified country
                 or error message if country not found

        Raises:
            ValueError: If country_name is invalid or not found
            FileNotFoundError: If GeoJSON file is not found
            TypeError: If content is not in expected format
            json.JSONDecodeError: If GeoJSON file is malformed
        """
        country_name = content.get("country_name")
        country_code = content.get("country_code")
        self.logger.info("Fetching GeoJSON data for country: %s", country_name)
        try:
            self.load_world_geojson()
            # Determine which property to use based on country code length
            country_data = None
            if country_code is not None:
                code_length = len(country_code)
                if code_length == 2:
                    country_data = next(
                        (
                            feature
                            for feature in self.world_data["gdf"].get("features", [])
                            if feature["properties"]["ISO_A2"].lower()
                            == country_code.lower()
                        ),
                        None,
                    )
                elif code_length == 3:
                    country_data = next(
                        (
                            feature
                            for feature in self.world_data["gdf"].get("features", [])
                            if feature["properties"]["ISO_A3"].lower()
                            == country_code.lower()
                        ),
                        None,
                    )
            elif country_name is not None:
                country_data = next(
                    (
                        feature
                        for feature in self.world_data["gdf"].get("features", [])
                        if feature["properties"]["NAME_EN"].lower()
                        == country_name.lower()
                    ),
                    None,
                )

            if country_data is None:
                country_identifier = (
                    country_name if country_name is not None else country_code
                )
                self.logger.warning("Country not found: %s", country_identifier)
                return {"error": f"Country not found: {country_identifier}"}
            # Convert to GeoJSON
            country_geojson = country_data
            self.logger.info("Successfully retrieved GeoJSON for %s", country_name)
            self.logger.info("GeoJSON data: %s", country_geojson)
            return json.dumps(country_geojson)
        except FileNotFoundError as e:
            self.logger.error("GeoJSON file not found: %s", str(e), exc_info=True)
            raise
        except json.JSONDecodeError as e:
            self.logger.error("Invalid GeoJSON format: %s", str(e), exc_info=True)
            raise
        except (TypeError, ValueError) as e:
            self.logger.error("Invalid input: %s", str(e), exc_info=True)
            raise

    def load_world_geojson(self):
        """
        Loads world GeoJSON data.

        Raises:
            requests.RequestException: If GeoJSON download fails
            json.JSONDecodeError: If GeoJSON is malformed
        """
        try:
            if "gdf" not in st.session_state or st.session_state.gdf is None:
                response = requests.get(self.world_data["geojson_url"], timeout=30)
                response.raise_for_status()
                self.world_data["gdf"] = response.json()
                st.session_state.gdf = self.world_data["gdf"]
        except requests.RequestException as e:
            self.logger.error("Failed to download GeoJSON: %s", str(e), exc_info=True)
            raise
        except json.JSONDecodeError as e:
            self.logger.error("Invalid GeoJSON format: %s", str(e), exc_info=True)
            raise

    def get_protected_areas_geojson(self, content: dict) -> str:
        """Get GeoJSON data for protected areas in a country.

        Args:
            content (dict): Dictionary containing three letter country_code (str)

        Returns:
            str: JSON string containing protected areas GeoJSON data

        Raises:
            ValueError: If country_code is invalid
            google.api_core.exceptions.GoogleAPIError: If BigQuery query fails
        """
        try:
            country_code = content["country_code"]
            client = bigquery.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
            query = self.build_geojson_query()
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "country_code", "STRING", country_code
                    )
                ]
            )
            query_job = client.query(query, job_config=job_config)
            results = [
                {
                    "name": row.name,
                    "category": row.IUCN_CAT,
                    "geojson": json.loads(row.geojson),
                }
                for row in query_job
            ]
            return json.dumps(results)
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            self.logger.error("Invalid input or GeoJSON: %s", str(e))
            raise
        except google.api_core.exceptions.GoogleAPIError as e:
            self.logger.error("BigQuery error: %s", str(e))
            raise

    def normalize_protected_area_name(self, name: str) -> str:
        """
        Normalize protected area name by removing common suffixes and extra spaces.
        Args:
            name (str): Name of protected area to normalize

        Returns:
            str: Normalized name
        """
        suffixes = [
            "national park",
            "national reserve",
            "game reserve",
            "conservation area",
            "marine park",
            "wildlife sanctuary",
            "nature reserve",
            "park",
            "reserve",
        ]

        name = name.lower().strip()

        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[: -(len(suffix))].strip()
                break

        return name

    def build_geojson_query(self) -> str:
        """Build the BigQuery query string for geojson query."""
        query = """
            SELECT name, IUCN_CAT, ST_ASGEOJSON(geometry) as geojson
            FROM `{project_id}.biodiversity.protected_areas_africa`
            WHERE ISO3 = @country_code
            AND IUCN_CAT IS NOT NULL
            AND IUCN_CAT IN ('Ia', 'Ib', 'II', 'III', 'IV', 'V', 'VI')
        """
        return self.build_query(query, where_clause="")

    def read_indicator_values(self, content):
        """
        Retrieves property data (e.g., terrestrial_hci, popden) for specified countries.

        Args:
            content (dict): Dictionary containing:
                - country_names (list, optional): List of country names
                - country_codes (list, optional): List of 2 or 3-letter country codes
                - property_name: Name of the property to query (e.g., 'terrestrial_hci', 'popden')

        Returns:
            dict: Dictionary containing data for each country or error message
        """
        property_name = content.get("property_name", "terrestrial_hci")
        country_names = content.get("country_names", [])
        country_codes = content.get("country_codes", [])

        if not country_names and not country_codes:
            return {"error": "No countries specified"}

        try:
            self.load_world_geojson()
            results = {}

            # Process countries specified by name
            for country_name in country_names:
                self.logger.info("Processing country name: %s", country_name)
                country_data = next(
                    (
                        feature
                        for feature in self.world_data["gdf"].get("features", [])
                        if feature["properties"]["NAME_EN"].lower()
                        == country_name.lower()
                    ),
                    None,
                )
                if country_data:
                    results[country_name] = self._process_country_data(
                        country_data, property_name
                    )
                else:
                    self.logger.warning("Country not found: %s", country_name)
                    results[country_name] = {
                        "error": f"Country not found: {country_name}"
                    }

            # Process countries specified by code
            for country_code in country_codes:
                code_length = len(country_code)
                country_data = None

                if code_length == 2:
                    country_data = next(
                        (
                            feature
                            for feature in self.world_data["gdf"].get("features", [])
                            if feature["properties"]["ISO_A2"].lower()
                            == country_code.lower()
                        ),
                        None,
                    )
                elif code_length == 3:
                    country_data = next(
                        (
                            feature
                            for feature in self.world_data["gdf"].get("features", [])
                            if feature["properties"]["ISO_A3"].lower()
                            == country_code.lower()
                        ),
                        None,
                    )

                if country_data:
                    country_name = country_data["properties"]["NAME_EN"]
                    results[country_name] = self._process_country_data(
                        country_data, property_name
                    )
                else:
                    self.logger.warning("Country not found for code: %s", country_code)
                    results[country_code] = {
                        "error": f"Country not found for code: {country_code}"
                    }

            if all("error" in country_data for country_data in results.values()):
                return {"error": "No data found for any of the specified countries"}

            return {
                "countries": results,
                "property_name": property_name,
                "type": "comparison",
            }

        except Exception as e:
            self.logger.error(
                "Error fetching %s: %s", property_name, str(e), exc_info=True
            )
            raise

    def _process_country_data(self, country_data, property_name):
        """Helper method to process country data and run the query."""
        country_geojson = country_data.get("geometry")
        country_properties = country_data.get("properties", {})
        country_name = country_properties.get("NAME_EN")

        query = f"""
            SELECT {property_name}, decimallongitude, decimallatitude
            FROM `tribal-logic-351707.biodiversity.hci` 
            WHERE decimallatitude is not null
            AND decimallongitude is not null
            AND {property_name} is not null
            AND ST_DWithin(
                ST_GeogFromGeoJson(
                    @geojson
                ),
                ST_GeogPoint(decimallongitude, decimallatitude),
                100
            )
        """

        client = bigquery.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "geojson", "STRING", json.dumps(country_geojson)
                )
            ]
        )
        query_job = client.query(query, job_config=job_config)

        country_results = [
            {
                property_name: getattr(row, property_name),
                "decimallatitude": row.decimallatitude,
                "decimallongitude": row.decimallongitude,
            }
            for row in query_job
        ]

        return {
            "data": country_results,
            "geojson": country_data,
            "property_name": property_name,
            "country_codes": {
                "iso_a2": country_properties.get("ISO_A2", ""),
                "iso_a3": country_properties.get("ISO_A3", ""),
            },
            "country_name": country_name,
        }

    def read_terrestrial_hci(self, content):
        """
        Retrieves terrestrial human coexistence index data for a specified country.

        Args:
            content (dict): Dictionary containing country_name or country_code

        Returns:
            list: List of HCI values with coordinates or dict with error message
        """
        content["property_name"] = "terrestrial_hci"
        return self.read_indicator_values(content)

    def read_population_density(self, content):
        """
        Retrieves population density data for a specified country.

        Args:
            content (dict): Dictionary containing country_name or country_code

        Returns:
            list: List of population density values with coordinates or dict with error message
        """
        content["property_name"] = "popden2010"
        return self.read_indicator_values(content)

    def _query_country_occurrences(
        self, project_id: str, base_query: str, country_code: str, scientific_name: str
    ) -> list:
        """Query species occurrences for a specific country."""
        code_length = len(country_code)
        code_field = "ISO_A2" if code_length == 2 else "ISO_A3"

        query = base_query.format(
            project=project_id,
            join_clause=f"""
                INNER JOIN `{project_id}.biodiversity.countries` c
                ON ST_CONTAINS(c.geometry, 
                    ST_GEOGPOINT(o.decimallongitude, o.decimallatitude))
            """,
            where_clause=f"AND c.{code_field} = @country_code",
        )

        client = bigquery.Client(project=project_id)
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "species_name", "STRING", scientific_name
                ),
                bigquery.ScalarQueryParameter("country_code", "STRING", country_code),
            ]
        )

        query_job = client.query(query, job_config=job_config)
        return [{"year": row.year, "count": row.count} for row in query_job]

    def get_yearly_occurrences(self, content):
        """
        Get yearly occurrence counts for a species, optionally filtered by countries.

        Args:
            content (dict): Dictionary containing:
                - species_name (str): Common name of the species
                - country_codes (list, optional): List of 2 or 3-letter country codes

        Returns:
            dict: Yearly occurrence data or error message
        """
        try:
            if not content.get("species_name"):
                return {"error": "Species name is required"}

            # Translate name and validate
            translated = json.loads(
                self.translate_to_scientific_name_from_api(
                    {"name": content["species_name"]}
                )
            )
            if "error" in translated or not translated.get("scientific_name"):
                return {
                    "error": f"Could not find valid scientific name for: {content['species_name']}"
                }

            # Base query template
            base_query = """
                SELECT 
                    EXTRACT(YEAR FROM eventdate) as year,
                    COUNT(*) as count
                FROM `{project}.biodiversity.occurances_endangered_species_mammals` o
                {join_clause}
                WHERE LOWER(o.species) = LOWER(@species_name)
                AND eventdate IS NOT NULL and eventdate > '1980-01-01' 
                {where_clause}
                GROUP BY year
                ORDER BY year
            """

            results = {}
            if content.get("country_codes"):
                # Process each country
                for code in content["country_codes"]:
                    country_data = self._query_country_occurrences(
                        os.getenv("GOOGLE_CLOUD_PROJECT"),
                        base_query,
                        code,
                        translated["scientific_name"],
                    )
                    if country_data:
                        results[code] = country_data
            else:
                # Query without country filter
                client = bigquery.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
                query_job = client.query(
                    base_query.format(
                        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                        join_clause="",
                        where_clause="",
                    ),
                    bigquery.QueryJobConfig(
                        query_parameters=[
                            bigquery.ScalarQueryParameter(
                                "species_name", "STRING", translated["scientific_name"]
                            )
                        ]
                    ),
                )
                results = [{"year": row.year, "count": row.count} for row in query_job]

            if not results:
                return {
                    "error": f"No occurrence data found for "
                    f"{content['species_name']} ({translated['scientific_name']})"
                }

            return {
                "common_name": content["species_name"],
                "scientific_name": translated["scientific_name"],
                "yearly_data": results,
                "type": "temporal",
            }

        except Exception as e:
            self.logger.error(
                "Error getting yearly occurrences: %s", str(e), exc_info=True
            )
            raise
