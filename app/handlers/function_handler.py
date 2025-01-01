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
from typing import List
import requests
import google.api_core.exceptions
from google.cloud import bigquery
from pygbif import species
#from EcoNameTranslator import to_common
from langchain_google_community import GoogleSearchAPIWrapper
import streamlit as st
from app.models.function_declarations import FUNCTION_DECLARATIONS

class FunctionHandler:
    """
    Handles the declaration and implementation of biodiversity-related functions.

    This class manages various functions related to species information, geographical data,
    and endangered species queries. It provides caching mechanisms for performance optimization
    and handles error logging.

    Attributes:
        logger (Logger): Class-specific logger instance
        declarations (dict): Dictionary of function declarations for Vertex AI
        function_handler (dict): Mapping of function names to their implementations
        world_gdf (dict): Cached GeoJSON data for world geographical features
        search (GoogleSearchAPIWrapper): Instance of Google Search API wrapper
    """

    def __init__(self):
        """
        Initializes the FunctionHandler.

        Raises:
            FileNotFoundError: If required files cannot be loaded
            ValueError: If configuration is invalid
        """
        try:
            self.logger = logging.getLogger("BioChat." + self.__class__.__name__)
            self.logger.info("Initializing FunctionHandler")
            self.setup_function_declarations()
            self.search = GoogleSearchAPIWrapper()
            self.world_gdf = None
            self.world_geojson_url = (
                "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
                "master/geojson/ne_110m_admin_0_countries.geojson"
            )
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
                "endangered_species_for_family": self.endangered_species_for_family,
                "endangered_classes_for_kingdom": self.endangered_classes_for_kingdom,
                "endangered_families_for_order": self.endangered_families_for_order,
                "endangered_orders_for_class": self.endangered_orders_for_class,
                "endangered_species_for_country": self.endangered_species_for_country,
                "number_of_endangered_species_by_conservation_status":
                self.number_of_endangered_species_by_conservation_status,
                "google_search": self.google_search
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
            query_string = content.get('query')
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
            # Translation timing
            translation_start = time.time()
            species_name = content['species_name']
            scientific_name = _self.translate_to_scientific_name_from_api({'name': species_name})
            if scientific_name:
                species_name = json.loads(scientific_name).get('scientific_name')
            _self.logger.info("Translation took %.2f seconds", time.time() - translation_start)

            # Query setup timing
            query_setup_start = time.time()
            if 'country_code' in content:
                country_code = content['country_code']
                _self.logger.info("Fetching occurrences for species: %s and country: %s",
                              species_name, country_code)
            else:
                country_code = None
                _self.logger.info("Fetching occurrences for species: %s", species_name)
            client = bigquery.Client(
                    project=os.getenv('GOOGLE_CLOUD_PROJECT'),
                )
            # Base query with parameterization
            query = """
                SELECT 
                    decimallatitude,
                    decimallongitude
                FROM `{}.biodiversity.cached_occurrences`
                WHERE LOWER(species) = LOWER(@species_name)
                    AND decimallatitude IS NOT NULL
                    AND decimallongitude IS NOT NULL
            """
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
            query = query.format(project_id)
            if country_code is not None:
                query += " AND countrycode = @country_code"
                job_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("species_name", "STRING", species_name),
                        bigquery.ScalarQueryParameter("country_code", "STRING", country_code)
                    ]
                )
            else:
                job_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("species_name", "STRING", species_name)
                    ]
                )
            _self.logger.info("Query setup took %.2f seconds", time.time() - query_setup_start)

            # Query execution timing
            query_start = time.time()
            query_job = client.query(
                query,
                job_config=job_config
            )

            # Efficient result processing
            results = [{
                "species": species_name,
                "decimallatitude": row.decimallatitude,
                "decimallongitude": row.decimallongitude
            } for row in query_job]
            _self.logger.info("Query execution took %.2f seconds", time.time() - query_start)

            _self.logger.info(
                "Successfully fetched %d occurrences for species %s%s in %.2f seconds",
                len(results),
                species_name,
                ' and country ' + country_code if country_code else '',
                time.time() - start_time
            )
            return results

        except google.api_core.exceptions.GoogleAPIError as e:
            _self.logger.error("BigQuery error (took %.2f seconds): %s",
                             time.time() - start_time, str(e), exc_info=True)
            raise
        except KeyError as e:
            _self.logger.error("Missing required field (took %.2f seconds): %s",
                             time.time() - start_time, str(e), exc_info=True)
            raise
        except (TypeError, ValueError) as e:
            _self.logger.error("Invalid input (took %.2f seconds): %s",
                             time.time() - start_time, str(e), exc_info=True)
            raise

#    @st.cache_data(
#        ttl=3600,  # Cache for 1 hour
#        show_spinner="Translating species name...",
#        max_entries=100
#    )
    def translate_to_scientific_name_from_api(_self, content: dict) -> str:  # pylint: disable=no-self-argument
        """
        Translates a common species name to its scientific name using the EBI Taxonomy REST API.
        Results are cached for 1 hour to improve performance and reduce API calls.

        Args:
            content (dict): A dictionary containing:
                - name (str): The common name of the species to translate

        Returns:
            str: A JSON-formatted string containing either:
                - Success: {"scientific_name": "<scientific_name>"}
                - Error: {"error": "<error_message>"}

        Raises:
            requests.Timeout: If the API request exceeds 10 seconds
            requests.RequestException: If the API request fails
            json.JSONDecodeError: If the API response cannot be parsed
            KeyError, ValueError: If the response format is unexpected

        """
        species_name = content.get('name', '').strip()
        if not species_name:
            return json.dumps({"error": "No species name provided"})

        try:
            # Configure request with timeout and headers
            _self.logger.info("Fetching scientific name for species: %s", species_name)
            url = f"https://www.ebi.ac.uk/ena/taxonomy/rest/any-name/{species_name}"
            headers = {
                'Accept': 'application/json'
            }
            # Make request with timeout
            response = requests.get(
                url,
                headers=headers,
                timeout=5
            )
            response.raise_for_status()
            # Parse response
            data = response.json()
            if data and isinstance(data, list) and len(data) > 0:
                scientific_name = data[0].get("scientificName")
                if scientific_name:
                    _self.logger.info("Successfully translated '%s' to '%s'",
                                     species_name, scientific_name)
                    return json.dumps({"scientific_name": scientific_name})
            _self.logger.warning("Could not translate name: %s", species_name)
            return json.dumps({"error": "Name could not be translated to scientific name"})
        except requests.Timeout:
            _self.logger.error("Request timeout for species: %s", species_name)
            return json.dumps({"error": "Request timed out"})
        except requests.RequestException as e:
            _self.logger.error("API request failed for species %s: %s", species_name, str(e))
            return json.dumps({"error": f"API request failed: {str(e)}"})
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            _self.logger.error("Failed to parse response for species %s: %s", species_name, str(e))
            return json.dumps({"error": f"Failed to parse response: {str(e)}"})


    def get_species_info_from_api(self, content):
        """
        Retrieves species information from GBIF API.
        
        Raises:
            ValueError: If species name is invalid
            pygbif.gbif.GbifError: If GBIF API request fails
        """
        start_time = time.time()
        try:
            species_name = content['name']
            self.logger.info("Fetching species info for: %s", species_name)
            # GBIF API call timing
            api_call_start = time.time()
            species_info = species.name_backbone(species_name)
            self.logger.info("GBIF API call took %.2f seconds", time.time() - api_call_start)
            # JSON serialization timing
            json_start = time.time()
            result = json.dumps(species_info)
            self.logger.info("JSON serialization took %.2f seconds", time.time() - json_start)
            self.logger.info("Total get_species_info took %.2f seconds", time.time() - start_time)
            return result
        except KeyError as e:
            self.logger.error("Missing species name (took %.2f seconds): %s",
                            time.time() - start_time, str(e), exc_info=True)
            raise ValueError("Species name is required") from e
        except Exception as e:  # GBIF doesn't expose specific exceptions
            self.logger.error("GBIF API error (took %.2f seconds): %s",
                            time.time() - start_time, str(e), exc_info=True)
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
        country_name = content.get('country_name')
        country_code = content.get('country_code')
        self.logger.info("Fetching GeoJSON data for country: %s", country_name)
        try:
            self.load_world_geojson()
            country_data = None
            if country_code is not None:
                country_data = next((feature for feature in  self.world_gdf.get("features", [])
                                     if feature["properties"]["ISO_A2"].lower()
                                        == country_code.lower()), None)
            elif country_name is not None:
                country_data = next((feature for feature in self.world_gdf.get("features", [])
                                     if feature["properties"]["NAME_EN"].lower()
                                        == country_name.lower()), None)
            if country_data is None or len(country_data) == 0:
                country_identifier = country_name if country_name is not None else country_code
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
            if 'world_gdf' not in st.session_state or st.session_state.world_gdf is None:
                response = requests.get(self.world_geojson_url, timeout=30)
                response.raise_for_status()
                self.world_gdf = response.json()
                st.session_state.world_gdf = self.world_gdf
        except requests.RequestException as e:
            self.logger.error("Failed to download GeoJSON: %s", str(e), exc_info=True)
            raise
        except json.JSONDecodeError as e:
            self.logger.error("Invalid GeoJSON format: %s", str(e), exc_info=True)
            raise


    def endangered_classes_for_kingdom(self, content) -> str:
        """
        Retrieves endangered classes within a specified kingdom and their species counts.

        Args:
            content (dict): Dictionary containing:
                - kingdom_name (str): Name of the kingdom to query (e.g., 'Animalia', 'Plantae')

        Returns:
            str: Formatted string with classes and their endangered species counts

        Raises:
            ValueError: If kingdom_name is invalid or not found
            google.api_core.exceptions.GoogleAPIError: If BigQuery query fails
            TypeError: If content is not in expected format
        """
        try:
            kingdom_name = content['kingdom_name']
            self.logger.info("Fetching classes for kingdom from BigQuery")

            client = bigquery.Client(
                project=os.getenv('GOOGLE_CLOUD_PROJECT'),
            )
            query = """
                SELECT class, count(class) as cnt 
                FROM `{}.biodiversity.endangered_species` 
                WHERE LOWER(kingdom) = LOWER(@kingdom) AND class IS NOT NULL 
                GROUP BY class ORDER BY class
            """
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
            query = query.format(project_id)
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("kingdom", "STRING", kingdom_name)
                ]
            )
            query_job = client.query(
                query,
                job_config=job_config
            )
            results = []
            intro = f"Here are the classes within the {kingdom_name} kingdom:\n\n"
            for row in query_job:
                formatted_entry = f"* **{row['class']}**: {row['cnt']} endangered species"
                results.append(formatted_entry)
            final_text = intro + '\n'.join(results)
            return final_text
        except google.api_core.exceptions.GoogleAPIError as e:
            self.logger.error("BigQuery error: %s", str(e), exc_info=True)
            raise
        except (TypeError, ValueError) as e:
            self.logger.error("Invalid input: %s", str(e), exc_info=True)
            raise

    def endangered_orders_for_class(self, content) -> str:
        """
        Retrieves endangered orders within a specified class and their species counts.

        Args:
            content (dict): Dictionary containing:
                - class_name (str): Name of the class to query (e.g., 'Mammalia', 'Aves')

        Returns:
            str: Formatted string with orders and their endangered species counts

        Raises:
            ValueError: If class_name is invalid or not found
            google.api_core.exceptions.GoogleAPIError: If BigQuery query fails
            TypeError: If content is not in expected format
        """
        try:
            clazz = content['class_name']
            self.logger.info("Fetching families for classes from BigQuery")

            client = bigquery.Client(
                project=os.getenv('GOOGLE_CLOUD_PROJECT'),
            )
            query = """
                SELECT order_name, count(order_name) as cnt 
                FROM `{}.biodiversity.endangered_species` 
                WHERE LOWER(class) = LOWER(@class) 
                    AND order_name IS NOT NULL 
                GROUP BY order_name 
                ORDER BY order_name
            """
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
            query = query.format(project_id)
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("class", "STRING", clazz)
                ]
            )
            query_job = client.query(
                query,
                job_config=job_config
            )
            results = []
            intro = f"Here are the orders within the {clazz} class:\n\n"
            for row in query_job:
                formatted_entry = f"* **{row['order_name']}**: {row['cnt']} endangered species"
                results.append(formatted_entry)
            final_text = intro + '\n'.join(results)
            return final_text
        except google.api_core.exceptions.GoogleAPIError as e:
            self.logger.error("BigQuery error: %s", str(e), exc_info=True)
            raise
        except (TypeError, ValueError) as e:
            self.logger.error("Invalid input: %s", str(e), exc_info=True)
            raise

    def endangered_families_for_order(self, content) -> str:
        """
        Retrieves endangered families within a specified order and their species counts.

        Args:
            content (dict): Dictionary containing:
                - order_name (str): Name of the order to query (e.g., 'Primates', 'Carnivora')

        Returns:
            str: Formatted string with families and their endangered species counts

        Raises:
            ValueError: If order_name is invalid or not found
            google.api_core.exceptions.GoogleAPIError: If BigQuery query fails
            TypeError: If content is not in expected format
        """
        try:
            self.logger.info("Fetching families for order from BigQuery")
            order_name = content['order_name']
            client = bigquery.Client(
                project=os.getenv('GOOGLE_CLOUD_PROJECT'),
            )
            query = """
                SELECT family_name, count(family_name) as cnt 
                FROM `{}.biodiversity.endangered_species` 
                WHERE LOWER(order_name) = LOWER(@order_name) 
                    AND family_name IS NOT NULL 
                GROUP BY family_name 
                ORDER BY family_name
            """
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
            query = query.format(project_id)
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("order_name", "STRING", order_name)
                ]
            )
            query_job = client.query(
                query,
                job_config=job_config
            )
            intro = (f"Here are the families within the {order_name} order, along with "
                    "the number of endangered species in each:\n\n")
            results = []
            for row in query_job:
                formatted_entry = f"* **{row['family_name']}**: {row['cnt']} endangered species"
                results.append(formatted_entry)

            final_text = intro + '\n'.join(results)
            return final_text
        except google.api_core.exceptions.GoogleAPIError as e:
            self.logger.error("BigQuery error: %s", str(e), exc_info=True)
            raise
        except (TypeError, ValueError) as e:
            self.logger.error("Invalid input: %s", str(e), exc_info=True)
            raise

    def endangered_species_for_family(self, content) -> str:
        """
        Retrieves endangered species within a specified family.

        Args:
            content (dict): Dictionary containing:
                - family_name (str): Name of the family to query (e.g., 'Hominidae', 'Lemuridae')
                - conservation_status (str, optional):
                    Status to filter by (e.g., 'Critically Endangered')

        Returns:
            str: Formatted string with endangered species and their IUCN links

        Raises:
            ValueError: If family_name is invalid or not found
            google.api_core.exceptions.GoogleAPIError: If BigQuery query fails
            TypeError: If content is not in expected format
        """
        try:
            self.logger.info("Fetching endangered species for family from BigQuery")
            family_name = content['family_name']
            conservation_status = None
            if 'conservation_status' in content:
                conservation_status = content['conservation_status']

            results_data = self._query_endangered_species(family_name, conservation_status)
            return self._format_species_results(results_data)
        except google.api_core.exceptions.GoogleAPIError as e:
            self.logger.error("BigQuery error: %s", str(e), exc_info=True)
            raise
        except (TypeError, ValueError) as e:
            self.logger.error("Invalid input: %s", str(e), exc_info=True)
            raise


    def _query_endangered_species(self, family_name: str,
                                conservation_status: str = None) -> List[tuple]:
        """Execute BigQuery to fetch endangered species data."""
        client = bigquery.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'))
        query = self._build_species_query(conservation_status)
        parameters = self._get_query_parameters(family_name, conservation_status)
        job_config = bigquery.QueryJobConfig(query_parameters=parameters)
        query_job = client.query(query, job_config=job_config)
        return [(row['species_header'], row['urls']) for row in query_job]

    def _build_species_query(self, conservation_status: str = None) -> str:
        """Build the BigQuery query string."""
        base_query = """
           SELECT 
               CONCAT(genus_name, ' ', species_name, ':') as species_header,
               STRING_AGG(url, '||' ORDER BY url) as urls
           FROM `{}.biodiversity.endangered_species` 
           WHERE LOWER(family_name) = LOWER(@family_name) 
           AND species_name IS NOT NULL 
           AND genus_name IS NOT NULL 
           {conservation_status_filter}
           GROUP BY genus_name, species_name
           ORDER BY species_header
        """
        conservation_status_filter = (
            "AND LOWER(conservation_status) = LOWER(@conservation_status)"
            if conservation_status
            else ""
        )
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        return base_query.format(project_id, conservation_status_filter=conservation_status_filter)

    def _get_query_parameters(self, family_name: str,
                            conservation_status: str = None) -> List[bigquery.ScalarQueryParameter]:
        """Create query parameters for BigQuery."""
        parameters = [
            bigquery.ScalarQueryParameter("family_name", "STRING", family_name)
        ]
        if conservation_status:
            parameters.append(
                bigquery.ScalarQueryParameter("conservation_status", "STRING", conservation_status)
            )
        return parameters

    def _format_species_results(self, results_data: List[tuple]) -> str:
        """Format the species results into a readable string."""
        formatted_entries = []
        for scientific_name, urls in results_data:
            urls_formatted = self._format_urls(urls)
            formatted_entry = self._format_species_entry(scientific_name, urls_formatted)
            formatted_entries.append(formatted_entry)

        return '\n'.join(formatted_entries)

    def _format_urls(self, urls: str) -> str:
        """Format URLs into a bulleted list."""
        return '\n'.join(f'    * {url}' for url in urls.split('||'))

    def _format_species_entry(self, scientific_name: str, urls_formatted: str) -> str:
        """Format a single species entry."""
        # Remove the colon from scientific name
        scientific_name = scientific_name.split(':')[0]
        return f"* **{scientific_name}**:\n{urls_formatted}"

    def endangered_species_for_country(self, content) -> str:
        """
        Retrieves endangered species for a specified country.

        Args:
            content (dict): Dictionary containing:
                - country_code (str): Two-letter country code to query (e.g., 'DE', 'ES')
                - conservation_status (str, optional):
                    Status to filter by (e.g., 'Critically Endangered')

        Returns:
            str: Formatted string with endangered species in the country

        Raises:
            ValueError: If country_code is invalid or not found
            google.api_core.exceptions.GoogleAPIError: If BigQuery query fails
            TypeError: If content is not in expected format
        """
        try:
            self.logger.info("Fetching species for country from BigQuery")
            country_code = content['country_code']
            if 'conservation_status' in content:
                conservation_status = content['conservation_status']
            else:
                conservation_status = None

            results_data = self._query_country_species(country_code, conservation_status)
            return self._format_country_species_results(results_data)
        except google.api_core.exceptions.GoogleAPIError as e:
            self.logger.error("BigQuery error: %s", str(e), exc_info=True)
            raise
        except (TypeError, ValueError) as e:
            self.logger.error("Invalid input: %s", str(e), exc_info=True)
            raise

    def _query_country_species(self, country_code: str,
                                conservation_status: str = None) -> List[tuple]:
        """Execute BigQuery to fetch country's endangered species data."""
        client = bigquery.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'))
        query = self._build_country_species_query(conservation_status)
        parameters = self._get_country_query_parameters(country_code, conservation_status)
        job_config = bigquery.QueryJobConfig(query_parameters=parameters)
        start_query = time.time()
        query_job = client.query(query, job_config=job_config)
        results = query_job.result()
        self.logger.info("BigQuery query completed in %.2f seconds", time.time() - start_query)
        return [(row['species_name'], row['family'], row['status'], row['url']) for row in results]

    def _build_country_species_query(self, conservation_status: str = None) -> str:
        """Build the BigQuery query string for country species."""
        base_query = """
            SELECT DISTINCT CONCAT(genus_name, ' ', species_name) as species_name, 
                   family_name as family, conservation_status as status, url
            FROM `{}.biodiversity.endangered_species` sp
            JOIN `{}.biodiversity.occurances_endangered_species_mammals` oc 
                ON CONCAT(genus_name, ' ', species_name) = oc.species 
            WHERE species_name IS NOT NULL 
                AND genus_name IS NOT NULL 
                AND oc.countrycode = @country_code
                {conservation_status_filter}
            GROUP BY genus_name, species_name, family_name, conservation_status, url
            ORDER BY species_name
        """
        conservation_status_filter = (
            "AND LOWER(conservation_status) = LOWER(@conservation_status)"
            if conservation_status
            else ""
        )
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        return base_query.format(project_id, project_id,
                                 conservation_status_filter=conservation_status_filter)

    def _get_country_query_parameters(self, country_code: str,
                            conservation_status: str = None) -> List[bigquery.ScalarQueryParameter]:
        """Create query parameters for country species query."""
        parameters = [
            bigquery.ScalarQueryParameter("country_code", "STRING", country_code)
        ]
        if conservation_status:
            parameters.append(
                bigquery.ScalarQueryParameter("conservation_status", "STRING", conservation_status)
            )
        return parameters

    def _format_country_species_results(self, results_data: List[tuple]) -> str:
        """Format the country species results into a readable string, grouped by family."""
        result = "**Only Mammals are included in the list.**\n\n"
        # Group results by family
        family_groups = {}
        for scientific_name, family, status, url in sorted(results_data, key=lambda x: x[1]):
            if family not in family_groups:
                family_groups[family] = []
            family_groups[family].append((scientific_name, status, url))
        # Format each family group
        for family in sorted(family_groups.keys()):
            result += f"**Family: {family}**\n"
            for scientific_name, status, url in sorted(family_groups[family]):
                result += self._format_country_species_entry(scientific_name, status, url)
            result += "\n"
        return result

    def _format_country_species_entry(self, scientific_name: str, status: str, url: str) -> str:
        """Format a single country species entry."""
        return f"* **{scientific_name}** ({status}):\n  {url}\n"

    def number_of_endangered_species_by_conservation_status(self, content) -> str:
        """
        Retrieves count of endangered species by conservation status.

        Args:
            content (dict): Dictionary containing:
                - country_code (str, optional):
                    Two-letter country code to filter by (e.g., 'DE', 'ES')
                - conservation_status (str, optional):
                    Status to filter by (e.g., 'Critically Endangered')

        Returns:
            str: Formatted string with species counts per conservation status

        Raises:
            ValueError: If country_code is invalid
            google.api_core.exceptions.GoogleAPIError: If BigQuery query fails
            TypeError: If content is not in expected format
        """
        try:
            country_code = content.get('country_code')
            conservation_status = content.get('conservation_status')
            if not country_code and not conservation_status:
                # If neither country_code nor conservation_status is provided,
                # return global statistics
                results = self._query_conservation_status_counts()
            else:
                # If either country_code or conservation_status is provided,
                #  return statistics for the specified country
                results = self._query_conservation_status_counts(country_code)
            return self._format_conservation_counts(results)
        except google.api_core.exceptions.GoogleAPIError as e:
            self.logger.error("BigQuery error: %s", str(e), exc_info=True)
            raise
        except (TypeError, ValueError) as e:
            self.logger.error("Invalid input: %s", str(e), exc_info=True)
            raise

    def _query_conservation_status_counts(self, country_code: str = None) -> List[tuple]:
        """Execute BigQuery to fetch conservation status counts."""
        client = bigquery.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'))
        query = self._build_conservation_count_query(country_code)
        parameters = self._get_conservation_count_parameters(country_code)
        job_config = bigquery.QueryJobConfig(query_parameters=parameters)
        start_query = time.time()
        query_job = client.query(query, job_config=job_config)
        results = query_job.result()
        self.logger.info("BigQuery query completed in %.2f seconds", time.time() - start_query)
        return [(row.conservation_status, row.species_count) for row in results]

    def _build_conservation_count_query(self, country_code: str = None) -> str:
        """Build the BigQuery query string for conservation status counts."""
        base_query = """
            SELECT conservation_status, 
                   COUNT(DISTINCT CONCAT(genus_name, ' ', species_name)) as species_count
            FROM `{0}.biodiversity.endangered_species` sp
            JOIN `{0}.biodiversity.occurances_endangered_species_mammals` oc 
                ON CONCAT(genus_name, ' ', species_name) = oc.species 
            WHERE species_name IS NOT NULL 
                AND genus_name IS NOT NULL 
                {country_code_filter}
            GROUP BY conservation_status
            ORDER BY species_count DESC
        """
        country_code_filter = "AND oc.countrycode = @country_code" if country_code else ""
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        return base_query.format(project_id, country_code_filter=country_code_filter)

    def _get_conservation_count_parameters(self,
                            country_code: str = None) -> List[bigquery.ScalarQueryParameter]:
        """Create query parameters for conservation count query."""
        if not country_code:
            return []
        return [bigquery.ScalarQueryParameter("country_code", "STRING", country_code)]

    def _format_conservation_counts(self, results: List[tuple]) -> str:
        """Format the conservation status counts into a readable string."""
        output = ["**Only Mammals are included in the list.**\n"]
        for status, count in results:
            output.append(f"* {status}: {count} species")
        return "\n".join(output)
