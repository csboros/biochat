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
from vertexai.generative_models import FunctionDeclaration
import requests
from google.cloud import bigquery
from pygbif import species
#from EcoNameTranslator import to_common
from langchain_google_community import GoogleSearchAPIWrapper
import streamlit as st


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
        Initializes the FunctionHandler with necessary components and configurations.

        Sets up logging, loads geographical data, and initializes function declarations
        and handlers. Caches world geographical data in Streamlit session state for
        improved performance.

        Raises:
            Exception: If initialization of any component fails, particularly during
                      world GeoJSON data loading
        """
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.info("Initializing FunctionHandler")
        self.setup_function_declarations()
        self.search = GoogleSearchAPIWrapper()
        self.world_gdf = None  # Initialize the attribute
        # Add URL for natural earth data
        self.world_geojson_url = (
            "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
            "master/geojson/ne_110m_admin_0_countries.geojson"
        )

    def setup_function_declarations(self):
        """
        Sets up function declarations and their corresponding handlers.

        Initializes two main dictionaries:
        - declarations: Contains FunctionDeclaration objects for Vertex AI
        - function_handler: Maps function names to their implementing methods

        Each function declaration includes:
        - Name
        - Description
        - Parameter specifications
        - Required parameters
        """
        # Store declarations separately from handlers
        self.declarations = {
            "translate_to_scientific_name": FunctionDeclaration(
                name="translate_to_scientific_name",
                description="Translate a given species name to a scientific name",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                "english name of the species to be translated "
                                "into scientific name"
                            ),
                        }
                    },
                },
            ),
            "get_species_info": FunctionDeclaration(
                name="get_species_info",
                description="Get info about a given species, get taxonomy of a given species",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "get taxonomy of a given species",
                        }
                    },
                },
            ),
            "get_occurences": FunctionDeclaration(
                name="get_occurences",
                description=(
                    "Get occurences, distribution for a given species, "
                    "show where species is found, where it lives."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "species_name": {
                            "type": "string",
                            "description": (
                                "name of the species to get occurences for, "
                                "if it is a common name, use the scientific name"
                            ),
                        },
                        "country_code": {
                            "type": "string",
                            "description": "2 letter country code to get occurences for",
                        },
                         "chart_type": {
                            "type": "string",
                            "description": "type of chart to display",
                        },
                    },
                    "required": ["species_name"]
                },
            ),
            "get_country_geojson": FunctionDeclaration(
                name="get_country_geojson",
                description="Get GeoJSON data for a specific country",
                parameters={
                    "type": "object",
                    "properties": {
                        "country_name": {
                            "type": "string",
                            "description": "name of the country to get GeoJSON data for",
                        }
                    },
                    "required": ["country_name"]
                },
            ),
            "endangered_species_for_family": FunctionDeclaration(
                name="endangered_species_for_family",
                description=(
                    "Get list of endangered species for a given family name, "
                    "with link to ICFN"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "family_name": {
                            "type": "string",
                            "description": "name of the family to get endangered species for",
                        },
                        "conservation_status": {
                            "type": "string",
                            "description": (
                                "optional conservation status to filter by, "
                                "possible values are: "
                                "Least Concern, Endangered, Near Threatened, Vulnerable, "
                                "Data Deficient, Critically Endangered, Extinct,"
                            ),
                        }
                    },
                    "required": ["family_name"]  # conservation_status is optional
                },
            ),
            "endangered_classes_for_kingdom": FunctionDeclaration(
                name="endangered_classes_for_kingdom",
                description="Get list of endangered classes for a given kingdom name",
                parameters={
                    "type": "object",
                    "properties": {
                        "kingdom_name": {
                            "type": "string",
                            "description": "name of the kingdom to get endangered classes for",
                        }
                    },
                },
            ),
            "endangered_families_for_order": FunctionDeclaration(
                name="endangered_families_for_order",
                description="Get list of endangered families for a given order name",
                parameters={
                    "type": "object",
                    "properties": {
                        "order_name": {
                            "type": "string",
                            "description": "name of the order to get endangered families for",
                        }
                    },
                },
            ),
            "endangered_orders_for_class": FunctionDeclaration(
                name="endangered_orders_for_class",
                description="Get list of endangered orders for a given class name",
                parameters={
                    "type": "object",
                    "properties": {
                        "class_name": {
                            "type": "string",
                            "description": "name of the class to get endangered orders for",
                        }
                    },
                },
            ),
            "google_search": FunctionDeclaration(
                name="google_search",
                description=(
                    "Search Google for the given query and return relevant results, "
                    "try this as last resort before giving up please."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "query to search for",
                        }
                    },
                },
            ),
            "endangered_species_for_country": FunctionDeclaration(
                name="endangered_species_for_country",
                description="Get list of endangered species for a given country code",
                parameters={
                    "type": "object",
                    "properties": {
                        "country_code": {
                            "type": "string",
                            "description": "country code to get endangered species for",
                        },
                        "conservation_status": {
                            "type": "string",
                            "description": (
                                "optional conservation status to filter by, "
                                "possible values are: "
                                "Least Concern, Endangered, Near Threatened, Vulnerable, "
                                "Data Deficient, Critically Endangered, Extinct,"
                            ),
                        }
                    },
                },
            ),
            "number_of_endangered_species_by_conservation_status": FunctionDeclaration(
                name="number_of_endangered_species_by_conservation_status",
                description=(
                    "Get number of endangered species for different conservational "
                    "status for a given country code or the whole world if country "
                    "code is not specified"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "country_code": {
                            "type": "string",
                            "description": "country code to get endangered species for",
                        },
                        "conservation_status": {
                            "type": "string",
                            "description": (
                                "optional conservation status to filter by, "
                                "possible values are:"
                                "Least Concern, Endangered, Near Threatened, Vulnerable, "
                                "Data Deficient, Critically Endangered, Extinct,"
                            ),
                        }

                    },
                },
            ),
        }

        # Function handlers point to actual methods
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

    def google_search(self, content) -> str:
        """
        Performs a Google search focused on IUCN Red List results.
        
        Args:
            content (dict): Dictionary containing:
                - query (str): Search query string
            
        Returns:
            str: Search results from Google, filtered to IUCN Red List content
            
        Note:
            This is used as a fallback when other API methods don't return results.
        """
        query_string = content.get('query')
        query = f"site:https://www.iucnredlist.org/ {query_string}"
        return self.search.run(query)

    @st.cache_data(
        ttl=3600,  # Cache for 1 hour
        show_spinner="Fetching data...",
        max_entries=100
    )
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
            Exception: If BigQuery query fails or returns invalid data
        """
        species_name = content['species_name']
        if 'country_code' in content:
            country_code = content['country_code']
            _self.logger.info("Fetching occurrences for species: %s and country: %s",
                              species_name, country_code)
        else:
            country_code = None
            _self.logger.info("Fetching occurrences for species: %s", species_name)
        try:
            client = bigquery.Client(
                project=os.getenv('GOOGLE_CLOUD_PROJECT'),
            )

            # Base query with parameterization
            query = """
                SELECT 
                    decimallatitude,
                    decimallongitude
                FROM `{}.biodiversity.occurances_endangered_species_mammals`
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
            _self.logger.info(
                "Successfully fetched %d occurrences for species %s%s",
                len(results),
                species_name,
                ' and country ' + country_code if country_code else ''
            )
            return results

        except Exception as e:
            _self.logger.error(
                "Error fetching terrestrial human coexistence index: %s",
                str(e),
                exc_info=True
            )
            raise

    @st.cache_data(
        ttl=3600,  # Cache for 1 hour
        show_spinner="Translating species name...",
        max_entries=100
    )
    def translate_to_scientific_name_from_api(_self, content):  # pylint: disable=no-self-argument
        """
        Translates common species names to scientific names using the EBI Taxonomy API.

        Args:
            content (dict): Dictionary containing:
                - name (str): Common name of the species to translate

        Returns:
            str: JSON string containing either:
                - scientific_name: The translated scientific name
                - error: Error message if translation fails

        Raises:
            requests.Timeout: If the API request times out
            requests.RequestException: If the API request fails
            JSONDecodeError: If the API response cannot be parsed
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
        Retrieves detailed species information from the GBIF API.

        Args:
            content (dict): Dictionary containing:
                - name (str): Scientific or common name of the species

        Returns:
            str: JSON string containing species taxonomic and classification information

        Raises:
            JSONDecodeError: If the API response cannot be parsed
        """
        species_name = content['name']
        species_info = species.name_backbone(species_name)
        return json.dumps(species_info)
    @st.cache_data(
        ttl=3600,  # Cache for 1 hour
        show_spinner="Translating species name...",
        max_entries=100
    )


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
            Exception: If there is an error processing the GeoJSON data
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
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            self.logger.error("Error getting GeoJSON for country %s: %s",
                              country_name, str(e), exc_info=True)
            return {"error": f"Error processing request: {str(e)}"}


    def load_world_geojson(self):
        """
        Loads the world GeoJSON data from the URL.
        """
        try:
            if 'world_gdf' not in st.session_state or st.session_state.world_gdf is None:
                self.logger.info("Loading world GeoJSON data")
                response = requests.get(self.world_geojson_url, timeout=30)  # 30 second timeout
                self.world_gdf = response.json()
                st.session_state.world_gdf = self.world_gdf
                self.logger.info("Loaded GeoJSON data for %d countries",
                                 len(self.world_gdf.get('features', [])))
            else:
                self.world_gdf = st.session_state.world_gdf

        except Exception as e:
            self.logger.error("Failed to load world GeoJSON data: %s", str(e), exc_info=True)
            raise


    def endangered_classes_for_kingdom(self, content) -> List[str]:
        """
        Retrieves endangered classes within a specified kingdom.

        Args:
            content (dict): Dictionary containing:
                - kingdom (str): Name of the kingdom to query

        Returns:
            List[dict]: List of dictionaries containing:
                - class (str): Name of the endangered class
                - count (int): Number of endangered species in that class

        Raises:
            Exception: If there is an error querying BigQuery
        """
        kingdom = content['kingdom']
        self.logger.info("Fetching classes for kingdom from BigQuery")

        try:
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
                    bigquery.ScalarQueryParameter("kingdom", "STRING", kingdom)
                ]
            )
            query_job = client.query(
                query,
                job_config=job_config
            )
            results = [{'class': row['class'], 'count': row['cnt']} for row in query_job]
            return results
        except Exception as e:
            self.logger.error(
                "Error fetching terrestrial human coexistence index: %s",
                str(e),
                exc_info=True
            )
            raise

    def endangered_orders_for_class(self, content) -> List[str]:
        """
        Retrieves endangered orders within a specified class.

        Args:
            content (dict): Dictionary containing:
                - class (str): Name of the class to query

        Returns:
            List[dict]: List of dictionaries containing:
                - family_name (str): Name of the endangered order
                - count (int): Number of endangered species in that order

        Raises:
            Exception: If there is an error querying BigQuery
        """
        clazz = content['class']
        self.logger.info("Fetching families for classes from BigQuery")

        try:
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
            results = [
                {'family_name': row['family_name'], 'count': row['cnt']}
                for row in query_job
            ]
            return results
        except Exception as e:
            self.logger.error(
                "Error fetching terrestrial human coexistence index: %s",
                str(e),
                exc_info=True
            )
            raise

    def endangered_families_for_order(self, content) -> List[str]:
        """
        Retrieves endangered families within a specified order.

        Args:
            content (dict): Dictionary containing:
                - order_name (str): Name of the order to query

        Returns:
            List[dict]: List of dictionaries containing:
                - family_name (str): Name of the endangered family
                - count (int): Number of endangered species in that family

        Raises:
            Exception: If there is an error querying BigQuery
        """
        self.logger.info("Fetching families for order from BigQuery")
        order_name = content['order_name']
        try:
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
        except Exception as e:
            self.logger.error(
                "Error fetching endangered families for order: %s",
                str(e),
                exc_info=True
            )
            raise

    def endangered_species_for_family(self, content) -> List[str]:
        """
        Retrieves endangered species within a specified family.
        
        Args:
            content (dict): Dictionary containing:
                - family_name (str): Name of the family to query
            
        Returns:
            List[dict]: List of dictionaries containing:
                - species_name (str): Full scientific name (genus + species)
                - url (str): URL to species information
            
        Raises:
            Exception: If there is an error querying BigQuery
        """
        self.logger.info("Fetching families for classes from BigQuery")

        family_name = content['family_name']
        conservation_status = None
        if 'conservation_status' in content:
            conservation_status = content['conservation_status']

        try:
            client = bigquery.Client(
                project=os.getenv('GOOGLE_CLOUD_PROJECT'),
            )
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
            query = base_query.format(project_id,
                                        conservation_status_filter=conservation_status_filter)
            parameters = [
                bigquery.ScalarQueryParameter("family_name", "STRING", family_name)
            ]
            if conservation_status:
                parameters.append(
                    bigquery.ScalarQueryParameter("conservation_status", "STRING",
                                                    conservation_status)
                )
            job_config = bigquery.QueryJobConfig(query_parameters=parameters)
            query_job = client.query(query, job_config=job_config)
            # Collect all scientific names first
            scientific_names = []
            results_data = []
            for row in query_job:
                scientific_name = row['species_header'].split(':')[0]  # Remove the colon
                scientific_names.append(scientific_name)
                results_data.append((scientific_name, row['urls']))

            # Translation takes too long, so we skip it
#            common_names_dict = to_common(scientific_names)
            common_names_dict = {}
            results = []
            for scientific_name, urls in results_data:
                urls_formatted = '\n'.join(f'    * {url}' for url in urls.split('||'))
                common_names = common_names_dict.get(scientific_name, [])
                print(common_names)
                common_names_str = (f" ({', '.join(common_names[1])})"
                                    if common_names and len(common_names) > 1
                                    else "")
                formatted_entry = (f"* **{scientific_name}**{common_names_str}:\n"
                                   f"{urls_formatted}")
                results.append(formatted_entry)

            final_text = '\n'.join(results)
            self.logger.info("Results: %s", final_text)
            return final_text
        except Exception as e:
            self.logger.error(
                "Error fetching terrestrial human coexistence index: %s",
                str(e),
                exc_info=True
            )
            raise

    def endangered_species_for_country(self, content) -> List[str]:
        """
        Retrieves endangered species found in a specific country.
        
        Args:
            content (dict): Dictionary containing:
                - country_code (str): Two-letter ISO country code
                - conservation_status (str, optional): Filter by conservation status
            
        Returns:
            str: Formatted string containing list of endangered species with:
                - Scientific name
                - Family name
                - Conservation status
                - URL to species information
            
        Note:
            Currently only includes mammal species.
            
        Raises:
            Exception: If there is an error querying BigQuery
        """
        self.logger.info("Fetching species for country from BigQuery")

        country_code = content['country_code']
        if 'conservation_status' in content:
            conservation_status = content['conservation_status']
        else:
            conservation_status = None

        try:
            client = bigquery.Client(
                project=os.getenv('GOOGLE_CLOUD_PROJECT'),
            )

            base_query = """
                SELECT DISTINCT CONCAT(genus_name, ' ', species_name) as species_name, 
                       family_name, conservation_status, url
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

            # Add conservation status filter only if it's provided
            conservation_status_filter = (
                "AND LOWER(conservation_status) = LOWER(@conservation_status)"
                if conservation_status
                else ""
            )
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
            query = base_query.format(project_id, project_id,
                                      conservation_status_filter=conservation_status_filter)
            # Set up query parameters based on whether conservation_status is provided
            parameters = [
                bigquery.ScalarQueryParameter("country_code", "STRING", country_code)
            ]
            if conservation_status:
                parameters.append(
                    bigquery.ScalarQueryParameter("conservation_status", "STRING",
                                                  conservation_status)
                )
            job_config = bigquery.QueryJobConfig(query_parameters=parameters)
            # Time the BigQuery query execution
            start_query = time.time()
            query_job = client.query(query, job_config=job_config)
            query_time = time.time() - start_query
            # Log query timing information
            self.logger.info(
                "BigQuery query completed in %.2f seconds",
                query_time
            )

            # Collect scientific names and data
            scientific_names = []
            results_data = []
            for row in query_job:
                scientific_names.append(row['species_name'])
                results_data.append((
                    row['species_name'],
                    row['family_name'],
                    row['conservation_status'],
                    row['url']
                ))
            # Time the common name translation
            start_translation = time.time()
            # Translation takes too long, so we skip it
#            common_names_dict = to_common(scientific_names)
            common_names_dict = {}
            translation_time = time.time() - start_translation
            # Log timing information
            self.logger.info(
                "Name translation completed in %.2f seconds for %d species",
                translation_time,
                len(scientific_names)
            )

            result = "**Only Mammals are included in the list.**\n"
            for scientific_name, family, status, url in results_data:
                common_names = common_names_dict.get(scientific_name, [])
                common_names_str = (f" ({', '.join(common_names[1])})"
                                    if common_names and len(common_names) > 1
                                    else "")
                result += (f"* **{scientific_name}**{common_names_str} ({family}, "
                           f"{status}):\n{url}\n")
            return result

        except Exception as e:
            self.logger.error(
                "Error fetching terrestrial human coexistence index: %s",
                str(e),
                exc_info=True
            )
            raise

    def number_of_endangered_species_by_conservation_status(self, content) -> str:
        """
        Retrieves count of endangered species grouped by conservation status.
        
        Args:
            content (dict): Dictionary containing:
                - country_code (str, optional): Two-letter ISO country code
                                              If not provided, returns global statistics
            
        Returns:
            str: Formatted string containing:
                - Conservation status
                - Number of species in each status category
            
        Note:
            Currently only includes mammal species.
            
        Raises:
            Exception: If there is an error querying BigQuery
        """
        country_code = content.get('country_code')
        try:
            client = bigquery.Client(
                project=os.getenv('GOOGLE_CLOUD_PROJECT'),
            )

            base_query = """
                SELECT conservation_status, COUNT(DISTINCT CONCAT(genus_name, ' ', species_name)) as species_count
                FROM `{0}.biodiversity.endangered_species` sp
                JOIN `{0}.biodiversity.occurances_endangered_species_mammals` oc 
                    ON CONCAT(genus_name, ' ', species_name) = oc.species 
                WHERE species_name IS NOT NULL 
                    AND genus_name IS NOT NULL 
                    {country_code_filter}
                GROUP BY conservation_status
                ORDER BY species_count DESC
            """

            # Add country code filter only if it's provided
            country_code_filter = "AND oc.countrycode = @country_code" if country_code else ""
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
            query = base_query.format(project_id, country_code_filter=country_code_filter)

            # Set up query parameters only if country_code is provided
            parameters = []
            if country_code:
                parameters.append(bigquery.ScalarQueryParameter("country_code", "STRING",
                                                                country_code))
            job_config = bigquery.QueryJobConfig(query_parameters=parameters)

            # Time the BigQuery query execution
            start_query = time.time()
            query_job = client.query(query, job_config=job_config)
            results = query_job.result()  # Wait for query to complete
            query_time = time.time() - start_query
            # Log query timing information
            self.logger.info(
                "BigQuery query completed in %.2f seconds",
                query_time
            )

            # Format the results as a string
            output = ["**Only Mammals are included in the list.**\n"]
            for row in results:
                output.append(f"* {row.conservation_status}: {row.species_count} species\n")
            return "\n".join(output)

        except (ValueError, KeyError) as e:
            logging.error(
                "Error in number_of_endangered_species_by_conservation_status: %s",
                str(e)
            )
            return f"Error retrieving conservation status counts: {str(e)}"
