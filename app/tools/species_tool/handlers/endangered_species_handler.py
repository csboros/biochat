"""Module for handling endangered species related queries."""

import os
import logging
from typing import List, Optional
import pycountry
import google.api_core.exceptions
from google.cloud import bigquery
from app.tools.message_bus import message_bus
from ...base_handler import BaseHandler

# pylint: disable=broad-except
class EndangeredSpeciesHandler(BaseHandler):
    """Handles queries related to endangered species."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("BioChat." + self.__class__.__name__)

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

            message_bus.publish("status_update", {
                "message": f"'🦁' Fetching endangered classes for {kingdom_name} kingdom...",
                "state": "running",
                "progress": 0
            })

            self.logger.info("Fetching classes for kingdom from BigQuery")

            client = bigquery.Client(
                project=os.getenv('GOOGLE_CLOUD_PROJECT'),
            )
            query = """
                SELECT class, count(class) as cnt
                FROM `{project_id}.biodiversity.endangered_species`
                WHERE LOWER(kingdom) = LOWER(@kingdom)
                    AND class IS NOT NULL
                GROUP BY class
                ORDER BY class
            """

            query = self.build_query(query)

            parameters = self.get_parameters(
                kingdom=kingdom_name
            )

            job_config = bigquery.QueryJobConfig(query_parameters=parameters)
            query_job = client.query(
                query,
                job_config=job_config
            )

            # Processing results
            message_bus.publish("status_update", {
                "message": "📊 Processing results...",
                "state": "running",
                "progress": 70
            })

            results = []
            intro = f"Here are the classes within the {kingdom_name} kingdom:\n\n"
            for row in query_job:
                formatted_entry = f"* **{row['class']}**: {row['cnt']} endangered species"
                results.append(formatted_entry)

            message_bus.publish("status_update", {
                "message": "✅ Data retrieved successfully",
                "state": "complete",
                "progress": 100
            })

            final_text = intro + '\n'.join(results)
            return final_text
        except google.api_core.exceptions.GoogleAPIError as e:
            message_bus.publish("status_update", {
                "message": f"Database error: {str(e)}",
                "state": "error",
                "progress": 0
            })
            self.logger.error("BigQuery error: %s", str(e), exc_info=True)
            raise
        except (TypeError, ValueError) as e:
            message_bus.publish("status_update", {
                "message": f"Invalid input: {str(e)}",
                "state": "error",
                "progress": 0
            })
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
            # Check if 'class_name' or 'animal_class' exists in content
            if 'class_name' not in content and 'animal_class' not in content:
                raise BusinessException("Either 'class_name' or 'animal_class' key is required in the content.")

            # Use 'class_name' if it exists, otherwise use 'animal_class'
            clazz = content.get('class_name') or content.get('animal_class')

            message_bus.publish("status_update", {
                "message": "🦁 Fetching endangered orders for class...",
                "state": "running",
                "progress": 0
            })

            self.logger.info("Fetching orders for class from BigQuery")
            client = bigquery.Client(
                project=os.getenv('GOOGLE_CLOUD_PROJECT'),
            )
            query = """
                SELECT
                    order_name,
                    COUNT(DISTINCT CONCAT(genus_name, ' ', species_name)) as cnt
                FROM `{project_id}.biodiversity.endangered_species`
                WHERE class_name = @class_name
                GROUP BY order_name
                ORDER BY cnt DESC
            """
            query = self.build_query(query)
            parameters = [
                bigquery.ScalarQueryParameter("class_name", "STRING", clazz)
            ]
            job_config = bigquery.QueryJobConfig(query_parameters=parameters)
            query_job = client.query(
                query,
                job_config=job_config
            )

            message_bus.publish("status_update", {
                "message": "📊 Processing results...",
                "state": "running",
                "progress": 70
            })

            results = []
            intro = f"Here are the orders within the {clazz} class:\n\n"
            for row in query_job:
                formatted_entry = f"* **{row['order_name']}**: {row['cnt']} endangered species"
                results.append(formatted_entry)

            message_bus.publish("status_update", {
                "message": "✅ Data retrieved successfully",
                "state": "complete",
                "progress": 100
            })

            final_text = intro + '\n'.join(results)
            return final_text
        except BusinessException as e:
            self.logger.error("Business exception occurred: %s", str(e))
            raise
        except google.api_core.exceptions.GoogleAPIError as e:
            message_bus.publish("status_update", {
                "message": f"'❌'Database error: {str(e)}",
                "state": "error",
                "progress": 0
            })
            self.logger.error("BigQuery error: %s", str(e), exc_info=True)
            raise
        except (TypeError, ValueError) as e:
            message_bus.publish("status_update", {
                "message": f"'❌' Invalid input: {str(e)}",
                "state": "error",
                "progress": 0
            })
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
            # Check if 'order_name' or 'order' exists in content
            if 'order_name' not in content and 'order' not in content:
                raise BusinessException("Either 'order_name' or 'order' key is required in the content.")

            # Use 'order_name' if it exists, otherwise use 'order'
            order_name = content.get('order_name') or content.get('order')

            message_bus.publish("status_update", {
                "message": f"'🐘' Fetching endangered families for {order_name} order...",
                "state": "running",
                "progress": 0
            })

            self.logger.info("Fetching families for order from BigQuery")
            client = bigquery.Client(
                project=os.getenv('GOOGLE_CLOUD_PROJECT'),
            )
            query = self.build_query(
                self.SPECIES_QUERY_TEMPLATE,
                where_clause="AND LOWER(order_name) = LOWER(@order_name) "
                    "AND family_name IS NOT NULL"
            )
            parameters = self.get_parameters(
                order_name=order_name
            )
            job_config = bigquery.QueryJobConfig(query_parameters=parameters)
            query_job = client.query(
                query,
                job_config=job_config
            )
            res = [(row.species_name, row.family, row.status, row.order_name, row['class'],
                   getattr(row, 'species_name_en', None))  # Handle species_name_en
                   for row in query_job]

            message_bus.publish("status_update", {
                "message": "✅ Data retrieved successfully",
                "state": "complete",
                "progress": 100
            })

            return self._format_hierarchy_data(res)
        except BusinessException as e:
            self.logger.error("Business exception occurred: %s", str(e))
            raise
        except google.api_core.exceptions.GoogleAPIError as e:
            message_bus.publish("status_update", {
                "message": f"Database error: {str(e)}",
                "state": "error",
                "progress": 0
            })
            self.logger.error("BigQuery error: %s", str(e), exc_info=True)
            raise
        except (TypeError, ValueError) as e:
            message_bus.publish("status_update", {
                "message": f"'❌' Invalid input: {str(e)}",
                "state": "error",
                "progress": 0
            })
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
            str: Formatted string with hierarchical structure of species data

        Raises:
            ValueError: If family_name is invalid or not found
            google.api_core.exceptions.GoogleAPIError: If BigQuery query fails
            TypeError: If content is not in expected format
        """
        try:
            family_name = content['family_name']
            conservation_status = content.get('conservation_status')

            message_bus.publish("status_update", {
                "message": f"'🐘' Fetching endangered species for {family_name} family...",
                "state": "running",
                "progress": 0
            })


            client = bigquery.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'))
            query = self.build_query(
                self.SPECIES_QUERY_TEMPLATE,
                where_clause="AND LOWER(family_name) = LOWER(@family_name)"
            )
            parameters = self.get_parameters(
                family_name=family_name,
                conservation_status=conservation_status
            )
            job_config = bigquery.QueryJobConfig(query_parameters=parameters)
            query_job = client.query(query, job_config=job_config)

            message_bus.publish("status_update", {
                "message": "📊 Processing results...",
                "state": "running",
                "progress": 70
            })

            res = [(row.species_name, row.family, row.status, row.order_name, row['class'],
                   getattr(row, 'species_name_en', None))
                   for row in query_job]

            message_bus.publish("status_update", {
                "message": "✅ Data retrieved successfully",
                "state": "complete",
                "progress": 100
            })

            return self._format_hierarchy_data(res)

        except google.api_core.exceptions.GoogleAPIError as e:
            message_bus.publish("status_update", {
                "message": f"'❌' Database error: {str(e)}",
                "state": "error",
                "progress": 0
            })
            self.logger.error("BigQuery error: %s", str(e), exc_info=True)
            raise
        except (TypeError, ValueError) as e:
            message_bus.publish("status_update", {
                "message": f"'❌' Invalid input: {str(e)}",
                "state": "error",
                "progress": 0
            })
            self.logger.error("Invalid input: %s", str(e), exc_info=True)
            raise

    def endangered_species_for_country(self, content) -> str:
        """
        Retrieves endangered species for a SINGLE specified country.
        For multiple countries, use endangered_species_for_countries() instead.

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
            # Use country_code if provided, otherwise convert country_name
            country_code = content.get('country_code')
            if not country_code and 'country_name' in content:
                country_code = pycountry.countries.lookup(content['country_name']).alpha_2

            if not country_code:
                raise BusinessException("No valid country code or name provided")

            conservation_status = content.get('conservation_status')

            message_bus.publish("status_update", {
                "message": f"🔍 Fetching endangered species for country {country_code}...",
                "state": "running",
                "progress": 0
            })

            self.logger.info("Fetching species for country from BigQuery")

            message_bus.publish("status_update", {
                "message": "Querying database...",
                "state": "running",
                "progress": 30
            })

            results_data = self._query_country_species(country_code, conservation_status)


            formatted_data = self._format_hierarchy_data(results_data)

            message_bus.publish("status_update", {
                "message": "✅ Data retrieved successfully",
                "state": "complete",
                "progress": 100
            })

            return formatted_data

        except google.api_core.exceptions.GoogleAPIError as e:
            message_bus.publish("status_update", {
                "message": f"'❌'Database error: {str(e)}",
                "state": "error",
                "progress": 0
            })
            self.logger.error("BigQuery error: %s", str(e), exc_info=True)
            raise
        except (TypeError, ValueError) as e:
            message_bus.publish("status_update", {
                "message": f"'❌' Invalid input: {str(e)}",
                "state": "error",
                "progress": 0
            })
            self.logger.error("Invalid input: %s", str(e), exc_info=True)
            raise

    def _query_country_species(self, country_code: str,
                             conservation_status: Optional[str] = None) -> list:
        """Execute BigQuery to fetch country's endangered species data."""
        client = bigquery.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'))
        query = self._build_country_species_query(conservation_status)
        parameters = self._get_country_query_parameters(country_code, conservation_status)
        job_config = bigquery.QueryJobConfig(query_parameters=parameters)
        query_job = client.query(query, job_config=job_config)
        results = query_job.result()
        return [(row.species_name, row.family, row.status,
                row.order_name, row['class'],
                getattr(row, 'species_name_en', None))  # Add species_name_en
                for row in results]

    def _build_country_species_query(self, conservation_status: Optional[str] = None) -> str:
        """Build the BigQuery query string for country species."""
        return self.build_query(
            self.SPECIES_QUERY_TEMPLATE,
            where_clause="AND oc.countrycode = @country_code",
            conservation_status=conservation_status
        )

    def _get_country_query_parameters(
            self,
            country_code: str,
            conservation_status: Optional[str] = None
    ) -> List[bigquery.ScalarQueryParameter]:
        """Create query parameters for country species query."""
        return self.get_parameters(
            country_code=country_code,
            conservation_status=conservation_status
        )

    def _format_hierarchy_data(self, data):
        """Format the data into a hierarchical structure."""
        hierarchy = {
            'name': 'Endangered Species',
            'children': []  # Classes
        }

        classes = {}
        orders = {}
        families = {}

        # First pass: organize all data
        for row in data:
            # Unpack with default None for species_name_en if it's not provided
            species_name, family_name, status, order_name, class_name, *extra = row
            species_name_en = extra[0] if extra else None

            # Create class if it doesn't exist
            if class_name not in classes:
                classes[class_name] = {
                    'name': class_name,
                    'children': []
                }
                hierarchy['children'].append(classes[class_name])

            # Create order if it doesn't exist
            if order_name not in orders:
                orders[order_name] = {
                    'name': order_name,
                    'children': []
                }
                classes[class_name]['children'].append(orders[order_name])

            # Create family if it doesn't exist
            if family_name not in families:
                families[family_name] = {
                    'name': family_name,
                    'children': []
                }
                orders[order_name]['children'].append(families[family_name])

            # Add species with English name if available
            species_info = {
                'name': species_name,
                'status': status,
                'value': 1
            }
            if species_name_en:  # Only add English name if it exists
                species_info['species_name_en'] = species_name_en

            families[family_name]['children'].append(species_info)

        # Sort all levels
        for family in families.values():
            family['children'].sort(key=lambda x: x['name'])

        for order in orders.values():
            order['children'].sort(key=lambda x: x['name'])

        for class_group in classes.values():
            class_group['children'].sort(key=lambda x: x['name'])

        hierarchy['children'].sort(key=lambda x: x['name'])

        return hierarchy

    def endangered_species_by_conservation_status(self, content) -> str:
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
            country_codes = content.get('country_codes')
            conservation_status = content.get('conservation_status')
            if not country_code and not conservation_status and not country_codes:
                # If neither country_code nor conservation_status is provided,
                # return global statistics
                results = self._query_conservation_status_counts()
            elif country_codes:
                all_results = []
                for code in country_codes:
                    results = self._query_conservation_status_counts(code)
                    all_results.append(self._format_conservation_counts(results, code))
                return "\n\n".join(all_results)
            else:
                # If either country_code or conservation_status is provided,
                #  return statistics for the specified country
                results = self._query_conservation_status_counts(country_code)
            return self._format_conservation_counts(results, country_code)
        except google.api_core.exceptions.GoogleAPIError as e:
            self.logger.error("BigQuery error: %s", str(e), exc_info=True)
            raise
        except (TypeError, ValueError) as e:
            self.logger.error("Invalid input: %s", str(e), exc_info=True)
            raise

    def _query_conservation_status_counts(self, country_code: str = None) -> list:
        """Execute BigQuery to fetch conservation status counts."""
        client = bigquery.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'))
        query = self._build_conservation_count_query(country_code)
        parameters = self.get_parameters(
            country_code=country_code
        )
        job_config = bigquery.QueryJobConfig(query_parameters=parameters)
        query_job = client.query(query, job_config=job_config)
        results = query_job.result()
        return [(row['conservation_status'], row['species_count']) for row in results]

    def _build_conservation_count_query(self, country_code: str = None) -> str:
        """Build the BigQuery query string for conservation status counts."""
        # pylint: disable=duplicate-code
        base_query = """
            SELECT conservation_status,
                   COUNT(DISTINCT CONCAT(genus_name, ' ', species_name)) as species_count
            FROM `{project_id}.biodiversity.endangered_species` sp
            JOIN `{project_id}.biodiversity.occurances_endangered_species_mammals` oc
                ON CONCAT(genus_name, ' ', species_name) = oc.species
            WHERE species_name IS NOT NULL
                AND genus_name IS NOT NULL
                {where_clause}
            GROUP BY conservation_status
            ORDER BY species_count DESC
        """
        where_clause = "AND oc.countrycode = @country_code" if country_code else ""
        return self.build_query(
            base_query,
            where_clause=where_clause
        )
        # pylint: enable=duplicate-code

    def _format_conservation_counts(self, results: list, country_code: str) -> str:
        """Format the conservation status counts into a readable string."""
        output = [
            f"**These are the endangered species in the {country_code}. "
            "(Only Mammals are included in the list)**\n"
        ]

        for status, count in results:
            output.append(f"* {status}: {count} species")
        return "\n".join(output)

    def endangered_species_hci_correlation(self, content) -> str:
        """Get correlation data between endangered species and human impact factors."""
        try:
            self.logger.info("Received content: %s", content)

            continent = content.get('continent', 'Africa')

            query = """
               SELECT
                    countrycode,
                    iso3,
                    continent,
                    AVG(terrestrial_hci) AS terrestrial_hci,
                    AVG(population_density) AS population_density,
                    SUM(CAST(species_count AS INT64)) AS species_count
                FROM
                    `{project_id}.biodiversity.endangered_species_country`
                WHERE
                    continent = @continent
                GROUP BY
                    countrycode,
                    iso3,
                    continent
                ORDER BY
                    species_count DESC
            """

            parameters = [
                bigquery.ScalarQueryParameter("continent", "STRING", continent)
            ]

            self.logger.info("Query parameters: %s", parameters)  # Debug print

            client = bigquery.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'))
            query = self.build_query(query)
            job_config = bigquery.QueryJobConfig(query_parameters=parameters)
            results = client.query(query, job_config=job_config).result()
            result = self._format_correlation_results(results, continent)
            return result
        except Exception as e:
            self.logger.error("Error in endangered_species_hci_correlation: %s", str(e))
            raise

    def _format_correlation_results(self, results, continent=None, conservation_status=None) -> str:
        """Format the correlation results into a JSON structure."""
        data = {
            "continent": continent,
            "countries": [
                {
                    "iso3": row['iso3'],
                    "species_count": row['species_count'],
                    "hci": row['terrestrial_hci'],
                    "popden": row['population_density']
                }
                for row in results
            ]
        }
        return data

    def endangered_species_for_countries(self, content) -> str:
        """
        Retrieves endangered species for multiple specified countries.

        Args:
            content (dict): Dictionary containing:
                - country_codes (list): List of two-letter country codes to query
                - conservation_status (str, optional): Status to filter by

        Returns:
            dict: Dictionary containing visualization data for all countries
        """
        try:
            country_codes = content.get('country_codes', [])
            conservation_status = content.get('conservation_status')

            if not country_codes:
                return {"error": "No country codes provided"}

            # Collect results for each country
            all_results = []
            for country_code in country_codes:
                params = {
                    'country_code': country_code,
                    'conservation_status': conservation_status
                }
                country_results = self.endangered_species_for_country(params)
                all_results.append(country_results)

            # Return the combined results as a dictionary
            return {
                "type": "multiple_countries",
                "data": all_results
            }

        except Exception as e:
            self.logger.error("Error in endangered_species_for_countries: %s", str(e))
            raise

    # pylint: disable=no-self-argument
    def get_occurrences(_self, content):
        """
        Retrieve occurrence data for a specified species.

        Args:
            content: The species data to process

        Returns:
            dict: Processed occurrence data

        Raises:
            ValueError: If content is empty or malformed
            TypeError: If content is not of the expected type
        """
        # Input validation
        if content is None:
            raise ValueError("Content cannot be None")

        if not isinstance(content, (dict, str)):
            raise TypeError(f"Expected dict or str, got {type(content)}")

        if isinstance(content, dict):
            if not content:
                raise ValueError("Content dictionary cannot be empty")
            if "species_name" not in content and \
               "species" not in content and \
               "scientific_name" not in content and \
               "name" not in content:
                raise ValueError("Dictionary must contain 'species_name', 'species', or 'name' key")

        if isinstance(content, str) and not content.strip():
            raise ValueError("Content string cannot be empty or whitespace")

        # Validate for special characters
        def contains_special_characters(text):
            return any(ord(char) > 127 for char in str(text))

        if isinstance(content, str) and contains_special_characters(content):
            raise ValueError("Invalid characters in content")
        species_name = None
        if isinstance(content, dict):
            species_name = (content.get("species_name") or
                           content.get("species") or
                           content.get("name") or
                           content.get("common_name") or
                           content.get("scientific_name"))
            if species_name and contains_special_characters(species_name):
                raise BusinessException("Invalid characters in content")

        try:
            # Initial status update
            message_bus.publish("status_update", {
                "message": "Starting to fetch occurrence data...",
                "state": "running",
                "progress": 0
            })

            if not species_name:
                raise BusinessException("Species name is required "
                                "(provide either 'species_name' or 'species')")

            # Translating species name
            message_bus.publish("status_update", {
                "message": f"Translating species name: {species_name}",
                "state": "running",
                "progress": 20
            })

            scientific_name = _self.translate_to_scientific_name_from_api({"name": species_name})

            if ("error" in scientific_name or
                    "scientific_name" not in scientific_name or
                    not scientific_name["scientific_name"]):
                message_bus.publish("status_update", {
                    "message": f"Could not translate species name: {species_name}",
                    "state": "error",
                    "progress": 0
                })
                return {
                    "species": species_name,
                    "occurrence_count": 0,
                    "occurrences": [],
                }

            species_name = scientific_name["scientific_name"]

            # Querying database
            message_bus.publish("status_update", {
                "message": "Querying database for occurrences...",
                "state": "running",
                "progress": 50
            })
            country_codes = content.get("country_codes") or content.get("country_code") or content.get("country")
            # Query setup timing
            if country_codes:
                # Handle both string and list inputs
                if isinstance(content["country_code"], str):
                    country_codes = [code.strip() for code in content["country_code"].split(',')]
                elif isinstance(content["country_code"], list):
                    # Filter out None values and empty strings
                    country_codes = [code for code in content["country_code"] if code]
                else:
                    country_codes = [content["country_code"]]

                # Only proceed with country filter if we have valid codes
                if country_codes:
                    _self.logger.info(
                        "Fetching occurrences for species: %s and countries: %s (type: %s)",
                        species_name,
                        country_codes,
                        type(country_codes)
                    )
                else:
                    country_codes = None
                    _self.logger.info("Fetching occurrences for species: %s", species_name)
            else:
                country_codes = None
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

            if country_codes:
                where_clause = "AND countrycode IN UNNEST(@country_codes)"
                # Ensure country_codes is a list of strings
                country_codes = [str(code) for code in country_codes]
                _self.logger.info("Query parameters - country_codes: %s", country_codes)
                parameters = [
                    bigquery.ScalarQueryParameter("species_name", "STRING", species_name),
                    bigquery.ArrayQueryParameter("country_codes", "STRING", country_codes)
                ]
            else:
                where_clause = ""
                parameters = [
                    bigquery.ScalarQueryParameter("species_name", "STRING", species_name)
                ]

            query = _self.build_query(
                base_query, where_clause=where_clause
            )

            job_config = bigquery.QueryJobConfig(query_parameters=parameters)
            query_job = client.query(query, job_config=job_config)

            message_bus.publish("status_update", {
                "message": "Processing results...",
                "state": "running",
                "progress": 80
            })

            total_occurrences = [
                {
                    "species": species_name,
                    "decimallatitude": row.decimallatitude,
                    "decimallongitude": row.decimallongitude,
                }
                for row in query_job
            ]

            # Final success message
            message_bus.publish("status_update", {
                "message": f"Found {len(total_occurrences)} occurrences for {species_name}",
                "state": "complete",
                "progress": 100
            })

            return {
                "species": species_name,
                "occurrence_count": len(total_occurrences),
                "occurrences": total_occurrences,
            }

        except Exception as e:
            message_bus.publish("status_update", {
                "message": f"'❌' Error processing request: {str(e)}",
                "state": "error",
                "progress": 0
            })
            _self.logger.error("Error in get_occurrences: %s", str(e), exc_info=True)
            raise

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
            species_name = (content.get("species_name") or
                           content.get("species") or
                           content.get("name") or
                           content.get("common_name") or
                           content.get("scientific_name"))
            if not species_name:
                raise BusinessException("Species name is required")

            message_bus.publish("status_update", {
                "message": f"'🐘' Fetching yearly occurrences for {species_name}...",
                "state": "running",
                "progress": 0
            })

            # Translate name and validate
            message_bus.publish("status_update", {
                "message": "'🐘' Validating species name...",
                "state": "running",
                "progress": 20
            })

            translated = self.translate_to_scientific_name_from_api(
                    {"name": species_name}
            )
            if "error" in translated or not translated.get("scientific_name"):
                message_bus.publish("status_update", {
                    "message": f"'❌' Could not find valid scientific name for: {species_name}",
                    "state": "error",
                    "progress": 0
                })
                raise BusinessException(f"Could not find valid scientific name for: {species_name}")

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
            country_codes = content.get("country_codes") or content.get("country_code") or content.get("country") or content.get("countries")
            self.logger.info("Country codes: %s", country_codes)

            if country_codes:
                # Process each country
                for code in country_codes:
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
                    f"{species_name} ({translated['scientific_name']})"
                }

            message_bus.publish("status_update", {
                "message": "Data retrieved successfully",
                "state": "complete",
                "progress": 100
            })

            return {
                "common_name": species_name,
                "scientific_name": translated["scientific_name"],
                "yearly_data": results,
                "type": "temporal",
            }

        except Exception as e:
            message_bus.publish("status_update", {
                "message": f"Error getting yearly occurrences: {str(e)}",
                "state": "error",
                "progress": 0
            })
            self.logger.error(
                "Error getting yearly occurrences: %s", str(e), exc_info=True
            )
            raise

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

class BusinessException(Exception):
    """Custom exception for business logic errors."""
