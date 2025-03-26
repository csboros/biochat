"""Module for handling endangered species related queries."""

import os
import logging
import json
from typing import List, Optional
import google.api_core.exceptions
from google.cloud import bigquery
from ...base_handler import BaseHandler

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
                FROM `{project_id}.biodiversity.endangered_species`
                WHERE LOWER(class) = LOWER(@class_name)
                    AND order_name IS NOT NULL
                GROUP BY order_name
                ORDER BY order_name
            """

            query = self.build_query(
                query,
                where_clause=""
            )

            parameters = self.get_parameters(
                class_name=clazz
            )

            job_config = bigquery.QueryJobConfig(query_parameters=parameters)
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
            return self._format_hierarchy_data(res)
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
            str: Formatted string with hierarchical structure of species data

        Raises:
            ValueError: If family_name is invalid or not found
            google.api_core.exceptions.GoogleAPIError: If BigQuery query fails
            TypeError: If content is not in expected format
        """
        try:
            self.logger.info("Fetching endangered species for family from BigQuery")
            family_name = content['family_name']
            conservation_status = content.get('conservation_status')

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

            res = [(row.species_name, row.family, row.status, row.order_name, row['class'],
                   getattr(row, 'species_name_en', None))  # Add species_name_en
                   for row in query_job]
            return self._format_hierarchy_data(res)
        except google.api_core.exceptions.GoogleAPIError as e:
            self.logger.error("BigQuery error: %s", str(e), exc_info=True)
            raise
        except (TypeError, ValueError) as e:
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
            self.logger.info("Fetching species for country from BigQuery")
            country_code = content['country_code']
            conservation_status = content.get('conservation_status')

            results_data = self._query_country_species(country_code, conservation_status)
            return self._format_hierarchy_data(results_data)
        except google.api_core.exceptions.GoogleAPIError as e:
            self.logger.error("BigQuery error: %s", str(e), exc_info=True)
            raise
        except (TypeError, ValueError) as e:
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

    def get_occurrences(_self, content):  # pylint: disable=no-self-argument
        """
        Retrieves species occurrence data from BigQuery.
        If country_code is provided, the function will return the occurrences
        for the specified country.
        If country_code is not provided, the function will return the distribution for the species.

        Args:
            content (dict): Dictionary containing:
                - species_name (str): Name of the species
                - country_code (str): Single country code or comma-separated list of country codes
                - chart_type (str): Type of visualization

        Returns:
            list: List of dictionaries containing occurrence data with latitude and longitude

        Raises:
            ValueError: If species_name is invalid or not found
            google.api_core.exceptions.GoogleAPIError: If BigQuery query fails
            TypeError: If content is not in expected format
            KeyError: If required fields are missing from the response
        """
        try:
            species_name = content["species_name"]
              # Query setup timing
            if "country_code" in content:
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

            scientific_name = _self.translate_to_scientific_name_from_api(
                {"name": species_name}
            )

            # Parse the JSON response and check for errors
            translated_result = json.loads(scientific_name)
            if ("error" in translated_result or
                    "scientific_name" not in translated_result or
                    not translated_result["scientific_name"]):
                _self.logger.warning(
                    "Could not translate species name: %s - %s",
                    species_name,
                    translated_result["error"],
                )
                return {
                    "species": species_name,
                    "occurrence_count": 0,
                    "occurrences": [],
                }

            species_name = translated_result["scientific_name"]

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
            total_occurrences = [
                {
                    "species": species_name,
                    "decimallatitude": row.decimallatitude,
                    "decimallongitude": row.decimallongitude,
                }
                for row in query_job
            ]
            return {
                "species": species_name,
                "occurrence_count": len(total_occurrences),
                "occurrences": total_occurrences,
            }

        except google.api_core.exceptions.GoogleAPIError as e:
            _self.logger.error(
                "BigQuery error: %s",
                str(e),
                exc_info=True,
            )
            raise
        except KeyError as e:
            _self.logger.error(
                "Missing required field: %s",
                str(e),
                exc_info=True,
            )
            raise
        except (TypeError, ValueError) as e:
            _self.logger.error(
                "Invalid input: %s",
                str(e),
                exc_info=True,
            )
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
