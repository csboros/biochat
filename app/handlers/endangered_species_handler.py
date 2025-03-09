"""Module for handling endangered species related queries."""

import os
import logging
from typing import List, Optional
import google.api_core.exceptions
from google.cloud import bigquery
from .base_handler import BaseHandler

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
                WHERE LOWER(kingdom) = LOWER(@kingdom) AND class IS NOT NULL 
                GROUP BY class ORDER BY class
            """

            query = self.build_query(
                query,
                where_clause=""
            )

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
            query = """
                SELECT family_name, count(family_name) as cnt 
                FROM `{project_id}.biodiversity.endangered_species` 
                {where_clause}
                GROUP BY family_name 
                ORDER BY family_name
            """

            query = self.build_query(
                query,
                where_clause=("WHERE LOWER(order_name) = LOWER(@order_name) "
                            "AND family_name IS NOT NULL")
            )

            parameters = self.get_parameters(
                order_name=order_name
            )

            job_config = bigquery.QueryJobConfig(query_parameters=parameters)
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
                                conservation_status: str = None) -> list:
        """Execute BigQuery to fetch endangered species data."""
        client = bigquery.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'))
        query = self._build_species_query(conservation_status)
        parameters = self.get_parameters(
            family_name=family_name,
            conservation_status=conservation_status
        )
        job_config = bigquery.QueryJobConfig(query_parameters=parameters)
        query_job = client.query(query, job_config=job_config)
        return [(row['species_header'], row['urls']) for row in query_job]

    def _build_species_query(self, conservation_status: str = None) -> str:
        """Build the BigQuery query string."""
        base_query = """
           SELECT 
               CONCAT(genus_name, ' ', species_name, ':') as species_header,
               STRING_AGG(url, '||' ORDER BY url) as urls
           FROM `{project_id}.biodiversity.endangered_species` 
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
        return self.build_query(
            base_query,
            conservation_status_filter=conservation_status_filter
        )

    def _format_species_results(self, results_data: list) -> str:
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
        scientific_name = scientific_name.split(':')[0]
        return f"* **{scientific_name}**:\n{urls_formatted}"

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
        return [(row.species_name, row.family, row.status, row.order_name, row['class']) for row in results]

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
        for species_name, family_name, status, order_name, class_name in data:
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

            # Add species
            families[family_name]['children'].append({
                'name': species_name,
                'status': status,
                'value': 1
            })

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
        Retrieves endangered species for MULTIPLE countries.
        Use this function when comparing species across two or more countries.

        Args:
            content (dict): Dictionary containing:
                - country_codes (list[str]): List of two-letter country codes (e.g., ['NA', 'KE'])
                - conservation_status (str, optional):
                    Status to filter by (e.g., 'Critically Endangered')

        Returns:
            str: Formatted string with endangered species for all specified countries
        """
        try:
            # Get list of countries to process
            country_codes = content.get('country_codes', [])  # Remove default European countries
            conservation_status = content.get('conservation_status')

            if not country_codes:  # Add validation for empty country list
                raise ValueError("No country codes provided")

            # Rest of the method remains the same
            all_results = []
            for country_code in country_codes:
                params = {
                    'country_code': country_code,  # Remove list wrapping
                    'conservation_status': conservation_status
                }
                country_results = self.endangered_species_for_country(params)
                all_results.append(country_results)

            return ("\n\n".join(all_results) if all_results 
                    else "No results found for any of the specified countries.")

        except Exception as e:
            self.logger.error("Error in endangered_species_for_countries: %s", str(e))
            raise
