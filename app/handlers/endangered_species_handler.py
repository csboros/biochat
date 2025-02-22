"""Module for handling endangered species related queries."""

import os
import logging
from typing import List, Optional
import google.api_core.exceptions
from google.cloud import bigquery
from .query_builder import BaseQueryBuilder

class EndangeredSpeciesHandler:
    """Handles queries related to endangered species."""

    def __init__(self):
        self.logger = logging.getLogger("BioChat." + self.__class__.__name__)
        self.query_builder = BaseQueryBuilder()

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
                                conservation_status: str = None) -> list:
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
                            conservation_status: str = None) -> list:
        """Create query parameters for BigQuery."""
        parameters = [
            bigquery.ScalarQueryParameter("family_name", "STRING", family_name)
        ]
        if conservation_status:
            parameters.append(
                bigquery.ScalarQueryParameter("conservation_status", "STRING", conservation_status)
            )
        return parameters

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
                                conservation_status: str = None) -> list:
        """Execute BigQuery to fetch country's endangered species data."""
        client = bigquery.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'))
        query = self._build_country_species_query(conservation_status)
        parameters = self._get_country_query_parameters(country_code, conservation_status)
        job_config = bigquery.QueryJobConfig(query_parameters=parameters)
        query_job = client.query(query, job_config=job_config)
        results = query_job.result()
        return [(row.species_name, row.family, row.status, row.url) for row in results]

    def _build_country_species_query(self, conservation_status: Optional[str] = None) -> str:
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        params = {
            'where_clause': "AND oc.countrycode = @country_code",
            'conservation_status': conservation_status
        }
        return self.query_builder.build_query(
            self.query_builder.SPECIES_QUERY_TEMPLATE,
            project_id,
            **params
        )

    def _get_country_query_parameters(
            self,
            country_code: str,
            conservation_status: Optional[str] = None
    ) -> List[bigquery.ScalarQueryParameter]:
        """Create query parameters for country species query."""
        return self.query_builder.get_parameters(
            country_code=country_code,
            conservation_status=conservation_status
        )

    def _format_country_species_results(self, results: list) -> str:
        """Format the country species results into a readable string."""
        output = ["**Only Mammals are included in the list.**\n"]
        # Group species by family
        families = {}
        for species_name, family, status, url in results:
            if family not in families:
                families[family] = []
            families[family].append(f"* **{species_name}** ({status})\n    * {url}")
        # Format output
        for family, species_list in sorted(families.items()):
            output.append(f"\n**Family: {family}**")
            output.extend(species_list)
        return "\n".join(output) + "\n"

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

    def _query_conservation_status_counts(self, country_code: str = None) -> list:
        """Execute BigQuery to fetch conservation status counts."""
        client = bigquery.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'))
        query = self._build_conservation_count_query(country_code)
        parameters = self._get_conservation_count_parameters(country_code)
        job_config = bigquery.QueryJobConfig(query_parameters=parameters)
        query_job = client.query(query, job_config=job_config)
        results = query_job.result()
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
                            country_code: str = None) -> list:
        """Create query parameters for conservation count query."""
        if not country_code:
            return []
        return [bigquery.ScalarQueryParameter("country_code", "STRING", country_code)]

    def _format_conservation_counts(self, results: list) -> str:
        """Format the conservation status counts into a readable string."""
        output = ["**Only Mammals are included in the list.**\n"]
        for status, count in results:
            output.append(f"* {status}: {count} species")
        return "\n".join(output)
