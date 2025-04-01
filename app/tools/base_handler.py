"""Base handler class with common functionality."""

import logging
import os
import json
from typing import Dict, List
import requests
from google.cloud import bigquery

class BaseHandler:
    """Base handler class that provides common functionality for API interactions and logging."""

    # Add base query templates
    SPECIES_QUERY_TEMPLATE = """
            WITH RankedSpecies AS (
                SELECT
                    CONCAT(genus_name, ' ', species_name) as species_name,
                    species_name_en,
                    family_name as family,
                    conservation_status as status,
                    order_name,
                    class,
                    ROW_NUMBER() OVER (
                        PARTITION BY species_name
                        ORDER BY
                            CASE conservation_status
                                WHEN 'Extinct' THEN 7
                                WHEN 'Critically Endangered' THEN 6
                                WHEN 'Endangered' THEN 5
                                WHEN 'Vulnerable' THEN 4
                                WHEN 'Near Threatened' THEN 3
                                WHEN 'Least Concern' THEN 2
                                WHEN 'Data Deficient' THEN 1
                                ELSE 0
                            END DESC
                    ) as rank
                FROM `{project_id}.biodiversity.endangered_species` sp
                JOIN `{project_id}.biodiversity.occurances_endangered_species_mammals` oc
                    ON CONCAT(genus_name, ' ', species_name) = oc.species
                WHERE species_name IS NOT NULL
                    AND genus_name IS NOT NULL
                    {where_clause}
                    {conservation_status_filter}
            )
            SELECT
                species_name,
                species_name_en,
                family,
                status,
                order_name,
                class
            FROM RankedSpecies
            WHERE rank = 1
            ORDER BY class, order_name, family, species_name
            """

    def __init__(self):
        """Initialize the base handler."""
        self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable must be set")
        self.logger = logging.getLogger(f"BioChat.{self.__class__.__name__}")
        self.client = bigquery.Client(project=self.project_id)

    def translate_to_scientific_name_from_api(self, content: Dict) -> str:
        """Translates common species name to scientific name using EBI Taxonomy API."""
        species_name = content.get("name", "").strip()
        if not species_name:
            return json.dumps({"error": "No species name provided"})

        try:
            self.logger.info("Fetching scientific name for species: %s", species_name)
            url = f"https://www.ebi.ac.uk/ena/taxonomy/rest/any-name/{species_name}"
            response = requests.get(
                url, headers={"Accept": "application/json"}, timeout=5
            )
            response.raise_for_status()

            data = response.json()
            if data and isinstance(data, list) and len(data) > 0:
                scientific_name = data[0].get("scientificName")
                if scientific_name:
                    self.logger.info(
                        "Successfully translated '%s' to '%s'",
                        species_name,
                        scientific_name,
                    )
                    return json.dumps({"scientific_name": scientific_name})

            return json.dumps({"error": "Name could not be translated"})

        except requests.Timeout:
            self.logger.error("Request timeout for species: %s", species_name)
            return json.dumps({"error": "Request timed out"})
        except Exception as e: # pylint: disable=broad-except
            self.logger.error("Error translating species name: %s", str(e))
            return json.dumps({"error": "An error occurred"})

    def translate_to_common_name_from_api(self, content: Dict) -> str:
        """Translates scientific species name to common name using EBI Taxonomy API."""
        scientific_name = content.get("name", "").strip()
        if not scientific_name:
            return json.dumps({"error": "No species name provided"})

        try:
            self.logger.info("Fetching common name for species: %s", scientific_name)
            url = f"https://www.ebi.ac.uk/ena/taxonomy/rest/scientific-name/{scientific_name}"
            response = requests.get(
                url, headers={"Accept": "application/json"}, timeout=5
            )
            response.raise_for_status()

            data = response.json()
            if data and isinstance(data, list) and len(data) > 0:
                common_name = data[0].get("commonName")
                if common_name:
                    self.logger.info(
                        "Successfully translated '%s' to '%s'",
                        scientific_name,
                        common_name,
                    )
                    return json.dumps({"common_name": common_name})

            return json.dumps({"error": "Name could not be translated"})

        except requests.Timeout:
            self.logger.error("Request timeout for species: %s", scientific_name)
            return json.dumps({"error": "Request timed out"})
        except Exception as e:  # pylint: disable=broad-except
            self.logger.error("Error translating species name: %s", str(e))
            return json.dumps({"error": "An error occurred"})

    def _build_filters(self, **kwargs) -> dict:
        """Build query filters based on parameters."""
        filters = {'conservation_status_filter': ''}

        if kwargs.get('conservation_status'):
            filters['conservation_status_filter'] = (
                "AND LOWER(conservation_status) = LOWER(@conservation_status)"
            )

        return filters

    def build_query(self, base_query: str, **kwargs) -> str:
        """Build a query with proper formatting and filters."""
        # Prepare all parameters
        params = {
            'project_id': self.project_id,
            'where_clause': kwargs.get('where_clause', ''),
            'conservation_status_filter': ''
        }

        # Build filters separately
        filters = self._build_filters(**kwargs)
        params.update(filters)

        # Apply all parameters
        return base_query.format(**params)

    def get_parameters(self, **kwargs) -> List[bigquery.ScalarQueryParameter]:
        """Create query parameters."""
        parameters = []
        for key, value in kwargs.items():
            if value is not None:
                parameters.append(
                    bigquery.ScalarQueryParameter(key, "STRING", value)
                )
        return parameters
