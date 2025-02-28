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
        SELECT DISTINCT CONCAT(genus_name, ' ', species_name) as species_name,
               family_name as family, conservation_status as status, url
        FROM `{project_id}.biodiversity.endangered_species` sp
        JOIN `{project_id}.biodiversity.occurances_endangered_species_mammals` oc
            ON CONCAT(genus_name, ' ', species_name) = oc.species
        WHERE species_name IS NOT NULL
            AND genus_name IS NOT NULL
            {where_clause}
            {conservation_status_filter}
        GROUP BY genus_name, species_name, family_name, conservation_status, url
        ORDER BY species_name
    """

    def __init__(self):
        self.logger = logging.getLogger(f"BioChat.{self.__class__.__name__}")
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
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
        except Exception as e:
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
