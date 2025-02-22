"""Module for handling BigQuery query building."""

import os
from typing import List
from google.cloud import bigquery

class BaseQueryBuilder:
    """Base class for building BigQuery queries."""

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
        """Initialize the query builder."""
        self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT')

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
