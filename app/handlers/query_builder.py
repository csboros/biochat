"""Module for handling BigQuery query building."""

from typing import List
from google.cloud import bigquery

class BaseQueryBuilder:
    """Base class for building BigQuery queries."""

    def build_query(self, base_query: str, project_id: str, **kwargs) -> str:
        """Build a query with proper formatting and filters."""
        # Replace all project_id placeholders with the same value
        query = base_query.format(*[project_id] * base_query.count('{}'))

        # Then apply filters
        filters = self._build_filters(**kwargs)
        if filters:
            query = query.format(**filters)

        return query

    def _build_filters(self, **kwargs) -> dict:
        """Build query filters based on parameters."""
        filters = {'conservation_status_filter': ''}  # Default empty filter

        if kwargs.get('conservation_status'):
            filters['conservation_status_filter'] = (
                "AND LOWER(conservation_status) = LOWER(@conservation_status)"
            )

        return filters

    def get_parameters(self, **kwargs) -> List[bigquery.ScalarQueryParameter]:
        """Create query parameters."""
        parameters = []
        for key, value in kwargs.items():
            if value is not None:
                parameters.append(
                    bigquery.ScalarQueryParameter(key, "STRING", value)
                )
        return parameters
