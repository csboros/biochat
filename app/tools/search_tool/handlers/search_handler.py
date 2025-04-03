"""Module for handling search-related queries."""

import os
import logging
from typing import Dict, Any
from langchain_google_community import GoogleSearchAPIWrapper
from app.tools.base_handler import BaseHandler

class SearchHandler(BaseHandler):
    """Handles search-related queries."""

    def __init__(self):
        """Initialize the search handler."""
        super().__init__()
        self.logger = logging.getLogger("BioChat." + self.__class__.__name__)
        self.search = GoogleSearchAPIWrapper()

    def google_search(self, content: Dict[str, Any]) -> str:
        """
        Performs a Google search focused on biodiversity-related results.

        Args:
            content (Dict[str, Any]): Dictionary containing:
                - query (str): The search query

        Returns:
            str: Search results as a string

        Raises:
            ValueError: If query is invalid
            google.api_core.exceptions.GoogleAPIError: If search API fails
        """
        try:
            query_string = content.get("query")
            if not query_string:
                raise ValueError("Query is required")

            # Create a search query across multiple authoritative biodiversity sites
            sites = [
                "iucnredlist.org",           # IUCN Red List
                "gbif.org",                  # Global Biodiversity Information Facility
                "eol.org",                   # Encyclopedia of Life
                "biodiversitya-z.org",       # Biodiversity A-Z
                "unep-wcmc.org",             # UN Environment World Conservation Monitoring Centre
                "protectedplanet.net",       # Protected Planet
                "worldwildlife.org",         # World Wildlife Fund
                "fauna-flora.org",           # Fauna & Flora International
                "blogs.worldbank.org",       # World Bank Blogs
                "statistics.laerd.com",      # Statistics explanations
                "statisticshowto.com",       # Statistics definitions
                "scholar.google.com",        # Academic papers
                "researchgate.net",          # Research papers and definitions
                "link.springer.com",         # Academic publications
                "sciencedirect.com",         # Scientific articles
            ]
            site_query = " OR ".join(f"site:{site}" for site in sites)
            query = f"({site_query}) {query_string}"
            return self.search.run(query)

        except ValueError as e:
            self.logger.error("Invalid query: %s", str(e), exc_info=True)
            raise
        except Exception as e:
            self.logger.error("Error performing Google search: %s", str(e), exc_info=True)
            raise
