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

    def google_search(self, params: dict) -> str:
        """Perform a Google search."""
        # Accept multiple parameter variations for the search query
        query = (params.get('query') or params.get('q') or
                params.get('search_term') or params.get('search_query') or
                params.get('search_terms') or params.get('queries'))

        if not query:
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
            "en.wikipedia.org",          # Wikipedia
        ]
        site_query = " OR ".join(f"site:{site}" for site in sites)
        query = f"({site_query}) {query}"
        return self.search.run(query)
