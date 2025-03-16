"""Module for handling species-HCI correlation queries."""

import os
import logging
from typing import Dict, Any
from google.cloud import bigquery
from vertexai.preview.generative_models import (GenerativeModel, GenerationConfig)
from .base_handler import BaseHandler

class CorrelationHandler(BaseHandler):
    """Handles queries related to species-HCI correlations."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("BioChat." + self.__class__.__name__)

    def get_species_hci_correlation(self, country_code: str = 'KEN') -> Dict[str, Any]:
        """
        Retrieves correlation data between species occurrence and HCI for a country.

        Args:
            country_code (str): ISO Alpha-3 country code (default: 'KEN')

        Returns:
            Dict[str, Any]: Dictionary containing correlation data

        Raises:
            google.api_core.exceptions.GoogleAPIError: If BigQuery query fails
        """
        try:
            # If country_code is a dict, extract the value
            if isinstance(country_code, dict):
                country_code = country_code.get('country_code', 'KEN')

            self.logger.info("Fetching species-HCI correlation data for country: %s", country_code)

            query = """
            WITH species_hci AS (
              SELECT 
                 CONCAT(sp.genus_name, ' ', sp.species_name) as species_name,
                sp.species_name_en,
                sp.conservation_status,
                oc.countrycode,
                ROUND(oc.decimallongitude * 4) / 4 as grid_lon,
                ROUND(oc.decimallatitude * 4) / 4 as grid_lat,
                SUM(oc.individualcount) as total_individuals_in_cell,
                AVG(h.terrestrial_hci) as cell_hci
              FROM `{project_id}.biodiversity.endangered_species` sp
              INNER JOIN `{project_id}.biodiversity.occurances_endangered_species_mammals` oc 
                ON CONCAT(sp.genus_name, ' ', sp.species_name) = oc.species
              INNER JOIN `{project_id}.biodiversity.hci` h
                ON ROUND(oc.decimallongitude * 4) / 4 = ROUND(h.decimallongitude * 4) / 4
                AND ROUND(oc.decimallatitude * 4) / 4 = ROUND(h.decimallatitude * 4) / 4
              INNER JOIN `{project_id}.biodiversity.countries` c
                ON c.iso_a3 = @country_code
                AND ST_CONTAINS(
                  c.geometry,
                  ST_GEOGPOINT(oc.decimallongitude, oc.decimallatitude)
                )
              GROUP BY 
                sp.species_name,
                sp.genus_name,
                sp.species_name_en,
                sp.conservation_status,
                oc.countrycode,
                c.name,
                grid_lon,
                grid_lat
            ),

            species_stats AS (
              SELECT 
                species_name,
                species_name_en,
                conservation_status,
                CORR(total_individuals_in_cell, cell_hci) as correlation_coefficient,
                COUNT(DISTINCT CONCAT(grid_lon, ',', grid_lat)) as number_of_grid_cells,
                SUM(total_individuals_in_cell) as total_individuals,
                AVG(cell_hci) as avg_hci,
                AVG(total_individuals_in_cell) as avg_individuals_per_cell,
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
                ) as status_rank
              FROM species_hci
              GROUP BY 
                species_name,
                species_name_en,
                conservation_status
              HAVING COUNT(DISTINCT CONCAT(grid_lon, ',', grid_lat)) >= 5
            )

            SELECT 
              species_name,
              species_name_en,
              conservation_status,
              correlation_coefficient,
              number_of_grid_cells,
              total_individuals,
              avg_hci,
              avg_individuals_per_cell
            FROM species_stats
            WHERE correlation_coefficient IS NOT NULL 
              AND IS_NAN(correlation_coefficient) = FALSE
              AND status_rank = 1
            ORDER BY correlation_coefficient DESC 
            """

            # Build query with project ID
            query = self.build_query(query)

            client = bigquery.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'))

            # Set up query parameters
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("country_code", "STRING", country_code)
                ]
            )

            # Execute query
            query_job = client.query(query, job_config=job_config)

            # Print query job ID for debugging

            results = query_job.result()

            # Print raw results for debugging
            results_list = list(results)  # Materialize results

            # Format results
            correlation_data = []
            for row in results_list:
                correlation_data.append({
                    'species_name': row.species_name,
                    'species_name_en': row.species_name_en,
                    'conservation_status': row.conservation_status,
                    'correlation_coefficient': row.correlation_coefficient,
                    'number_of_grid_cells': row.number_of_grid_cells,
                    'total_individuals': row.total_individuals,
                    'avg_hci': row.avg_hci,
                    'avg_individuals_per_cell': row.avg_individuals_per_cell
                })

            print("Formatted correlation data:", correlation_data)

            return {
                'country_code': country_code,
                'correlations': correlation_data
            }

        except Exception as e:
            self.logger.error("Error fetching correlation data: %s", str(e), exc_info=True)
            print("Error details:", str(e))
            raise

    def analyze_correlation_data_with_llm(self, country_code: str = 'KEN'):
        """
        Analyzes species correlation data using LLM to draw conclusions about conservation patterns.

        Args:
            country_code (str): ISO Alpha-3 country code (default: 'KEN')

        Returns:
            str: LLM analysis of correlation patterns and conservation implications
        """
        # Get correlation data first
        correlation_data = self.get_species_hci_correlation(country_code)

        # Format the data into a clear prompt
        prompt = """You are a conservation biology expert. Analyze this species 
            correlation data between species occurrence and Human Coexistence Index (HCI).

Context:
- HCI measures human impact (higher values = more human impact)
- Correlation shows if species occur more (+ve) or less (-ve) in human-impacted areas
- Focus on conservation status patterns and their implications

Please analyze:
1. Patterns for each conservation status (Critically Endangered, Endangered, Vulnerable, Near Threatened, Least Concern, Extinct)
2. Notable correlations (both positive and negative)
3. Conservation implications
4. Data limitations and caveats
5. Please summarize the results in a few sentences and show the summary at the end of your response.

Data:
"""
        # Send to LLM and return response
        prompt += f"\n{correlation_data['correlations']}"
        return self.send_to_llm(prompt)

    def send_to_llm(self, prompt: str) -> str:
        """Sends a prompt to Google's Gemini LLM and returns the response."""
        try:
            self.logger.info("Sending prompt to Gemini")

            # Lower temperature for more analytical responses
            model = GenerativeModel(
                "gemini-2.0-flash-lite", 
                generation_config=GenerationConfig(
                    temperature=0.5,  # Lower temperature for more precise analysis
                    max_output_tokens=8192,  # Ensure we get detailed response
                    top_p=0.95,  # More focused token selection
                )
            )

            # Add explicit instruction for quantitative analysis
            response = model.generate_content(prompt)

            self.logger.info("Received response from Gemini")
            return response.text

        except Exception as e:
            self.logger.error("Error getting response from Gemini: %s", str(e), exc_info=True)
            raise
