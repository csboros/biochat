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

    def get_species_hci_correlation(self, country_code: str) -> Dict[str, Any]:
        """
        Retrieves correlation data between species occurrence and HCI for a country.

        Args:
            country_code (str): ISO Alpha-3 country code

        Returns:
            Dict[str, Any]: Dictionary containing correlation data

        Raises:
            google.api_core.exceptions.GoogleAPIError: If BigQuery query fails
        """
        try:
            # If country_code is a dict, extract the value
            if isinstance(country_code, dict):
                country_code = country_code.get('country_code')

            if not country_code:
                raise ValueError("Country code must be provided")

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
                SUM(COALESCE(oc.individualcount, 1)) as total_individuals_in_cell,
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

    def get_species_hci_correlation_by_status(self, conservation_status: str = 'Critically Endangered') -> Dict[str, Any]:
        """
        Retrieves correlation data between species occurrence and HCI for a specific conservation status.

        Args:
            conservation_status (str): Conservation status to filter by (default: 'Critically Endangered')
                                     Valid values: 'Critically Endangered', 'Endangered', 'Vulnerable', 
                                                 'Near Threatened', 'Least Concern', 'Data Deficient', 'Extinct'

        Returns:
            Dict[str, Any]: Dictionary containing correlation data for the specified conservation status

        Raises:
            google.api_core.exceptions.GoogleAPIError: If BigQuery query fails
            ValueError: If invalid conservation status is provided
        """
        try:
            # Normalize input string by stripping whitespace
            if isinstance(conservation_status, dict):
                conservation_status = conservation_status.get('conservation_status', '').strip()
            else:
                conservation_status = str(conservation_status).strip()

            # Debug logging
            self.logger.info("Received conservation status: '%s'", conservation_status)

            # Validate conservation status
            valid_statuses = [
                'Critically Endangered', 'Endangered', 'Vulnerable', 
                'Near Threatened', 'Least Concern', 'Data Deficient', 'Extinct'
            ]

            # Debug logging for comparison
            self.logger.info("Valid statuses: %s", valid_statuses)
            self.logger.info("Status in valid_statuses: %s", conservation_status in valid_statuses)
            if conservation_status not in valid_statuses:
                raise ValueError(f"Invalid conservation status '{conservation_status}'. "
                                 f"Must be one of: {', '.join(valid_statuses)}")

            self.logger.info("Fetching species-HCI correlation data for status: %s",
                             conservation_status)

            query = """
            WITH species_hci AS (
              SELECT 
                CONCAT(sp.genus_name, ' ', sp.species_name) as species_name,
                sp.species_name_en,
                sp.conservation_status,
                ROUND(oc.decimallongitude * 4) / 4 as grid_lon,
                ROUND(oc.decimallatitude * 4) / 4 as grid_lat,
                SUM(COALESCE(oc.individualcount, 1)) as total_individuals_in_cell,
                AVG(h.terrestrial_hci) as cell_hci
              FROM `{project_id}.biodiversity.endangered_species` sp
              INNER JOIN `{project_id}.biodiversity.occurances_endangered_species_mammals` oc 
                ON CONCAT(sp.genus_name, ' ', sp.species_name) = oc.species
              INNER JOIN `{project_id}.biodiversity.hci` h
                ON ROUND(oc.decimallongitude * 4) / 4 = ROUND(h.decimallongitude * 4) / 4
                AND ROUND(oc.decimallatitude * 4) / 4 = ROUND(h.decimallatitude * 4) / 4
              WHERE sp.conservation_status = @conservation_status
              GROUP BY 
                species_name,
                species_name_en,
                conservation_status,
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
                AVG(total_individuals_in_cell) as avg_individuals_per_cell
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
            ORDER BY ABS(correlation_coefficient) DESC
            """

            # Build query with project ID
            query = self.build_query(query)

            client = bigquery.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'))

            # Set up query parameters
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("conservation_status", "STRING", conservation_status)
                ]
            )

            # Execute query
            query_job = client.query(query, job_config=job_config)
            results = query_job.result()

            # Format results
            correlation_data = []
            for row in results:
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

            return {
                'conservation_status': conservation_status,
                'correlations': correlation_data
            }

        except Exception as e:
            self.logger.error("Error fetching correlation data by status: %s", str(e),
                              exc_info=True)
            raise

    def analyze_species_correlations(self, country_code: str = None, conservation_status: str = None) -> Dict[str, Any]:
        """
        Analyzes correlation patterns between species occurrences and HCI, either by country or conservation status.

        Args:
            country_code (str, optional): ISO Alpha-3 country code (e.g., 'KEN' for Kenya)
            conservation_status (str, optional): Conservation status to filter by
                Valid values: 'Critically Endangered', 'Endangered', 'Vulnerable',
                            'Near Threatened', 'Least Concern', 'Data Deficient', 'Extinct'

        Returns:
            Dict[str, Any]: Dictionary containing correlation analysis results

        Raises:
            ValueError: If neither country_code nor conservation_status is provided
        """
        try:
            # Debug logging for initial parameters
            self.logger.info("Initial parameters in analyze_species_correlations:")
            self.logger.info("country_code: %s (type: %s)", country_code, type(country_code))
            self.logger.info("conservation_status: %s (type: %s)", conservation_status, type(conservation_status))

            # Special handling for misplaced parameters
            if isinstance(country_code, dict) and 'conservation_status' in country_code:
                conservation_status = country_code['conservation_status']
                country_code = None
                self.logger.info("Extracted misplaced conservation_status from country_code dict: %s", conservation_status)
            elif isinstance(country_code, dict):
                country_code = country_code.get('country_code')
                self.logger.info("Extracted country_code from dict: %s", country_code)

            if isinstance(conservation_status, dict):
                conservation_status = conservation_status.get('conservation_status')
                self.logger.info("Extracted conservation_status from dict: %s", conservation_status)
            elif isinstance(conservation_status, str):
                conservation_status = conservation_status.strip()
                self.logger.info("Stripped conservation_status string: %s", conservation_status)

            # Debug logging for processed parameters
            self.logger.info("Processed parameters:")
            self.logger.info("country_code: %s", country_code)
            self.logger.info("conservation_status: %s", conservation_status)

            # Validate parameters
            if not country_code and not conservation_status:
                self.logger.error("Both parameters are empty or None")
                raise ValueError("Either country_code or conservation_status must be provided")

            # Get correlation data based on provided parameter
            correlation_data = {}
            if conservation_status:
                self.logger.info("Using conservation status: %s", conservation_status)
                correlation_data = self.get_species_hci_correlation_by_status(conservation_status)
            elif country_code:
                self.logger.info("Using country code: %s", country_code)
                correlation_data = self.get_species_hci_correlation(country_code)

            # Send to LLM for analysis
            prompt = """You are a conservation biology expert. Analyze this species 
                correlation data between species occurrence and Human Coexistence Index (HCI).

            Context:
            - HCI measures human impact (higher values = more human impact)
            - Correlation shows if species occur more (+ve) or less (-ve) in human-impacted areas
            """

            if country_code:
                prompt += f"\nAnalyzing correlations for country: {country_code}\n"
            else:
                prompt += f"\nAnalyzing correlations for {conservation_status} species\n"

            prompt += """
            Please analyze:
            1. Overall correlation patterns
            2. If we have different conservation statuses, please analyze the correlations for each conservation status.
            3. Notable correlations (both positive and negative)
            4. Conservation implications
            5. Data limitations and caveats
            6. Please summarize the results in a few sentences and show the summary at the end of your response.

            Data:
            """
            prompt += f"\n{correlation_data['correlations']}"

            analysis = self.send_to_llm(prompt)

            # Return both the analysis and the correlation data for visualization
            return {
                'correlation_data': correlation_data,
                'analysis': analysis,
                'type': 'country' if country_code else 'conservation_status',
                'filter_value': country_code or conservation_status
            }

        except Exception as e:
            self.logger.error("Error analyzing species correlations: %s", str(e), exc_info=True)
            raise

    def get_species_shared_habitat(self, species_name: str) -> Dict[str, Any]:
        """
        Retrieves correlation data between the specified species and other species that share its habitat.

        Args:
            species_name (str): Scientific name of the species (e.g., 'Panthera leo')

        Returns:
            Dict[str, Any]: Dictionary containing correlation data with other species

        Raises:
            google.api_core.exceptions.GoogleAPIError: If BigQuery query fails
            ValueError: If species name is not provided or invalid
        """
        try:
            # Handle case where species_name is passed as a dictionary
            if isinstance(species_name, dict):
                species_name = species_name.get('species_name')
                if isinstance(species_name, dict):  # Handle double nesting if it occurs
                    species_name = species_name.get('species_name')

            if not species_name:
                raise ValueError("Species name must be provided")

            print(f"Fetching shared habitat correlations for species: {species_name}")  # Debug log

            query = """
            SELECT 
                species_1,
                species_1_en,
                species_1_status,
                species_2,
                species_2_en,
                species_2_status,
                correlation_coefficient,
                overlapping_cells
            FROM `{project_id}.biodiversity.species_correlations`
            WHERE species_1 = @species_name and correlation_coefficient > 0.2
            ORDER BY ABS(correlation_coefficient) DESC
            """

            # Build query with project ID
            query = self.build_query(query)
            print(f"Executing query: {query}")  # Debug log

            client = bigquery.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'))

            # Set up query parameters
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("species_name", "STRING", species_name)
                ]
            )

            # Execute query
            query_job = client.query(query, job_config=job_config)
            results = query_job.result()

            # Format results
            correlation_data = []
            for row in results:
                correlation_data.append({
                    'species_1': row.species_1,
                    'species_1_en': row.species_1_en,
                    'species_1_status': row.species_1_status,
                    'species_2': row.species_2,
                    'species_2_en': row.species_2_en,
                    'species_2_status': row.species_2_status,
                    'correlation_coefficient': row.correlation_coefficient,
                    'overlapping_cells': row.overlapping_cells
                })

            print(f"Found {len(correlation_data)} correlations")  # Debug log

            response = {
                'species_name': species_name,
                'correlations': correlation_data
            }
            print("Returning response:", response)  # Debug log
            return response

        except Exception as e:
            print(f"Error in get_species_shared_habitat: {str(e)}")  # Debug log
            self.logger.error("Error fetching shared habitat data: %s", str(e), exc_info=True)
            raise
