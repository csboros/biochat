"""
Handler for species-related operations.
"""
from typing import Dict
import json
import logging
import os
import time
import urllib.parse
import requests
import google.api_core.exceptions
from google.cloud import bigquery
from pygbif import species
from ...base_handler import BaseHandler
try:
    from fuzzywuzzy import fuzz
except ImportError:
    fuzz = None

class SpeciesHandler(BaseHandler):
    """Handler for species-related operations."""

    def __init__(self):
        """Initialize the species handler."""
        super().__init__()
        self.logger = logging.getLogger("BioChat." + self.__class__.__name__)

    def get_species_info_from_api(self, content):
        """
        Retrieves species information from GBIF API.

        Raises:
            ValueError: If species name is invalid
            pygbif.gbif.GbifError: If GBIF API request fails
        """
        start_time = time.time()
        try:
            species_name = content["name"]
            self.logger.info("Fetching species info for: %s", species_name)
            # GBIF API call timing
            api_call_start = time.time()
            species_info = species.name_suggest(species_name)
            self.logger.info(
                "GBIF API call took %.2f seconds", time.time() - api_call_start
            )
            # JSON serialization timing
            json_start = time.time()
            result = json.dumps(species_info)
            self.logger.info(
                "JSON serialization took %.2f seconds", time.time() - json_start
            )
            self.logger.info(
                "Total get_species_info took %.2f seconds", time.time() - start_time
            )
            print(result)
            return result
        except KeyError as e:
            self.logger.error(
                "Missing species name (took %.2f seconds): %s",
                time.time() - start_time,
                str(e),
                exc_info=True,
            )
            raise ValueError("Species name is required") from e
        except Exception as e:  # GBIF doesn't expose specific exceptions
            self.logger.error(
                "GBIF API error (took %.2f seconds): %s",
                time.time() - start_time,
                str(e),
                exc_info=True,
            )
            raise

    def get_species_images(self, content):
        """
        Retrieves images for a species from GBIF API.

        Args:
            content (dict): Dictionary containing:
                - species_name (str): Name of the species

        Returns:
            list: List of image URLs and metadata
        """
        try:
            species_name = content["species_name"]

            # GBIF API endpoint for species search
            url = f"https://api.gbif.org/v1/species/search?q={urllib.parse.quote(species_name)}"

            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data.get("results"):
                return {"species": species_name, "image_count": 0, "images": []}

            taxon_key = data["results"][0]["key"]

            # Get media items for this taxon
            media_url = (
                f"https://api.gbif.org/v1/occurrence/search?"
                f"taxonKey={taxon_key}&mediaType=StillImage&limit=5"
            )
            print(media_url)

            response = requests.get(media_url, timeout=10)
            response.raise_for_status()
            media_data = response.json()

            images = []
            for result in media_data.get("results", []):
                media = result.get("media", [])
                for item in media:
                    if item.get("type") == "StillImage":
                        images.append({
                            "url": item.get("identifier"),
                            "title": result.get("scientificName", species_name),
                            "creator": item.get("creator", "Unknown"),
                            "license": item.get("license", "Unknown"),
                            "publisher": result.get("publisher", "GBIF")
                        })
                        break  # Only get first image from each occurrence

            return {
                "species": species_name,
                "image_count": len(images),
                "images": images
            }

        except requests.RequestException as e:
            self.logger.error(
                "Error fetching from GBIF API: %s",
                str(e),
                exc_info=True
            )
            raise
        except Exception as e:
            self.logger.error(
                "Error processing species images: %s",
                str(e),
                exc_info=True
            )
            raise

    def normalize_protected_area_name(self, name: str) -> str:
        """Normalize protected area name by removing common suffixes and extra spaces."""
        suffixes = [
            "national park",
            "national reserve",
            "game reserve",
            "conservation area",
            "marine park",
            "wildlife sanctuary",
            "nature reserve",
            "park",
            "reserve",
        ]
        name = name.lower().strip()
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[: -(len(suffix))].strip()
                break
        return name

    def get_endangered_species_in_protected_area(self, content: dict) -> dict:
        """
        Get endangered species in a protected area with fuzzy name matching.
        Args:
            content (dict): Dictionary containing protected_area_name
        Returns:
            dict: List of endangered species and their details
        Raises:
            ValueError: If protected area name is invalid or not found
            google.api_core.exceptions.GoogleAPIError: If BigQuery query fails
        """
        try:
            protected_area_name = content.get("protected_area_name")
            if not protected_area_name:
                raise ValueError("Protected area name is required")

            # Get best matching protected area name
            best_match = self._find_matching_protected_area(protected_area_name)
            if not best_match:
                self.logger.warning(
                    "No matching protected area found for: %s", protected_area_name
                )
                return {
                    "error": f"No matching protected area found for: {protected_area_name}"
                }

            # Query for endangered species
            client = bigquery.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
            query = """
                WITH endangered_species_lookup AS (
                  SELECT CONCAT(genus_name, ' ', species_name) as full_name,
                         conservation_status,
                         genus_name,
                         family_name,
                         scientific_name
                  FROM `{project_id}.biodiversity.endangered_species`
                ),
                matching_occurrences AS (
                  SELECT o.species, o.decimallatitude, o.decimallongitude
                  FROM `{project_id}.biodiversity.cached_occurrences` o
                  INNER JOIN endangered_species_lookup e
                  ON o.species = e.full_name
                  WHERE o.decimallongitude IS NOT NULL
                    AND o.decimallatitude IS NOT NULL
                ),
                protected_area AS (
                  SELECT geometry
                  FROM `{project_id}.biodiversity.protected_areas_africa`
                  WHERE name = @protected_area_name
                  LIMIT 1
                )
                SELECT
                  o.species,
                  ANY_VALUE(e.genus_name) as genus_name,
                  ANY_VALUE(e.family_name) as family_name,
                  STRING_AGG(DISTINCT CONCAT(e.scientific_name, ' (', e.conservation_status, ')'), ';\\r\\n') as scientific_names_with_status,
                  COUNT(DISTINCT FORMAT("%f,%f", o.decimallatitude, o.decimallongitude)) as observation_count
                FROM matching_occurrences o
                CROSS JOIN protected_area p
                JOIN endangered_species_lookup e
                  ON o.species = e.full_name
                WHERE ST_CONTAINS(
                  p.geometry,
                  ST_GEOGPOINT(o.decimallongitude, o.decimallatitude)
                )
                GROUP BY o.species
                ORDER BY observation_count DESC, o.species ASC
            """

            query = self.build_query(query)
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "protected_area_name", "STRING", best_match
                    )
                ]
            )

            return [
                {
                    "species": row.species,
                    "observation_count": row.observation_count,
                    "genus_name": row.genus_name,
                    "family_name": row.family_name,
                    "scientific_names_with_status": row.scientific_names_with_status,
                }
                for row in client.query(query, job_config=job_config)
            ]

        except (KeyError, ValueError) as e:
            self.logger.error("Invalid input: %s", str(e))
            raise
        except google.api_core.exceptions.GoogleAPIError as e:
            self.logger.error("BigQuery error: %s", str(e), exc_info=True)
            raise
        except Exception as e:
            self.logger.error("Error processing request: %s", str(e), exc_info=True)
            raise

    def _find_matching_protected_area(self, protected_area_name: str) -> str:
        """Find best matching protected area name using fuzzy matching."""
        normalized_input = self.normalize_protected_area_name(protected_area_name)
        client = bigquery.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
        areas_query = self.build_query(
            "SELECT DISTINCT name FROM `{project_id}.biodiversity.protected_areas_africa` "
            "WHERE name IS NOT NULL"
        )

        best_match = None
        highest_ratio = 0

        for row in client.query(areas_query):
            ratio = fuzz.ratio(
                normalized_input, self.normalize_protected_area_name(row.name)
            )
            if ratio > highest_ratio and ratio > 80:
                highest_ratio = ratio
                best_match = row.name

        return best_match

    def get_species_occurrences_in_protected_area(self, content: dict) -> Dict:
        """Get occurrence data for a specific species in a protected area."""
        try:
            protected_area_name = content.get("protected_area_name")
            species_name = content.get("species_name")

            if not protected_area_name or not species_name:
                raise ValueError(
                    "Both protected area name and species name are required"
                )

            best_match = self._find_matching_protected_area(protected_area_name)
            if not best_match:
                self.logger.warning(
                    "No matching protected area found for: %s", protected_area_name
                )
                return []

            translated = json.loads(
                self.translate_to_scientific_name_from_api({"name": species_name})
            )
            if "error" in translated or not translated.get("scientific_name"):
                self.logger.warning(
                    "Could not translate species name: %s", species_name
                )
                return []

            query = f"""
                WITH protected_area AS (
                  SELECT geometry
                  FROM `{self.project_id}.biodiversity.protected_areas_africa`
                  WHERE LOWER(name) = LOWER(@protected_area_name)
                  LIMIT 1
                )
                SELECT
                  o.species,
                  o.decimallatitude,
                  o.decimallongitude,
                  COUNT(*) OVER() as total_occurrences
                FROM `{self.project_id}.biodiversity.occurances_endangered_species_mammals` o
                CROSS JOIN protected_area p
                WHERE LOWER(o.species) = LOWER(@species_name)
                  AND ST_CONTAINS(
                    p.geometry,
                    ST_GEOGPOINT(o.decimallongitude, o.decimallatitude)
                  )
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "protected_area_name", "STRING", best_match
                    ),
                    bigquery.ScalarQueryParameter(
                        "species_name", "STRING", translated["scientific_name"]
                    ),
                ]
            )

            results = []
            total_occurrences = 0
            for row in self.client.query(query, job_config=job_config):
                results.append(
                    {
                        "species": row.species,
                        "decimallatitude": row.decimallatitude,
                        "decimallongitude": row.decimallongitude,
                    }
                )
                total_occurrences = row.total_occurrences

            return {
                "protected_area": best_match,
                "species": translated["scientific_name"],
                "occurrence_count": total_occurrences,
                "occurrences": results,
            }

        except (KeyError, ValueError) as e:
            self.logger.error("Invalid input: %s", str(e))
            raise
        except Exception as e:
            self.logger.error("BigQuery error: %s", str(e))
            raise

    def get_endangered_species_by_country(self, content):
        """
        Get endangered species data for a specific country.

        Args:
            content (dict): Dictionary containing:
                - country_code (str): Two-letter country code

        Returns:
            list: List of endangered species data
        """
        try:
            country_code = content["country_code"]
            client = bigquery.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))

            query = """
            SELECT
                decimallongitude,
                decimallatitude,
                conservation_status,
                o.species,
                sp.species_name_en
            FROM
                `{project_id}.biodiversity.occurances_endangered_species_mammals` o
            INNER JOIN
                `{project_id}.biodiversity.countries` c
                ON ST_CONTAINS(c.geometry, ST_GEOGPOINT(o.decimallongitude, o.decimallatitude))
            INNER JOIN
                `{project_id}.biodiversity.endangered_species` sp
                ON CONCAT(sp.genus_name, ' ', sp.species_name) = o.species
            WHERE
                eventdate IS NOT NULL
                AND eventdate > '1980-01-01'
                AND countrycode = @country_code
            """.format(project_id=os.getenv("GOOGLE_CLOUD_PROJECT"))

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "country_code", "STRING", country_code
                    )
                ]
            )

            query_job = client.query(query, job_config=job_config)
            results = []
            total_occurrences = 0

            # Execute query
            for row in query_job:
                results.append({
                    "species": row.species,
                    "species_name_en": row.species_name_en,
                    "decimallatitude": row.decimallatitude,
                    "decimallongitude": row.decimallongitude,
                    "conservation_status": row.conservation_status
                })
                total_occurrences += 1

            return {
                "country_code": country_code,
                "total_occurrences": total_occurrences,
                "occurrences": results
            }

        except Exception as e:
            self.logger.error(
                "Error getting endangered species by country: %s",
                str(e),
                exc_info=True
            )
            raise

    def get_protected_areas_geojson(self, content: dict) -> str:
        """Get GeoJSON data for protected areas in a country.

        Args:
            content (dict): Dictionary containing three letter country_code (str)

        Returns:
            str: JSON string containing protected areas GeoJSON data

        Raises:
            ValueError: If country_code is invalid
            google.api_core.exceptions.GoogleAPIError: If BigQuery query fails
        """
        try:
            country_code = content["country_code"]
            client = bigquery.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
            query = self.build_geojson_query()
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "country_code", "STRING", country_code
                    )
                ]
            )
            query_job = client.query(query, job_config=job_config)
            results = [
                {
                    "name": row.name,
                    "category": row.IUCN_CAT,
                    "geojson": json.loads(row.geojson),
                }
                for row in query_job
            ]
            return json.dumps(results)
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            self.logger.error("Invalid input or GeoJSON: %s", str(e))
            raise
        except Exception as e:
            self.logger.error("Error: %s", str(e))
            raise

    def build_geojson_query(self) -> str:
        """Build the BigQuery query string for geojson query."""
        query = """
            SELECT name, IUCN_CAT, ST_ASGEOJSON(geometry) as geojson
            FROM `{project_id}.biodiversity.protected_areas_africa`
            WHERE ISO3 = @country_code
            AND IUCN_CAT IS NOT NULL
            AND IUCN_CAT IN ('Ia', 'Ib', 'II', 'III', 'IV', 'V', 'VI')
        """
        return self.build_query(query, where_clause="")

