"""Handler for species-related operations."""

from typing import Dict
import json
import os
from pygbif import species
from google.cloud import bigquery
import google.api_core.exceptions
from .base_handler import BaseHandler

try:
    from fuzzywuzzy import fuzz
except ImportError:
    fuzz = None


class SpeciesHandler(BaseHandler):
    """Handles species-related operations including name translation and occurrence data."""


    def get_species_info_from_api(self, content: Dict) -> str:
        """Retrieves species information from GBIF API."""
        try:
            species_name = content["name"]
            self.logger.info("Fetching species info for: %s", species_name)
            species_info = species.name_backbone(species_name)
            return json.dumps(species_info)
        except KeyError as exc:
            raise ValueError("Species name is required") from exc
        except Exception as e:
            self.logger.error("GBIF API error: %s", str(e))
            raise

    def get_yearly_occurrences(self, content: Dict) -> Dict:
        """Get yearly occurrence counts for a species."""
        if not content.get("species_name"):
            return {"error": "Species name is required"}

        try:
            translated = json.loads(
                self.translate_to_scientific_name_from_api(
                    {"name": content["species_name"]}
                )
            )

            if "error" in translated:
                return {"error": f"Invalid species name: {content['species_name']}"}

            query = f"""
                SELECT 
                    EXTRACT(YEAR FROM eventdate) as year,
                    COUNT(*) as count
                FROM `{self.project_id}.biodiversity.occurances_endangered_species_mammals`
                WHERE LOWER(species) = LOWER(@species_name)
                AND eventdate IS NOT NULL 
                AND eventdate > '1980-01-01'
                GROUP BY year
                ORDER BY year
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "species_name", "STRING", translated["scientific_name"]
                    )
                ]
            )

            results = [
                {"year": row.year, "count": row.count}
                for row in self.client.query(query, job_config=job_config)
            ]

            return {
                "common_name": content["species_name"],
                "scientific_name": translated["scientific_name"],
                "yearly_data": results,
                "type": "temporal",
            }

        except Exception as e:
            self.logger.error("Error getting yearly occurrences: %s", str(e))
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
