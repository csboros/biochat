"""
Tool for species-related functionality.
"""

from typing import Dict, Any, List
from vertexai.generative_models import FunctionDeclaration
from ..tool import Tool
from .handlers.species_handler import SpeciesHandler
from .handlers.endangered_species_handler import EndangeredSpeciesHandler

class SpeciesTool(Tool):
    """Tool for species-related functionality."""

    def __init__(self):
        """Initialize the species tool."""
        self.species_handler = SpeciesHandler()
        self.endangered_handler = EndangeredSpeciesHandler()

    def get_handlers(self) -> Dict[str, Any]:
        """Returns the handlers for this tool."""
        return {
            "species": self.species_handler,
            "endangered": self.endangered_handler
        }

    def get_function_declarations(self) -> List[FunctionDeclaration]:
        """Returns the function declarations for species-related functions."""
        return [
            FunctionDeclaration(
                name="translate_to_scientific_name",
                description=(
                    "IMPORTANT: This is step 1 for any species query using common names. "
                    "Translates a common/English name to scientific name, "
                    "which must then be used with get_species_info. "
                    "Examples: 'Tell me about Bengal Tiger' requires: "
                    "1. First call this to convert 'Bengal Tiger' → 'Panthera tigris tigris' "
                    "2. Then call get_species_info with the result"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                "common/English name of the species that needs "
                                "translation to scientific name"
                            ),
                        },
                    },
                    "required": ["name"],
                },
            ),
            FunctionDeclaration(
                name="translate_to_common_name",
                description=(
                    "IMPORTANT: Use this function to translate scientific names to "
                    "common/English names. This is useful when you have a scientific name "
                    "and need to find its common name. Examples:\n"
                    "- 'What is the common name for Panthera tigris?' → 'Tiger'\n"
                    "- 'Translate Leporidae to common name' → 'Hares and Rabbits' \n"
                    "- 'What do we call Ursus arctos in English?' → "
                    "'Brown Bear' \n"
                    "For ANY question asking to translate or explain what a scientific name "
                    "means in common terms, use this function first."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "scientific name of the species that needs "
                            "translation to common/English name",
                        },
                    },
                    "required": ["name"],
                },
            ),
            FunctionDeclaration(
                name="get_species_info",
                description=(
                    "Use this function to get detailed information about a species when "
                    "users ask questions like 'Tell me about Bengal Tiger', "
                    "'What can you tell me about Lions?', or similar queries. "
                    "IMPORTANT: This is a two-step process: "
                    "1. First translate_to_scientific_name(common_name) → scientific_name "
                    "2. Then call this function with the scientific_name "
                    "Example: "
                    "For 'Tell me about Bengal Tiger': "
                    "1. translate_to_scientific_name('Bengal Tiger') → "
                    "'Panthera tigris tigris' "
                    "2. Then call this function with 'Panthera tigris tigris'"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                "SCIENTIFIC NAME of the species (obtained from "
                                "translate_to_scientific_name). For queries like "
                                "'Tell me about Bengal Tiger', you must: "
                                "1. First get scientific name from translate_to_scientific_name "
                                "2. Then use that result here"
                            ),
                        }
                    },
                    "required": ["name"],
                },
            ),
            FunctionDeclaration(
                name="get_species_images",
                description="Get images of a species",
                parameters={
                    "type": "object",
                    "properties": {"species_name": {"type": "string"}},
                },
            ),
            FunctionDeclaration(
                name="get_endangered_species_in_protected_area",
                description=(
                    "ALWAYS USE THIS FUNCTION for any questions about species "
                    "in protected areas/parks/reserves. "
                    "Examples that should use this function:\n"
                    "- 'What endangered species live in Serengeti?'\n"
                    "- 'What animals are in Yellowstone?'\n"
                    "- 'Show me species in Kruger National Park'\n"
                    "- 'Tell me about wildlife in Masai Mara'\n"
                    "The function will return a list of endangered species found in the "
                    "specified area."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "protected_area_name": {
                            "type": "string",
                            "description": "name of the protected area "
                            "(e.g., 'Serengeti', 'Yellowstone', 'Kruger')",
                        }
                    },
                    "required": ["protected_area_name"],
                },
            ),
            FunctionDeclaration(
                name="get_species_occurrences_in_protected_area",
                description=(
                    "Get occurrence data (coordinates) for a specific species within a "
                    "protected area. Shows where a particular species is found within "
                    "the protected area. "
                    "Use this for questions like: 'Where are elephants in Kruger?', "
                    "'Show me lions in Serengeti', 'Find tigers in Ranthambore'"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "protected_area_name": {
                            "type": "string",
                            "description": "name of the protected area to search within, "
                            "please use the name as provided, do not append anything to it",
                        },
                        "species_name": {
                            "type": "string",
                            "description": "name of the species to find occurrences for "
                            "(common name or scientific name)",
                        },
                    },
                    "required": ["protected_area_name", "species_name"],
                },
            ),
            FunctionDeclaration(
                name="get_endangered_species_by_country",
                description=(
                    "Show endangered species locations in a country."
                    "⚠️ MUST USE THIS FUNCTION for ANY of these cases:\n"
                    "1. Questions using words: 'locations', 'where', 'show on map', 'display'\n"
                    "2. Questions about WHERE endangered species are found\n"
                    "3. Requests for MAP visualization of endangered species\n\n"
                    "Example queries that MUST use this function:\n"
                    "- 'Show endangered species locations in Kenya' ✓\n"
                    "- 'Where are endangered animals in Tanzania?' ✓\n"
                    "- 'Display locations of endangered species in Rwanda' ✓\n"
                    "- 'Show where endangered species are found in Uganda' ✓\n\n"
                    "This function returns COORDINATES for MAP DISPLAY.\n"
                    "DO NOT use endangered_species_for_country for these types of queries!"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "country_code": {
                            "type": "string",
                            "description": (
                                "TWO LETTER country code ONLY (e.g., 'KE' for Kenya, "
                                "'TZ' for Tanzania). DO NOT use three letter codes."
                            ),
                            "pattern": "^[A-Z]{2}$",
                        }
                    },
                    "required": ["country_code"],
                },
            ),
            FunctionDeclaration(
                name="endangered_species_for_family",
                description=(
                    "Get list of endangered species for a given family name, "
                    "with link to IUCN"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "family_name": {
                            "type": "string",
                            "description": "name of the family to get endangered species for (e.g., Felidae, Canidae, Ursidae)",
                        },
                        "conservation_status": {
                            "type": "string",
                            "description": (
                                "optional conservation status to filter by, possible values are: "
                                "Least Concern, Endangered, Near Threatened, Vulnerable, "
                                "Data Deficient, Critically Endangered, Extinct"
                            ),
                            "enum": [
                                "Least Concern",
                                "Endangered",
                                "Near Threatened",
                                "Vulnerable",
                                "Data Deficient",
                                "Critically Endangered",
                                "Extinct"
                            ]
                        },
                        "chart_type": {
                            "type": "string",
                            "description": (
                                "type of chart to display, supported values: "
                                "'force_directed_graph', 'tree_chart'"
                            ),
                            "enum": ["force_directed_graph", "tree_chart"]
                        },
                    },
                    "required": ["family_name", "chart_type"]
                }
            ),
            FunctionDeclaration(
                name="endangered_classes_for_kingdom",
                description=(
                    "Get list of endangered classes for a given kingdom name "
                    "with count of endangered species in each class"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "kingdom_name": {
                            "type": "string",
                            "description": (
                                "name of the kingdom (e.g., Animalia, only Animalia is available in the dataset) "
                                "to get endangered classes for"
                            ),
                        }
                    },
                }
            ),
            FunctionDeclaration(
                name="endangered_families_for_order",
                description="Get list of endangered families for a given order name",
                parameters={
                    "type": "object",
                    "properties": {
                        "order_name": {
                            "type": "string",
                            "description": (
                                "name of the order to get endangered families for "
                                "(e.g., Primates, Carnivora, etc.)"
                            )
                        },
                        "chart_type": {
                            "type": "string",
                            "description": ("type of chart to display, supported values: "
                                "'force_directed_graph', 'tree_chart'"),
                            "enum": ["force_directed_graph", "tree_chart"]
                        },
                    },
                    "required": ["order_name", "chart_type"]
                }
            ),
            FunctionDeclaration(
                name="endangered_orders_for_class",
                description=(
                    "Retrieves endangered orders within a specified class and "
                    "their species counts. "
                    "Use this for questions about:\n"
                    "- List of orders with endangered species in a class\n"
                    "- Orders containing endangered species for a class\n"
                    "- Endangered species distribution across orders\n"
                    "- Which orders have endangered species\n"
                    "Examples:\n"
                    "- 'List all orders with endangered species for class Mammalia'\n"
                    "- 'Show me endangered orders in Aves'\n"
                    "- 'What orders have endangered species in Reptilia?'\n"
                    "- 'List orders with endangered species in class Mammalia'"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "class_name": {
                            "type": "string",
                            "description": "Name of the class to query (e.g., 'Mammalia', 'Aves')"
                        }
                    },
                    "required": ["class_name"]
                }
            ),
            FunctionDeclaration(
                name="endangered_species_for_country",
                description=(
                    "Get endangered species information for  a country. "
                    "⚠️ Use this function ONLY for getting a simple LIST of species "
                    "(NO locations/maps).\n"
                    "DO NOT use this function if the question asks about:\n"
                    "- Locations of species\n"
                    "- Where species are found\n"
                    "- Showing/displaying on a map\n\n"
                    "Example queries for this function:\n"
                    "- 'List endangered species in Kenya'\n"
                    "- 'What endangered animals live in Tanzania?'\n"
                    "- 'Tell me the endangered species in Uganda'\n\n"
                    "- 'Show endangered species locations...' "
                    "(use get_endangered_species_by_country)\n"
                    "- 'Where are endangered species...' "
                    "(use get_endangered_species_by_country)\n"
                    "- 'Display locations of...' "
                    "(use get_endangered_species_by_country)"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "country_code": {
                            "type": "string",
                            "description": (
                                "TWO LETTER country code ONLY (e.g., 'KE' for Kenya, "
                                "'TZ' for Tanzania). DO NOT use three letter codes. "
                                "Examples: Kenya → 'KE' (not 'KEN'), "
                                "Tanzania → 'TZ' (not 'TZA'), "
                                "Uganda → 'UG' (not 'UGA'). "
                                "If unsure, use google_search to find the correct 2-letter code."
                            ),
                            "pattern": "^[A-Z]{2}$",
                        },
                        "conservation_status": {
                            "type": "string",
                            "description": (
                                "Filter by conservation status "
                                "(e.g., 'Critically Endangered')"
                            ),
                            "enum": [
                                "Least Concern",
                                "Endangered",
                                "Near Threatened",
                                "Vulnerable",
                                "Data Deficient",
                                "Critically Endangered",
                                "Extinct"
                            ]
                        },
                        "chart_type": {
                            "type": "string",
                            "description": (
                                "type of chart to display, supported values: "
                                "'force_directed_graph', 'tree_chart'"
                            ),
                            "enum": ["force_directed_graph", "tree_chart"]
                        },
                    },
                    "required": ["country_code", "chart_type"]
                }
            ),
            FunctionDeclaration(
                name="endangered_species_for_countries",
                description=(
                    "Get endangered species information for MULTIPLE countries at once. "
                    "Use this function for comparing species across countries. "
                    "Examples: 'Compare endangered species between Kenya and Tanzania', "
                    "'Show endangered species in both Uganda and Rwanda'. "
                    "For single country queries, use endangered_species_for_country instead."
                    "Supported chart types: 'force_directed_graph', 'tree_chart'"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "country_codes": {
                            "type": "array",
                            "description": (
                                "List of country codes to get data for "
                                "(e.g., ['DE', 'FR', 'IT'])"
                            ),
                            "items": {
                                "type": "string"
                            }
                        },
                        "conservation_status": {
                            "type": "string",
                            "description": (
                                "Filter by conservation status "
                                "(e.g., 'Critically Endangered')"
                            ),
                            "enum": [
                                "Least Concern",
                                "Endangered",
                                "Near Threatened",
                                "Vulnerable",
                                "Data Deficient",
                                "Critically Endangered",
                                "Extinct"
                            ]
                        },
                        "chart_type": {
                            "type": "string",
                            "description": (
                                "type of chart to display, supported values: "
                                "'force_directed_graph', 'tree_chart'"
                            ),
                             "enum": ["force_directed_graph", "tree_chart"]
                        },
                    },
                    "required": ["country_codes", "chart_type"]
                }
            ),
            FunctionDeclaration(
                name="endangered_species_by_conservation_status",
                description=(
                    "Get number of endangered species for different conservational "
                    "status for a given country code or the whole world if country "
                    "code is not specified"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "country_code": {
                            "type": "string",
                            "description": "two letter country code to get endangered species for",
                        },
                        "conservation_status": {
                            "type": "string",
                            "description": (
                                "optional conservation status to filter by, possible values are: "
                                "Least Concern, Endangered, Near Threatened, Vulnerable, "
                                "Data Deficient, Critically Endangered, Extinct,"
                            ),
                            "enum": [
                                "Least Concern",
                                "Endangered",
                                "Near Threatened",
                                "Vulnerable",
                                "Data Deficient",
                                "Critically Endangered",
                                "Extinct"
                            ]
                        },

                    },
                    "required": ["conservation_status"]
                }
            ),
            FunctionDeclaration(
                name="endangered_species_hci_correlation",
                description=(
                    "Tool to analyze correlation between HCI and endangered species. "
                    "Required for questions about: correlation, relationship, or connection "
                    "between HCI (Human Coexistence Index) and endangered species counts. "
                    "Input: None required - defaults to Africa and Critically Endangered species. "
                    "Output: Scatter plot showing correlation."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "continent": {"type": "string", "default": "Africa", "description": "Only Africa is available in the dataset"}
                    }
                }
            ),
            FunctionDeclaration(
                name="get_occurrences",
                description=(
                    "This function is for MAP VISUALIZATION of specific species "
                    "distribution. It returns coordinates and data needed to display "
                    "species locations on a map. ⚠️ IMPORTANT: DO NOT use this for text-based answers "
                    "about where species live OR for analyzing Human Coexistence Index (HCI) data. "
                    "Use this function ONLY when you need to SHOW "
                    "the distribution of a specific species on a map.\n\n"
                    "Examples that should use this function:\n"
                    "- 'Where can I find Orangutans in the wild?'\n"
                    "- 'Show me a map of where lions live'\n"
                    "- 'Display the distribution of pandas on a map'\n"
                    "- 'Map where tigers are found'\n"
                    "- 'Show the geographic range of elephants'\n\n"
                    "IMPORTANT: For common names, first use translate_to_scientific_name, "
                    "then use the result here. "
                    "Example workflow for 'Show habitat of Mountain Gorillas': "
                    "1. First translate_to_scientific_name('Mountain Gorilla') → "
                    "scientific name "
                    "2. Then call this function with the result to show the map."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "species_name": {
                            "type": "string",
                            "description": (
                                "⚠️ Name of the species to find "
                                "(MUST use species_name as parameter)"
                            ),
                        },
                        "country_code": {
                            "type": "string",
                            "description": "Optional: 2 letter country code to get occurrences for",
                        },
                        "chart_type": {
                            "type": "string",
                            "description": (
                                "type of chart to display. For species distribution "
                                "visualization, use 'HEXAGON_MAP' (default) or 'HEATMAP'. "
                            ),
                            "enum": ["HEATMAP", "HEXAGON_MAP"]
                        },
                    },
                    "required": ["species_name", "chart_type"],
                },
            ),
            FunctionDeclaration(
                name="get_yearly_occurrences",
                description=(
                    "Get yearly occurrence counts for a species to show how sightings have "
                    "changed over time. "
                    "Use this for questions about temporal trends, historical sightings, or "
                    "changes in species observations in specific countries or regions. "
                    "Examples: 'How have Lion sightings changed over time?', "
                    "'Show elephant sightings in Kenya', 'What is the trend of rhino observations "
                    "in Tanzania?', 'Compare gorilla populations between Uganda and Rwanda', "
                    "'Show yearly data for tiger sightings in India', "
                    "'How have elephant numbers changed in Kenya over the years?'"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "species_name": {
                            "type": "string",
                            "description": (
                                "Name of the species to analyze (e.g., 'Lion', 'Elephant'). "
                                "Common names will be automatically translated to scientific names."
                            ),
                        },
                        "country_codes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "List of 2 or 3-letter country codes to filter observations. "
                                "For example: ['KE'] for Kenya, ['TZ'] for Tanzania, ['UG', 'RW'] "
                                "for Uganda and Rwanda. "
                                "Leave empty to get global data."
                            ),
                        },
                    },
                    "required": ["species_name"],
                },
            ),
            FunctionDeclaration(
                name="get_protected_areas_geojson",
                description=(
                    "Get protected areas (parks, reserves, sanctuaries) for a country. "
                    "IMPORTANT: Use this function for ANY questions about:\n"
                    "- Showing protected areas in a country\n"
                    "- Listing national parks in a country\n"
                    "- Displaying wildlife reserves\n"
                    "- Mapping conservation areas\n\n"
                    "Examples:\n"
                    "- 'Show protected areas in Kenya'\n"
                    "- 'Display national parks in Tanzania'\n"
                    "- 'Map wildlife reserves in Uganda'\n"
                    "- 'What protected areas are in Rwanda?'\n"
                    "- 'Show me the parks in Kenya'\n"
                    "\n"
                    "IMPORTANT: Must use 3-letter ISO country codes (e.g., KEN, TZA, UGA, USA, GBR)"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "country_code": {
                            "type": "string",
                            "description": (
                                "THREE LETTER country code ONLY (e.g., 'KEN' for Kenya, "
                                "'TZA' for Tanzania). "
                                "DO NOT use two letter codes. Examples: "
                                "Kenya → 'KEN' (not 'KE'), "
                                "Tanzania → 'TZA' (not 'TZ'), "
                                "Uganda → 'UGA' (not 'UG'). "
                            ),
                            "pattern": "^[A-Z]{3}$",  # Enforce exactly 3 uppercase letters
                        }
                    },
                    "required": ["country_code"],
                },
            ),
        ]

    def get_function_mappings(self) -> Dict[str, Any]:
        """Returns the mapping of function names to their implementations."""
        return {
            "translate_to_scientific_name":
                self.species_handler.translate_to_scientific_name_from_api,
            "translate_to_common_name":
                self.species_handler.translate_to_common_name_from_api,
            "get_species_info": self.species_handler.get_species_info_from_api,
            "get_species_images": self.species_handler.get_species_images,
            "get_endangered_species_in_protected_area":
                self.species_handler.get_endangered_species_in_protected_area,
            "get_species_occurrences_in_protected_area":
                self.species_handler.get_species_occurrences_in_protected_area,
            "get_endangered_species_by_country":
                self.species_handler.get_endangered_species_by_country,
            "endangered_species_for_family":
                self.endangered_handler.endangered_species_for_family,
            "endangered_classes_for_kingdom":
                self.endangered_handler.endangered_classes_for_kingdom,
            "endangered_families_for_order":
                self.endangered_handler.endangered_families_for_order,
            "endangered_orders_for_class":
                self.endangered_handler.endangered_orders_for_class,
            "endangered_species_for_country":
                self.endangered_handler.endangered_species_for_country,
            "endangered_species_for_countries":
                self.endangered_handler.endangered_species_for_countries,
            "endangered_species_by_conservation_status":
                self.endangered_handler.endangered_species_by_conservation_status,
            "endangered_species_hci_correlation":
                self.endangered_handler.endangered_species_hci_correlation,
            "get_occurrences": self.endangered_handler.get_occurrences,
            "get_yearly_occurrences": self.endangered_handler.get_yearly_occurrences,
            "get_protected_areas_geojson": self.species_handler.get_protected_areas_geojson,
        }
