"""
Function declarations for the Biodiversity API.
Contains all available function definitions that can be called by the AI model,
including species information, geographic data, and endangered species queries.
"""

from vertexai.generative_models import FunctionDeclaration

FUNCTION_DECLARATIONS = [
    FunctionDeclaration(
        name="translate_to_scientific_name",
        description="Translate a given species name to a scientific name",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "english name of the species to be translated into scientific name"
                    ),
                },
            },
            "required": ["name"]
        },
    ),
   FunctionDeclaration(
        name="get_species_info",
        description="Get info about a given species, get taxonomy of a given species",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "get taxonomy of a given species",
                }
            },
            "required": ["name"]
        },
    ),
    FunctionDeclaration(
        name="get_occurences",
        description=(
            "Get occurences, distribution for a given species, "
            "show where species is found, where species live. Answer to questions like: "
            "Where do species live? Where are species found?"
        ),
        parameters={
            "type": "object",
            "properties": {
                "species_name": {
                    "type": "string",
                    "description": (
                        "name of the species to get occurences, distribution for, "
                        "if it is a common name, use the scientific name. "
                        "If you do not know the scientific name, translate it to scientific name "
                        "using the translate_to_scientific_name function."
                    ),
                },
                "country_code": {
                    "type": "string",
                    "description": "2 letter country code to get occurences for",
                },
                "chart_type": {
                    "type": "string",
                    "description": "type of chart to display",
                },
            },
#            "required": ["species_name"]
        },
    ),
    FunctionDeclaration(
        name="get_country_geojson",
        description="Get GeoJSON data for a specific country",
        parameters={
            "type": "object",
            "properties": {
                "country_name": {
                    "type": "string",
                    "description": "name of the country to get GeoJSON data for",
                }
            },
            "required": ["country_name"]
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
                    "description": "name of the family to get endangered species for",
                },
                "conservation_status": {
                    "type": "string",
                    "description": (
                        "optional conservation status to filter by, possible values are: "
                        "Least Concern, Endangered, Near Threatened, Vulnerable, "
                        "Data Deficient, Critically Endangered, Extinct,"
                    ),
                },
            },
        },
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
                        "name of the kingdom (e.g., Animalia, Plantae) "
                        "to get endangered classes for"
                    ),
                }
            },
        },
    ),
    FunctionDeclaration(
        name="endangered_families_for_order",
        description="Get list of endangered families for a given order name",
        parameters={
            "type": "object",
            "properties": {
                "order_name": {
                    "type": "string",
                    "description": "name of the order to get endangered families for "
                    "(e.g., Primates, Carnivora, etc.)",
                }
            },
        },
    ),
    FunctionDeclaration(
        name="endangered_orders_for_class",
        description="Get list of endangered orders for a given class name",
        parameters={
            "type": "object",
            "properties": {
                "class_name": {
                    "type": "string",
                    "description": "name of the class to get endangered orders for",
                }
            },
        },
    ),
    FunctionDeclaration(
        name="google_search",
        description=(
            "Search Google for the given query and return relevant results, "
            "try this as last resort before giving up please."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "query to search for",
                }
            },
        },
    ),
    FunctionDeclaration(
        name="endangered_species_for_country",
        description="Get list of endangered species for a given country code",
        parameters={
            "type": "object",
            "properties": {
                "country_code": {
                    "type": "string",
                    "description": "country code to get endangered species for",
                },
                "conservation_status": {
                    "type": "string",
                    "description": (
                        "optional conservation status to filter by, possible values are: "
                        "Least Concern, Endangered, Near Threatened, Vulnerable, Data Deficient, "
                         "Critically Endangered, Extinct,"
                    ),
                },
            },
        },
    ),
    FunctionDeclaration(
        name="number_of_endangered_species_by_conservation_status",
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
                    "description": "country code to get endangered species for",
                },
                "conservation_status": {
                    "type": "string",
                    "description": (
                        "optional conservation status to filter by, possible values are: "
                        "Least Concern, Endangered, Near Threatened, Vulnerable, "
                        "Data Deficient, Critically Endangered, Extinct,"
                    ),
                },
            },
        },
    ),
    FunctionDeclaration(
        name="get_protected_areas_geojson",
        description=(
            "Get GeoJSON data for protected areas in a country "
            "with three letter country code"
        ),
        parameters={
            "type": "object",
            "properties": {
                "country_code": {
                    "type": "string",
                    "description": "three letter country code to get protected areas for, "
                    "(KEN for Kenya, USA for United States of America, etc.) "
                    "if you do not know the three letter country code, "
                    "use google_search function to find it",
                }
            },
        },
    ),
    FunctionDeclaration(
        name="get_endangered_species_in_protected_area",
        description=(
            "Get list of endangered species in a protected area, please use the name "
            "the user provided, do not append anything to it, like National Park, "
            "or Protected Area, etc."
        ),
        parameters={
            "type": "object",
            "properties": {
                "protected_area_name": {
                    "type": "string",
                    "description": "name of the protected area to get endangered species for, "
                    "please use the name the user provided, do not append anything to it",
                }
            },
        },
    ),
    FunctionDeclaration(
        name="get_species_occurrences_in_protected_area",
        description=(
            "Get occurrence data (coordinates) for a specific species within a protected area. "
            "Shows where a particular species is found within the protected area."
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
                    "description": "scientific name of the species to find occurrences for",
                }
            },
            "required": ["protected_area_name", "species_name"]
        },
    ),
]
