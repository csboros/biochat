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
            "show where species is found, where it lives."
        ),
        parameters={
            "type": "object",
            "properties": {
                "species_name": {
                    "type": "string",
                    "description": (
                        "name of the species to get occurences for, "
                        "if it is a common name, use the scientific name"
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
            "required": ["species_name"]
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
                    "description": "name of the order to get endangered families for",
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
]
