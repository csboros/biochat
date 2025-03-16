"""
Function declarations for the Biodiversity API.
Contains all available function definitions that can be called by the AI model,
including species information, geographic data, and endangered species queries.
"""

from vertexai.generative_models import FunctionDeclaration

FUNCTION_DECLARATIONS = [
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
                    "description": "common/English name of the species that needs "
                    "translation to scientific name",
                },
            },
            "required": ["name"],
        },
    ),
    FunctionDeclaration(
        name="translate_to_common_name",
        description=(
            "IMPORTANT: Use this function to translate scientific names to common/English names. "
            "This is useful when you have a scientific name and need to find its common name. "
            "Examples: \n"
            "- 'What is the common name for Panthera tigris?' → 'Tiger' \n"
            "- 'Translate Leporidae to common name' → 'Hares and Rabbits' \n"
            "- 'What do we call Ursus arctos in English?' → 'Brown Bear' \n"
            "For ANY question asking to translate or explain what a scientific name means in common terms, "
            "use this function first."
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
            "IMPORTANT: This is step 2 - must be called after translate_to_scientific_name "
            "for common names. "
            "Gets conservation status, habitat, and other details using scientific name. "
            "Example workflow for 'Tell me about Bengal Tiger': "
            "1. First translate_to_scientific_name('Bengal Tiger') → 'Panthera tigris tigris' "
            "2. Then call this function with 'Panthera tigris tigris'"
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "SCIENTIFIC NAME of the species (obtained from "
                        "translate_to_scientific_name). "
                        "For queries like 'Tell me about Bengal Tiger', you must: "
                        "1. First get scientific name from translate_to_scientific_name "
                        "2. Then use that result here"
                    ),
                }
            },
            "required": ["name"],
        },
    ),
    FunctionDeclaration(
        name="get_occurences",
        description=(
            "Get geographic distribution and occurrence data for a species GLOBALLY "
            "or in a specific country. "
            "IMPORTANT: For common names, first use translate_to_scientific_name, "
            "then use the result here. "
            "Example workflow for 'Show habitat of Mountain Gorillas': "
            "1. First translate_to_scientific_name('Mountain Gorilla') → scientific name "
            "2. Then call this function with the result. "
            "\n\n"
            "Use this function for ANY questions about: "
            "- where species live or can be found "
            "- which countries have specific species "
            "- presence/absence of species in locations "
            "- questions like 'Are there X in Y?' "
            "\n\n"
            "Examples: "
            "'Where do lions live?', 'Where can I find pandas?', "
            "'In which countries can I find Giant Pandas?', "
            "'Which countries have tigers?', "
            "'Are there any lions in Africa?', "
            "'Show me where tigers are found globally', "
            "'Where are orangutans in the wild?', "
            "'Can I find elephants in Kenya?'"
        ),
        parameters={
            "type": "object",
            "properties": {
                "species_name": {
                    "type": "string",
                    "description": "name of the species (e.g., 'Lion', 'Panda')",
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
            "required": ["species_name"],
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
            "required": ["country_name"],
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
               "chart_type": {
                    "type": "string",
                    "description": "type of chart to display, supported values: "
                    "'force_directed_graph', 'tree'",
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
                    "description": (
                        "name of the order to get endangered families for "
                        "(e.g., Primates, Carnivora, etc.)"
                    )
                },
               "chart_type": {
                    "type": "string",
                    "description": "type of chart to display, supported values: "
                    "'force_directed_graph', 'tree'",
                },
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
        description=(
            "Get list of endangered species in a SINGLE country using "
            "TWO LETTER country code ONLY. "
            "For comparing multiple countries, use endangered_species_for_countries instead. "
            "Examples: 'Show endangered species in Kenya' → use 'KE', "
            "'List endangered animals in Tanzania' → use 'TZ'. "
            "IMPORTANT: Must use 2-letter ISO country codes (e.g., KE, TZ, UG, US, GB), "
            "3-letter codes will not work! "
            "Supported chart types: 'tree', 'force_directed_graph'"
        ),
        parameters={
            "type": "object",
            "properties": {
                "country_code": {
                    "type": "string",
                    "description": (
                        "TWO LETTER country code ONLY (e.g., 'KE' for Kenya, 'TZ' for Tanzania). "
                        "DO NOT use three letter codes. Examples: "
                        "Kenya → 'KE' (not 'KEN'), "
                        "Tanzania → 'TZ' (not 'TZA'), "
                        "Uganda → 'UG' (not 'UGA'). "
                        "If unsure, use google_search to find the correct 2-letter code."
                    ),
                    "pattern": "^[A-Z]{2}$",  # Enforce exactly 2 uppercase letters
                },
                "conservation_status": {
                    "type": "string",
                    "description": (
                        "optional conservation status to filter by, possible values are: "
                        "Least Concern, Endangered, Near Threatened, Vulnerable, Data Deficient, "
                        "Critically Endangered, Extinct"
                    ),
                },
                "chart_type": {
                    "type": "string",
                    "description": "type of chart to display, supported values: "
                    "'force_directed_graph', 'tree'",
                    "enum": ["force_directed_graph", "tree"]
                },
            },
            "required": ["country_code"],
        },
    ),
    FunctionDeclaration(
        name="endangered_species_for_countries",
        description=(
            "Get endangered species information for MULTIPLE countries at once. "
            "Use this function for comparing species across countries. "
            "Examples: 'Compare endangered species between Kenya and Tanzania', "
            "'Show endangered species in both Uganda and Rwanda'. "
            "For single country queries, use endangered_species_for_country instead."
            "Supported chart types: 'force_directed_graph', 'tree'"
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
                    "description": "Filter by conservation status (e.g., 'Critically Endangered')"
                },
                "chart_type": {
                    "type": "string",
                    "description": "type of chart to display, supported values: "
                    "'force_directed_graph', 'tree'",
                },
            },
            "required": ["country_codes"]
        }
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
            "Get GeoJSON data for protected areas in a country using THREE LETTER "
            "country code ONLY. "
            "Examples: 'Show protected areas in Kenya' → use 'KEN', "
            "'Map reserves in Tanzania' → use 'TZA', "
            "'Display parks in Uganda' → use 'UGA'. "
            "IMPORTANT: Must use 3-letter ISO country codes (e.g., KEN, TZA, UGA, USA, GBR), "
            "2-letter codes will not work!"
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
                        "If unsure, use google_search to find the correct 3-letter code."
                    ),
                    "pattern": "^[A-Z]{3}$",  # Enforce exactly 3 uppercase letters
                }
            },
            "required": ["country_code"],
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
            "The function will return a list of endangered species found in the specified area."
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
            "Get occurrence data (coordinates) for a specific species within a protected area. "
            "Shows where a particular species is found within the protected area. "
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
        name="read_terrestrial_hci",
        description=(
            "Read and compare terrestrial human coexistence index (HCI) data between countries. "
            "IMPORTANT: Use this function for ANY questions about: "
            "- terrestrial HCI data "
            "- human impact on wildlife "
            "- human-wildlife coexistence comparisons "
            "- wildlife impact between countries. "
            "Examples: "
            "'What is the terrestrial HCI data for Kenya and Uganda?', "
            "'Compare human impact on wildlife between Kenya and Uganda', "
            "'Show me the human coexistence index for Tanzania and Kenya', "
            "'How does human impact on wildlife compare in East Africa', "
            "'What is the human-wildlife coexistence situation in Kenya vs Tanzania'"
        ),
        parameters={
            "type": "object",
            "properties": {
                "country_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of country names to compare. For queries about HCI data, "
                        "human impact on wildlife, or coexistence comparisons between countries."
                    ),
                },
                "country_codes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of three letter country codes (e.g., ['KEN', 'UGA'])",
                },
            },
            "required": ["country_names"],
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
        name="read_population_density",
        description="Read population density data for a specified country",
        parameters={
            "type": "object",
            "properties": {
                "country_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of country names to compare (e.g., ['Kenya', 'Uganda']). "
                        "For single country queries, provide just one country name."
                    ),
                },
                "country_codes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of three letter country codes (e.g., ['KEN', 'UGA'])",
                },
            },
            "required": ["country_names"],  # At least country names are required
        },
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
                "continent": {"type": "string", "default": "Africa"},
            }
        }
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
        name="get_endangered_species_by_country",
        description=(
            "Get detailed occurrence data for all endangered species in a specific country. "
            "Returns location data and conservation status for each occurrence. "
            "IMPORTANT: Use TWO LETTER country code ONLY. "
            "Examples: 'Show endangered species locations in Kenya' → use 'KE', "
            "'Where are endangered animals in Tanzania' → use 'TZ'. "
            "For comparing multiple countries, use endangered_species_for_countries instead."
        ),
        parameters={
            "type": "object",
            "properties": {
                "country_code": {
                    "type": "string",
                    "description": "TWO LETTER country code "
                    "(e.g., 'KE' for Kenya, 'TZ' for Tanzania)",
                    "pattern": "^[A-Z]{2}$",  # Enforce exactly 2 uppercase letters
                }
            },
            "required": ["country_code"],
        },
    ),
    FunctionDeclaration(
        name="get_species_hci_correlation",
        description=(
            "Get correlation data between species occurrence and Human Coexistence Index (HCI) "
            "for a specific country. Use this for questions about: \n"
            "- How species distribution relates to human impact\n"
            "- Where endangered species live relative to human activity\n"
            "- Correlation between species presence and HCI\n"
            "Examples:\n"
            "- 'How do endangered species relate to human activity in Kenya?'\n"
            "- 'Show correlation between species and HCI in Tanzania'\n"
            "- 'What's the relationship between endangered animals and human impact in Uganda?'"
        ),
        parameters={
            "type": "object",
            "properties": {
                "country_code": {
                    "type": "string",
                    "description": "ISO Alpha-3 country code "
                    "(e.g., 'KEN' for Kenya, 'TZA' for Tanzania)",
                }
            },
            "required": ["country_code"]
        }
    ),
    FunctionDeclaration(
        name="analyze_species_correlations",
        description=(
            "Analyze correlation patterns between species occurrences and human impact (HCI). "
            "Use this for questions about: \n"
            "- How different species relate to human presence\n"
            "- Whether endangered species avoid or prefer human areas\n"
            "- Understanding conservation implications of species-human relationships\n"
            "Examples: 'Analyze correlation patterns for endangered species', "
            "'How do species distributions relate to human impact?', "
            "'Do threatened species avoid human areas?'"
        ),
        parameters={
            "type": "object",
            "properties": {
                "country_code": {
                    "type": "string",
                    "description": "ISO Alpha-3 country code "
                    "(e.g., 'KEN' for Kenya, 'TZA' for Tanzania)",
                }
            },
            "required": ["country_code"]
        }
    ),
]
