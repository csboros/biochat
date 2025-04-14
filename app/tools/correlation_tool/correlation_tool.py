"""Module for correlation tool integration."""

from typing import Dict, Any, List
from vertexai.generative_models import FunctionDeclaration
from app.tools.tool import Tool
from app.tools.correlation_tool.handlers.correlation_handler import CorrelationHandler


class CorrelationTool(Tool):
    """Tool for species-HCI correlation analysis."""

    def __init__(self):
        """Initialize the correlation tool."""
        self.correlation_handler = CorrelationHandler()

    def get_handlers(self) -> Dict[str, Any]:
        """Get all handlers associated with this tool.

        Returns:
            Dict[str, Any]: Dictionary mapping handler names to their instances
        """
        return {
            "correlation": self.correlation_handler
        }

    def get_function_declarations(self) -> List[FunctionDeclaration]:
        """Get function declarations for correlation operations.

        Returns:
            List[FunctionDeclaration]: List of function declarations for correlation operations
        """
        return [
             FunctionDeclaration(
                name="read_terrestrial_hci",
                description=(
                    "Read and compare the overall terrestrial Human Coexistence Index (HCI) "
                    "value between different countries. Use this ONLY for questions asking "
                    "to compare the general HCI index score or overall human coexistence "
                    "levels across specified countries. "
                    "Examples: "
                    "'What is the terrestrial HCI data for Kenya and Uganda?', "
                    "'Compare the HCI index between Kenya and Uganda', "
                    "'Show me the human coexistence index score for Tanzania and Kenya'. "
                    "IMPORTANT: This function provides ONLY the country-level HCI index value. "
                    "It does NOT analyze specific species data or map species locations. "
                    "Do NOT use for questions about individual species."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "country_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "List of country names to compare their HCI index values."
                            ),
                            "minItems": 1
                        },
                    },
                    "required": ["country_names"],
                },
            ),
            FunctionDeclaration(
                name="get_species_hci_correlation",
                description=(
                    "Get correlation data between species occurrence and Human "
                    "Coexistence Index (HCI) for a specific country. Use this for "
                    "questions about: \n"
                    "- How species distribution relates to human impact\n"
                    "- Where endangered species live relative to human activity\n"
                    "- Correlation between species presence and HCI\n"
                    "Examples:\n"
                    "- 'How do endangered species relate to human activity in Kenya?'\n"
                    "- 'Show correlation between species and HCI in Tanzania'\n"
                    "- 'What's the relationship between endangered animals and human "
                    "impact in Uganda?'"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "country_code": {
                            "type": "string",
                            "description": (
                                "ISO Alpha-3 country code (e.g., 'KEN' for Kenya, "
                                "'TZA' for Tanzania)"
                            )
                        }
                    },
                    "required": ["country_code"]
                }
            ),
            FunctionDeclaration(
                name="get_species_hci_correlation_by_status",
                description=(
                    "Get correlation data between species occurrence and Human "
                    "Coexistence Index (HCI) filtered by conservation status. Use this "
                    "for questions about: \n"
                    "- How specific conservation status species relate to human impact\n"
                    "- Correlation patterns for critically endangered/endangered/"
                    "vulnerable species\n"
                    "- Understanding how different threat levels relate to human presence\n"
                    "- General queries about critically endangered species\n"
                    "Examples:\n"
                    "- 'Show correlation for critically endangered species'\n"
                    "- 'How do vulnerable species relate to human activity?'\n"
                    "- 'What's the relationship between endangered species and HCI?'\n"
                    "- 'Tell me about critically endangered species'\n"
                    "- 'Show data for critically endangered species'\n"
                    "- 'Analyze critically endangered species patterns'"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "conservation_status": {
                            "type": "string",
                            "description": (
                                "Conservation status to filter by. Valid values: "
                                "'Critically Endangered', 'Endangered', 'Vulnerable', "
                                "'Near Threatened', 'Least Concern', 'Data Deficient', "
                                "'Extinct'"
                            ),
                            "enum": [
                                "Critically Endangered",
                                "Endangered",
                                "Vulnerable",
                                "Near Threatened",
                                "Least Concern",
                                "Data Deficient",
                                "Extinct"
                            ]
                        }
                    },
                    "required": ["conservation_status"]
                }
            ),
            FunctionDeclaration(
                name="analyze_species_correlations",
                description=(
                    "Analyze correlation patterns between species occurrences and human "
                    "impact (HCI). Can analyze by either country or conservation status. "
                    "Use this for questions about:\n"
                    "- How different species relate to human presence in a specific "
                    "country\n"
                    "- How species of a particular conservation status relate to human "
                    "areas globally\n"
                    "- Understanding conservation implications of species-human "
                    "relationships\n"
                    "Examples: (Shows only the most significant and notable correlations "
                    "to avoid information overload.)\n"
                    "- 'Analyze correlation patterns for endangered species in Kenya'\n"
                    "- 'How do critically endangered species relate to human impact?'\n"
                    "- 'Analyze species-HCI relationships in Tanzania'\n"
                    "- 'Show correlation analysis for vulnerable species'"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "country_code": {
                            "type": "string",
                            "description": (
                                "ISO Alpha-3 country code (e.g., 'KEN' for Kenya, "
                                "'TZA' for Tanzania). Optional if conservation_status "
                                "is provided."
                            )
                        },
                        "conservation_status": {
                            "type": "string",
                            "description": (
                                "Conservation status to analyze. Optional if country_code "
                                "is provided. Valid values: 'Critically Endangered', "
                                "'Endangered', 'Vulnerable', 'Near Threatened', "
                                "'Least Concern', 'Data Deficient', 'Extinct'"
                            ),
                            "enum": [
                                "Critically Endangered",
                                "Endangered",
                                "Vulnerable",
                                "Near Threatened",
                                "Least Concern",
                                "Data Deficient",
                                "Extinct"
                            ]
                        }
                    }
                }
            ),
            FunctionDeclaration(
                name="get_species_shared_habitat",
                description=(
                    "Get correlation data between a specific species and other species "
                    "that share its habitat. Use this for questions about:\n"
                    "- Which species share habitat with a particular species\n"
                    "- How different species correlate in their occurrence patterns\n"
                    "- Understanding species relationships and co-occurrence\n"
                    "Examples:\n"
                    "- 'Which species share habitat with lion?'\n"
                    "- 'Show species that live alongside tigers'\n"
                    "- 'What animals coexist with elephants?'\n"
                    "- 'Find species that correlate with pandas'\n"
                    "IMPORTANT: For common names, first use translate_to_scientific_name "
                    "to get the scientific name."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "species_name": {
                            "type": "string",
                            "description": (
                                "Scientific name of the species (e.g., 'Panthera leo' "
                                "for Lion). For common names, first use "
                                "translate_to_scientific_name."
                            )
                        }
                    },
                    "required": ["species_name"]
                }
            )
        ]

    def get_function_mappings(self) -> Dict[str, Any]:
        """Get function mappings for correlation operations.

        Returns:
            Dict[str, Any]: Dictionary mapping function names to their implementations
        """
        return {
            "get_species_hci_correlation": (
                self.correlation_handler.get_species_hci_correlation
            ),
            "get_species_hci_correlation_by_status": (
                self.correlation_handler.get_species_hci_correlation_by_status
            ),
            "analyze_species_correlations": (
                self.correlation_handler.analyze_species_correlations
            ),
            "get_species_shared_habitat": (
                self.correlation_handler.get_species_shared_habitat
            ),
            "read_terrestrial_hci": self.correlation_handler.read_terrestrial_hci
        }