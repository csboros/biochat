"""Module for Earth Engine tool integration."""

from typing import Dict, Any, List
from vertexai.generative_models import FunctionDeclaration
from app.tools.tool import Tool
from app.tools.earth_engine_tool.handlers.forest_handler import ForestHandlerEE
from .handlers.human_modification_handler import HumanModificationHandlerEE
from .handlers.habitat_analyzer import HabitatAnalyzer
from .handlers.topography_analyzer import TopographyAnalyzer
from .handlers.climate_analyzer import ClimateAnalyzer

class EarthEngineTool(Tool):
    """Tool for Earth Engine data processing and analysis."""

    def __init__(self):
        """Initialize the EarthEngineTool with its handlers."""
        self.forest_handler = ForestHandlerEE()
        self.human_modification_handler = HumanModificationHandlerEE()
        self.habitat_analyzer = HabitatAnalyzer()
        self.topography_analyzer = TopographyAnalyzer()
        self.climate_analyzer = ClimateAnalyzer()

    def get_handlers(self) -> Dict[str, Any]:
        """Get all handlers associated with this tool.

        Returns:
            Dict[str, Any]: Dictionary mapping handler names to their instances
        """
        return {
            "forest": self.forest_handler,
            "human_modification": self.human_modification_handler,
            "habitat": self.habitat_analyzer,
            "topography": self.topography_analyzer,
            "climate": self.climate_analyzer
        }

    def get_function_declarations(self) -> List[FunctionDeclaration]:
        """Get function declarations for Earth Engine operations.

        Returns:
            List[FunctionDeclaration]: List of function declarations for Earth Engine operations
        """
        return [
            FunctionDeclaration(
                name="calculate_species_forest_correlation",
                description=(
                    "Calculate correlation between species occurrence and forest cover/loss. "
                    "This analyzes how species distribution relates to forest metrics using "
                    "Hansen Global Forest Change data. "
                    "Examples:\n"
                    "- 'How does lion distribution relate to forest cover?'\n"
                    "- 'How does orangutan distribution relate to forest cover?'\n"
                    "- 'Show forest correlation for Panthera leo'\n"
                    "- 'Analyze forest habitat relationship for tigers'"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "species_name": {
                            "type": "string",
                            "description": "Common or scientific name of the species"
                        }
                    },
                    "required": ["species_name"]
                }
            ),
            FunctionDeclaration(
                name="calculate_species_humanmod_correlation",
                description=(
                    "Calculate correlation between species occurrence and human modification. "
                    "This analyzes how species distribution relates to human impact using "
                    "the Global Human Modification dataset. "
                    "Examples:\n"
                    "- 'How does lion distribution relate to human modification?'\n"
                    "- 'How does elephant distribution relate to human impact?'\n"
                    "- 'Show human modification correlation for Panthera leo'\n"
                    "- 'Analyze human impact relationship for tigers'"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "species_name": {
                            "type": "string",
                            "description": "Common or scientific name of the species"
                        }
                    },
                    "required": ["species_name"]
                }
            ),
            FunctionDeclaration(
                name="analyze_habitat_distribution",
                description=(
                    "Analyze species habitat distribution and preferences using "
                    "Copernicus land cover data. This function is essential for any questions regarding:\n"
                    "- Habitat analysis, preferences, or distribution\n"
                    "- Types of habitats a species uses or prefers\n"
                    "- Habitat dependencies or requirements\n"
                    "- Habitat fragmentation analysis\n"
                    "- Primary or preferred habitat types\n"
                    "This function will return insights into the habitat characteristics and may include visualizations.\n"
                    "Examples of valid queries:\n"
                    "- 'What habitats does the lion (Panthera leo) use?'\n"
                    "- 'Analyze habitat distribution for Bornean orangutans (Pongo pygmaeus)'\n"
                    "- 'Show habitat preferences for gorillas (Gorilla gorilla)'\n"
                    "- 'What type of habitat do orangutans prefer?'\n"
                    "- 'Study the habitat of Bornean orangutans'\n"
                    "- 'Analyze habitat fragmentation for tigers (Panthera tigris)'\n"
                    "IMPORTANT: This is the primary function for ALL habitat analysis queries."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "species_name": {
                            "type": "string",
                            "description": "Common or scientific name of the species"
                        }
                    },
                    "required": ["species_name"]
                }
            ),
            FunctionDeclaration(
                name="analyze_topography",
                description=(
                    "Analyze topography characteristics of species habitat. "
                    "This analyzes the topographical features of a species' habitat. "
                    "Examples:\n"
                    "- 'Analyze topography for Gorilla beringei'\n"
                    "- 'Show topography for Bornean orangutans'"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "species_name": {
                            "type": "string",
                            "description": "Common or scientific name of the species"
                        }
                    },
                    "required": ["species_name"]
                }
            ),
            FunctionDeclaration(
                name="analyze_climate",
                description=(
                    "Analyze climate characteristics of species habitat "
                    "using ERA5-Land and CHIRPS data. "
                    "This analyzes temperature and precipitation patterns, "
                    "climate trends, and species-climate relationships. "
                    "Use this for questions about:\n"
                    "- What are the temperature and precipitation patterns in the species' range\n"
                    "- How does climate vary across the species' distribution\n"
                    "- What are the climate preferences of the species\n"
                    "- How might climate change affect the species\n"
                    "Examples:\n"
                    "- 'Analyze climate for Panthera leo'\n"
                    "- 'Show climate patterns for Bornean orangutans'\n"
                    "- 'What are the climate preferences of tigers?'\n"
                    "- 'How does climate affect elephant distribution?'"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "species_name": {
                            "type": "string",
                            "description": "Common or scientific name of the species"
                        }
                    },
                    "required": ["species_name"]
                }
            )
        ]

    def get_function_mappings(self) -> Dict[str, Any]:
        """Get function mappings for Earth Engine operations.

        Returns:
            Dict[str, Any]: Dictionary mapping function names to their implementations
        """
        return {
            "calculate_species_forest_correlation":
                self.forest_handler.calculate_species_forest_correlation,
            "calculate_species_humanmod_correlation":
                self.human_modification_handler.calculate_species_humanmod_correlation,
            "analyze_habitat_distribution": self.habitat_analyzer.analyze_habitat_distribution,
            "analyze_topography": self.topography_analyzer.analyze_topography,
            "analyze_climate": self.climate_analyzer.analyze
        }
