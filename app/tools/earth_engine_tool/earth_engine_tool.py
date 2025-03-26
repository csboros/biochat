"""Module for Earth Engine tool integration."""

from typing import Dict, Any, List
from vertexai.generative_models import FunctionDeclaration
from app.tools.tool import Tool
from app.tools.earth_engine_tool.handlers.forest_handler import ForestHandlerEE
from app.tools.earth_engine_tool.handlers.human_modification_handler import HumanModificationHandlerEE

class EarthEngineTool(Tool):
    """Tool for Earth Engine data processing and analysis."""

    def __init__(self):
        """Initialize the EarthEngineTool with its handlers."""
        self.forest_handler = ForestHandlerEE()
        self.human_modification_handler = HumanModificationHandlerEE()

    def get_handlers(self) -> Dict[str, Any]:
        """Get all handlers associated with this tool.

        Returns:
            Dict[str, Any]: Dictionary mapping handler names to their instances
        """
        return {
            "forest": self.forest_handler,
            "human_modification": self.human_modification_handler
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
                            "description": "Scientific name of the species (e.g., 'Panthera leo'), "
                            "for common names, first use translate_to_scientific_name"
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
                            "description": "Scientific name of the species (e.g., 'Panthera leo'), "
                            "for common names, first use translate_to_scientific_name"
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
            "calculate_species_forest_correlation": self.forest_handler.calculate_species_forest_correlation,
            "calculate_species_humanmod_correlation": self.human_modification_handler.calculate_species_humanmod_correlation
        }