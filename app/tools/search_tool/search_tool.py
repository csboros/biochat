"""Module for search tool integration."""

from typing import Dict, Any, List
from vertexai.generative_models import FunctionDeclaration
from app.tools.tool import Tool
from app.tools.search_tool.handlers.search_handler import SearchHandler

class SearchTool(Tool):
    """Tool for performing search operations."""

    def __init__(self):
        """Initialize the search tool."""
        self.search_handler = SearchHandler()

    def get_handlers(self) -> Dict[str, Any]:
        """Get all handlers associated with this tool.

        Returns:
            Dict[str, Any]: Dictionary mapping handler names to their instances
        """
        return {
            "search": self.search_handler
        }

    def get_function_declarations(self) -> List[FunctionDeclaration]:
        """Get function declarations for search operations.

        Returns:
            List[FunctionDeclaration]: List of function declarations for search operations
        """
        return [
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
                    "required": ["query"]
                }
            )
        ]

    def get_function_mappings(self) -> Dict[str, Any]:
        """Get function mappings for search operations.

        Returns:
            Dict[str, Any]: Dictionary mapping function names to their implementations
        """
        return {
            "google_search": self.search_handler.google_search
        }