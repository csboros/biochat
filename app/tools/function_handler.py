"""
Function handler that consolidates all tool definitions and makes them available for external use.
"""

from typing import Dict, Any, List
from vertexai.generative_models import FunctionDeclaration

from .species_tool import SpeciesTool
from .search_tool import SearchTool
from .correlation_tool import CorrelationTool
from .earth_engine_tool import EarthEngineTool

class FunctionHandler:
    """Handler that consolidates all tool definitions and makes them available for external use."""

    def __init__(self):
        """Initialize the function handler with all available tools."""
        self.species_tool = SpeciesTool()
        self.search_tool = SearchTool()
        self.correlation_tool = CorrelationTool()
        self.earth_engine_tool = EarthEngineTool()

    def get_all_function_declarations(self) -> List[FunctionDeclaration]:
        """
        Get all function declarations from all tools.

        Returns:
            List[FunctionDeclaration]: List of all function declarations
        """
        all_declarations = []

        # Collect declarations from each tool
        all_declarations.extend(self.species_tool.get_function_declarations())
        all_declarations.extend(self.search_tool.get_function_declarations())
        all_declarations.extend(self.correlation_tool.get_function_declarations())
        all_declarations.extend(self.earth_engine_tool.get_function_declarations())

        return all_declarations

    def get_all_function_mappings(self) -> Dict[str, Any]:
        """
        Get all function mappings from all tools.

        Returns:
            Dict[str, Any]: Dictionary mapping function names to their implementations
        """
        all_mappings = {}

        # Collect mappings from each tool
        all_mappings.update(self.species_tool.get_function_mappings())
        all_mappings.update(self.search_tool.get_function_mappings())
        all_mappings.update(self.correlation_tool.get_function_mappings())
        all_mappings.update(self.earth_engine_tool.get_function_mappings())

        return all_mappings

    def get_tool_by_name(self, tool_name: str) -> Any:
        """
        Get a specific tool by its name.

        Args:
            tool_name (str): Name of the tool to retrieve

        Returns:
            Any: The requested tool instance

        Raises:
            ValueError: If the tool name is not found
        """
        tools = {
            "species": self.species_tool,
            "search": self.search_tool,
            "correlation": self.correlation_tool,
            "earth_engine": self.earth_engine_tool
        }

        if tool_name not in tools:
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {list(tools.keys())}")

        return tools[tool_name]

    def get_handler_by_name(self, tool_name: str, handler_name: str) -> Any:
        """
        Get a specific handler from a tool by its name.

        Args:
            tool_name (str): Name of the tool
            handler_name (str): Name of the handler within the tool

        Returns:
            Any: The requested handler instance

        Raises:
            ValueError: If the tool or handler name is not found
        """
        tool = self.get_tool_by_name(tool_name)
        handlers = tool.get_handlers()

        if handler_name not in handlers:
            raise ValueError(
                f"Handler '{handler_name}' not found in tool '{tool_name}'. "
                f"Available handlers: {list(handlers.keys())}"
            )

        return handlers[handler_name]