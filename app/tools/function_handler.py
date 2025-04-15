"""
Function handler that consolidates all tool definitions and makes them available for external use.
"""

from typing import Dict, Any, List, Union
from enum import Enum
import logging
from vertexai.generative_models import FunctionDeclaration
from app.tools.message_bus import message_bus
from app.tools.help_command_handler import HelpCommandHandler

from .species_tool import SpeciesTool
from .search_tool import SearchTool
from .correlation_tool import CorrelationTool
from .earth_engine_tool import EarthEngineTool


class ToolName(Enum):
    """Enumeration of available tools."""
    SPECIES = "species"
    SEARCH = "search"
    CORRELATION = "correlation"
    EARTH_ENGINE = "earth_engine"
    HELP = "help"

# pylint: disable=broad-except
class FunctionHandler:
    """Handler that consolidates all tool definitions and makes them available for external use."""

    # Class variable to implement the singleton pattern
    _instance = None

    def __new__(cls):
        """Implement singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super(FunctionHandler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the function handler with all available tools."""
        # Initialize the attribute at the beginning to satisfy the linter
        self._initialized = getattr(self, '_initialized', False)

        # Only initialize once
        if self._initialized:
            return

        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initializing FunctionHandler")

        # Initialize tools with strong references
        try:
            self.species_tool = SpeciesTool()
            self.logger.debug("SpeciesTool initialized")
        except Exception as e:
            self.logger.error("Failed to initialize SpeciesTool: %s", str(e))
            self.species_tool = None

        try:
            self.search_tool = SearchTool()
            self.logger.debug("SearchTool initialized")
        except Exception as e:
            self.logger.error("Failed to initialize SearchTool: %s", str(e))
            self.search_tool = None

        try:
            self.correlation_tool = CorrelationTool()
            self.logger.debug("CorrelationTool initialized")
        except Exception as e:
            self.logger.error("Failed to initialize CorrelationTool: %s", str(e))
            self.correlation_tool = None

        try:
            self.earth_engine_tool = EarthEngineTool()
            self.logger.debug("EarthEngineTool initialized")
        except Exception as e:
            self.logger.error("Failed to initialize EarthEngineTool: %s", str(e))
            self.earth_engine_tool = None

        self.help_handler = HelpCommandHandler()  # Initialize without tools first

        # Store tools in a dictionary for easy access
        self._tools_dict = {
            ToolName.SPECIES.value: self.species_tool,
            ToolName.SEARCH.value: self.search_tool,
            ToolName.CORRELATION.value: self.correlation_tool,
            ToolName.EARTH_ENGINE.value: self.earth_engine_tool,
            ToolName.HELP.value: self.help_handler
        }

        # Later, after all tools are initialized:
        self.help_handler.set_tools(list(self._tools_dict.values()))


        # Mark as initialized
        self._initialized = True
        self.logger.debug("FunctionHandler fully initialized")

    def get_all_tools(self) -> Dict[str, Any]:
        """
        Get a dictionary of all available tools.

        Returns:
            Dict[str, Any]: Dictionary mapping tool names to their instances
        """
        return self._tools_dict

    def get_all_function_declarations(self) -> List[FunctionDeclaration]:
        """
        Get all function declarations from all tools.

        Returns:
            List[FunctionDeclaration]: List of all function declarations
        """
        all_declarations = []

        # Add help function declaration
        help_declaration = FunctionDeclaration(
            name="help",
            description="Get help about available tools, functions, and system capabilities. Use this for any questions about what the system can do, how to use it, or to learn about available features.",
            parameters={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Category name or topic to get help about"
                    }
                }
            }
        )

        all_declarations.append(help_declaration)

        # Collect declarations from each tool - with null checks
        if self.species_tool is not None:
            try:
                all_declarations.extend(self.species_tool.get_function_declarations())
            except Exception as e:
                self.logger.error("Error getting species tool declarations: %s", str(e))
        else:
            self.logger.warning("Species tool is None, skipping its declarations")

        if self.search_tool is not None:
            try:
                all_declarations.extend(self.search_tool.get_function_declarations())
            except Exception as e:
                self.logger.error("Error getting search tool declarations: %s", str(e))
        else:
            self.logger.warning("Search tool is None, skipping its declarations")

        if self.correlation_tool is not None:
            try:
                all_declarations.extend(self.correlation_tool.get_function_declarations())
            except Exception as e:
                self.logger.error("Error getting correlation tool declarations: %s", str(e))
        else:
            self.logger.warning("Correlation tool is None, skipping its declarations")

        if self.earth_engine_tool is not None:
            try:
                all_declarations.extend(self.earth_engine_tool.get_function_declarations())
            except Exception as e:
                self.logger.error("Error getting earth engine tool declarations: %s", str(e))
        else:
            self.logger.warning("Earth engine tool is None, skipping its declarations")

        return all_declarations

    def get_all_function_mappings(self) -> Dict[str, Any]:
        """
        Get all function mappings from all tools.

        Returns:
            Dict[str, Any]: Dictionary mapping function names to their implementations
        """
        all_mappings = {
            "help": self.handle_help_command,
        }

        # Collect mappings from each tool - with null checks
        if self.species_tool is not None:
            try:
                all_mappings.update(self.species_tool.get_function_mappings())
            except Exception as e:
                self.logger.error("Error getting species tool mappings: %s", str(e))

        if self.search_tool is not None:
            try:
                all_mappings.update(self.search_tool.get_function_mappings())
            except Exception as e:
                self.logger.error("Error getting search tool mappings: %s", str(e))

        if self.correlation_tool is not None:
            try:
                all_mappings.update(self.correlation_tool.get_function_mappings())
            except Exception as e:
                self.logger.error("Error getting correlation tool mappings: %s", str(e))

        if self.earth_engine_tool is not None:
            try:
                all_mappings.update(self.earth_engine_tool.get_function_mappings())
            except Exception as e:
                self.logger.error("Error getting earth engine tool mappings: %s", str(e))

        return all_mappings

    def get_tool_by_name(self, tool_name) -> Any:
        """
        Get a specific tool instance by its name or enum value.

        Args:
            tool_name: Name of the tool (string or ToolName enum)

        Returns:
            The requested tool instance

        Raises:
            ValueError: If the tool name is not found
        """
        self.logger.debug("Requested tool: %s", tool_name)

        # Convert enum to string value if needed
        if isinstance(tool_name, ToolName):
            tool_key = tool_name.value
        else:
            tool_key = str(tool_name)

        # Check if tool exists in our dictionary
        if tool_key not in self._tools_dict:
            available_tools = list(self._tools_dict.keys())
            self.logger.error("Tool '%s' not found. Available tools: %s",
                             tool_key, available_tools)

            # Try case-insensitive matching
            for key, value in self._tools_dict.items():
                if key.lower() == tool_key.lower():
                    self.logger.debug("Found tool with case-insensitive match: %s", key)
                    return value

            raise ValueError(f"Tool '{tool_key}' not found. Available tools: {available_tools}")

        # Get the tool from our dictionary
        tool = self._tools_dict[tool_key]

        # Verify the tool is not None
        if tool is None:
            self.logger.error("Tool '%s' exists but is not initialized", tool_key)
            raise ValueError(f"Tool '{tool_key}' is not properly initialized")

        return tool

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

    def handle_help_command(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle help command requests.

        Args:
            arguments: Dictionary containing help command arguments

        Returns:
            Dictionary containing the help response
        """
        self.logger.info("Handling help command with arguments: %s", arguments)

        # If no type is specified, default to general help
        if 'type' not in arguments:
            arguments['type'] = 'general'

        # Process the help request using the help handler
        result = self.help_handler.handle_help_command(arguments)
        self.logger.info("Help command result: %s", result.get('success'))

        return result

    def get_system_overview(self) -> str:
        """
        Get a comprehensive overview of the system's capabilities.

        Returns:
            str: A formatted string describing the system's capabilities
        """
        help_info = self.help_handler.handle_help_command({'type': 'general'})

        if not help_info.get('success'):
            return "I apologize, but I couldn't retrieve the system overview at this time."

        overview = "Here's an overview of the system's capabilities:\n\n"

        # Group tools by category
        tools_by_category = {}
        for tool in help_info['data']['tools']:
            category = tool['category']
            if category not in tools_by_category:
                tools_by_category[category] = []
            tools_by_category[category].append(tool)

        # Format overview by category
        for category, tools in tools_by_category.items():
            overview += f"## {category}\n"
            for tool in tools:
                overview += f"\n### {tool['name']}\n"
                overview += f"{tool['description']}\n"
                overview += "\nAvailable functions:\n"
                for func in tool['available_functions']:
                    overview += f"- {func['name']}: {func['description']}\n"
                    overview += f"  Parameters: {', '.join(func['parameters'])}\n"
                    overview += f"  Returns: {func['returns']}\n"
                overview += "\n"

        return overview

    def handle_function_call(self, input_text: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Handle function calls from both structured commands and natural language.
        """

        try:
            # Get all function mappings
            function_mappings = self.get_all_function_mappings()

            # Get the function name and arguments
            function_name = input_text.get('name')
            arguments = input_text.get('arguments', {})

            # Check if the function exists
            if function_name not in function_mappings:
                raise ValueError(f"Function '{function_name}' not found")

            # Execute the function with the provided arguments
            result = function_mappings[function_name](arguments)

            return {
                'success': True,
                'result': result
            }

        except Exception as e:
            message_bus.publish("status_update", {
                "message": f"❌ Error in function call: {str(e)}",
                "state": "error"
            })
            return {
                'success': False,
                'error': str(e)
            }

    def get_available_functions(self) -> Dict[str, Any]:
        """
        Get information about available functions.

        Returns:
            Dictionary containing available functions information
        """
        return self.help_handler.handle_help_command({'type': 'general'})

