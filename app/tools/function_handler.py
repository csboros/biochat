"""
Function handler that consolidates all tool definitions and makes them available for external use.
"""

from typing import Dict, Any, List, Union
from vertexai.generative_models import FunctionDeclaration
from app.tools.message_bus import message_bus
from app.tools.help_command_handler import HelpCommandHandler

from .species_tool import SpeciesTool
from .search_tool import SearchTool
from .correlation_tool import CorrelationTool
from .earth_engine_tool import EarthEngineTool

# pylint: disable=broad-except
class FunctionHandler:
    """Handler that consolidates all tool definitions and makes them available for external use."""

    def __init__(self):
        """Initialize the function handler with all available tools."""
        self.species_tool = SpeciesTool()
        self.search_tool = SearchTool()
        self.correlation_tool = CorrelationTool()
        self.earth_engine_tool = EarthEngineTool()
        self.help_handler = HelpCommandHandler()
        self.help_handler.function_handler = self  # Set the function handler reference

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
            description="Get help about available tools, functions, and system capabilities",
            parameters={
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Type of help requested "
                                "(general, category, tool, or function)",
                        "enum": ["general", "category", "tool", "function"]
                    },
                    "category": {
                        "type": "string",
                        "description": "Category name when requesting category-specific help"
                    },
                    "tool": {
                        "type": "string",
                        "description": "Tool name when requesting tool-specific help"
                    },
                    "function": {
                        "type": "string",
                        "description": "Function name when requesting function-specific help"
                    }
                }
            }
        )

        # Add category classification declaration
        classify_category_declaration = FunctionDeclaration(
            name="classify_help_category",
            description="Classify a help request into the appropriate category",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The help request to classify"
                    }
                },
                "required": ["query"]
            }
        )

        all_declarations.extend([help_declaration, classify_category_declaration])

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
        all_mappings = {
            "help": self.handle_help_command,
            "classify_help_category": self.classify_help_category,
        }

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

    def handle_help_command(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle help-related commands and queries.

        Args:
            arguments: Dictionary containing help command arguments

        Returns:
            Dictionary containing help information
        """
        return self.help_handler.handle_help_command(arguments)

    def classify_help_category(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a help request into the appropriate category.

        Args:
            arguments: Dictionary containing the query to classify

        Returns:
            Dictionary containing the classified category
        """
        query = arguments.get('query', '').lower()

        # Return the classification result
        return {
            'success': True,
            'category': self.help_handler.parse_natural_language_query(query)
        }

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
        # If input is a string and contains help-related keywords
        help_keywords = ['help', 'how', 'what']
        if (isinstance(input_text, str) and
                any(word in input_text.lower() for word in help_keywords)):
            return self.help_handler.handle_help_command(input_text)

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
