"""Handler for help-related commands in the application."""

from typing import Dict, Any
from app.tools.message_bus import message_bus
from app.tools.help_system import ApplicationHelpSystem, ToolCategory

# pylint: disable=broad-except
class HelpCommandHandler:
    """Handles help-related commands and queries."""

    def __init__(self):
        """Initialize the help command handler."""
        self.help_system = ApplicationHelpSystem()
        self._function_handler = None  # Will be set later via dependency injection
        # Update keywords to match ToolCategory enum values
        self.category_keywords = {
            'habitat': 'HABITAT_ANALYSIS',
            'species': 'SPECIES_ANALYSIS',
            'correlation': 'CORRELATION_ANALYSIS',
            'search': 'SEARCH',
            'visualization': 'VISUALIZATION',
            'utility': 'UTILITY'
            # Add more mappings as needed
        }

    @property
    def function_handler(self):
        """Get the function handler instance."""
        return self._function_handler

    @function_handler.setter
    def function_handler(self, handler: Any):
        """Set the function handler instance."""
        self._function_handler = handler

    def parse_natural_language_query(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query into a structured help command.

        Args:
            query: Natural language help request

        Returns:
            Dictionary containing the structured help command
        """
        query = query.lower()

        # Check for multi-word category phrases first
        category_phrases = {
            'habitat analysis': 'HABITAT_ANALYSIS',
            'species analysis': 'SPECIES_ANALYSIS',
            'correlation analysis': 'CORRELATION_ANALYSIS'
        }

        # Check for full phrases first
        for phrase, category in category_phrases.items():
            if phrase in query:
                return {
                    'type': 'category',
                    'category': category
                }

        # Fall back to single keyword matching
        for keyword, category in self.category_keywords.items():
            if keyword in query:
                return {
                    'type': 'category',
                    'category': category
                }

        # If no specific category is found, return general help
        return {
            'type': 'general'
        }

    def handle_help_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle help-related commands."""
        try:
            # Handle natural language queries
            if isinstance(command, str):
                if self.function_handler is None:
                    command = {'type': 'general'}
                else:
                    result = self.function_handler.handle_function_call({
                        'name': 'classify_help_category',
                        'arguments': {
                            'query': command
                        }
                    })

                    if result.get('success'):
                        command = {
                            'type': 'category',
                            'category': result['result']['category']
                        }
                    else:
                        command = {'type': 'general'}

            command_type = command.get('type', 'general')

            if command_type == 'general':
                return self._handle_general_help()
            elif command_type == 'category':
                category = command.get('category')
                return self._handle_category_help(category)
            elif command_type == 'tool':
                tool_name = command.get('tool')
                return self._handle_tool_help(tool_name)
            elif command_type == 'function':
                tool_name = command.get('tool')
                function_name = command.get('function')
                return self._handle_function_help(tool_name, function_name)
            else:
                return {
                    'success': False,
                    'error': f"Unknown help command type: {command_type}",
                    'available_commands': ['general', 'category', 'tool', 'function']
                }

        except Exception as e:
            message_bus.publish("status_update", {
                "message": f"❌ Error in help command: {str(e)}",
                "state": "error"
            })
            return {
                'success': False,
                'error': str(e)
            }

    def _handle_general_help(self) -> Dict[str, Any]:
        """Handle general help request."""
        help_info = self.help_system.get_help()

        message_bus.publish("status_update", {
            "message": "📚 Showing available tools and categories",
            "state": "complete"
        })

        return {
            'success': True,
            'data': help_info,
            'message': "Available tools and categories retrieved successfully"
        }

    def _handle_category_help(self, category: str) -> Dict[str, Any]:
        """Handle category-specific help request."""
        try:
            category_enum = ToolCategory[category.upper()]
            help_info = self.help_system.get_help(category_enum)

            message_bus.publish("status_update", {
                "message": f"📚 Showing tools in category: {category}",
                "state": "complete"
            })

            return {
                'success': True,
                'data': help_info,
                'message': f"Tools in category '{category}' retrieved successfully"
            }
        except KeyError:
            return {
                'success': False,
                'error': f"Unknown category: {category}",
                'available_categories': [cat.name for cat in ToolCategory]
            }

    def _handle_tool_help(self, tool_name: str) -> Dict[str, Any]:
        """Handle tool-specific help request."""
        help_info = self.help_system.get_tool_help(tool_name)

        if 'error' in help_info:
            message_bus.publish("status_update", {
                "message": f"❌ Tool not found: {tool_name}",
                "state": "error"
            })
            return {
                'success': False,
                'error': help_info['error'],
                'available_tools': help_info['available_tools']
            }

        message_bus.publish("status_update", {
            "message": f"📚 Showing help for tool: {tool_name}",
            "state": "complete"
        })

        return {
            'success': True,
            'data': help_info,
            'message': f"Help information for tool '{tool_name}' retrieved successfully"
        }

    def _handle_function_help(self, tool_name: str, function_name: str) -> Dict[str, Any]:
        """Handle function-specific help request."""
        help_info = self.help_system.get_function_help(tool_name, function_name)

        if 'error' in help_info:
            message_bus.publish("status_update", {
                "message": f"❌ Function not found: {function_name} in tool {tool_name}",
                "state": "error"
            })
            return {
                'success': False,
                'error': help_info['error'],
                'available_functions': help_info.get('available_functions', [])
            }

        message_bus.publish("status_update", {
            "message": f"📚 Showing help for function: {function_name} in tool {tool_name}",
            "state": "complete"
        })

        return {
            'success': True,
            'data': help_info,
            'message': "Help information for function "
                    "'{function_name}' in tool '{tool_name}' retrieved successfully"
        }
