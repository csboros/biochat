"""Handler for help-related commands in the application."""

from typing import Dict, Any
from app.tools.message_bus import message_bus
from app.tools.help_system import ApplicationHelpSystem, ToolCategory

class HelpCommandHandler:
    """Handles help-related commands and queries."""

    def __init__(self):
        """Initialize the help command handler."""
        self.help_system = ApplicationHelpSystem()

    def handle_help_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle help-related commands.

        Args:
            command: Dictionary containing the help command and its parameters

        Returns:
            Dictionary containing the help response
        """
        try:
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
            'message': f"Help information for function '{function_name}' in tool '{tool_name}' retrieved successfully"
        }