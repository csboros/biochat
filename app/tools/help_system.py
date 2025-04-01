"""Comprehensive help system for all tools in the application."""

from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

class ToolCategory(Enum):
    """Categories of tools available in the system."""
    HABITAT_ANALYSIS = "Habitat Analysis"
    SPECIES_ANALYSIS = "Species Analysis"
    CORRELATION_ANALYSIS = "Correlation Analysis"
    SEARCH = "Search"
    VISUALIZATION = "Visualization"
    UTILITY = "Utility"

@dataclass
class Tool:
    """Represents a tool with its description and available functions."""
    name: str
    category: ToolCategory
    description: str
    module_path: str
    available_functions: List[Dict[str, Any]]
    example_usage: str

class ApplicationHelpSystem:
    """Provides comprehensive help and documentation about all available tools."""

    def __init__(self):
        """Initialize the help system with all available tools."""
        self.tools = self._initialize_tools()

    def _initialize_tools(self) -> List[Tool]:
        """Initialize the list of available tools by scanning the tools directory."""
        tools = []

        # Earth Engine Tool
        tools.append(Tool(
            name="Earth Engine Tool",
            category=ToolCategory.HABITAT_ANALYSIS,
            description="""
            Provides habitat analysis capabilities using Google Earth Engine.
            Includes habitat distribution, connectivity, and fragmentation analysis.
            """,
            module_path="app.tools.earth_engine_tool.handlers.earth_engine_handler",
            available_functions=[
                {
                    'name': 'analyze_habitat_distribution',
                    'description': "Analyzes species habitat distribution "
                            "using Copernicus land cover data",
                    'parameters': ['species_name: str'],
                    'returns': 'Dict[str, Any]'
                },
                {
                    'name': 'analyze_habitat_connectivity',
                    'description': "Evaluates habitat connectivity across "
                                        "different land cover types",
                    'parameters': ['points_with_landcover', 'habitat_usage: Dict[str, float]'],
                    'returns': 'Dict[str, Any]'
                }
            ],
            example_usage="""
            from app.tools.earth_engine_tool.handlers.earth_engine_handler import EarthEngineHandler

            handler = EarthEngineHandler()
            results = handler.analyze_habitat_distribution('Species Name')
            """
        ))

        # Species Tool
        tools.append(Tool(
            name="Species Tool",
            category=ToolCategory.SPECIES_ANALYSIS,
            description="""
            Provides species-related functionality including species search,
            occurrence data retrieval, and species information analysis.
            """,
            module_path="app.tools.species_tool.handlers.species_handler",
            available_functions=[
                {
                    'name': 'search_species',
                    'description': 'Searches for species by name or scientific name',
                    'parameters': ['query: str'],
                    'returns': 'List[Dict[str, Any]]'
                },
                {
                    'name': 'get_species_occurrences',
                    'description': 'Retrieves occurrence data for a species',
                    'parameters': ['species_id: str'],
                    'returns': 'List[Dict[str, Any]]'
                }
            ],
            example_usage="""
            from app.tools.species_tool.handlers.species_handler import SpeciesHandler

            handler = SpeciesHandler()
            species = handler.search_species('Panthera leo')
            """
        ))

        # Correlation Tool
        tools.append(Tool(
            name="Correlation Tool",
            category=ToolCategory.CORRELATION_ANALYSIS,
            description="""
            Analyzes correlations between species occurrences and environmental variables,
            including climate data and land cover types.
            """,
            module_path="app.tools.correlation_tool.handlers.correlation_handler",
            available_functions=[
                {
                    'name': 'analyze_correlation',
                    'description': "Analyzes correlation between species occurrences "
                        "and environmental variables",
                    'parameters': ['species_id: str', 'variables: List[str]'],
                    'returns': 'Dict[str, Any]'
                }
            ],
            example_usage="""
            from app.tools.correlation_tool.handlers.correlation_handler import CorrelationHandler

            handler = CorrelationHandler()
            correlation = handler.analyze_correlation('species_id', ['temperature', 'precipitation'])
            """
        ))

        # Search Tool
        tools.append(Tool(
            name="Search Tool",
            category=ToolCategory.SEARCH,
            description="""
            Provides search functionality across various data sources,
            including species, habitats, and environmental data.
            """,
            module_path="app.tools.search_tool.handlers.search_handler",
            available_functions=[
                {
                    'name': 'search',
                    'description': 'Performs a search across multiple data sources',
                    'parameters': ['query: str', 'filters: Dict[str, Any]'],
                    'returns': 'Dict[str, Any]'
                }
            ],
            example_usage="""
            from app.tools.search_tool.handlers.search_handler import SearchHandler

            handler = SearchHandler()
            results = handler.search('query', {'type': 'species'})
            """
        ))

        # Visualization Tool
        tools.append(Tool(
            name="Visualization Tool",
            category=ToolCategory.VISUALIZATION,
            description="""
            Creates various visualizations including maps, charts, and plots
            for species distribution, habitat analysis, and correlation results.
            """,
            module_path="app.tools.visualization.renderers.base",
            available_functions=[
                {
                    'name': 'render',
                    'description': 'Renders visualizations based on provided data',
                    'parameters': ['data: Dict[str, Any]', 'parameters: Dict[str, Any]'],
                    'returns': 'None'
                }
            ],
            example_usage="""
            from app.tools.visualization.renderers.habitat_viz import HabitatViz

            renderer = HabitatViz()
            renderer.render(data, parameters)
            """
        ))

        return tools

    def get_help(self, category: ToolCategory = None) -> Dict[str, Any]:
        """
        Get help information about available tools.

        Args:
            category: Optional category to filter tools by

        Returns:
            Dictionary containing help information
        """
        if category:
            tools = [t for t in self.tools if t.category == category]
        else:
            tools = self.tools

        help_info = {
            'categories': [cat.value for cat in ToolCategory],
            'tools': [
                {
                    'name': t.name,
                    'category': t.category.value,
                    'description': t.description.strip(),
                    'module_path': t.module_path,
                    'available_functions': t.available_functions,
                    'example_usage': t.example_usage.strip()
                }
                for t in tools
            ]
        }

        return help_info

    def get_tool_help(self, tool_name: str) -> Dict[str, Any]:
        """
        Get detailed help information about a specific tool.

        Args:
            tool_name: Name of the tool to get help for

        Returns:
            Dictionary containing detailed help information for the tool
        """
        tool = next(
            (t for t in self.tools if t.name.lower() == tool_name.lower()),
            None
        )

        if not tool:
            return {
                'error': f"Tool '{tool_name}' not found",
                'available_tools': [t.name for t in self.tools]
            }

        return {
            'name': tool.name,
            'category': tool.category.value,
            'description': tool.description.strip(),
            'module_path': tool.module_path,
            'available_functions': tool.available_functions,
            'example_usage': tool.example_usage.strip()
        }

    def get_function_help(self, tool_name: str, function_name: str) -> Dict[str, Any]:
        """
        Get detailed help information about a specific function in a tool.

        Args:
            tool_name: Name of the tool
            function_name: Name of the function

        Returns:
            Dictionary containing detailed help information for the function
        """
        tool = next(
            (t for t in self.tools if t.name.lower() == tool_name.lower()),
            None
        )

        if not tool:
            return {
                'error': f"Tool '{tool_name}' not found",
                'available_tools': [t.name for t in self.tools]
            }

        function = next(
            (f for f in tool.available_functions if f['name'].lower() == function_name.lower()),
            None
        )

        if not function:
            return {
                'error': f"Function '{function_name}' not found in tool '{tool_name}'",
                'available_functions': [f['name'] for f in tool.available_functions]
            }

        return {
            'tool_name': tool.name,
            'function_name': function['name'],
            'description': function['description'],
            'parameters': function['parameters'],
            'returns': function['returns'],
            'example_usage': tool.example_usage.strip()
        }
