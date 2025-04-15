"""Comprehensive help system for all tools in the application."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

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
    """Provides help information about the application's tools and functions."""

    def __init__(self):
        """Initialize the help system with all available tools."""
        self.logger = logging.getLogger("BioChat.HelpSystem")
        self.tools = self._initialize_tools()
        self._current_help_query = ""

        # Initialize category descriptions
        self.category_descriptions = {
            ToolCategory.HABITAT_ANALYSIS: """
            Tools for analyzing habitats, including distribution, connectivity, and fragmentation.
            These tools help understand the spatial characteristics of species habitats and how
            they are affected by various environmental factors.
            """,
            ToolCategory.SPECIES_ANALYSIS: """
            Tools for analyzing species data, distributions, and characteristics.
            These tools provide insights into species occurrence, behavior, and relationships
            with their environment.
            """,
            ToolCategory.CORRELATION_ANALYSIS: """
            Tools for analyzing correlations between different biodiversity factors.
            These tools help identify relationships between species, habitats, and environmental
            variables.
            """,
            ToolCategory.SEARCH: """
            Tools for searching and retrieving biodiversity data from various sources.
            These tools provide access to species information, occurrence data, and other
            biodiversity-related information.
            """,
            ToolCategory.VISUALIZATION: """
            Tools for visualizing biodiversity data in various formats including maps, charts,
            and graphs. These tools help communicate complex biodiversity information in an
            intuitive way.
            """,
            ToolCategory.UTILITY: """
            Utility tools for data processing, transformation, and management.
            These tools support the other categories by providing common functionality
            for data handling.
            """
        }

        # Initialize specialized category content
        self.specialized_category_content = {
            "Human Impact Analysis": {
                "text": """# Human Impact Analysis

The Human Impact Analysis tools allow you to explore the relationship between human activities and biodiversity.

## Key Capabilities:

### 1. Human Coexistence Index (HCI)
- Measure the level of human impact on natural environments
- Visualize HCI data across different regions
- Compare HCI values between protected and non-protected areas

### 2. Species-HCI Correlation
- Analyze how species distribution correlates with human activity
- Identify species that are sensitive to human presence
- Discover species that coexist well with humans

### 3. Human Modification Analysis
- Examine how human land modification affects species
- Visualize human modification gradients
- Compare species presence across different modification levels

## Available Functions:
- **read_terrestrial_hci**: Retrieves Human Coexistence Index data for a specific region
- **get_species_hci_correlation**: Analyzes correlation between species presence and HCI
- **endangered_species_hci_correlation**: Examines how endangered species relate to human impact
"""
            },
            # Add other specialized categories here as needed
        }

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

    def get_help(self, category: Optional[ToolCategory] = None, query: Optional[str] = None) -> Dict[str, Any]:
        """
        Get help information for the application.

        Args:
            category: Optional category to filter tools by
            query: Optional natural language query to provide context

        Returns:
            Dictionary containing help information
        """
        # Check for specialized content based on query
        if query:
            for special_category, content in self.specialized_category_content.items():
                if special_category.lower() in query.lower():
                    self.logger.info("Providing specialized help for: %s", special_category)
                    return content

        # Regular help processing
        if category is None:
            # Return general help overview
            return self.get_system_overview()

        # Get tools for the category
        tools = self._get_tools_by_category(category)

        # Get the category description
        category_description = self.category_descriptions.get(category, "No description available.")

        return {
            'category': category.value,
            'description': category_description.strip(),
            'tools': tools
        }

    def get_system_overview(self) -> Dict[str, Any]:
        """
        Get a comprehensive overview of the system's capabilities.

        Returns:
            Dictionary containing the system overview
        """
        overview = """# Biodiversity Chat System Overview

This system provides comprehensive tools for analyzing and understanding biodiversity data. Here are the main capabilities:

## 1. Species Analysis
- Search and retrieve information about species
- Get species occurrence data
- Analyze species distribution patterns
- View species images
- Track yearly observation trends

## 2. Habitat Analysis
- Analyze habitat distribution
- Evaluate habitat connectivity
- Study habitat fragmentation
- Assess forest dependency
- Analyze topography and climate

## 3. Conservation Status
- Get endangered species information
- Analyze conservation status by country
- Track species by conservation category
- Monitor protected areas

## 4. Human Impact Analysis
- Calculate Human Coexistence Index (HCI)
- Analyze species-HCI correlations
- Study human modification effects
- Evaluate population density impacts

## 5. Geographic Analysis
- Country-specific analysis
- Protected area analysis
- Multi-country comparisons
- Geographic distribution mapping

## 6. Data Visualization
- Interactive maps
- Statistical charts
- Correlation plots
- Distribution visualizations
- Time series analysis

## Example Queries
1. "Show me endangered species in Kenya"
2. "What is the habitat distribution for African elephants?"
3. "Analyze the correlation between human activity and species presence"
4. "Show me species distribution in Serengeti National Park"
5. "What is the forest dependency of gorillas?"

For more specific information about any of these capabilities, you can ask:
- "Help me with species analysis"
- "Tell me about habitat analysis tools"
- "What conservation status functions are available?"
- "Show me human impact analysis capabilities"
- "What geographic analysis can you do?"
- "What visualization options are available?"

Would you like to know more about any specific aspect of the system?
"""

        return {
            'text': overview,
            'categories': self._get_categories()
        }

    def _get_categories(self) -> List[Dict[str, Any]]:
        """Get a list of all available categories."""
        categories = []
        for category in ToolCategory:
            categories.append({
                'name': category.name.replace('_', ' ').title(),
                'description': self._get_category_description(category)
            })
        return categories

    def _get_category_description(self, category: ToolCategory) -> str:
        """Get a description for a category."""
        return self.category_descriptions.get(category, "No description available.")

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

    def _get_tools_by_category(self, category: ToolCategory) -> List[Dict[str, Any]]:
        """
        Get all tools in a specific category.

        Args:
            category: The category to filter tools by

        Returns:
            List of tools in the specified category
        """
        tools_in_category = []
        for tool in self.tools:
            if tool.category == category:
                # Convert Tool object to dictionary
                tool_dict = {
                    'name': tool.name,
                    'description': tool.description.strip(),
                    'functions': tool.available_functions
                }
                tools_in_category.append(tool_dict)
        return tools_in_category
