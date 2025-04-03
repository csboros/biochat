"""
Base class for all tools in the application.
Each tool contains its own handler(s) and function declarations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from vertexai.generative_models import FunctionDeclaration

class Tool(ABC):
    """Base class for all tools in the application."""

    @abstractmethod
    def get_handlers(self) -> Dict[str, Any]:
        """
        Returns the handlers for this tool.

        Returns:
            Dict[str, Any]: Dictionary mapping handler names to their instances
        """

    @abstractmethod
    def get_function_declarations(self) -> List[FunctionDeclaration]:
        """
        Returns the function declarations for this tool.

        Returns:
            List[FunctionDeclaration]: List of function declarations
        """

    @abstractmethod
    def get_function_mappings(self) -> Dict[str, Any]:
        """
        Returns the mapping of function names to their implementations.

        Returns:
            Dict[str, Any]: Dictionary mapping function names to their implementations
        """
