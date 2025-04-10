"""
This module provides a FunctionSelector class for creating a dynamic UI
to select and execute functions with parameters in a Streamlit application.
"""
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass
import streamlit as st

@dataclass
class FunctionParameter:
    """Represents a parameter for a function."""
    name: str
    param_type: str
    description: str
    required: bool
    enum_values: Optional[List[str]] = None


# pylint: disable=no-member
# pylint: disable=broad-except
class FunctionSelector:
    """Creates a dynamic UI for selecting and executing functions with parameters."""

    def __init__(self, function_handler):
        """Initialize the FunctionSelector."""
        self.logger = logging.getLogger(__name__)
        self.function_handler = function_handler
        self.available_functions = self._get_available_functions()

    def _get_available_functions(self):
        """Get available functions from the function handler."""
        available_functions = {}

        try:
            # Get all function declarations from the function handler
            function_declarations = self.function_handler.get_all_function_declarations()
            function_mappings = self.function_handler.get_all_function_mappings()

            # Convert function declarations to the format expected by the UI
            for declaration in function_declarations:
                # Access the name from the raw function declaration or from the to_dict() method
                function_info = declaration.to_dict()
                function_name = function_info.get('name')

                if not function_name:
                    self.logger.warning("Function declaration missing name: %s", declaration)
                    continue

                # Skip help and classification functions as they're internal
                if function_name in ['help', 'classify_help_category']:
                    continue

                # Check if we have a mapping for this function
                if function_name in function_mappings:
                    # Extract parameter information
                    parameters = []
                    # Get parameters from the function declaration dictionary
                    if 'parameters' in function_info and 'properties' in function_info['parameters']:
                        parameters = list(function_info['parameters']['properties'].keys())

                    # Create function details
                    available_functions[function_name] = {
                        'description': function_info.get('description', ''),
                        'handler': function_name,
                        'inputs': parameters,
                        'aliases': [function_name]  # Use function name as alias
                    }
        except Exception as e:
            self.logger.error("Error getting available functions: %s", str(e), exc_info=True)

        return available_functions

    def render(self) -> Dict[str, Any]:
        """Render the function selector UI."""
        st.subheader("Function Selector")

        # Create function selection dropdown using the mappings
        function_names = list(self.available_functions.keys())

        if not function_names:
            st.warning("No functions available")
            return None

        selected_function_index = st.selectbox(
            "Select Function",
            range(len(function_names)),
            format_func=lambda x: f"{function_names[x]} - {self.available_functions[function_names[x]]['description']}"
        )

        if selected_function_index is None:
            return None

        # Get the selected function details
        selected_function = function_names[selected_function_index]
        function_details = self.available_functions[selected_function]

        # Create a form for parameter inputs
        with st.form("function_parameters"):
            st.markdown(f"### Parameters for {selected_function}")

            # Create input fields for each parameter
            param_values = {}
            for param_name in function_details['inputs']:
                # For now, treat all parameters as required string inputs
                param_values[param_name] = st.text_input(f"{param_name} (required)")

            # Add execute button
            submitted = st.form_submit_button("Execute Function")

            if submitted:
                # Filter out parameters that are not filled out
                param_values = {name: value for name, value in param_values.items() if value}

                return {
                    "function_name": function_details['handler'],
                    "parameters": param_values,
                    "function_details": function_details
                }

        return None

    def execute_function(self, function_name: Any, parameters: Dict[str, Any]) -> Any:
        """Execute the selected function with the provided parameters."""
        try:
            # Get the function from the function handler
            function_mappings = self.function_handler.get_all_function_mappings()

            if function_name in function_mappings:
                self.logger.info(f"Executing function: {function_name}")
                return function_mappings[function_name](parameters)
            else:
                st.error(f"Function '{function_name}' not found")
                return None
        except Exception as e:
            st.error(f"Error executing function: {str(e)}")
            return None