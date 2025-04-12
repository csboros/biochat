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

        # Sort the available_functions dictionary by keys (function names) alphabetically
        available_functions = dict(sorted(available_functions.items()))

        return available_functions

    def render(self) -> Dict[str, Any]:
        """Render the function selector UI."""
        st.subheader("Function Selector")

        # Create function selection dropdown using only the function names
        function_names = list(self.available_functions.keys())

        if not function_names:
            st.warning("No functions available")
            return None

        selected_function_index = st.selectbox(
            "Select Function",
            range(len(function_names)),
            format_func=lambda x: function_names[x]  # Show only the function name
        )

        if selected_function_index is None:
            return None

        # Get the selected function details
        selected_function = function_names[selected_function_index]
        function_details = self.available_functions[selected_function]

        # Display the description below the selector
        st.write(f"**Description**: {function_details['description'].split('.')[0]}.")  # First sentence only

        # Create a form for parameter inputs
        with st.form("function_parameters"):
            st.markdown(f"### Parameters for {selected_function}")

            # Get function declaration from all declarations
            function_declaration = None
            function_info = {}
            try:
                # Get all declarations and find the one for the selected function
                all_declarations = self.function_handler.get_all_function_declarations()
                for decl in all_declarations:
                    if decl.to_dict().get('name') == selected_function:
                        function_declaration = decl
                        function_info = function_declaration.to_dict()
                        break
            except Exception as e:
                st.warning(f"Could not get function declaration: {str(e)}")

            # Create input fields for each parameter
            param_values = {}
            for param_name in function_details['inputs']:
                # Get parameter details from the function declaration
                param_info = {}
                if 'parameters' in function_info and 'properties' in function_info['parameters']:
                    param_info = function_info['parameters']['properties'].get(param_name, {})

                # Check if this parameter has enum values
                enum_values = param_info.get('enum', None)

                # For chart_type specifically, check if it's supposed to be an enum
                if param_name == 'chart_type' and not enum_values:
                    # Try to find enum values in a different location or hardcode common chart types
                    enum_values = ['bar', 'line', 'scatter', 'pie', 'map']
                    st.info(f"Using predefined chart types for {param_name}")

                if enum_values:
                    # Use a selectbox for enum parameters
                    st.write(f"{param_name} (Dropdown selection)")
                    param_values[param_name] = st.selectbox(
                        f"Select {param_name}",
                        options=enum_values,
                        key=f"enum_{param_name}"
                    )
                else:
                    # Use text input for non-enum parameters
                    param_values[param_name] = st.text_input(
                        f"{param_name} (required)",
                        key=f"text_{param_name}"
                    )

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