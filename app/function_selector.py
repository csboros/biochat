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
        # Create a simple layout
        st.subheader("Function Selector")

        # Create function selection dropdown using only the function names
        function_names = list(self.available_functions.keys())

        if not function_names:
            st.warning("No functions available")
            return None

        # Function dropdown: Use function names directly as options
        # The return value will be the selected name, not the index.
        selected_function_name = st.selectbox(
            "Select Function",
            options=function_names,  # Use names as options
            index=0, # Default to the first function if state is invalid
            key="function_selector_dropdown" # Keep the key for state
        )

        # No need to check for None index anymore, selectbox returns the name
        if not selected_function_name:
             # This case might happen if function_names is empty, though we check above.
             return None

        # Get the selected function details using the name
        function_details = self.available_functions[selected_function_name]

        # Get function declaration from all declarations
        function_declaration = None
        function_info = {}
        try:
            # Get all declarations and find the one for the selected function
            all_declarations = self.function_handler.get_all_function_declarations()
            for decl in all_declarations:
                # Use the selected_function_name for comparison
                if decl.to_dict().get('name') == selected_function_name:
                    function_declaration = decl
                    function_info = function_declaration.to_dict()
                    break
        except Exception as e:
            st.warning(f"Could not get function declaration: {str(e)}")

        # Display the description
        description = function_details['description'].split('.')[0] + '.'  # First sentence only
        st.markdown(f"**Description**: {description}")

        # Create a form for parameter inputs
        result = None

        # Use a unique key for the form based on the selected function name
        form_key = f"function_form_{selected_function_name}"

        with st.form(key=form_key):
            st.markdown("### Parameters")

            # Create input fields for each parameter
            param_values = {}
            for param_name in function_details['inputs']:
                # Get parameter details from the function declaration
                param_info = {}
                if 'parameters' in function_info and 'properties' in function_info['parameters']:
                    param_info = function_info['parameters']['properties'].get(param_name, {})

                # Check if this parameter is an array type
                is_array = param_info.get('type') == 'array'

                # Also check if the parameter name suggests it should be an array
                array_name_patterns = ['codes', 'names', 'ids', 'list', 'array']
                name_suggests_array = any(pattern in param_name.lower() for pattern in array_name_patterns)

                # Check if this parameter has enum values
                enum_values = param_info.get('enum', None)

                # For chart_type specifically, check if it's supposed to be an enum
                if param_name == 'chart_type' and not enum_values:
                    # Try to find enum values in a different location or hardcode common chart types
                    enum_values = ['bar', 'line', 'scatter', 'pie', 'map']
                    st.info(f"Using predefined chart types for {param_name}")

                # Create a unique key for each input field
                input_key = f"{form_key}_{param_name}"

                if is_array or name_suggests_array:
                    # Special handling for array parameters
                    st.write(f"{param_name} (Comma-separated list)")
                    # Add description if available
                    if 'description' in param_info:
                        st.info(param_info['description'])
                    param_values[param_name] = st.text_input(
                        f"Enter {param_name} as comma-separated values",
                        key=input_key
                    )
                elif enum_values:
                    # Use a selectbox for enum parameters
                    st.write(f"{param_name} (Dropdown selection)")
                    param_values[param_name] = st.selectbox(
                        f"Select {param_name}",
                        options=enum_values,
                        key=input_key
                    )
                else:
                    # Use text input for non-enum parameters
                    param_values[param_name] = st.text_input(
                        f"{param_name} (required)",
                        key=input_key
                    )

            # Add execute button
            submitted = st.form_submit_button("Execute Function")

            if submitted:
                # Filter out parameters that are not filled out
                param_values = {name: value for name, value in param_values.items() if value}

                # Process parameters based on their types
                processed_params = {}
                for param_name, value in param_values.items():
                    # Get parameter info again
                    param_info = {}
                    if 'parameters' in function_info and 'properties' in function_info['parameters']:
                        param_info = function_info['parameters']['properties'].get(param_name, {})

                    # Check if this parameter is an array type
                    is_array = param_info.get('type') == 'array'

                    # Also check if the parameter name suggests it should be an array
                    array_name_patterns = ['codes', 'names', 'ids', 'list', 'array']
                    name_suggests_array = any(pattern in param_name.lower() for pattern in array_name_patterns)

                    if is_array or name_suggests_array:
                        # Convert comma-separated string to list
                        processed_params[param_name] = [item.strip() for item in value.split(',') if item.strip()]
                    else:
                        # Keep other parameters as is
                        processed_params[param_name] = value

                result = {
                    # Use selected_function_name here
                    "function_name": selected_function_name,
                    "parameters": processed_params,
                    "function_details": function_details
                }

        return result

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