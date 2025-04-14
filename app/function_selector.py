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
        self.logger.info("Starting _get_available_functions")
        available_functions = {}

        # --- Define functions to exclude from the UI ---
        excluded_functions = {
            'help',
            'classify_help_category',
            'translate_to_common_name',
            'translate_to_scientific_name',
            'google_search',
            'get_species_info',
            'get_species_images',
            'endangered_species_for_countries'
        }
        # --- End of definition ---

        try:
            # Get all function declarations from the function handler
            function_declarations = self.function_handler.get_all_function_declarations()
            function_mappings = self.function_handler.get_all_function_mappings()

            # --- Log the total number of declarations and mapping keys received ---
            self.logger.debug("Received %s declarations.", len(function_declarations))
            self.logger.debug("Received %s mappings. Keys: %s", len(function_mappings), list(function_mappings.keys()))
            # --- End of addition ---

            # Convert function declarations to the format expected by the UI
            for i, declaration in enumerate(function_declarations):
                function_info = {}
                function_name = None
                try:
                    # Access the name from the raw function declaration or from the to_dict() method
                    function_info = declaration.to_dict()
                    function_name = function_info.get('name')
                    # --- Log each function name being processed ---
                    self.logger.debug("Processing declaration %s: name='%s'", i+1, function_name)
                    # --- End of addition ---

                except Exception as e:
                    # --- Log errors during declaration processing ---
                    self.logger.error("Error processing declaration %s: %s. Error: %s", i+1, declaration, e, exc_info=True)
                    # --- End of addition ---
                    continue # Skip this declaration if conversion fails

                if not function_name:
                    self.logger.warning("Declaration %s missing name: %s", i+1, declaration)
                    continue

                # --- Updated check to skip excluded functions ---
                if function_name in excluded_functions:
                    self.logger.debug("Skipping excluded function: '%s'", function_name)
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
                        'handler': function_name, # Keep handler name for potential execution logic
                        'inputs': parameters,
                        'aliases': [function_name]  # Use function name as alias
                    }
                else:
                    # --- Log missing mapping ---
                    self.logger.warning("No mapping found for declared function: '%s'. Skipping.", function_name)
                    # --- End of addition ---

        except Exception as e:
            self.logger.error("Error getting available functions: %s", str(e), exc_info=True)

        # Sort the available_functions dictionary by keys (function names) alphabetically
        available_functions = dict(sorted(available_functions.items()))

        # --- Log the final list of available function keys ---
        self.logger.info("Finished _get_available_functions. Available keys: %s", list(available_functions.keys()))
        # --- End of addition ---

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
            st.warning("Could not get function declaration: %s", str(e))

        # Display the description
        description = function_details['description'].split('.')[0] + '.'  # First sentence only
        st.markdown(f"**Description**: {description}")

        # Create a form for parameter inputs
        result = None

        # Use a unique key for the form based on the selected function name
        form_key = f"function_form_{selected_function_name}"

        with st.form(key=form_key):
            st.markdown("### Parameters")

            required_params = []
            if function_info and 'parameters' in function_info and 'required' in function_info['parameters']:
                required_params = function_info['parameters'].get('required', [])

            # --- Define a placeholder for optional enums ---
            placeholder = "-- Select --"
            # --- End of addition ---

            param_values = {}
            for param_name in function_details['inputs']:
                param_info = {}
                if function_info and 'parameters' in function_info and 'properties' in function_info['parameters']:
                    param_info = function_info['parameters']['properties'].get(param_name, {})

                is_required = param_name in required_params
                required_label_suffix = " (required)" if is_required else ""

                # --- Check for country code format hints ---
                format_hint = ""
                # Check only parameters related to country codes
                if 'country_code' in param_name.lower():
                    pattern = param_info.get('pattern', '')
                    description_lower = param_info.get('description', '').lower()

                    if pattern == '^[A-Z]{2}$' or '2-letter' in description_lower or 'alpha-2' in description_lower:
                        format_hint = "(expects 2-letter ISO codes)"
                    elif pattern == '^[A-Z]{3}$' or '3-letter' in description_lower or 'alpha-3' in description_lower:
                        format_hint = "(expects 3-letter ISO codes)"

                # Combine original description and format hint
                original_description = param_info.get('description', '')
                display_description = f"{original_description.strip()} {format_hint}".strip()
                # --- End of country code format check ---

                is_array = param_info.get('type') == 'array'
                array_name_patterns = ['codes', 'names', 'ids', 'list', 'array']
                name_suggests_array = any(pattern in param_name.lower() for pattern in array_name_patterns)
                enum_values = param_info.get('enum', None)

                if param_name == 'chart_type' and not enum_values:
                    enum_values = ['bar', 'line', 'scatter', 'pie', 'map']
                    # --- Update description display ---
                    if not display_description: # Add info only if no other description exists
                         st.info(f"Using predefined chart types for {param_name}")
                    # --- End of update ---

                input_key = f"{form_key}_{param_name}"

                if is_array or name_suggests_array:
                    st.write(f"{param_name} (Comma-separated list){required_label_suffix}")
                    if display_description:
                        st.info(display_description)
                    param_values[param_name] = st.text_input(
                        f"Enter {param_name} as comma-separated values",
                        key=input_key
                    )
                elif enum_values:
                    st.write(f"{param_name} (Dropdown selection){required_label_suffix}")
                    if display_description:
                        st.info(display_description)

                    # --- Modify options and default index for selectbox ---
                    display_options = [placeholder] + enum_values
                    # Default to placeholder (index 0) if not required,
                    # else default to the first actual option (index 1)
                    default_index = 0 if not is_required else 1

                    param_values[param_name] = st.selectbox(
                        f"Select {param_name}",
                        options=display_options,
                        index=default_index, # Use calculated default index
                        key=input_key
                    )
                    # --- End of modification ---

                else:
                    # --- Update description display ---
                    # Display description/hint above the input field
                    if display_description:
                        st.info(display_description)
                    # --- End of update ---
                    param_values[param_name] = st.text_input(
                        f"{param_name}{required_label_suffix}", # Label includes required suffix here
                        key=input_key
                    )

            submitted = st.form_submit_button("Execute Function")

            if submitted:
                # --- Modify filtering to exclude placeholder ---
                # Initial filter for non-empty values
                filtered_values = {name: value for name, value in param_values.items() if value}

                processed_params = {}
                for param_name, value in filtered_values.items():
                    # Get parameter info again to check if it was an enum
                    param_info = {}
                    if function_info and 'parameters' in function_info and 'properties' in function_info['parameters']:
                         param_info = function_info['parameters']['properties'].get(param_name, {})
                    is_enum = 'enum' in param_info

                    # Skip if it's an enum and the value is the placeholder
                    if is_enum and value == placeholder:
                        continue # Don't include this parameter

                    # Check if this parameter is an array type
                    is_array = param_info.get('type') == 'array'
                    array_name_patterns = ['codes', 'names', 'ids', 'list', 'array']
                    name_suggests_array = any(pattern in param_name.lower() for pattern in array_name_patterns)

                    if is_array or name_suggests_array:
                        # Convert comma-separated string to list
                        processed_params[param_name] = [item.strip() for item in value.split(',') if item.strip()]
                    else:
                        # Keep other parameters as is
                        processed_params[param_name] = value
                # --- End of modification ---

                result = {
                    "function_name": selected_function_name,
                    "parameters": processed_params, # Use the newly processed params
                    "function_details": function_details
                }

        return result

    def execute_function(self, function_name: Any, parameters: Dict[str, Any]) -> Any:
        """Execute the selected function with the provided parameters."""
        try:
            # Get the function from the function handler
            function_mappings = self.function_handler.get_all_function_mappings()

            if function_name in function_mappings:
                self.logger.info("Executing function: %s", function_name)
                return function_mappings[function_name](parameters)
            st.error("Function '%s' not found", function_name)
            return None
        except Exception as e:
            st.error(f"Error executing function: {str(e)}")
            return None
