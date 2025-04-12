"""Response handler for processing different types of data responses in the BioChat application."""

import logging
import time
import json
from typing import Dict, Any
from vertexai.preview.generative_models import (
    Part
)
import streamlit as st
from app.tools.visualization.chart_types import ChartType


# pylint: disable=no-member
# pylint: disable=broad-except
class ResponseHandler:
    """Handles processing of different types of responses from the chat interface."""

    def __init__(self):
        """Initialize the response handler."""
        self.logger = logging.getLogger("BioChat.ResponseHandler")
        self.function_handler = None
    def add_message_to_history(self, role, content):
        """Add a message to the session state history."""
        st.session_state.messages.append({"role": role, "content": content})

    def process_translation(self, data_response, parameters):
        """Processes and visualizes GeoJSON data for protected areas."""
        self.logger.debug(data_response)
        st.session_state.messages.append({
            "role": "assistant",
            "content": {"text": data_response["scientific_name"]}
        })

    def process_geojson_data(self, data_response, parameters):
        """Processes and visualizes GeoJSON data for protected areas."""
        st.session_state.messages.append({
            "role": "assistant",
            "content": {
                "chart_data": data_response,
                "type": ChartType.GEOJSON_MAP,
                "parameters": parameters
            }
        })

    def process_json_data(self, data_response, parameters):
        """
        Processes and visualizes JSON data for protected areas.

        Args:
            data_response (Union[dict, list, None]): The response data
            parameters (dict): Parameters for visualization
        """
        try:
            # Check if data_response is None or empty
            if not data_response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {"text": "No data available for this query."}
                })
                return

            # Handle dictionary responses
            if isinstance(data_response, dict):
                if data_response.get("error"):
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": {"text":
                                    str(data_response.get("error", "Unknown error occurred"))}
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": {
                            "chart_data": data_response,
                            "type": ChartType.JSON,
                            "parameters": parameters
                        }
                    })
            # Handle list responses
            elif isinstance(data_response, list):
                if not data_response:  # Empty list
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": {"text": "No results found for this query."}
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": {
                            "chart_data": data_response,
                            "type": ChartType.JSON,
                            "parameters": parameters
                        }
                    })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {"text": f"Unexpected data format received: {type(data_response)}"}
                })

        except Exception as e:  # pylint: disable=broad-except
            self.logger.error("Error processing JSON data: %s", str(e), exc_info=True)
            st.session_state.messages.append({
                "role": "assistant",
                "content": {"text": "An error occurred while processing the data."}
            })

    def process_occurrences_data(self, data_response, parameters):
        """
        Processes and visualizes occurrence data for species.

        Args:
            data_response (dict): The response data containing occurrence information
            parameters (dict): Parameters for chart generation including:
                - chart_type (str): Type of chart to generate
                - country_code (str, optional): Country code for geographical data

        Raises:
            TypeError: If data_response format is invalid for JSON normalization
            ValueError: If data processing or visualization fails
            AttributeError: If required parameters are missing or session state is not initialized
        """
        start_time = time.time()
        self.logger.debug("Processing occurrences data")
        try:
            # Data normalization timing
            norm_start = time.time()
            self.logger.debug("Data normalization took %.2f seconds",
                            time.time() - norm_start)
            if data_response.get("occurrences") is None \
                or len(data_response.get("occurrences")) == 0:
                self.logger.info("No data to display (took %.2f seconds)",
                               time.time() - start_time)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {"text": "No data to display"}
                })
            else:
                # Get chart type from parameters or use default
                chart_type_str = parameters.get("chart_type", "HEXAGON_MAP")
                try:
                    chart_type = ChartType.from_string(chart_type_str)
                except ValueError:
                    self.logger.warning("Invalid chart type %s, defaulting to HEXAGON_MAP", chart_type_str)
                    chart_type = ChartType.HEXAGON_MAP

                self.logger.debug("Parameters: %s", parameters)
                self.logger.debug("Chart Type: %s", chart_type)
                # Session state update timing
                session_start = time.time()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {
                        "chart_data": data_response,  # Pass the entire data_response
                        "type": chart_type,
                        "parameters": parameters
                    }
                })
                self.logger.debug("Session state update took %.2f seconds",
                                time.time() - session_start)
                self.logger.info("Successfully processed and displayed data in %.2f seconds",
                               time.time() - start_time)
        except TypeError as e:
            self.logger.error("Invalid data format (took %.2f seconds): %s",
                            time.time() - start_time, str(e), exc_info=True)
            raise
        except ValueError as e:
            self.logger.error("Data processing error (took %.2f seconds): %s",
                            time.time() - start_time, str(e), exc_info=True)
            raise
        except AttributeError as e:
            self.logger.error("Missing required attribute (took %.2f seconds): %s",
                            time.time() - start_time, str(e), exc_info=True)
            raise

    def process_indicator_data(self, data_response, parameters):
        """Processes and visualizes terrestrial human coexistence index data."""
        self.logger.debug("Processing terrestrial HCI data")
        try:
            chart_type = parameters.get("chart_type", ChartType.SCATTER_PLOT_3D)
            if data_response.get("error"):
                st.session_state.messages.append({"role": "assistant",
                                    "content": {"text": data_response.get("error")}})
            else:
                st.session_state.messages.append({"role": "assistant",
                                    "content": {"chart_data": data_response, "type": chart_type,
                                    "parameters": parameters}})
        except Exception as e:
            self.logger.error("Error processing occurrences data: %s",
                                str(e), exc_info=True)
            raise

    def process_yearly_observations(self, data_response, parameters):
        """Process yearly observation data and adds it to session state."""
        try:
            if not data_response.get("yearly_data") or len(data_response.get("yearly_data")) == 0:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {
                        "text": (f"No yearly observations found for "
                                f"{parameters.get('species_name', 'Unknown')}")
                    }
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {
                        "chart_data": data_response,
                        "type": ChartType.YEARLY_OBSERVATIONS,
                        "parameters": parameters
                    }
                })
        except Exception as e:
            self.logger.error("Error processing yearly observations: %s", str(e), exc_info=True)
            raise

    def process_endangered_species_hci_correlation(self, data_response, parameters):
        """Process and visualize correlation data between HCI and endangered species."""
        try:
            if not data_response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {"text": "No correlation data available."}
                })
                return

            st.session_state.messages.append({
                "role": "assistant",
                "content": {
                    "chart_data": data_response,
                    "type": ChartType.CORRELATION_SCATTER,
                    "parameters": parameters
                }
            })
        except Exception as e:
            self.logger.error("Error processing correlation data: %s", str(e), exc_info=True)
            st.session_state.messages.append({
                "role": "assistant",
                "content": {"text": f"Error processing correlation data: {str(e)}"}
            })

    def process_species_images(self, images_data, parameters):
        """Display species images in the Streamlit interface."""
        if images_data["image_count"] > 0:
            st.session_state.messages.append({
                "role": "assistant",
                "content": {
                    "chart_data": images_data,
                    "type": ChartType.IMAGES,
                    "parameters": parameters
                }
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": {"text": "No images found for this species."}
            })

    def process_endangered_species(self, data_response, parameters):
        """Process endangered species data and visualize using specified chart type."""
        try:
            chart_type = parameters.get("chart_type", ChartType.FORCE_DIRECTED_GRAPH)
            if not data_response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {"text": "No endangered species data available."}
                })
                return

            st.session_state.messages.append({
                "role": "assistant",
                "content": {
                    "chart_data": data_response,
                    "type": chart_type,
                    "parameters": parameters
                }
            })
        except Exception as e:
            self.logger.error("Error processing endangered species data: %s", str(e), exc_info=True)
            st.session_state.messages.append({
                "role": "assistant",
                "content": {"text": f"Error processing endangered species data: {str(e)}"}
            })

    def process_endangered_species_by_country(self, data_response, parameters):
        """Process and visualize endangered species occurrence data for a specific country."""
        try:
            if not data_response or data_response.get("error"):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {
                        "text": data_response.get("error", "No endangered species data available.")
                    }
                })
                return

            st.session_state.messages.append({
                "role": "assistant",
                "content": {
                    "chart_data": data_response,
                    "type": ChartType.OCCURRENCE_MAP,
                    "parameters": parameters
                }
            })

        except Exception as e:
            self.logger.error(
                "Error processing endangered species by country data: %s",
                str(e),
                exc_info=True
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": {
                    "text": f"Error processing endangered species data: {str(e)}"
                }
            })

    def process_species_hci_correlation(self, data_response, parameters):
        """Process and visualize species-HCI correlation data."""
        try:
            if "error" in data_response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {"text": data_response["error"]}
                })
                return

            st.session_state.messages.append({
                "role": "assistant",
                "content": {
                    "chart_data": data_response,
                    "type": ChartType.SPECIES_HCI_CORRELATION,
                    "parameters": parameters
                }
            })

        except Exception as e:
            self.logger.error("Error processing correlation data: %s", str(e), exc_info=True)
            st.session_state.messages.append({
                "role": "assistant",
                "content": {"text": f"Error processing correlation data: {str(e)}"}
            })

    def process_species_hci_correlation_by_status(self, data_response, parameters):
        """Process and visualize species-HCI correlation data filtered by conservation status."""
        try:
            if "error" in data_response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {"text": data_response["error"]}
                })
                return

            st.session_state.messages.append({
                "role": "assistant",
                "content": {
                    "chart_data": data_response,
                    "type": ChartType.SPECIES_HCI_CORRELATION,
                    "parameters": parameters
                }
            })

        except Exception as e:
            self.logger.error(
                "Error processing correlation by status data: %s",
                str(e),
                exc_info=True
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": {"text": f"Error processing correlation data: {str(e)}"}
            })

    def process_species_shared_habitat(self, data_response, parameters):
        """Process and visualize species shared habitat data."""
        try:
            if "error" in data_response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {"text": data_response["error"]}
                })
                return

            st.session_state.messages.append({
                "role": "assistant",
                "content": {
                    "chart_data": data_response,
                    "type": ChartType.SPECIES_SHARED_HABITAT,
                    "parameters": parameters
                }
            })

        except Exception as e:
            self.logger.error(
                "Error processing species shared habitat data: %s",
                str(e),
                exc_info=True
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": {"text": f"Error processing species shared habitat data: {str(e)}"}
            })

    def process_species_translation(self, data_response, parameters):
        """Process and display species information."""
        try:
            # Parse the JSON string if it's a string
            if isinstance(data_response, str):
                data = json.loads(data_response)
            else:
                data = data_response

            # Extract the scientific name or error message
            if "scientific_name" in data:
                message = data["scientific_name"]
            elif "error" in data:
                message = data["error"]
            else:
                message = "Could not find scientific name"

            st.session_state.messages.append({
                "role": "assistant",
                "content": {"text": message}
            })
        except json.JSONDecodeError:
            st.session_state.messages.append({
                "role": "assistant",
                "content": {"text": str(data_response)}
            })

    def process_species_info(self, data_response, parameters):
        """Process and display species information."""
        st.session_state.messages.append({
            "role": "assistant",
            "content": {"text": data_response}
        })

    def process_species_correlation_analysis(self, data_response, parameters):
        """Process and display species correlation analysis results."""
        try:
            if isinstance(data_response, str):
                self.add_message_to_history("assistant", {"text": data_response})
                return

            if 'correlation_data' in data_response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {
                        "chart_data": data_response['correlation_data'],
                        "type": ChartType.SPECIES_HCI_CORRELATION,
                        "parameters": parameters
                    }
                })
            if 'analysis' in data_response:
                self.add_message_to_history("assistant", {"text": data_response['analysis']})

        except (ValueError, TypeError, KeyError) as e:
            self.logger.error("Error processing correlation analysis: %s", str(e), exc_info=True)
            self.add_message_to_history(
                "assistant",
                {"text": f"Error processing correlation analysis: {str(e)}"}
            )

    def process_correlation_result(self, data_response, parameters, chart_type):
        """Process and display species human modification correlation analysis results."""
        try:
            if isinstance(data_response, str):
                self.add_message_to_history("assistant", {"text": data_response})
                return

            if 'observations' in data_response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {
                        "chart_data": data_response,
                        "type": chart_type,
                        "parameters": parameters
                    }
                })

            if 'analysis' in data_response:
                self.add_message_to_history("assistant", {"text": data_response['analysis']})

        except (ValueError, TypeError, KeyError) as e:
            self.logger.error("Error processing correlation analysis: %s", str(e), exc_info=True)
            self.add_message_to_history(
                "assistant",
                {"text": f"Error processing correlation analysis: {str(e)}"}
            )

    def process_habitat_analysis(self, data_response, parameters):
        """Process and visualize habitat analysis results."""
        try:
            if not data_response or data_response.get("error"):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {
                        "text": data_response.get("error", "No habitat analysis data available.")
                    }
                })
                return

            st.session_state.messages.append({
                "role": "assistant",
                "content": {
                    "chart_data": data_response,
                    "type": ChartType.HABITAT_ANALYSIS,
                    "parameters": parameters
                }
            })

            if data_response.get("analysis"):
                self.add_message_to_history("assistant", {"text": data_response["analysis"]})

        except Exception as e:
            self.logger.error("Error processing habitat analysis data: %s", str(e), exc_info=True)
            st.session_state.messages.append({
                "role": "assistant",
                "content": {"text": f"Error processing habitat analysis data: {str(e)}"}
            })

    def process_topography_analysis(self, data_response, parameters):
        """Process and visualize topography analysis results."""
        try:
            if not data_response or data_response.get("error"):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {
                        "text": data_response.get("error", "No topography analysis data available.")
                    }
                })
                return

            st.session_state.messages.append({
                "role": "assistant",
                "content": {
                    "chart_data": data_response,
                    "type": ChartType.TOPOGRAPHY_ANALYSIS,
                    "parameters": parameters
                }
            })

            if data_response.get("analysis"):
                self.add_message_to_history("assistant", {"text": data_response["analysis"]})

        except Exception as e:
            self.logger.error("Error processing topography analysis data: %s", str(e),
                              exc_info=True)
            st.session_state.messages.append({
                "role": "assistant",
                "content": {"text": f"Error processing topography analysis data: {str(e)}"}
            })

    def process_climate_analysis(self, data_response, parameters):
        """Process and visualize climate analysis results."""
        try:
            if not data_response or data_response.get("error"):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {
                        "text": data_response.get("error", "No climate analysis data available.")
                    }
                })
                return

            st.session_state.messages.append({
                "role": "assistant",
                "content": {
                    "chart_data": data_response,
                    "type": ChartType.CLIMATE_ANALYSIS,
                    "parameters": parameters
                }
            })

            if data_response.get("analysis"):
                self.add_message_to_history("assistant", {"text": data_response["analysis"]})

        except Exception as e:
            self.logger.error("Error processing climate analysis data: %s", str(e), exc_info=True)
            st.session_state.messages.append({
                "role": "assistant",
                "content": {"text": f"Error processing climate analysis data: {str(e)}"}
            })

    def handle_function_call(self, call):
        """
        Helper method to handle different function call types.

        Args:
            call (dict): The function call details
        """
        try:
            handlers = {
                'help': lambda c: self.handle_help_command({
                    'type': 'category' if c.get('params', {}).get('category') else 'general',
                'category': c.get('params', {}).get('category'),
                'tool': c.get('params', {}).get('tool'),
                'function': c.get('params', {}).get('function')
            }),
            'get_yearly_occurrences': lambda c: (
                self.process_yearly_observations(
                    c['response'], c['params']
                )
            ),
            'get_occurrences': lambda c: self.process_occurrences_data(c['response'], c['params']),
            'get_species_occurrences_in_protected_area': lambda c: (
                self.process_occurrences_data(c['response'], c['params'])
            ),
            'get_protected_areas_geojson': lambda c: (
                self.process_geojson_data(c['response'], c['params'])
            ),
            'get_endangered_species_in_protected_area': lambda c: (
                self.process_json_data(c['response'], c['params'])
            ),
            'endangered_species_hci_correlation': lambda c: (
                self.process_endangered_species_hci_correlation(
                    c['response'], c['params']
                )
            ),
            'get_species_images': lambda c: (
                self.process_species_images(c['response'], c['params'])
            ),
            'read_terrestrial_hci': lambda c: (
                self.process_indicator_data(c['response'], c['params'])
            ),
            'read_population_density': lambda c: (
                self.process_indicator_data(c['response'], c['params'])
            ),
            'endangered_species_for_country': lambda c: (
                self.process_endangered_species(c['response'], c['params'])
            ),
            'endangered_species_for_countries': lambda c: (
                self.process_endangered_species(c['response'], c['params'])
            ),
            'endangered_families_for_order': lambda c: (
                self.process_endangered_species(c['response'], c['params'])
            ),
            'endangered_species_for_family': lambda c: (
                self.process_endangered_species(c['response'], c['params'])
            ),
            'get_endangered_species_by_country': lambda c: (
                self.process_endangered_species_by_country(
                    c['response'], c['params']
                )
            ),
            'get_species_hci_correlation': lambda c: (
                self.process_species_hci_correlation(
                    c['response'], c['params']
                )
            ),
            'get_species_hci_correlation_by_status': lambda c: (
                self.process_species_hci_correlation_by_status(
                    c['response'], c['params']
                )
            ),
            'get_species_shared_habitat': lambda c: (
                self.process_species_shared_habitat(
                    c['response'], c['params']
                )
            ),
            'analyze_species_correlations': lambda c: (
                self.process_species_correlation_analysis(
                    c['response'], c['params']
                )
            ),
            'calculate_species_forest_correlation': lambda c: (
                self.process_correlation_result(
                    c['response'], c['params'], ChartType.FOREST_CORRELATION
                )
            ),
            'calculate_species_humanmod_correlation': lambda c: (
                self.process_correlation_result(
                    c['response'], c['params'], ChartType.HUMANMOD_CORRELATION
                )
            ),
            'analyze_habitat_distribution': lambda c: (
                self.process_habitat_analysis(c['response'], c['params'])
            ),
            'analyze_topography': lambda c: (
                self.process_topography_analysis(c['response'], c['params'])
            ),
            'analyze_climate': lambda c: (
                    self.process_climate_analysis(c['response'], c['params'])
                )
            }

            if call['name'] in handlers:
                handlers[call['name']](call)
                return None

            # Functions that need to return results to Gemini
            chain_functions = (
                'translate_to_scientific_name',
                'google_search',
                'get_species_info'  # Add get_species_info to chain functions
            )
            print(call['name'])
            print(call['response'])
            if call['name'] in chain_functions:
                # Prepare response for Gemini based on type
                if isinstance(call['response'], list):
                    # Convert list to dictionary for Gemini
                    return Part.from_function_response(
                        name=call['name'],
                        response={"results": call['response']},
                    )
                elif isinstance(call['response'], str):
                    # Wrap string in dictionary
                    return Part.from_function_response(
                        name=call['name'],
                        response={"result": call['response']},
                    )
                else:
                    # Use response as is if it's already a dictionary
                    return Part.from_function_response(
                        name=call['name'],
                        response=call['response'],
                    )

            # Handle simple text response functions
            simple_text_functions = (
                'endangered_species_by_conservation_status',
                'endangered_orders_for_class',
                'endangered_classes_for_kingdom'
            )

            if call['name'] in simple_text_functions:
                if isinstance(call['response'], str):
                    self.add_message_to_history(
                        "assistant", {"text": call['response']}
                    )
                else:
                    self.add_message_to_history(
                        "assistant", {"text": str(call['response'])}
                    )
                return None

            # Default handling for other functions
            if isinstance(call['response'], list):
                # Convert list to dictionary
                return Part.from_function_response(
                    name=call['name'],
                    response={"results": call['response']},
                )
            elif isinstance(call['response'], str):
                # Wrap string in dictionary
                return Part.from_function_response(
                    name=call['name'],
                    response={"result": call['response']},
                )
            else:
                # Use response as is if it's already a dictionary
                return Part.from_function_response(
                    name=call['name'],
                    response=call['response'],
                )
        except Exception as e:
            self.logger.error("Error in function call handling: %s", str(e), exc_info=True)
            self.add_message_to_history(
                "assistant",
                {"text": f"Error processing request: {str(e)}"}
            )
            return None


    def handle_help_command(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle help-related commands and queries.
        """
        try:
            help_type = arguments.get('type', 'general')
            result = {"success": False, "error": "Invalid request"}

            if help_type == 'general':
                overview = self.get_system_overview()
                self.add_message_to_history("assistant", {"text": overview})
                result = {"success": True, "data": {"text": overview}}
            elif help_type in ['category', 'tool', 'function']:
                # Validate required parameters
                if help_type == 'function' and (not arguments.get('tool') or not arguments.get('function')):
                    result = {"success": False, "error": "Both tool and function names are required"}
                elif help_type in ['category', 'tool'] and not arguments.get(help_type):
                    result = {"success": False, "error": f"{help_type.capitalize()} name is required"}
                else:
                    # Get help info from function handler
                    help_info = self.function_handler.help_handler.handle_help_command(arguments)
                    if help_info.get('success'):
                        self.add_message_to_history("assistant", {"text": help_info['data']['text']})
                    result = help_info

            return result

        except Exception as e:
            self.logger.error("Error handling help command: %s", str(e), exc_info=True)
            error_message = f"Error processing help request: {str(e)}"
            self.add_message_to_history("assistant", {"text": error_message})
            return {"success": False, "error": error_message}

    def get_system_overview(self) -> str:
        """
        Get a comprehensive overview of the system's capabilities.

        Returns:
            str: A formatted string describing the system's capabilities
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
        return overview

