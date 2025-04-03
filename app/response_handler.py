"""Response handler for processing different types of data responses in the BioChat application."""

import logging
import time
import streamlit as st
from app.tools.visualization.chart_types import ChartType

# pylint: disable=no-member
# pylint: disable=broad-except
class ResponseHandler:
    """Handles processing of different types of responses from the chat interface."""

    def __init__(self):
        """Initialize the response handler."""
        self.logger = logging.getLogger("BioChat.ResponseHandler")

    def add_message_to_history(self, role, content):
        """Add a message to the session state history."""
        st.session_state.messages.append({"role": role, "content": content})

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
