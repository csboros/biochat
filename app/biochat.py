"""
    A Streamlit application for exploring biodiversity data using Vertex AI's Gemini model.

    This application provides an interactive chat interface where users can query information
    about endangered species, conservation status, and global biodiversity patterns. It
    integrates with various data sources and visualization tools to present biodiversity
    information.

"""
import logging
import time
import os
from typing import Dict, Any, List
import google.cloud.aiplatform as vertexai
from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
    ResponseValidationError
)
from google.api_core import exceptions as google_exceptions
import streamlit as st
from app.utils.logging_config import setup_logging
from app.tools.function_handler import FunctionHandler
from app.tools.visualization.chart_handler import ChartHandler
from app.tools.visualization.chart_types import ChartType
from app.tools.message_bus import message_bus
from app.response_handler import ResponseHandler


# Setup logging configuration at application startup
setup_logging()

# pylint: disable=no-member
# pylint: disable=broad-except
class BioChat:
    """
    Main application class for the Biodiversity Chat interface.

    Manages the Streamlit interface, Vertex AI model integration, and chat functionality.
    Handles user interactions, function calls, and visualization of biodiversity data.
    """

    SYSTEM_MESSAGE = """You are a biodiversity expert assistant. Your primary role is to help users
    understand endangered species, their conservation status, and global biodiversity patterns.

 Key definitions and concepts:
- HCI (Human Coexistence Index): A measure of human impact on an area, with higher values indicating greater human presence and activity
- Species-HCI correlation: In our analysis, this is calculated as:
  1. For each species, we look at grid cells where it occurs
  2. We calculate the Pearson correlation coefficient between:
     * Number of individuals of the species in each cell
     * HCI value of that cell
  3. Interpretation:
     * Positive correlation (+1): More individuals in high-HCI areas
     * Negative correlation (-1): More individuals in low-HCI areas
     * Zero correlation (0): No clear relationship
  4. Additional metrics provided:
     * avg_hci: Mean HCI value across all cells where species occurs
     * number_of_grid_cells: Number of cells where species is found
     * total_individuals: Total count of individuals observed

    IMPORTANT: For EVERY user query:

    1. For any question you can't answer directly, use the appropriate tool immediately - don't announce your intention to use it
    2. For general knowledge questions, use google_search directly
    3. Always use the most specific tool available for the task
    4. If multiple tools might help, use them in sequence without announcing each step

    5. For protected area queries (e.g. "What species live in X park/reserve?"):
       - ALWAYS use get_endangered_species_in_protected_area
       Example: "What endangered species live in Serengeti?" → use get_endangered_species_in_protected_area("Serengeti National Park")

    6. For SINGLE country queries:
       - Use endangered_species_for_country with TWO-letter country code
       Example: 'Show endangered species in Kenya' → use endangered_species_for_country with 'KE'

    7. For MULTIPLE country comparisons:
       - Use endangered_species_for_countries with list of TWO-letter country codes
       Example: 'Compare endangered species between Kenya and Tanzania' → use endangered_species_for_countries with ['KE', 'TZ']

    8. For species information:
       - First use translate_to_scientific_name for common names
       - Then use get_species_info with the result

    9. For species distribution:
       - Use get_occurrences for location data
       - Use get_yearly_occurrences for temporal trends

    Remember: Always use the appropriate function based on whether the query is about a single country or multiple countries.

    IMPORTANT: You must use the provided functions for any queries about species, countries, or biodiversity data.
    If no function matches the user's query or if you need additional general information, use the google_search function
    to find relevant information. Do not rely on your general knowledge alone - either use the provided functions or
    perform a Google search.

    Please do not announce your intention to use the functions, simple use it.
    """

    # pylint: disable=no-member
    @st.cache_resource(show_spinner=False)
    def initialize_app_resources(_self):  # pylint: disable=no-self-argument
        """
        Initializes and caches all application resources required
        for the Biodiversity Chat interface.

        This method handles the initialization of Vertex AI, function handlers, and the chat model.
        Resources are cached using Streamlit's cache_resource to persist across reruns.

        Returns
        -------
        dict
            Dictionary containing initialized resources:
                handler : FunctionHandler
                    Handles function declarations and executions
                chart_handler : ChartHandler
                    Manages chart generation and visualization
                model : GenerativeModel
                    Initialized Gemini model instance
                chat : ChatSession
                    Active chat session with system context

        Raises
        ------
        google.api_core.exceptions.ResourceExhausted
            If API quota is exceeded
        google.api_core.exceptions.TooManyRequests
            If request rate limit is exceeded
        Exception
            For other initialization failures
        """
        start_time = time.time()
        logger = logging.getLogger("BioChat")
        logger.info("Starting app resources initialization")
        try:
            # Add timeout to Vertex AI initialization
            vertex_start = time.time()
            with st.spinner("Initializing Vertex AI..."):
                vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location="us-central1")
                logger.info("Vertex AI initialized successfully in %.2f seconds",
                          time.time() - vertex_start)

            # Initialize handlers
            handlers_start = time.time()
            with st.spinner("Setting up handlers..."):
                function_handler = FunctionHandler()
                chart_handler = ChartHandler()
                logger.info("Handlers initialized in %.2f seconds",
                          time.time() - handlers_start)

            # Initialize model with timeout
            model_start = time.time()
            with st.spinner("Loading Gemini model..."):
                tools = Tool(function_declarations=function_handler.get_all_function_declarations())
                model = GenerativeModel(
                    "gemini-2.0-flash",
                    generation_config=GenerationConfig(temperature=0),
                    tools=[tools],
                    system_instruction=_self.SYSTEM_MESSAGE
                )
                logger.info("Model loaded successfully in %.2f seconds",
                          time.time() - model_start)

            # Initialize chat
            chat_start = time.time()
            with st.spinner("Setting up chat..."):
                chat = model.start_chat(response_validation=False)
                chat.send_message(_self.SYSTEM_MESSAGE)
                logger.info("Chat session initialized in %.2f seconds",
                          time.time() - chat_start)

            total_time = time.time() - start_time
            logger.info("Total initialization completed in %.2f seconds", total_time)
            return {
                'handler': function_handler,
                'chart_handler': chart_handler,
                'model': model,
                'chat': chat
            }
        except (google_exceptions.ResourceExhausted,
                google_exceptions.TooManyRequests) as e:
            logger.error("API quota exceeded (took %.2f seconds): %s",
                        time.time() - start_time, str(e), exc_info=True)
            raise
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error during initialization (took %.2f seconds): %s",
                        time.time() - start_time, str(e), exc_info=True)
            raise

    def __init__(self):
        """
        Initializes the BioChat Application.

        Raises:
            google.api_core.exceptions.ResourceExhausted: If API quota is exceeded
            ValueError: If initialization parameters are invalid
            RuntimeError: If required resources cannot be initialized
        """
        # Set page config to wide mode
        st.set_page_config(
            page_title="Biodiversity Chat",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        self.logger = logging.getLogger("BioChat." + self.__class__.__name__)
        try:
            resources = self.initialize_app_resources()
            self.function_handler = resources['handler'].get_all_function_mappings()
            self.function_declarations = resources['handler'].get_all_function_declarations()
            self.chart_handler = resources['chart_handler']
            self.chat = resources['chat']
            self.initialize_session_state()
            self.response_handler = ResponseHandler()

        except (google_exceptions.ResourceExhausted, google_exceptions.TooManyRequests) as e:
            self.logger.error("API quota exceeded: %s", str(e), exc_info=True)
            st.error("API quota has been exceeded. Please wait a few minutes and try again.")
        except Exception as e:  # pylint: disable=broad-except
            # Justified as initialization failure should catch all possible errors
            self.logger.error("Error during initialization: %s", str(e), exc_info=True)
            raise

    def initialize_session_state(self):
        """
        Initializes the Streamlit session state and sets up the chat message history.
        """
        if "messages" not in st.session_state:
            st.session_state.messages = []
            self.logger.debug("Initialized empty messages in session state")

        # Only subscribe to status updates if needed
        if "status_subscribed" not in st.session_state:
            message_bus.subscribe("status_update", self._handle_status_update)
            st.session_state.status_subscribed = True

    def run(self):
        """
        Main execution method for the Streamlit application.

        Handles the chat interface setup, message history display, and user input processing.
        Most error handling is delegated to process_assistant_response method.
        """
        self.logger.info("Starting BioChat Application")
        try:
            # Main UI setup
            st.title("Biodiversity Chat")
            # Example queries section
            st.write("[See example queries]"
                    "(https://github.com/csboros/biochat/blob/main/prompts.md)")
            # Core functionality
            self.handle_user_input()
            self.display_message_history()

        except ResponseValidationError as e:
            # response has been blocked by safety filters, show the message history
            # and ask for a different prompt
            st.session_state.messages.pop()
            self.logger.error("ResponseValidationError: %s", str(e), exc_info=True)
            st.error("No results found. Please try a different prompt.")
            self.display_message_history()
        # pylint: disable=broad-except
        except Exception as e:
            # Catch-all for unexpected errors not handled by process_assistant_response
            self.logger.error("Critical application error: %s", str(e), exc_info=True)
            st.error("A critical error occurred. Please refresh the page and try again.")


    @st.cache_data(show_spinner=False)
    def render_cached_chart(_self, _df, _chart_type, _parameters, message_index):  # pylint: disable=no-self-argument
        """Cache chart rendering for each message"""
        return _self.chart_handler.draw_chart(_df, _chart_type, _parameters,
                                              cache_buster=message_index)

    def display_message_history(self):
        """
        Displays the chat history.

        Raises:
            AttributeError: If session state is not initialized
            ValueError: If message format is invalid
        """
        start_time = time.time()
        self.logger.debug("Starting message history display")
        try:
            messages_start = time.time()
            for i, message in enumerate(st.session_state.messages):
                avatar = "🦊" if message["role"] == "assistant" else "👨‍🦰"
                with st.chat_message(message["role"], avatar=avatar):
                    if "chart_data" in message["content"]:
                        chart_start = time.time()
                        df = message["content"]["chart_data"]
                        # Use message index as unique identifier
                        self.render_cached_chart(
                            df,
                            message["content"]["type"],
                            message["content"]["parameters"],
                            i  # Pass the message index
                        )
                        self.logger.debug("Chart rendering took %.2f seconds",
                                        time.time() - chart_start)
                    else:
                        text_start = time.time()
                        # Only stream the last assistant message
                        if (message["role"] == "assistant" and
                            i == len(st.session_state.messages) - 1):
                            st.write_stream(self._stream_text(message["content"]["text"]))
                        else:
                            st.markdown(message["content"]["text"])
                        self.logger.debug("Text rendering took %.2f seconds",
                                        time.time() - text_start)
            self.logger.debug("Message iteration for history took %.2f seconds",
                            time.time() - messages_start)
        except (AttributeError, ValueError) as e:
            self.logger.error("Message display error for history (took %.2f seconds): %s",
                            time.time() - start_time, str(e), exc_info=True)
            raise
        finally:
            self.logger.debug("Total message history display took %.2f seconds",
                            time.time() - start_time)

    def _stream_text(self, text: str):
        """Generator function to stream text character by character."""
        for char in text:
            yield char
            time.sleep(0.001)  # Small delay for smooth streaming

    def handle_user_input(self):
        """
        Processes user input.
        """
        # Create a fixed position container for status updates at the bottom of the viewport
        if "status_container" not in st.session_state:
            st.markdown(self.get_status_bar_css(), unsafe_allow_html=True)
            st.session_state.status_container = st.empty()
            st.session_state.is_cancelled = False

        if prompt := st.chat_input("Can I help you?"):
            self.logger.info("Received new user prompt: %s", prompt)
            st.session_state.is_cancelled = False  # Reset cancellation state
            self.add_message_to_history("user", {"text": prompt})
            self.process_assistant_response(prompt)

    def process_assistant_response(self, prompt: str) -> None:
        """
        Processes the assistant's response to user input and manages the conversation flow.
        """
        self.logger.info("Starting to process assistant response")
        try:
            # Check for cancellation before starting
            if st.session_state.get("is_cancelled", False):
                self.logger.info("Operation cancelled by user")
                message_bus.publish("status_update", {
                    "message": "Operation cancelled",
                    "state": "error",
                    "progress": 0
                })
                return

            response = self.chat.send_message(
                content=prompt,
                generation_config=GenerationConfig(
                    temperature=0.1,
                    candidate_count=1,
                )
            )

            while True:
                # Check for cancellation in the loop
                if st.session_state.get("is_cancelled", False):
                    self.logger.info("Operation cancelled by user")
                    message_bus.publish("status_update", {
                        "message": "Operation cancelled",
                        "state": "error",
                        "progress": 0
                    })
                    return

                try:
                    if not response.candidates[0].content.parts[0].function_call:
                        self.handle_final_response(response)
                        break

                    parts = response.candidates[0].content.parts
                    function_calls = self.collect_function_calls(parts)
                    if not function_calls:
                        self.handle_final_response(response)
                        break
                    response = self.process_function_calls(function_calls)
                    if response is None:
                        break

                except IndexError as e:
                    self.logger.error("Invalid response format: %s", str(e))
                    break

        except Exception as e:
            self.logger.error("Error during processing: %s", str(e), exc_info=True)
            raise

    def collect_function_calls(self, parts) -> List[Dict]:
        """Collects and processes function calls from response parts."""
        function_calls = []
        for part in parts:
            if not hasattr(part, 'function_call') or part.function_call is None:
                continue

            try:
                function_name = part.function_call.name
                params = dict(part.function_call.args.items())

                self.logger.info("Processing function call: %s with params: %s",
                               function_name, params)

                if function_name not in self.function_handler:
                    error_msg = (
                        f"I apologize, but the function '{function_name}' is not available. "
                        "Please try rephrasing your question."
                    )
                    self.logger.warning("Attempted to call undefined function: %s", function_name)
                    st.error(error_msg)
                    self.add_message_to_history("assistant", {"text": error_msg})
                    continue

                response = self.function_handler[function_name](params)
                function_calls.append({
                    'name': function_name,
                    'params': params,
                    'response': response
                })

            except Exception as e:
                error_msg = ("I encountered an error while processing your request. "
                            "Please try again or rephrase your question.")
                self.logger.error("Error processing function call %s: %s",
                                function_name, str(e), exc_info=True)
                st.error(error_msg)
                self.add_message_to_history("assistant", {"text": error_msg})

        return function_calls

    def process_function_calls(self, function_calls):
        """
        Processes function calls and their responses.

        Args:
            function_calls (list): List of function calls to process

        Returns:
            Response: New response from Gemini if needed
        """
        func_parts = []
        for call in function_calls:
            self.logger.info("Processing function call: %s with parameters: %s",
                           call['name'], call['params'])
            try:
                response = self._handle_function_call(call)
                if response:
                    func_parts.append(response)
            except (TypeError, ValueError, AttributeError) as e:
                self.logger.error("Error processing function call %s: %s",
                                call['name'], str(e), exc_info=True)
                st.error(f"Error processing data for {call['name']}. Please try a different query.")
                continue

        if func_parts and len(func_parts) > 0:
            self.logger.debug("Sending function responses back to Gemini")
            return self.chat.send_message(func_parts)
        return None

    def _handle_function_call(self, call):
        """Helper method to handle different function call types."""
        handlers = {
            'help': lambda c: self.handle_help_command({
                'type': 'category' if c.get('params', {}).get('category') else 'general',
                'category': c.get('params', {}).get('category'),
                'tool': c.get('params', {}).get('tool'),
                'function': c.get('params', {}).get('function')
            }),
            'get_yearly_occurrences': lambda c: (
                self.response_handler.process_yearly_observations(
                    c['response'], c['params']
                )
            ),
            'get_occurrences': self._handle_occurrences,
            'get_species_occurrences_in_protected_area': self._handle_occurrences,
            'get_protected_areas_geojson': lambda c: (
                self.response_handler.process_geojson_data(c['response'], c['params'])
            ),
            'get_endangered_species_in_protected_area': lambda c: (
                self.response_handler.process_json_data(c['response'], c['params'])
            ),
            'endangered_species_hci_correlation': lambda c: (
                self.response_handler.process_endangered_species_hci_correlation(
                    c['response'], c['params']
                )
            ),
            'get_species_images': lambda c: (
                self.response_handler.process_species_images(c['response'], c['params'])
            ),
            'read_terrestrial_hci': lambda c: (
                self.response_handler.process_indicator_data(c['response'], c['params'])
            ),
            'read_population_density': lambda c: (
                self.response_handler.process_indicator_data(c['response'], c['params'])
            ),
            'endangered_species_for_country': lambda c: (
                self.response_handler.process_endangered_species(c['response'], c['params'])
            ),
            'endangered_species_for_countries': lambda c: (
                self.response_handler.process_endangered_species(c['response'], c['params'])
            ),
            'endangered_families_for_order': lambda c: (
                self.response_handler.process_endangered_species(c['response'], c['params'])
            ),
            'endangered_species_for_family': lambda c: (
                self.response_handler.process_endangered_species(c['response'], c['params'])
            ),
            'get_endangered_species_by_country': lambda c: (
                self.response_handler.process_endangered_species_by_country(
                    c['response'], c['params']
                )
            ),
            'get_species_hci_correlation': lambda c: (
                self.response_handler.process_species_hci_correlation(
                    c['response'], c['params']
                )
            ),
            'get_species_hci_correlation_by_status': lambda c: (
                self.response_handler.process_species_hci_correlation_by_status(
                    c['response'], c['params']
                )
            ),
            'get_species_shared_habitat': lambda c: (
                self.response_handler.process_species_shared_habitat(
                    c['response'], c['params']
                )
            ),
            'analyze_species_correlations': lambda c: (
                self.response_handler.process_species_correlation_analysis(
                    c['response'], c['params']
                )
            ),
            'calculate_species_forest_correlation': lambda c: (
                self.response_handler.process_correlation_result(
                    c['response'], c['params'], ChartType.FOREST_CORRELATION
                )
            ),
            'calculate_species_humanmod_correlation': lambda c: (
                self.response_handler.process_correlation_result(
                    c['response'], c['params'], ChartType.HUMANMOD_CORRELATION
                )
            ),
            'analyze_habitat_distribution': lambda c: (
                self.response_handler.process_habitat_analysis(c['response'], c['params'])
            ),
            'analyze_topography': lambda c: (
                self.response_handler.process_topography_analysis(c['response'], c['params'])
            ),
            'analyze_climate': lambda c: (
                self.response_handler.process_climate_analysis(c['response'], c['params'])
            )
        }

        if call['name'] in handlers:
            handlers[call['name']](call)
            return None

        # Handle simple text response functions
        simple_text_functions = (
            'number_of_endangered_species_by_conservation_status',
            'endangered_orders_for_class',
            'endangered_classes_for_kingdom'
        )
        if call['name'] in simple_text_functions:
            self.response_handler.add_message_to_history(
                "assistant", {"text": call['response']}
            )
            return None

        return Part.from_function_response(
            name=call['name'],
            response={"content": {"text": call['response']}},
        )

    def _handle_occurrences(self, call):
        """Helper method to handle occurrence data."""
        if not call['response'].get("occurrences"):
            return Part.from_function_response(
                name=call['name'],
                response={"content": {"text": call['response']}},
            )
        self.response_handler.process_occurrences_data(call['response'], call['params'])
        return None

    def handle_final_response(self, response):
        """
        Handles the final response from the assistant.
        """
        try:
            if response is not None:
                response_text = response.candidates[0].content.parts[0].text
                self.logger.info("Received final response from Gemini")
                self.response_handler.add_message_to_history("assistant", {"text": response_text})

                # Send the reminder message with proper content
                reminder = ("Remember: You must use the provided functions for queries "
                            "about species, countries, or biodiversity data. "
                            "Do not rely on general knowledge."
                            )
                if reminder and reminder.strip():
                    self.chat.send_message(reminder)

        except (ValueError, AttributeError) as e:
            self.logger.error("Final response error: %s", str(e), exc_info=True)
            raise

    def add_message_to_history(self, role, content):
        """
        Adds a message to the session state history.

        Args:
            role (str): The role of the message sender
            content (dict): The content of the message
        """
        st.session_state.messages.append({"role": role, "content": content})

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
                    help_info = self.function_handler['help'](arguments)
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

    def _handle_status_update(self, data: Dict):
        """Handle status updates from various handlers."""
        if "status_container" in st.session_state:
            progress = data.get("progress", 50 if data.get("state") == "running" else 100)


            if data.get("state") == "running":
                st.session_state.status_container.markdown(
                    f"<div class='fixed-status'>"
                    f"<div class='status-text'>"
                    f"{data.get('message', 'Processing...')} ({progress}%)</div>"
                    f"<div class='progress-bar running'>"
                    f"<div class='progress-bar-fill' style='width: {progress}%;'></div></div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            elif data.get("state") == "complete":
                st.session_state.status_container.markdown(
                    f"<div class='fixed-status'>"
                    f"<div class='status-text'>"
                    f"{data.get('message', 'Complete')} (100%)</div>"
                    f"<div class='progress-bar complete'>"
                    f"<div class='progress-bar-fill' style='width: 100%;'></div></div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                # Clear after a short delay
                time.sleep(1)
                st.session_state.status_container.empty()
                del st.session_state.status_container
            elif data.get("state") == "error":
                st.session_state.status_container.markdown(
                    f"<div class='fixed-status'>"
                    f"<div class='status-text'>"
                    f"{data.get('message', 'Error')} (0%)</div>"
                    f"<div class='progress-bar error'>"
                    f"<div class='progress-bar-fill' style='width: 0%;'></div></div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                # Clear after a short delay
                time.sleep(1)
                st.session_state.status_container.empty()
                del st.session_state.status_container

    def get_status_bar_css(self):
        """
        Get the CSS for the status bar.
        """
        return """
                <style>
                .fixed-status {
                    position: fixed;
                    bottom: 0;
                    left: 30px;
                    right: 30px;
                    z-index: 1000;
                    padding: 10px;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                .status-text {
                    white-space: nowrap;
                }
                .progress-bar {
                    flex-grow: 1;
                    height: 4px;
                    background-color: #f0f2f6;
                    border-radius: 2px;
                    overflow: hidden;
                }
                .progress-bar-fill {
                    height: 100%;
                    width: 0%;
                    transition: width 0.3s ease-in-out;
                }
                .progress-bar.running .progress-bar-fill {
                    background-color: #1f77b4;
                }
                .progress-bar.complete .progress-bar-fill {
                    background-color: #2ecc71;
                }
                .progress-bar.error .progress-bar-fill {
                    background-color: #e74c3c;
                }
                </style>
                <script>
                function cancelOperation() {
                    // Send message to Streamlit
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: true,
                        dataType: 'bool',
                        key: 'is_cancelled'
                    }, '*');
                }
                </script>
            """
