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
    ResponseValidationError,
    FinishReason,
    ToolConfig
)
from google.api_core import exceptions as google_exceptions
import streamlit as st
from app.utils.logging_config import setup_logging
from app.tools.function_handler import FunctionHandler
from app.tools.visualization.chart_handler import ChartHandler
from app.tools.message_bus import message_bus
from app.response_handler import ResponseHandler
from app.exceptions import BusinessException
from app.function_selector import FunctionSelector


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

    SYSTEM_MESSAGE = """You are an AI assistant designed to help users interact with ecological and environmental analysis tools.

You must ONLY make direct function calls. DO NOT:
- Write implementation code (loops, conditionals, etc.)
- Try to combine multiple function calls in one statement
- Write pseudocode or algorithms

⚠️ FUNCTION PARAMETERS:
For translate_to_scientific_name, ALWAYS use 'name' as the parameter:
✅ CORRECT: translate_to_scientific_name(name="Bornean Orangutan")
❌ INCORRECT: translate_to_scientific_name(common_name="Bornean Orangutan")
❌ INCORRECT: translate_to_scientific_name(species_name="Bornean Orangutan")

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
       - ALWAYS use get_species_in_protected_area
       Example: "What endangered species live in Serengeti?" → use get_species_in_protected_area ("Serengeti National Park")

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

    10. When processing google_search results:
        - ALWAYS extract and summarize the relevant information from the search results
        - NEVER tell the user to review the search results themselves
        - Provide a comprehensive answer based on the search results
        - If the search results contain a list (like countries, species, etc.), format it as a bulleted list
        - If the search results are incomplete, acknowledge this and provide what information is available

    Remember: Always use the appropriate function based on whether the query is about a single country or multiple countries.

    IMPORTANT: You must use the provided functions for any queries about species, countries, or biodiversity data.
    If no function matches the user's query or if you need additional general information, use the google_search function
    to find relevant information. Do not rely on your general knowledge alone - either use the provided functions or
    perform a Google search.

    Please do not announce your intention to use the functions, simple use it.

    """

    # pylint: disable=no-member
    # pylint: disable=no-self-argument
    @st.cache_resource(show_spinner=False)
    def initialize_app_resources(_self):
        """
        Initialize and cache application resources including handlers, Vertex AI, and chat model.

        Returns:
            dict: Dictionary containing initialized handlers, model, and chat session
        """
        start_time = time.time()
        logger = logging.getLogger("BioChat")
        logger.info("Starting app resources initialization")
        try:
            # Initialize handlers first
            handlers_start = time.time()
            with st.spinner("Setting up handlers..."):
                function_handler = FunctionHandler()
                chart_handler = ChartHandler()
                # Ensure function declarations are loaded
                function_declarations = function_handler.get_all_function_declarations()
                logger.info("Handlers initialized in %.2f seconds",
                          time.time() - handlers_start)

            # Initialize Vertex AI only after handlers are ready
            vertex_start = time.time()
            with st.spinner("Initializing Vertex AI..."):
                vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location="us-central1")
                logger.info("Vertex AI initialized successfully in %.2f seconds",
                          time.time() - vertex_start)

            # Initialize model with loaded declarations
            model_start = time.time()
            with st.spinner("Loading Gemini model..."):
                tools = Tool(function_declarations=function_declarations)

                model = GenerativeModel(
                    "gemini-2.0-flash",
#                    "gemini-2.5-pro-exp-03-25",
                    generation_config=GenerationConfig(temperature=0.01),
                    tools=[tools],
                    tool_config=ToolConfig(
                        function_calling_config=ToolConfig.FunctionCallingConfig(
                            mode=ToolConfig.FunctionCallingConfig.Mode.AUTO
                        )
                    )
                )
                logger.info("Model loaded successfully in %.2f seconds",
                          time.time() - model_start)

            logger.debug("Loading function declarations from tools:")

            logger.info("Total initialization completed in %.2f seconds",
                       time.time() - start_time)
            return {
                'handler': function_handler,
                'chart_handler': chart_handler,
                'model': model,
                'function_declarations': function_declarations
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
            initial_sidebar_state="collapsed"
        )

        self.logger = logging.getLogger("BioChat." + self.__class__.__name__)

        # Initialize session state first to ensure it exists
        self.initialize_session_state()

        try:
            # Check if resources are already in session state
            if "app_resources" not in st.session_state:
                self.logger.info("Initializing new app resources")
                resources = self.initialize_app_resources()
                # Store resources in session state for persistence across reloads
                st.session_state.app_resources = resources
            else:
                self.logger.info("Using existing app resources from session state")
                resources = st.session_state.app_resources

            # Validate that resources are still valid
            if not self._validate_resources(resources):
                self.logger.warning("Cached resources invalid, reinitializing...")
                resources = self.initialize_app_resources()
                st.session_state.app_resources = resources

            # Set instance variables from resources
            self.func_handler = resources['handler']
            self.function_handler = resources['handler'].get_all_function_mappings()
            self.function_declarations = resources['function_declarations']
            self.chart_handler = resources['chart_handler']

            # Chat sessions should be user-specific and stored in session state
            if "chat_session" not in st.session_state:
                self.logger.info("Creating new chat session")
                self.chat = resources['model'].start_chat(response_validation=False)
                # Send system message to initialize chat
                initial_response = self.chat.send_message(
                    [Part.from_text(self.SYSTEM_MESSAGE)],
                    generation_config=GenerationConfig(temperature=0.01)
                )
                if not initial_response:
                    raise RuntimeError("Chat initialization failed")
                st.session_state.chat_session = self.chat
            else:
                self.logger.info("Using existing chat session from session state")
                self.chat = st.session_state.chat_session

            self.generation_config = GenerationConfig(temperature=0.01)
            self.response_handler = ResponseHandler()

            # Initialize the function selector
            self.function_selector = FunctionSelector(self.func_handler)

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
        """
        self.logger.info("Starting BioChat Application")

        # Always display the title at the top
        st.title("Biodiversity Chat")
        st.write("[See example queries](https://github.com/csboros/biochat/blob/main/prompts.md)")

        # Add a separator
        st.markdown("<hr>", unsafe_allow_html=True)

        try:
            # Core functionality
            self.handle_user_input()
            self.display_message_history()

            # Handle function selector in the sidebar
            function_result = self.setup_sidebar_function_selector()

            # # Process the assistant's response in the main content area
            if function_result:
                # Add user message to history BEFORE processing
                self.add_message_to_history("user", {"text": f"Call the function {function_result['function_details']['aliases'][0]} with parameters {function_result['parameters']}"})

                # Process the assistant's response
                self.process_assistant_response(f"Call the function {function_result['function_details']['aliases'][0]} with parameters {function_result['parameters']}")

                # Display message history
                self.display_message_history()

        except ResponseValidationError as e:
            # Handle response validation error
            self.logger.debug("Current messages before pop: %s", st.session_state.messages)
            if st.session_state.messages:
                st.session_state.messages.pop()  # Only pop if there are messages
                self.logger.debug("Popped a message. Current messages: %s", st.session_state.messages)
            else:
                self.logger.warning("Attempted to pop from an empty messages list.")

            self.logger.error("ResponseValidationError: %s", str(e), exc_info=True)
            st.error("No results found. Please try a different prompt.")
            self.display_message_history()
        except Exception as e:
            # Catch-all for unexpected errors not handled by process_assistant_response
            self.logger.error("Critical application error: %s", str(e), exc_info=True)
            st.error("A critical error occurred. Please refresh the page and try again.")

    def setup_sidebar_function_selector(self):
        """
        Sets up the sidebar with CSS styling and function selector.

        Returns:
            dict or None: Function call information if a function was selected, None otherwise
        """
        # Add custom CSS for the sidebar tab styling and width
        st.markdown("""
            <style>
                /* Make the sidebar wider when open */
                [data-testid="stSidebar"][aria-expanded="true"] {
                    min-width: 350px !important;
                    max-width: 450px !important;
                }

                /* Adjust the main content area when sidebar is open */
                [data-testid="stSidebar"][aria-expanded="true"] ~ .main .block-container {
                    max-width: calc(100% - 450px) !important;
                    padding-left: 2rem;
                }

                /* When sidebar is closed, let main content use full width */
                [data-testid="stSidebar"][aria-expanded="false"] ~ .main .block-container {
                    max-width: 100% !important;
                    padding-left: 2rem;
                    padding-right: 2rem;
                }

                /* Original sidebar styling */
                .sidebar .sidebar-content {
                    background-color: #f0f2f6;
                }
                .sidebar-tab {
                    font-weight: bold;
                    padding: 10px;
                    background-color: #4e8df5;
                    color: white;
                    border-radius: 5px;
                    margin-bottom: 10px;
                    cursor: pointer;
                }
                .stExpander {
                    border: none !important;
                    box-shadow: none !important;
                }
            </style>
        """, unsafe_allow_html=True)

        function_result = None

        # Create the function selector in the sidebar
        with st.sidebar:
            st.markdown('<div class="sidebar-tab">Function Selector</div>', unsafe_allow_html=True)
            with st.expander("Select a function", expanded=False):
                result = self.function_selector.render()
                if result:
                    function_name = result["function_name"]
                    parameters = result["parameters"]
                    function_details = result["function_details"]

                    # Get the description and extract only the first sentence
                    description = function_details.get("description", "")
                    first_sentence = description.split('.')[0] + '.' if description else ""

                    # Display the first sentence of the description
                    st.write(f"**Description**: {first_sentence}")

                    # Store the function call information for later use
                    function_result = {
                        "function_name": function_name,
                        "parameters": parameters,
                        "function_details": function_details
                    }

        return function_result

    @st.cache_data(show_spinner=False)
    def render_cached_chart(_self, _df, _chart_type, _parameters, message_index):  # pylint: disable=no-self-argument
        """Cache chart rendering for each message"""
        return _self.chart_handler.draw_chart(_df, _chart_type, _parameters,
                                              cache_buster=message_index)

    def display_message_history(self):
        """
        Display the chat message history.
        """
        # Ensure messages list exists
        if "messages" not in st.session_state:
            st.session_state.messages = []
            return

        try:
            # Create a stable copy of the messages list
            messages_to_display = list(st.session_state.messages)

            # Display each message using the stable copy
            for i, message in enumerate(messages_to_display):
                role = message.get("role", "")
                content = message.get("content", {})

                if role == "user":
                    with st.chat_message("user", avatar = "👨‍🦰"):
                        st.markdown(content.get("text", ""))
                elif role == "assistant":
                    with st.chat_message("assistant", avatar = "🦊" ):
                        if "text" in content:
                            st.markdown(content["text"])
                        elif "chart_data" in content:
                            # Handle chart rendering using your implementation
                            try:
                                df = content.get("chart_data")
                                chart_type = content.get("type")
                                parameters = content.get("parameters")

                                if df is not None and chart_type is not None:
                                    # Generate a unique string ID based on the chart data
                                    # This avoids using indices that might cause errors
                                    chart_id = f"chart_{hash(str(df))}"
                                    self.render_cached_chart(df, chart_type, parameters, chart_id)
                                else:
                                    st.markdown("*Chart data could not be displayed*")
                            except Exception as chart_error:
                                self.logger.error(f"Error rendering chart: {str(chart_error)}")
                                st.markdown("*Error rendering chart*")
                        elif "image" in content:
                            st.image(content["image"])
        except Exception as e:
            # Catch any other errors that might occur
            self.logger.error(f"Error displaying message history: {str(e)}")

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

        if prompt := st.chat_input("What would you like to know about biodiversity?"):
            self.logger.info("Received new user prompt: %s", prompt)
            st.session_state.is_cancelled = False  # Reset cancellation state

            # Add user message to history BEFORE processing
            self.add_message_to_history("user", {"text": prompt})

            # Process the assistant's response
            self.process_assistant_response(prompt)

    def process_assistant_response(self, prompt: str, retry_count=0) -> None:
        """Process the assistant's response."""

        self.logger.info("Prompt: %s", prompt)
        if not prompt or not prompt.strip():
            st.error("Empty prompt received. Please try again.")
            return

        max_retries = 2  # Maximum number of retries
        try:
            # Ensure prompt is not empty and properly formatted
            if not prompt or not isinstance(prompt, str) or not prompt.strip():
                self.logger.error("Empty or invalid prompt detected before sending to API")
                st.error("Invalid prompt format. Please try again.")
                return

            # Log the exact content being sent to the API
            self.logger.debug("Sending to API: prompt='%s', type=%s, length=%d",
                              prompt, type(prompt), len(prompt))

            # Create the Part explicitly with validation
            text_part = Part.from_text(prompt.strip())
            if not hasattr(text_part, 'text') or not text_part.text:
                self.logger.error("Created Part has empty text: %s", text_part)
                raise ValueError("Failed to create valid Part from prompt")

            response = self.chat.send_message(
                [text_part],  # Send the validated Part
                tools=[Tool(function_declarations=self.function_declarations)],
                generation_config=self.generation_config
            )
            self.logger.info("Response: %s", str(response))

            if response and response.candidates:
                self.logger.debug("Processing response candidates")
                candidate = response.candidates[0]

                if candidate.finish_reason == FinishReason.MALFORMED_FUNCTION_CALL:
                    self.handle_malformed_function_call(candidate)
                    return

                elif candidate.content.parts:
                    # Call the new method to process the response parts
                    self.process_response_parts(candidate.content.parts, prompt)

        except google_exceptions.InvalidArgument as e:
            self.logger.error("Invalid argument error: %s", str(e))
            if retry_count >= max_retries:
                self.logger.error("Max retries reached. Unable to process response.")
                st.error("Sorry, there was a problem processing your request. Please try again.")
                return
            self.logger.warning(
                "Chat session invalid, reinitializing... (Attempt %d)",
                retry_count + 1
            )
            self.reinitialize_chat_and_send_prompt(prompt, retry_count)
        except Exception as e:
            self.logger.error("Error processing response: %s", str(e))
            st.error("An error occurred while processing the response. Please try again.")

    def process_response_parts(self, parts, prompt):
        """Processes the response parts from the assistant."""
        # Process the response
        if st.session_state.get("is_cancelled", False):
            self.logger.info("Operation cancelled by user")
            message_bus.publish("status_update", {
                "message": "Operation cancelled",
                "state": "error",
                "progress": 0
            })
            return
        part_to_process_next_round = []
        for part in parts:
            try:
                self.logger.info("Processing response part: %s", part)
                if not part:
                    self.logger.error("No content part in response")
                    st.error("Received an empty response. Please try again.")
                    break

                if part.function_call:
                    # If it's a function call, collect and process it
                    function_calls = self.collect_function_calls([part], prompt)  # Pass the single part
                    if function_calls:
                        response = self.process_function_calls(function_calls)
                        # If the response is still a function call, continue processing
                        if response and response.candidates and response.candidates[0].content.parts:
                            # Extend the list with individual parts instead of appending the whole list
                            part_to_process_next_round.extend(response.candidates[0].content.parts)
                        else:
                            # Handle the final response if it's not a function call
                            self.handle_final_response(part)
                    break  # Exit the loop after processing the function call

                else:
                    # If it's not a function call, handle the final response
                    self.handle_final_response(part)
                    break
            except BusinessException as e:
                self.logger.warning("Business logic error: %s", str(e))
                self.add_message_to_history("assistant", {"text": str(e)})
                break
            except IndexError as e:
                self.logger.error("Invalid response structure: %s", str(e))
                st.error("Received an invalid response format. Please try again.")
                break
        # process the next round of parts if any
        if part_to_process_next_round:
            self.process_response_parts(part_to_process_next_round, prompt)

    def reinitialize_chat_and_send_prompt(self, prompt: str, retry_count: int) -> None:
        """Reinitializes the chat and sends the user prompt."""
        try:
            # Reinitialize chat
            self.logger.info("Reinitializing chat session (Attempt %d)", retry_count + 1)

            # Create a new chat session
            model = st.session_state.app_resources['model']
            self.chat = model.start_chat(response_validation=False)

            # Send system message to initialize chat
            self.logger.debug("Sending system message to initialize chat")
            initial_response = self.chat.send_message(
                [Part.from_text(self.SYSTEM_MESSAGE)],
                generation_config=GenerationConfig(temperature=0.01)
            )

            if not initial_response:
                self.logger.error("Failed to initialize chat with system message")
                raise RuntimeError("Chat initialization failed")

            # Update the session state with the new chat
            st.session_state.chat_session = self.chat
            self.logger.info("Chat session successfully reinitialized")

            # Now send the user prompt
            if not prompt or not prompt.strip():
                self.logger.error("User prompt is empty. Cannot send to chat.")
                return

            self.logger.debug("Sending user prompt: %s", prompt)

            # Send the message with proper tools configuration
            response = self.chat.send_message(
                [Part.from_text(prompt.strip())],
                tools=[Tool(function_declarations=self.function_declarations)],
                generation_config=self.generation_config
            )

            # Process the response
            if response and response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    self.process_response_parts(candidate.content.parts, prompt)
                else:
                    self.logger.warning("Response has no content parts")
                    st.error("I couldn't process your request. Please try again with a different question.")
            else:
                self.logger.warning("No valid response received after reinitialization")
                st.error("I couldn't generate a response. Please try again.")

        except Exception as reinit_error:
            self.logger.error("Failed to reinitialize chat (Attempt %d): %s",
                              retry_count + 1, str(reinit_error), exc_info=True)

            if retry_count < 2:  # Limit retries to prevent infinite loops
                self.logger.info("Attempting another retry...")
                time.sleep(1)  # Add a small delay before retrying
                self.reinitialize_chat_and_send_prompt(prompt, retry_count + 1)
            else:
                st.error("I'm having trouble processing your request. Please try again later.")

    def collect_function_calls(self, parts, prompt) -> List[Dict]:
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
                        "Here are some available functions that might help you:\n"
                    )

                    # Retrieve and list all available functions
                    available_functions = self.get_available_functions(prompt) or []
                    if available_functions:
                        for func in available_functions:
                            error_msg += f"- **{func['name']}**: {func['description']}\n"
                    else:
                        error_msg += "No available functions found."

                    self.logger.warning("Attempted to call undefined function: %s", function_name)
                    self.add_message_to_history("assistant", {"text": error_msg})
                    continue

                try:
                    response = self.function_handler[function_name](params)
                except BusinessException as e:
                    error_msg = str(e)  # Get the error message from the exception
                    self.logger.error("Business error in %s: %s", function_name, error_msg)
                    self.add_message_to_history("assistant", {"text": error_msg})
                    continue

                function_calls.append({
                    'name': function_name,
                    'params': params,
                    'response': response
                })

            except Exception:
                error_msg = ("I couldn't process the data for {function_name}. "
                            "Let me try a different approach.")
                self.add_message_to_history("assistant", {"text": error_msg})
                continue

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

        # Process function calls with the verified tools
        for call in function_calls:
            self.logger.info("Processing function call: %s with parameters: %s",
                           call['name'], call['params'])
            try:
                # Check if the function call was successful and has a response
                if 'response' not in call or call['response'] is None:
                    self.logger.warning("No response for function call: %s", call['name'])
                    continue

                response = self.response_handler.handle_function_call(
                    call
                )

                # Validate the response before adding it
                if response:
                    self.logger.debug("Got valid response for %s: %s", call['name'], response)
                    func_parts.append(response)
                else:
                    self.logger.warning("Empty response from handler for %s", call['name'])
            except Exception as e:
                self.logger.error("Error processing function call %s: %s",
                            call['name'], str(e), exc_info=True)
                # Add error message to chat instead of showing error popup
                error_msg = (f"I couldn't process the data for {call['name']}. "
                            f"Let me try a different approach.")
                self.add_message_to_history("assistant", {"text": error_msg})
                continue

        if len(func_parts) != len(function_calls):
            self.logger.warning("Mismatch in function calls and responses: expected %d, got %d",
                             len(function_calls), len(func_parts))
            return None

        if func_parts and len(func_parts) > 0:
            self.logger.debug("Sending function responses back to Gemini with parts: %s",
                              func_parts)

            # Determine if this is likely a final response that needs higher temperature
            needs_higher_temp = self._should_use_higher_temperature(function_calls)

            if needs_higher_temp:
                # Use higher temperature for final responses (like google_search results)
                response = self.chat.send_message(
                    func_parts,
                    generation_config=GenerationConfig(
                        temperature=0.9,
                        max_output_tokens=1200
                    )
                )
            else:
                # Use default (low) temperature for potential intermediate function calls
                response = self.chat.send_message(func_parts)

            self.logger.debug("Received response from Gemini: %s",
                         str(response.candidates[0].content.parts
                             if response.candidates else "No candidates"))
            return response

        return None

    def _should_use_higher_temperature(self, function_calls):
        """
        Determine if we should use a higher temperature based on the function responses.

        Args:
            function_calls: List of function calls

        Returns:
            bool: True if higher temperature should be used, False otherwise
        """
        # Functions that typically produce final responses that benefit from higher temperature
        final_response_functions = [
            'google_search',
            'get_species_info'
        ]

        # Check if any of the function calls are from final response functions
        for call in function_calls:
            if call['name'] in final_response_functions:
                return True

        return False

    def handle_final_response(self, part):
        """Handles the final response from the assistant."""
        try:
            self.logger.info("Part in handle_final_response: %s", part)

            # Check if part has text
            if hasattr(part, 'text') and part.text:
                self.add_message_to_history("assistant", {"text": part.text})

            # Reminder to the model to use the tools
            if not hasattr(part, 'function_call') or part.function_call is None:
                try:
                    self.chat.send_message([Part.from_text(self.SYSTEM_MESSAGE)], generation_config=GenerationConfig(temperature=0.01))
                except Exception as e:
                    self.logger.warning("Failed to send reminder to model: %s", str(e))

        except Exception as e:
            self.logger.error("Final response error: %s", str(e), exc_info=True)
            st.error("An error occurred while processing the response. Please try again.")

    def add_message_to_history(self, role, content):
        """
        Add a message to the chat history.

        Args:
            role (str): The role of the message sender ('user' or 'assistant')
            content (dict): The content of the message
        """
        # Ensure messages list exists in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Add the message to the history
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

    def handle_malformed_function_call(self, candidate):
        """Handles the case of a malformed function call."""
        self.logger.warning("Malformed function call detected")
        error_msg = candidate.finish_message

        # Extract the attempted function call
        function_call = error_msg.split("Malformed function call: ")[1] if "Malformed function call: " in error_msg else error_msg

        self.logger.warning("MALFORMED_FUNCTION_CALL detected: %s", function_call)


        # Create a helpful error message with alternatives
        error_message = (
            "I apologize, but I tried to use a function that doesn't exist. "
            f"I attempted to call: `{function_call.strip()}`\n\n"
            "Here are some available functions that might help you:\n"
        )

        # Get available functions to include in the error message
        available_functions = self.get_available_functions("function call") or []

        if available_functions:
            for func in available_functions:
                error_message += f"- **{func['name']}**: {func['description']}\n"
        else:
            error_message += "No available functions found."

        error_message += (
            "\nLet me try a different approach to answer your question. "
            "Please ask again and I'll use one of my available functions."
        )

        # Log the error message
        self.logger.warning("Showing error to user: %s", error_message)

        # Add the error message to chat history
        self.add_message_to_history("assistant", {"text": error_message})

        # Send a reminder to the model about available functions
        try:
            self.chat.send_message([Part.from_text(self.SYSTEM_MESSAGE)],
                                  generation_config=GenerationConfig(temperature=0.01))
        except Exception as e:
            self.logger.warning("Failed to send reminder to model: %s", str(e))

    def _validate_resources(self, resources):
        """Validate that cached resources are still usable"""
        try:
            # Simple validation check - ensure key components exist
            if not all(k in resources for k in ['handler', 'chart_handler', 'model']):
                return False

            # Optional: Test that the model is still responsive
            # This might add overhead but ensures the connection is still valid
            # test_response = resources['model'].generate_content("test")
            # if not test_response:
            #     return False

            return True
        except Exception:
            return False

    def get_available_functions(self, context: str) -> List[Dict]:
        """Returns a list of available functions with their short descriptions that match the context."""
        functions = []

        # Extract keywords from the context
        keywords = self.extract_keywords(context)  # Implement this method to extract relevant keywords

        # Split the context keywords into a set for easier matching
        context_words = set(keyword.lower() for keyword in keywords)

        for func_name, func in self.function_handler.items():
            # Split the function name into words by removing underscores
            func_words = set(func_name.lower().split())

            # Check if any keyword is present in the function name or description
            if context_words.intersection(func_words) or any(keyword.lower() in (func.__doc__ or "").lower() for keyword in keywords):
                # Extract a short description without detailed argument information
                short_description = func.__doc__.splitlines()[0] if func.__doc__ else "No description available."
                functions.append({
                    "name": func_name,
                    "description": short_description  # Only keep the short description
                })
        return None
#       return functions

    def extract_keywords(self, context: str) -> List[str]:
        """Extracts relevant keywords from the context."""
        # Simple keyword extraction logic (this can be improved with NLP techniques)
        # Split the context into words by spaces and filter out common stop words
        stop_words = set(["the", "is", "at", "which", "on", "and", "a", "to", "in", "for", "with", "of", "by"])

        # Split by spaces only
        words = context.split()

        # Filter out stop words and convert to lowercase
        keywords = [word for word in words if word.lower() not in stop_words]

        return keywords
