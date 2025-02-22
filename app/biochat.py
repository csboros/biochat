"""
    A Streamlit application for exploring biodiversity data using Vertex AI's Gemini model.
    
    This application provides an interactive chat interface where users can query information
    about endangered species, conservation status, and global biodiversity patterns. It
    integrates with various data sources and visualization tools to present biodiversity
    information.

"""

import logging
import time
import pandas as pd
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
    ResponseValidationError
)
import requests
from google.api_core import exceptions as google_exceptions
import streamlit as st
from app.utils.logging_config import setup_logging
from app.handlers.function_handler import FunctionHandler
from app.handlers.chart_handler import ChartHandler

# Setup logging configuration at application startup
setup_logging()

class BioChat:
    """
    Main application class for the Biodiversity Chat interface.
    
    Manages the Streamlit interface, Vertex AI model integration, and chat functionality.
    Handles user interactions, function calls, and visualization of biodiversity data.
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
                vertexai.init(location="us-central1")
                logger.info("Vertex AI initialized successfully in %.2f seconds",
                          time.time() - vertex_start)

            # Initialize handlers
            handlers_start = time.time()
            with st.spinner("Setting up handlers..."):
                handler = FunctionHandler()
                chart_handler = ChartHandler()
                logger.info("Handlers initialized in %.2f seconds",
                          time.time() - handlers_start)

            # Initialize model with timeout
            model_start = time.time()
            with st.spinner("Loading Gemini model..."):
                tools = Tool(function_declarations=handler.declarations)
                model = GenerativeModel(
                    model_name="gemini-pro",
                    generation_config=GenerationConfig(temperature=0),
                    tools=[tools],
                )
                logger.info("Model loaded successfully in %.2f seconds",
                          time.time() - model_start)

            # Initialize chat
            chat_start = time.time()
            with st.spinner("Setting up chat..."):
                chat = model.start_chat()
                system_message = """You are a biodiversity expert assistant.
                Your role is to help users
                understand endangered species, their conservation status, 
                and global biodiversity patterns. 
                When providing information:
                - First, Use the provided tools to get the data and information you need 
                - Second, if you don't have the answer, use the google_search tool to find the answer. 
                - Third, if you don't have the answer based on the tools and google search, 
                    use your own knowledge to answer the questions. 
                - Use available tools to show data visualizations when relevant
                """
                chat.send_message(system_message)
                logger.info("Chat session initialized in %.2f seconds",
                          time.time() - chat_start)

            total_time = time.time() - start_time
            logger.info("Total initialization completed in %.2f seconds", total_time)
            return {
                'handler': handler,
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
            page_title="AI Chat Assistant",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        self.logger = logging.getLogger("BioChat." + self.__class__.__name__)
        try:
            resources = self.initialize_app_resources()
            self.function_handler = resources['handler'].function_handler
            self.function_declarations = resources['handler'].declarations
            self.chart_handler = resources['chart_handler']
            self.gemini_model = resources['model']
            self.chat = resources['chat']
            self.initialize_session_state()
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
            with st.expander("Examples Queries"):
                st.write('''
                    - Show all families for primates as list with the number of endangered species 
                    - Show all endangered species in the family of HOMINIDAE with link to IUCN
                    - Where do Bornean Orangutans live?
                    - Show Bornean Orangutans distribution as heatmap
                    - List the number of endangered species per conservation status for Germany
                    - List the endangered species for Germany with status Critically Endangered
                    - Show the distribution of Common Hamster
                    - Give me more details about the Common Hamster such as conservation status and threats based on the IUCN website
                    - Show all protected areas in Kenya
                    - List all endangered species in Masai Mara
                    - Show the distribution of lions in Masai Mara     
                ''')
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
    def render_cached_chart(_self, df, chart_type, parameters):  # pylint: disable=no-self-argument
        """Cache chart rendering for each message"""
        return _self.chart_handler.draw_chart(df, chart_type, parameters)

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
            for _, message in enumerate(st.session_state.messages):
                avatar = "🦊" if message["role"] == "assistant" else "👨‍🦰"
                with st.chat_message(message["role"], avatar=avatar):
                    if "chart_data" in message["content"]:
                        chart_start = time.time()
                        df = message["content"]["chart_data"]
                        # Use message index as part of cache key
                        self.render_cached_chart(
                            df,
                            message["content"]["type"],
                            message["content"]["parameters"]
                        )
                        self.logger.debug("Chart rendering took %.2f seconds",
                                        time.time() - chart_start)
                    else:
                        text_start = time.time()
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

    def handle_user_input(self):
        """
        Processes user input.
        """
        if prompt := st.chat_input("Can I help you?"):
            self.logger.info("Received new user prompt: %s", prompt)
            self.add_message_to_history("user", {"text": prompt})
            self.process_assistant_response(prompt)

    def process_assistant_response(self, prompt: str) -> None:
        """
        Processes the assistant's response to user input and manages the conversation flow.

        This method handles the interaction with the Gemini model, processes any function
        calls requested by the model, and manages the display of responses in the chat interface.

        Parameters
        ----------
        prompt : str
            The user's input text to be processed by the assistant

        Raises
        ------
        google_exceptions.ResourceExhausted
            When the API quota has been exceeded
        google_exceptions.TooManyRequests
            When too many requests are made in a short time period
        google_exceptions.GoogleAPIError
            When there's an error in the API communication
        requests.exceptions.RequestException
            When there's an error in the API communication
        """
        start_time = time.time()
        self.logger.info("Starting to process assistant response")
        try:
            # Send initial message
            message_start = time.time()
            response = self.chat.send_message(content=prompt)
            self.logger.info("Initial message sent in %.2f seconds",
                           time.time() - message_start)
            while True:
                try:
                    # Process response parts
                    parts_start = time.time()
                    parts = response.candidates[0].content.parts
                    function_calls = self.collect_function_calls(parts)
                    self.logger.info("Response parts processed in %.2f seconds",
                                   time.time() - parts_start)
                    if not function_calls:
                        final_start = time.time()
                        self.handle_final_response(response)
                        self.logger.info("Final response handled in %.2f seconds",
                                       time.time() - final_start)
                        break
                    # Process function calls
                    func_start = time.time()
                    response = self.process_function_calls(function_calls)
                    self.logger.info("Function calls processed in %.2f seconds",
                                   time.time() - func_start)
                    if response is None:
                        break
                except IndexError as e:
                    self.logger.error("Invalid response format (took %.2f seconds): %s",
                                    time.time() - start_time, str(e), exc_info=True)
                    st.error("Received an invalid response format. Please try again.")
                    break
                except (ValueError, AttributeError) as e:
                    self.logger.error("Missing response attributes (took %.2f seconds): %s",
                                    time.time() - start_time, str(e), exc_info=True)
                    st.error("Received an incomplete response. Please try again.")
                    break
        except (google_exceptions.ResourceExhausted,
                google_exceptions.TooManyRequests) as e:
            self.logger.error("API quota exceeded (took %.2f seconds): %s",
                            time.time() - start_time, str(e), exc_info=True)
            st.error("API quota has been exceeded. Please wait a few minutes and try again.")
        except google_exceptions.GoogleAPIError as e:
            self.logger.error("API error (took %.2f seconds): %s",
                            time.time() - start_time, str(e), exc_info=True)
            st.error(f"Failed to get data from Google BigQuery: {str(e)}")
        except requests.exceptions.RequestException as e:
            self.logger.error("Network error (took %.2f seconds): %s",
                            time.time() - start_time, str(e), exc_info=True)
            st.error("Network error occurred. Please check your connection and try again.")
        except Exception as e:  # pylint: disable=broad-except
            self.logger.error("Unexpected error (took %.2f seconds): %s",
                            time.time() - start_time, str(e), exc_info=True)
            raise
        finally:
            self.logger.info("Total assistant response processing took %.2f seconds",
                           time.time() - start_time)

    def collect_function_calls(self, parts):
        """
        Collects function calls from response parts.
        """
        function_calls = []
        for part in parts:
            if part.function_call is not None:
                function_name = part.function_call.name
                params = dict(part.function_call.args.items())
                self.logger.info("Processing function call: %s with params: %s",
                                 function_name, params)
                response = self.function_handler[function_name](params)
                function_calls.append({
                    'name': function_name,
                    'params': params,
                    'response': response
                })
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
            self.logger.info("Processing function call: %s", call['name'])
            self.logger.info("Processing function call with parameters: %s", call['params'])
            try:
                if call['name'] in ('get_occurences', 'get_species_occurrences_in_protected_area'):
                    self.process_occurrences_data(call['response'], call['params'])
                elif call['name'] == "get_protected_areas_geojson":
                    self.process_geojson_data(call['response'], call['params'])
                elif call['name'] == "get_endangered_species_in_protected_area":
                    self.process_json_data(call['response'], call['params'])
                elif call['name'] in ('endangered_species_for_country',
                                  'number_of_endangered_species_by_conservation_status',
                                  'endangered_species_for_family', 
                                  'endangered_families_for_order',
                                  'endangered_orders_for_class',
                                  'endangered_classes_for_kingdom'):
                    self.add_message_to_history("assistant", {"text": call['response']})
                else:
                    func_parts.append(Part.from_function_response(
                        name=call['name'],
                        response={"content": {"text": call['response']}},
                    ))
            except (TypeError, ValueError, AttributeError) as e:
                self.logger.error("Error processing function call %s: %s",
                                    call['name'], str(e), exc_info=True)
                st.error(f"Error processing data for {call['name']}. Please try a different query.")
                continue  # Skip this function call but continue processing others

        if func_parts:
            self.logger.debug("Sending function responses back to Gemini")
            return self.chat.send_message(func_parts)
        return None

    def handle_final_response(self, response):
        """
        Handles the final response.
        
        Raises:
            ValueError: If response format is invalid
            AttributeError: If required attributes are missing
        """
        try:
            if response is not None:
                response_text = response.candidates[0].content.parts[0].text
                self.logger.info("Received final response from Gemini")
                self.add_message_to_history("assistant", {"text": response_text})
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

    def process_geojson_data(self, data_response, parameters):
        """
        Processes and visualizes GeoJSON data for protected areas.
        """
        st.session_state.messages.append({
                    "role": "assistant",
                    "content": {
                        "chart_data": data_response,
                        "type": "geojson",
                        "parameters": parameters
                    }
                })

    def process_json_data(self, data_response, parameters):
        """
        Processes and visualizes JSON data for protected areas.
        """
        st.session_state.messages.append({
                    "role": "assistant",
                    "content": {
                        "chart_data": data_response,
                        "type": "json",
                        "parameters": parameters
                    }
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
            df = pd.json_normalize(data_response)
            self.logger.debug("Data normalization took %.2f seconds",
                            time.time() - norm_start)
            if df.empty:
                self.logger.info("No data to display (took %.2f seconds)",
                               time.time() - start_time)
#                st.markdown("No data to display")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {"text": "No data to display"}
                })
            else:
                chart_type = parameters.get("chart_type", "hexagon")
                self.logger.debug("Parameters: %s", parameters)
                # Chart rendering timing
#                render_start = time.time()
#                self.chart_handler.draw_chart(df, chart_type, parameters)
#                self.logger.debug("Chart rendering took %.2f seconds",
#                                time.time() - render_start)
                # Session state update timing
                session_start = time.time()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {
                        "chart_data": df,
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
