"""
    A Streamlit application for exploring biodiversity data using Vertex AI's Gemini model.
    
    This application provides an interactive chat interface where users can query information
    about endangered species, conservation status, and global biodiversity patterns. It
    integrates with various data sources and visualization tools to present biodiversity
    information.

"""

import logging
import pandas as pd
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
)
from google.api_core import exceptions as google_exceptions
import streamlit as st
from function_handler import FunctionHandler
from chart_handler import ChartHandler
from logging_config import setup_logging

# Setup logging configuration at application startup
setup_logging()

class BiodiversityApp:
    """
    Main application class for the Biodiversity Chat interface.
    
    Manages the Streamlit interface, Vertex AI model integration, and chat functionality.
    Handles user interactions, function calls, and visualization of biodiversity data.
    """

    def __init__(self):
        """
        Initializes the BiodiversityApp with necessary components and configurations.
        Sets up logging, Vertex AI, function handlers, and session state.
        
        The Google Cloud Project ID is loaded from Streamlit secrets configuration.
        See the .streamlit/secrets.toml file for configuration details.
        
        Raises:
            StreamlitAPIException: If the required secret GOOGLE_CLOUD_PROJECT is not found
            Exception: If initialization of any component fails
        """
        # Create a class-specific logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing BiodiversityApp")
        try:
            vertexai.init()
            self.logger.info("Initialized Vertex AI")
            handler = FunctionHandler()
            self.function_handler = handler.function_handler
            self.function_declarations = handler.declarations
            self.chart_handler = ChartHandler()
            self.setup_tools()
            self.setup_model()
            self.initialize_session_state()
        except Exception as e:
            self.logger.error("Error during initialization: %s", str(e), exc_info=True)
            raise

    def setup_tools(self):
        """
        Configures the biodiversity tools and function declarations for the application.
        Sets up various functions for species information, geographical data, and endangered species queries. 
        Raises:
            Exception: If tool setup fails
        """
        self.logger.info("Setting up tools and function declarations")
        try:
            self.biodiversity_tool = Tool(
                function_declarations=[
                    self.function_declarations['translate_to_scientific_name'],
                    self.function_declarations['get_species_info'],
                    self.function_declarations['get_country_geojson'],
                    self.function_declarations['endangered_species_for_family'],
                    self.function_declarations['endangered_classes_for_kingdom'],
                    self.function_declarations['endangered_families_for_order'],
                    self.function_declarations['endangered_orders_for_class'],
                    self.function_declarations['get_occurences'],
                    self.function_declarations['endangered_species_for_country'],
                    self.function_declarations['number_of_endangered_species_by_conservation_status'],
                    self.function_declarations['google_search'],
                ],
            )
            self.logger.debug("Successfully set up tools and function declarations")
        except Exception as e:
            self.logger.error("Error setting up tools: %s", str(e), exc_info=True)
            raise

    def setup_model(self):
        """
        Initializes the Gemini model with specific configuration parameters.
        Sets up the model with appropriate temperature and tools for generation.
        
        Raises:
            Exception: If model setup fails
        """
        self.logger.info("Setting up Gemini model")
        try:
            self.gemini_model = GenerativeModel(
#                "gemini-pro",
                "gemini-1.5-pro-002",
#                "gemini-1.5-flash",
                generation_config=GenerationConfig(temperature=0),
                tools=[ self.biodiversity_tool],
            )
            self.logger.debug("Successfully initialized Gemini model")
        except Exception as e:
            self.logger.error("Error setting up Gemini model: %s", str(e), exc_info=True)
            raise

    def initialize_session_state(self):
        """
        Initializes the Streamlit session state and sets up the chat history.
        Creates a new chat session with the Gemini model and sets the initial system context.
        
        Raises:
            Exception: If session state initialization fails
        """
        self.logger.info("Initializing session state")
        try:
            if "history" not in st.session_state:
                st.session_state.history = []
                self.logger.debug("Initialized empty history in session state")
            if "messages" not in st.session_state:
                st.session_state.messages = []
                self.logger.debug("Initialized empty messages in session state")

            # Add system message to set context     
            system_message = """You are a biodiversity expert assistant. Your role is to help users
            understand endangered species, their conservation status, and global biodiversity patterns. 
            When providing information:
            - First, Use the provided tools to get the data and information you need 
            - Second, if you don't have the answer, use the google_search tool to find the answer. 
            - Third, if you don't have the answer based on the tools and google search, use your own knowledge to answer the questions. 
            - Use available tools to show data visualizations when relevant
            """
            self.chat = self.gemini_model.start_chat(history=st.session_state.history)
            self.chat.send_message(system_message)
            self.chat = self.gemini_model.start_chat(history=st.session_state.history)
            self.logger.debug("Started new chat session with Gemini") 
        except google_exceptions.ResourceExhausted as e:
            self.logger.error("API quota exceeded: %s", str(e), exc_info=True)
            st.error("API quota has been exceeded. Please wait a few minutes and try again.")  
        except Exception as e:
            self.logger.error("Error initializing session state: %s", str(e), exc_info=True)
            raise

    def run(self):
        """
        Main execution method for the Streamlit application.
        Handles the chat interface, message history, and processes user inputs.
        Manages function calls and responses from the Gemini model.
        
        Raises:
            Exception: If any error occurs during application execution
        """
        self.logger.info("Starting BiodiversityApp")
        try:
            st.title("Biodiversity Chat")
            # Display message history
            self.logger.debug("Displaying %d previous messages", len(st.session_state.messages))
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if "chart_data" in message["content"]:
                        df = message["content"]["chart_data"]
                        self.logger.debug("Rendering chart of type: %s", message['content']['type'])
                        self.chart_handler.draw_chart(
                            df,
                            message["content"]["type"],
                            message["content"]["parameters"]
                        )
                    else:
                        st.markdown(message["content"]["text"])

            # Handle new input
            if prompt := st.chat_input("Can I help you?"):
                self.logger.info("Received new user prompt: %s", prompt)
                st.session_state.messages.append({"role": "user", "content": {"text": prompt}})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Process assistant response
                with st.chat_message("assistant"):
                    self.logger.debug("Sending message to Gemini")
                    response = self.chat.send_message(content=prompt)
                    function_calling_in_process = True
                    
                    while function_calling_in_process:
                        parts = response.candidates[0].content.parts
                        function_names = []
                        function_api_responses = []
                        function_api_parameters = []
                        
                        # Call functions one by one 
                        for part in parts:
                            if part.function_call is not None:

                                function_name = part.function_call.name
                                self.logger.info("Processing function call: %s", function_name)
                                function_names.append(function_name)
                                params = dict(part.function_call.args.items())
                                function_api_parameters.append(params)
                                self.logger.debug("Function parameters: %s", params)

                                function_api_response = self.function_handler[function_name](params)
                                self.logger.info("len(function_api_response): %d", len(function_api_response))
                                function_api_responses.append(function_api_response)
                                self.logger.debug("Received response for function: %s", function_name)

                        # Process function responses
                        if len(function_names) > 0:
                            num_responses = len(function_names)
                            self.logger.debug("Processing %d function responses", num_responses)
                            func_parts = []
                            for i, function_name in enumerate(function_names):
                                self.logger.debug("Processing response for function: %s", function_name)
                                # Anything that can be handled locally should be handled here 
                                if function_name == "get_occurences":
                                    self.process_occurrences_data(
                                        function_api_responses[i],
                                        function_api_parameters[i]
                                    )
                                elif function_name in ('endangered_species_for_country',
                                      'number_of_endangered_species_by_conservation_status',
                                      'endangered_species_for_family', 'endangered_families_for_order'):
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": {"text": function_api_responses[i]}
                                    })
                                    st.markdown(function_api_responses[i])
                                # Anything that can't be handled locally should be sent back to Gemini
                                else:
                                    func_parts.append(Part.from_function_response(
                                        name=function_name,
                                        response={"content": {"text": function_api_responses[i]}},
                                    ))
                            # Send function responses back to Gemini if any and use the response to check for function calls
                            if len(func_parts) > 0:
                                self.logger.debug("Sending function responses back to Gemini")
                                response = self.chat.send_message(func_parts)
                            # If no function responses, then we have the final response
                            else:
                                function_calling_in_process = False
                                self.logger.info("Received final response from Gemini")
                                st.session_state.history = self.chat.history

                        else:
                            # If no function responses, then we have the final response
                            function_calling_in_process = False
                            if response is not None:
                                response = response.candidates[0].content.parts[0]
                                response_text = response.text
                                self.logger.info("Received final response from Gemini: %s", response_text)
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": {"text": response_text}
                                })
                                st.write(response_text)
                                st.session_state.history = self.chat.history
        except (ValueError, RuntimeError, IOError) as e:
            self.logger.error("Error in main application loop: %s", str(e), exc_info=True)
            st.error("An unexpected error occurred. Please try again later.")
        except google_exceptions.ResourceExhausted as e:
            self.logger.error("API quota exceeded: %s", str(e), exc_info=True)
            st.error("API quota has been exceeded. Please wait a few minutes and try again.")


    def process_occurrences_data(self, data_response, parameters):
        """
        Processes and visualizes occurrence data for species.
        
        Args:
            data_response (dict): The response data containing occurrence information
            parameters (dict): Parameters for chart generation including:
                - chart_type (str): Type of chart to generate
                - country_code (str, optional): Country code for geographical data
                
        Raises:
            Exception: If data processing or visualization fails
        """
        self.logger.debug("Processing occurrences data")
        try:
            df = pd.json_normalize(data_response)
            chart_type = parameters.get("chart_type", "hexagon")
            self.logger.debug("Parameters: %s", parameters)
            if parameters.get("country_code", None) is not None:
                parameters["geojson"] = self.function_handler['get_country_geojson'](parameters)[:20000]
            self.chart_handler.draw_chart(df, chart_type, parameters)
            st.session_state.messages.append({"role": "assistant",
                                              "content": {"chart_data": df, "type": chart_type,
                                              "parameters": parameters}})
        except Exception as e:
            self.logger.error("Error processing occurrences data: %s", str(e), exc_info=True)
            raise

if __name__ == "__main__":
    app = BiodiversityApp()
    app.run()
