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
        Sets up various functions for species information, geographical data, 
        and endangered species queries. 
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
                    self.function_declarations
                        ['number_of_endangered_species_by_conservation_status'],
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
            self.display_message_history()
            self.handle_user_input()
        except google_exceptions.ResourceExhausted as e:
            self.logger.error("API quota exceeded: %s", str(e), exc_info=True)
            st.error("API quota has been exceeded. Please wait a few minutes and try again.")
        except (ValueError, RuntimeError, AttributeError) as e:
            self.logger.error("Error in main application loop: %s", str(e), exc_info=True)
            st.error("An unexpected error occurred. Please try again later.")

    def display_message_history(self):
        """
        Displays the chat history including text messages and charts.
        """
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

    def handle_user_input(self):
        """
        Processes new user input and generates responses.
        """
        if prompt := st.chat_input("Can I help you?"):
            self.logger.info("Received new user prompt: %s", prompt)
            self.add_message_to_history("user", {"text": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                self.process_assistant_response(prompt)

    def process_assistant_response(self, prompt):
        """
        Processes the assistant's response, including function calls.
        
        Args:
            prompt (str): The user's input prompt
        """
        self.logger.debug("Sending message to Gemini")
        response = self.chat.send_message(content=prompt)
        # Loop until all function calls are processed
        while True:
            parts = response.candidates[0].content.parts
            function_calls = self.collect_function_calls(parts)
            # If no function calls are found, handle the final response
            if not function_calls:
                self.handle_final_response(response)
                break
            # Process function calls and get a new response
            response = self.process_function_calls(function_calls)
            # If no new response is needed (all functions handled locally), exit the loop
            if response is None:
                break

    def collect_function_calls(self, parts):
        """
        Collects function calls from response parts.
        
        Args:
            parts (list): List of response parts from Gemini
            
        Returns:
            list: List of dictionaries containing function call information
        """
        function_calls = []
        for part in parts:
            if part.function_call is not None:
                function_name = part.function_call.name
                params = dict(part.function_call.args.items())
                self.logger.info("Processing function call: %s", function_name)
                self.logger.debug("Function parameters: %s", params)
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
            if call['name'] == "get_occurences":
                self.process_occurrences_data(call['response'], call['params'])
            elif call['name'] in ('endangered_species_for_country',
                              'number_of_endangered_species_by_conservation_status',
                              'endangered_species_for_family', 
                              'endangered_families_for_order'):
                self.add_message_to_history("assistant", {"text": call['response']})
                st.markdown(call['response'])
            else:
                func_parts.append(Part.from_function_response(
                    name=call['name'],
                    response={"content": {"text": call['response']}},
                ))
        if func_parts:
            self.logger.debug("Sending function responses back to Gemini")
            return self.chat.send_message(func_parts)
        return None

    def handle_final_response(self, response):
        """
        Handles the final response from the assistant.
        
        Args:
            response: The response from Gemini
        """
        if response is not None:
            response_text = response.candidates[0].content.parts[0].text
            self.logger.info("Received final response from Gemini: %s", response_text)
            self.add_message_to_history("assistant", {"text": response_text})
            st.write(response_text)
            st.session_state.history = self.chat.history

    def add_message_to_history(self, role, content):
        """
        Adds a message to the session state history.
        
        Args:
            role (str): The role of the message sender
            content (dict): The content of the message
        """
        st.session_state.messages.append({"role": role, "content": content})

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
                parameters["geojson"] = (
                    self.function_handler['get_country_geojson'](parameters)[:20000]
                )
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
