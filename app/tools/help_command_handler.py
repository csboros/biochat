"""Handler for help commands using Gemini to match user queries to tools."""

import logging
import re
from typing import Dict, Any, List, Optional
import requests
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
from app.tools.tool import Tool
from app.tools.message_bus import message_bus

# pylint: disable=broad-except
class HelpCommandHandler:
    """Handles help commands by matching queries to tools using Gemini."""

    def __init__(self, tools=None):
        """Initialize the help command handler.

        Args:
            tools: List of available tools (optional, can be set later)
            message_bus: Message bus for sending status updates
        """
        self.logger = logging.getLogger("BioChat.HelpCommandHandler")
        self.tools = tools or []
        self.gemini_model = GenerativeModel("gemini-pro")
        self.generation_config = GenerationConfig(
            temperature=0.2,
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048
        )

    def set_tools(self, tools: List[Tool]):
        """Set the tools after initialization.

        Args:
            tools: List of available tools
        """
        self.tools = tools


    def handle_help_command(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a help command by matching the query to available tools.

        Args:
            args: Dictionary containing help command arguments

        Returns:
            Dictionary with help information or error message
        """
        if message_bus:
            message_bus.publish("status_update", {
                "message": "🔍 Looking up help information...",
                "state": "running",
                "progress": 10
            })

        if not self.tools:
            if message_bus:
                message_bus.publish("status_update", {
                    "message": "❌ No tools available",
                    "state": "error",
                    "progress": 100
                })
            return {
                'success': False,
                'error': "No tools available",
                'message': "The help system has not been properly initialized with tools."
            }

        query = args.get('category', None)

        if not query:
            if message_bus:
                message_bus.publish("status_update", {
                    "message": "📚 Generating general help information...",
                    "state": "running",
                    "progress": 30
                })
            result = self._get_general_help()
        else:
            if message_bus:
                message_bus.publish("status_update", {
                    "message": f"🔍 Searching for help on '{query}'...",
                    "state": "running",
                    "progress": 30
                })

            # Match query to function declarations
            matched_tool = self._match_query_to_function_declarations(query)

            if matched_tool:
                if message_bus:
                    message_bus.publish("status_update", {
                        "message": f"📝 Generating help for {matched_tool.__class__.__name__}...",
                        "state": "running",
                        "progress": 50
                    })
                result = self._generate_tool_help(matched_tool, query)
            else:
                if message_bus:
                    message_bus.publish("status_update", {
                        "message": f"❌ Could not find help for '{query}'",
                        "state": "error",
                        "progress": 100
                    })
                result = {
                    'success': False,
                    'error': f"Could not find help information for '{query}'",
                    'message': "Try asking about a specific tool or category."
                }

        # Ensure the response is properly formatted for the caller
        if 'data' in result and 'text' in result['data']:
            result['response'] = result['data']['text']
            if message_bus and result.get('success', False):
                message_bus.publish("status_update", {
                    "message": "✅ Help information retrieved successfully",
                    "state": "complete",
                    "progress": 100
                })

        return result

    def _match_query_to_function_declarations(self, query: str) -> Optional[Tool]:
        """Match a query to function declarations across all tools.

        Args:
            query: User's help query

        Returns:
            Matched Tool object or None if no match found
        """
        if message_bus:
            message_bus.publish("status_update", {
                "message": "🔄 Matching query to available tools...",
                "state": "running",
                "progress": 40
            })

        # Collect all function declarations from all tools
        all_declarations = []
        for tool in self.tools:
            if (hasattr(tool, 'get_function_declarations') and
                callable(getattr(tool, 'get_function_declarations'))):
                try:
                    declarations = tool.get_function_declarations()
                    if declarations:
                        all_declarations.append({
                            'tool': tool,
                            'declarations': declarations
                        })
                except Exception as e:
                    self.logger.error("Error getting function declarations from %s: %s",
                                      tool.__class__.__name__, e)

        if not all_declarations:
            self.logger.warning("No function declarations found in any tools")
            return None

        # Create a prompt for Gemini to match the query to function declarations
        prompt = f"""
        Given the following function declarations and the user query "{query}",
        determine which tool best matches the query. Return ONLY the index number of the best matching tool.

        """

        for i, tool_data in enumerate(all_declarations):
            tool_name = tool_data['tool'].__class__.__name__
            prompt += f"\nTool {i}: {tool_name}\n"

            for func in tool_data['declarations']:
                # Convert FunctionDeclaration to dictionary
                func_dict = func.to_dict()
                prompt += f"- Function: {func_dict.get('name', 'Unknown')}\n"
                prompt += f"  Description: {func_dict.get('description', 'No description')}\n"

        prompt += (f"\nBased on the user query '{query}', which tool index number "
                   f"(0 to {len(all_declarations)-1}) is the best match? "
                   f"Return ONLY a single number and nothing else.")

        try:
            if message_bus:
                message_bus.publish("status_update", {
                    "message": "🤖 Asking Gemini to match your query to the right tool...",
                    "state": "running",
                    "progress": 45
                })

            response = self.gemini_model.generate_content(
                prompt,
                generation_config=self.generation_config
            )

            # Extract the tool index from the response
            response_text = response.text.strip()

            if message_bus:
                message_bus.publish("status_update", {
                    "message": "✅ Gemini has identified a matching tool",
                    "state": "running",
                    "progress": 50
                })

            # Try to extract just the number from the response
            number_match = re.search(r'\b(\d+)\b', response_text)
            if number_match:
                tool_index = int(number_match.group(1))
                if 0 <= tool_index < len(all_declarations):
                    self.logger.info(
                        "Matched query '%s' to tool: %s",
                        query,
                        all_declarations[tool_index]['tool'].__class__.__name__
                    )
                    return all_declarations[tool_index]['tool']
                else:
                    self.logger.error("Extracted tool index %s is out of range (0-%s)",
                                     tool_index, len(all_declarations)-1)

            # If we couldn't extract a valid number, try a different approach
            # Look for the tool name in the response
            for i, tool_data in enumerate(all_declarations):
                tool_name = tool_data['tool'].__class__.__name__
                if tool_name.lower() in response_text.lower():
                    self.logger.info("Matched query '%s' to tool by name: %s",
                                     query, tool_name)
                    return tool_data['tool']

            # Add special case handling for common categories
            if "habitat" in query.lower():
                for tool_data in all_declarations:
                    if "EarthEngineTool" in tool_data['tool'].__class__.__name__:
                        self.logger.info("Matched 'habitat' query to EarthEngineTool")
                        return tool_data['tool']

            if "species" in query.lower():
                for tool_data in all_declarations:
                    if "SpeciesTool" in tool_data['tool'].__class__.__name__:
                        self.logger.info("Matched 'species' query to SpeciesTool")
                        return tool_data['tool']

            self.logger.error("Could not parse tool index from response: %s", response_text)
            return None
        except Exception as e:
            self.logger.error("Error matching query to function declarations: %s", e)
            if message_bus:
                message_bus.publish("status_update", {
                    "message": "❌ Error matching query: %s",
                    "state": "error",
                    "progress": 100
                })
            return None

    def _generate_tool_help(self, tool: Tool, query: str) -> Dict[str, Any]:
        """Generate help information for a tool using Gemini.

        Args:
            tool: The tool to generate help for
            query: Original user query for context

        Returns:
            Dictionary with help information
        """
        try:
            if message_bus:
                message_bus.publish("status_update", {
                    "message": "📝 Generating detailed help for %s...",
                    "state": "running",
                    "progress": 60
                })

            function_declarations = tool.get_function_declarations()

            # Format function declarations for Gemini
            formatted_declarations = []
            for func in function_declarations:
                # Convert FunctionDeclaration to dictionary
                func_dict = func.to_dict()
                formatted_declarations.append({
                    'name': func_dict.get('name', 'Unknown'),
                    'description': func_dict.get('description', 'No description'),
                    'parameters': func_dict.get('parameters', {})
                })

            # Try to load additional information from prompts.md
            additional_info = self._get_additional_info_from_prompts_md(tool.__class__.__name__)

            # Extract screenshot references before generating help text
            screenshot_references = self._extract_screenshot_references(additional_info)
            self.logger.info("Found %s screenshots for %s",
                             len(screenshot_references), tool.__class__.__name__)

            if message_bus and screenshot_references:
                message_bus.publish("status_update", {
                    "message": "🖼️ Found %s screenshots",
                    "state": "running",
                    "progress": 70
                })

            # Format screenshot information for the prompt
            formatted_screenshots = []
            for ref in screenshot_references:
                formatted_screenshots.append({
                    'alt_text': ref.get('alt_text', 'Screenshot'),
                    'github_url': ref.get('github_url', ''),
                    'prompt_context': ref.get('prompt_context', '')
                })

            if message_bus:
                message_bus.publish("status_update", {
                    "message": "🤖 Asking Gemini to generate comprehensive documentation...",
                    "state": "running",
                    "progress": 75
                })

            # Use Gemini to generate a comprehensive description
            prompt = f"""
            Generate a comprehensive help description for the {tool.__class__.__name__}
            based on the following function declarations and additional information.
            The user asked about "{query}".

            Function declarations:
            {formatted_declarations}

            Additional information from documentation:
            {additional_info}

            Include:
            1. An detailed overview of what the tool does
            2. Descriptions of each function
            3. Example prompts that users can try (at least 3-5 specific examples)
            4. Any limitations or requirements

            DO NOT include screenshots or references to screenshots in your response.
            Format the response in Markdown with clear sections and examples.
            """

            response = self.gemini_model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            help_text = response.text

            if message_bus:
                message_bus.publish("status_update", {
                    "message": "📄 Documentation generated, preparing response...",
                    "state": "running",
                    "progress": 85
                })

            # Now we can safely append screenshots without duplication
            if formatted_screenshots:
                help_text += "\n\n## Screenshots\n\n"
                for i, screenshot in enumerate(formatted_screenshots):
                    help_text += f"### {screenshot['alt_text']}\n\n"
                    help_text += f"![{screenshot['alt_text']}]({screenshot['github_url']})\n\n"
                    if screenshot['prompt_context']:
                        help_text += f"*{screenshot['prompt_context']}*\n\n"

            # Prepare the response data
            response_data = {
                'text': help_text,
                'tool_name': tool.__class__.__name__,
                'functions': formatted_declarations,
                'screenshots': formatted_screenshots
            }

            # Immediately signal completion to improve UI responsiveness
            if message_bus:
                message_bus.publish("status_update", {
                    "message": "✅ Help information ready",
                    "state": "complete",
                    "progress": 100
                })

            return {
                'success': True,
                'data': response_data,
                'message': f"Help information for '{query}' retrieved successfully",
                'response': help_text
            }
        except Exception as e:
            self.logger.error("Error generating tool help: %s", e)
            if message_bus:
                message_bus.publish("status_update", {
                    "message": "❌ Error generating help: %s",
                    "state": "error",
                    "progress": 100
                })
            return {
                'success': False,
                'error': f"Error generating help information: {str(e)}",
                'message': "An error occurred while generating help information."
            }

    def _get_additional_info_from_prompts_md(self, tool_name: str) -> str:
        """Extract additional information about a tool from prompts.md.

        Args:
            tool_name: Name of the tool to find information for

        Returns:
            String containing relevant information from prompts.md
        """
        try:
            # Fetch prompts.md from GitHub
            prompts_content = ""
            github_url = "https://raw.githubusercontent.com/csboros/biochat/main/prompts.md"

            try:
                response = requests.get(github_url, timeout=10)
                if response.status_code == 200:
                    prompts_content = response.text
                    self.logger.info("Fetched prompts.md from GitHub URL: %s", github_url)
                else:
                    self.logger.warning("Failed to fetch prompts.md from GitHub. Status code: %s",
                                        response.status_code)
            except Exception as e:
                self.logger.error("Error fetching prompts.md from GitHub: %s", e)

            if not prompts_content:
                self.logger.warning("Could not fetch prompts.md from GitHub")
                return "No additional information available."

            # Extract relevant sections for this tool
            return self._extract_tool_sections(prompts_content, tool_name)
        except Exception as e:
            self.logger.error("Error extracting information from prompts.md: %s", e, exc_info=True)
            return "Error extracting additional information."

    def _extract_tool_sections(self, prompts_content: str, tool_name: str) -> str:
        """Extract sections related to a specific tool from prompts content.

        Args:
            prompts_content: Content of prompts.md
            tool_name: Name of the tool to find information for

        Returns:
            String containing relevant sections
        """
        # Look for sections that might be related to this tool
        tool_keywords = [
            tool_name,
            tool_name.replace("Tool", ""),
            # Add common variations of the tool name
            tool_name.replace("Tool", " Tool"),
            tool_name.replace("Tool", "s"),
            tool_name.replace("Tool", " Analysis"),
        ]

        relevant_sections = []

        # Try to find sections with headers containing the tool name
        for keyword in tool_keywords:
            # Look for markdown headers with the keyword
            header_pattern = re.compile(
                r'#{1,6}\s+([^#\n]*' + re.escape(keyword) + r'[^#\n]*)',
                re.IGNORECASE
            )
            headers = header_pattern.findall(prompts_content)

            for header in headers:
                # For each header, try to extract the section content
                header_pattern = re.compile(
                    r'#{1,6}\s+' + re.escape(header) + r'\s*(.*?)(?=#{1,6}\s+|$)',
                    re.DOTALL
                )
                sections = header_pattern.findall(prompts_content)
                relevant_sections.extend(sections)
                self.logger.info("Found section for '%s' with header '%s'",
                                 keyword, header)

        if not relevant_sections:
            for keyword in tool_keywords:
                paragraph_pattern = re.compile(
                    r'((?:[^\n]+\n){1,5}[^\n]*' + re.escape(keyword) +
                    r'[^\n]*(?:\n[^\n]+){1,5})',
                    re.IGNORECASE
                )
                paragraphs = paragraph_pattern.findall(prompts_content)
                if paragraphs:
                    relevant_sections.extend(paragraphs)
                    self.logger.info("Found %s paragraphs containing '%s'",
                                     len(paragraphs), keyword)

        if not relevant_sections:
            self.logger.info("No specific sections found for %s in prompts.md", tool_name)
            return "No specific information found in documentation."

        combined_sections = "\n\n".join(relevant_sections)
        self.logger.info("Extracted %s sections with total length %s",
                         len(relevant_sections), len(combined_sections))
        return combined_sections

    def _extract_screenshot_references(self, text: str) -> List[Dict[str, str]]:
        """Extract references to screenshots from text.

        Args:
            text: Text to search for screenshot references

        Returns:
            List of screenshot references with alt text and path
        """
        try:
            # Look for markdown image references - this pattern matches ![alt text](image_url)
            image_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)', re.DOTALL)
            images = image_pattern.findall(text)

            # Also look for HTML image tags - <img src="image_url" alt="alt text">
            html_image_pattern = re.compile(
                r'<img\s+src=[\'"]([^\'"]+)[\'"](?:\s+alt=[\'"]([^\'"]+)[\'"])?',
                re.DOTALL
            )
            html_images = html_image_pattern.findall(text)

            # Format as structured references
            references = []

            # Process markdown images
            for alt_text, image_path in images:
                # For GitHub URLs, convert relative paths to absolute GitHub URLs
                github_url = ""
                if image_path.startswith("http"):
                    # Already an absolute URL
                    github_url = image_path
                else:
                    # Convert relative path to GitHub URL
                    # Assuming the base repo URL is https://github.com/csboros/biochat
                    base_url = "https://raw.githubusercontent.com/csboros/biochat/main/"
                    # Remove leading ./ or ../ from the path
                    clean_path = re.sub(r'^\.{1,2}/', '', image_path)
                    github_url = base_url + clean_path

                # Extract the prompt context (text before the image)
                prompt_context = self._extract_prompt_context(text, f"![{alt_text}]({image_path})")

                references.append({
                    'alt_text': alt_text or 'Screenshot',
                    'path': image_path,
                    'github_url': github_url,
                    'prompt_context': prompt_context
                })

            for image_path, alt_text in html_images:
                # For GitHub URLs, convert relative paths to absolute GitHub URLs
                github_url = ""
                if image_path.startswith("http"):
                    # Already an absolute URL
                    github_url = image_path
                else:
                    # Convert relative path to GitHub URL
                    base_url = "https://raw.githubusercontent.com/csboros/biochat/main/"
                    # Remove leading ./ or ../ from the path
                    clean_path = re.sub(r'^\.{1,2}/', '', image_path)
                    github_url = base_url + clean_path

                # Extract the prompt context (text before the image)
                prompt_context = self._extract_prompt_context(text, f'<img src="{image_path}"')

                references.append({
                    'alt_text': alt_text or 'Screenshot',
                    'path': image_path,
                    'github_url': github_url,
                    'prompt_context': prompt_context
                })

            self.logger.info("Found %s screenshot references in documentation", len(references))
            return references
        except Exception as e:
            self.logger.error("Error extracting screenshot references: %s", e)
            return []

    def _extract_prompt_context(self, text: str, image_marker: str) -> str:
        """Extract the prompt context (text before an image).

        Args:
            text: Full text to search in
            image_marker: The image marker to find

        Returns:
            The prompt context text
        """
        try:
            # Find the position of the image marker
            pos = text.find(image_marker)
            if pos == -1:
                return ""

            # Look for the start of the paragraph or section containing this image
            # First try to find the nearest preceding newline + text
            paragraph_start = text.rfind("\n\n", 0, pos)
            if paragraph_start == -1:
                paragraph_start = 0
            else:
                paragraph_start += 2  # Skip the newlines

            # Extract the text from paragraph start to the image
            prompt_text = text[paragraph_start:pos].strip()

            # If the prompt text is too short, try to get more context
            if len(prompt_text) < 50:
                # Look for the previous paragraph
                prev_paragraph_start = text.rfind("\n\n", 0, paragraph_start - 2)
                if prev_paragraph_start != -1:
                    prompt_text = text[prev_paragraph_start + 2:pos].strip()

            return prompt_text
        except Exception as e:
            self.logger.error("Error extracting prompt context: %s", e)
            return ""

    def _get_general_help(self) -> Dict[str, Any]:
        """Generate general help information about all available tools.

        Returns:
            Dictionary with general help information
        """
        # Collect all function declarations from all tools
        all_functions = []
        for tool in self.tools:
            if (hasattr(tool, 'get_function_declarations') and
                callable(getattr(tool, 'get_function_declarations'))):
                try:
                    declarations = tool.get_function_declarations()
                    if declarations:
                        for func in declarations:
                            # Convert FunctionDeclaration to dictionary
                            func_dict = func.to_dict()
                            all_functions.append({
                                'tool': tool.__class__.__name__,
                                'name': func_dict.get('name', 'Unknown'),
                                'description': func_dict.get('description', 'No description')
                            })
                except Exception as e:
                    self.logger.error("Error getting function declarations from %s: %s",
                                      tool.__class__.__name__, e)

        # Use Gemini to generate a comprehensive overview
        prompt = """
        Generate a comprehensive overview of the BioChat system based on these available functions.
        Format the response in Markdown with clear sections and examples of what users can ask.

        Available functions:
        """ + str(all_functions)

        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            overview_text = response.text

            return {
                'success': True,
                'data': {
                    'text': overview_text,
                    'available_tools': list(set(f['tool'] for f in all_functions))
                },
                'message': "General help information retrieved successfully"
            }
        except Exception as e:
            self.logger.error("Error generating general help: %s", e)
            return {
                'success': False,
                'error': "Error generating help information: %s",
                'message': "An error occurred while generating help information."
            }
