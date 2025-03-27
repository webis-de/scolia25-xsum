"""
A client for interacting with Azure's GPT-4o Mini model. This client handles
authentication, request formatting, and response parsing for text generation
tasks through Azure OpenAI Services.

This module provides a simple interface for sending prompts to the GPT-4o Mini
model and retrieving the generated responses.
"""

import os
import requests
from dotenv import load_dotenv
from typing import Optional, Dict, Any


class GPT4OMini:
    """
    GPT-4o Mini Azure API client for text generation.

    This class provides methods to interact with Azure's GPT-4o Mini deployment,
    handling API authentication, request formatting, and response parsing.

    Attributes:
        endpoint (str): The Azure API endpoint for the GPT-4o Mini model.
        api_key (str): The API key for authentication.
        headers (dict): HTTP request headers including content type and API key.
        system_prompt (str): Instructions defining the AI assistant's behavior.
        temperature (float): Controls randomness in output generation (0.0-1.0).
        top_p (float): Controls diversity via nucleus sampling (0.0-1.0).
        max_tokens (int): Maximum number of tokens in the generated response.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key_env_var: str = "QED_OPENAI_API_KEY",
        system_prompt: str = "You are an AI assistant that helps people find information.",
        temperature: float = 0.3,
        top_p: float = 0.95,
        max_tokens: int = 800,
    ):
        """
        Initialize the GPT-4o Mini client.

        Args:
            endpoint: Optional custom API endpoint URL.
            api_key_env_var: Environment variable name containing the API key.
            system_prompt: Instructions that define the assistant's behavior.
            temperature: Controls randomness (lower is more deterministic).
            top_p: Controls diversity via nucleus sampling.
            max_tokens: Maximum tokens to generate in the response.

        Raises:
            ValueError: If the API key environment variable is not set.
        """
        # Load environment variables and get API key
        load_dotenv(override=True)
        self.api_key = os.getenv(api_key_env_var)
        if not self.api_key:
            raise ValueError(
                f"API key not found. Please set the {api_key_env_var} environment variable."
            )

        # Set up request headers with API key
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

        # Use provided endpoint or default to Azure deployment URL
        # Use provided endpoint or default to placeholder
        self.endpoint = endpoint or os.getenv(
            "AZURE_GPT4O_MINI_ENDPOINT",
            "https://your-resource-name.openai.azure.com/openai/deployments/your-deployment-name/"
            "chat/completions?api-version=2024-02-15-preview",
        )

        # Store model configuration parameters
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def build_payload(self, user_prompt: str) -> Dict[str, Any]:
        """
        Construct the request payload for the API.

        Args:
            user_prompt: The user's input text to send to the model.

        Returns:
            A dictionary containing the formatted request payload.
        """
        return {
            "messages": [
                # System message defines the assistant's behavior
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                },
                # User message contains the prompt to respond to
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}],
                },
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

    def send_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send the request to the API endpoint.

        Args:
            payload: The formatted request data to send.

        Returns:
            The JSON response from the API.

        Raises:
            SystemExit: If the request fails for any reason.
        """
        try:
            # Send POST request to the API endpoint
            response = requests.post(self.endpoint, headers=self.headers, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json()
        except requests.RequestException as e:
            raise SystemExit(f"Failed to make the request. Error: {e}")

    def get_response(self, user_prompt: str) -> str:
        """
        Get a response from GPT-4o Mini for the given prompt.

        Args:
            user_prompt: The text prompt to send to the model.

        Returns:
            The generated text response from the model.

        Raises:
            ValueError: If the response format is unexpected.
        """
        # Build and send the request
        payload = self.build_payload(user_prompt)
        response = self.send_request(payload)

        # Extract the generated text from the response
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected response structure: {e}")
