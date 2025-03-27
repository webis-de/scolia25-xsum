"""
This module provides a client for interacting with the Azure-hosted Phi-3 language model.
It handles authentication, request formatting, and response parsing.

The client is designed for the Phi-3 Small model but can be configured for other endpoints.
"""

import os
from typing import Optional, Dict, Any

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv


class Phi3Client:
    """
    Phi3 Azure API client for interacting with Azure-hosted Phi-3 language models.

    This client handles authentication, request formatting, and response parsing
    for the Azure Phi-3 API endpoints.

    Attributes:
        endpoint (str): The API endpoint URL
        api_key (str): The API authentication key
        client (ChatCompletionsClient): The Azure ChatCompletionsClient instance
        system_prompt (str): The system prompt that guides model behavior
        temperature (float): Sampling temperature (higher = more random)
        top_p (float): Nucleus sampling parameter (lower = more focused)
        max_tokens (int): Maximum number of tokens to generate in responses
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key_env_var: str = "PHI3_SMALL_API_KEY",
        system_prompt: str = "You are an AI assistant that helps people find information.",
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 4096,
    ):
        """
        Initialize the Phi3 client with the specified configuration.

        Args:
            endpoint: API endpoint URL (optional, defaults to Phi-3 Small endpoint)
            api_key_env_var: Environment variable name containing the API key
            system_prompt: Default system prompt for all conversations
            temperature: Sampling temperature for generation
            top_p: Top-p (nucleus) sampling parameter
            max_tokens: Maximum number of tokens to generate

        Raises:
            ValueError: If the API key is not found in environment variables
        """
        # Load environment variables from .env file
        load_dotenv(override=True)

        # Get API key from environment
        self.api_key = os.getenv(api_key_env_var)
        if not self.api_key:
            raise ValueError(
                f"API key not found. Please set the {api_key_env_var} environment variable."
            )

        # Use default endpoint if none provided
        self.endpoint = endpoint or os.getenv("PHI3_SMALL_ENDPOINT")
        if not self.endpoint:
            raise ValueError(
                "API endpoint not found. Please provide an endpoint or set the PHI3_SMALL_ENDPOINT environment variable."
            )

        # Initialize Azure client
        self.credential = AzureKeyCredential(self.api_key)
        self.client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=self.credential,
        )

        # Store generation parameters
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def build_payload(self, user_prompt: str) -> Dict[str, Any]:
        """
        Construct the API request payload with the user prompt.

        Args:
            user_prompt: The user's input text

        Returns:
            Dictionary containing the formatted API request
        """
        return {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    def send_request(
        self, payload: Dict[str, Any], verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Send the request to the Azure API and return the response.

        Args:
            payload: The formatted request payload
            verbose: Whether to print model information

        Returns:
            Raw API response

        Raises:
            SystemExit: If the API request fails
        """
        try:
            # Send completion request to API
            response = self.client.complete(payload)

            # Print model information if verbose mode is enabled
            if verbose:
                model_info = self.client.get_model_info()
                print("Model name:", model_info.model_name)
                print("Model type:", model_info.model_type)
                print("Model provider name:", model_info.model_provider_name)
            return response
        except Exception as e:
            raise SystemExit(f"Failed to make the request. Error: {e}")

    def get_response(self, user_prompt: str, verbose: bool = False) -> str:
        """
        Get a response from the model for the given user prompt.

        This method handles the full request-response cycle.

        Args:
            user_prompt: The user's input text
            verbose: Whether to print model information

        Returns:
            The model's text response

        Raises:
            ValueError: If the response structure is unexpected
        """
        # Build the request payload
        payload = self.build_payload(user_prompt)

        # Send the request to the API
        response = self.send_request(payload, verbose=verbose)

        # Extract text from response
        try:
            return response.choices[0].message.content
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected response structure: {e}")
