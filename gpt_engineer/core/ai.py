"""
AI Module

This module provides an AI class that interfaces with language models to perform various tasks such as
starting a conversation, advancing the conversation, and handling message serialization. It also includes
backoff strategies for handling rate limit errors from the OpenAI API.

Classes:
    AI: A class that interfaces with language models for conversation management and message serialization.
    ModelRegistry: A class that manages different LLM providers and their configurations.

Functions:
    serialize_messages(messages: List[Message]) -> str
        Serialize a list of messages to a JSON string.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import backoff
import openai
import pyperclip

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from gpt_engineer.core.token_usage import TokenUsageLog

# Type hint for a chat message
Message = Union[AIMessage, HumanMessage, SystemMessage]

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for a language model."""
    name: str
    provider: str
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    streaming: bool = True
    azure_endpoint: Optional[str] = None
    vision: bool = False

class ModelProvider(ABC):
    """Abstract base class for model providers."""
    
    @abstractmethod
    def create_model(self, config: ModelConfig) -> BaseChatModel:
        """Create a model instance with the given configuration."""
        pass

class OpenAIProvider(ModelProvider):
    """Provider for OpenAI models."""
    
    def create_model(self, config: ModelConfig) -> BaseChatModel:
        if config.azure_endpoint:
            return AzureChatOpenAI(
                azure_endpoint=config.azure_endpoint,
                model_name=config.name,
                temperature=config.temperature,
                streaming=config.streaming,
            )
        return ChatOpenAI(
            model_name=config.name,
            temperature=config.temperature,
            streaming=config.streaming,
        )

class AnthropicProvider(ModelProvider):
    """Provider for Anthropic models."""
    
    def create_model(self, config: ModelConfig) -> BaseChatModel:
        return ChatAnthropic(
            model_name=config.name,
            temperature=config.temperature,
            streaming=config.streaming,
        )

class ModelRegistry:
    """Registry for managing different LLM providers."""
    
    def __init__(self):
        self.providers: Dict[str, ModelProvider] = {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
        }
    
    def get_provider(self, provider_name: str) -> ModelProvider:
        """Get a provider by name."""
        if provider_name not in self.providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        return self.providers[provider_name]
    
    def create_model(self, config: ModelConfig) -> BaseChatModel:
        """Create a model instance with the given configuration."""
        provider = self.get_provider(config.provider)
        return provider.create_model(config)

class AI:
    """
    A class that interfaces with language models for conversation management and message serialization.

    This class provides methods to start and advance conversations, handle message serialization,
    and implement backoff strategies for rate limit errors when interacting with the OpenAI API.

    Attributes
    ----------
    model_config : ModelConfig
        The configuration for the language model.
    registry : ModelRegistry
        The registry for managing different LLM providers.
    llm : BaseChatModel
        The language model instance for conversation management.
    token_usage_log : TokenUsageLog
        A log for tracking token usage during conversations.
    """

    def __init__(
        self,
        model_name="gpt-4-turbo",
        temperature=0.1,
        azure_endpoint=None,
        streaming=True,
        vision=False,
        provider="openai",
    ):
        """
        Initialize the AI class.

        Parameters
        ----------
        model_name : str, optional
            The name of the language model to use, by default "gpt-4-turbo"
        temperature : float, optional
            The temperature setting for the language model, by default 0.1
        azure_endpoint : str, optional
            The endpoint URL for the Azure-hosted language model, by default None
        streaming : bool, optional
            A flag indicating whether to use streaming for the language model, by default True
        vision : bool, optional
            A flag indicating whether to use vision capabilities, by default False
        provider : str, optional
            The provider of the language model, by default "openai"
        """
        self.model_config = ModelConfig(
            name=model_name,
            provider=provider,
            temperature=temperature,
            azure_endpoint=azure_endpoint,
            streaming=streaming,
            vision=vision,
        )
        self.registry = ModelRegistry()
        self.llm = self._create_chat_model()
        self.token_usage_log = TokenUsageLog(model_name)

    def start(self, system: str, user: Any, *, step_name: str) -> List[Message]:
        """
        Start the conversation with a system message and a user message.

        Parameters
        ----------
        system : str
            The content of the system message.
        user : str
            The content of the user message.
        step_name : str
            The name of the step.

        Returns
        -------
        List[Message]
            The list of messages in the conversation.
        """

        messages: List[Message] = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
        return self.next(messages, step_name=step_name)

    def _extract_content(self, content):
        """
        Extracts text content from a message, supporting both string and list types.
        Parameters
        ----------
        content : Union[str, List[dict]]
            The content of a message, which could be a string or a list.
        Returns
        -------
        str
            The extracted text content.
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list) and content and "text" in content[0]:
            # Assuming the structure of list content is [{'type': 'text', 'text': 'Some text'}, ...]
            return content[0]["text"]
        else:
            return ""

    def _collapse_text_messages(self, messages: List[Message]):
        """
        Combine consecutive messages of the same type into a single message, where if the message content
        is a list type, the first text element's content is taken. This method keeps `combined_content` as a string.

        This method iterates through the list of messages, combining consecutive messages of the same type
        by joining their content with a newline character. If the content is a list, it extracts text from the first
        text element's content. This reduces the number of messages and simplifies the conversation for processing.

        Parameters
        ----------
        messages : List[Message]
            The list of messages to collapse.

        Returns
        -------
        List[Message]
            The list of messages after collapsing consecutive messages of the same type.
        """
        collapsed_messages = []
        if not messages:
            return collapsed_messages

        previous_message = messages[0]
        combined_content = self._extract_content(previous_message.content)

        for current_message in messages[1:]:
            if current_message.type == previous_message.type:
                combined_content += "\n\n" + self._extract_content(
                    current_message.content
                )
            else:
                collapsed_messages.append(
                    previous_message.__class__(content=combined_content)
                )
                previous_message = current_message
                combined_content = self._extract_content(current_message.content)

        collapsed_messages.append(previous_message.__class__(content=combined_content))
        return collapsed_messages

    def next(
        self,
        messages: List[Message],
        prompt: Optional[str] = None,
        *,
        step_name: str,
    ) -> List[Message]:
        """
        Advances the conversation by sending message history
        to LLM and updating with the response.

        Parameters
        ----------
        messages : List[Message]
            The list of messages in the conversation.
        prompt : Optional[str], optional
            The prompt to use, by default None.
        step_name : str
            The name of the step.

        Returns
        -------
        List[Message]
            The updated list of messages in the conversation.
        """

        if prompt:
            messages.append(HumanMessage(content=prompt))

        logger.debug(
            "Creating a new chat completion: %s",
            "\n".join([m.pretty_repr() for m in messages]),
        )

        if not self.model_config.vision:
            messages = self._collapse_text_messages(messages)

        response = self.backoff_inference(messages)

        self.token_usage_log.update_log(
            messages=messages, answer=response, step_name=step_name
        )
        messages.append(AIMessage(content=response))
        logger.debug(f"Chat completion finished: {messages}")

        return messages

    @backoff.on_exception(
        backoff.expo,
        (Exception,),  # Handle all exceptions for now
        max_tries=5,
        max_time=30,
    )
    def backoff_inference(self, messages: List[Message]) -> str:
        """
        Perform inference with exponential backoff for rate limits and API errors.

        Args:
            messages: List of messages to process.

        Returns:
            str: The model's response.

        Raises:
            Exception
                If there is an error during inference.
        """
        try:
            return self.llm.invoke(messages).content
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise

    @staticmethod
    def serialize_messages(messages: List[Message]) -> str:
        """
        Serialize a list of messages to a JSON string.

        Parameters
        ----------
        messages : List[Message]
            The list of messages to serialize.

        Returns
        -------
        str
            The serialized messages as a JSON string.
        """
        return json.dumps(messages_to_dict(messages))

    @staticmethod
    def deserialize_messages(jsondictstr: str) -> List[Message]:
        """
        Deserialize a JSON string to a list of messages.

        Parameters
        ----------
        jsondictstr : str
            The JSON string to deserialize.

        Returns
        -------
        List[Message]
            The deserialized list of messages.
        """
        data = json.loads(jsondictstr)
        # Modify implicit is_chunk property to ALWAYS false
        # since Langchain's Message schema is stricter
        prevalidated_data = [
            {**item, "tools": {**item.get("tools", {}), "is_chunk": False}}
            for item in data
        ]
        return list(messages_from_dict(prevalidated_data))  # type: ignore

    def _create_chat_model(self) -> BaseChatModel:
        """
        Create a chat model with the specified configuration.

        Returns
        -------
        BaseChatModel
            The created chat model instance.

        Raises
        ------
        ValueError
            If the model configuration is invalid.
        RuntimeError
            If there is an error creating the chat model.
        """
        try:
            return self.registry.create_model(self.model_config)
        except ValueError as e:
            logger.error(f"Invalid model configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Error creating chat model: {e}")
            raise RuntimeError(f"Failed to create chat model: {e}")


def serialize_messages(messages: List[Message]) -> str:
    return AI.serialize_messages(messages)


class ClipboardAI(AI):
    # Ignore not init superclass
    def __init__(self, **_):  # type: ignore
        self.model_config = ModelConfig(
            name="clipboard_llm",
            provider="openai",
            temperature=0.1,
            streaming=True,
            vision=False,
        )
        self.registry = ModelRegistry()
        self.token_usage_log = TokenUsageLog("clipboard_llm")

    @staticmethod
    def serialize_messages(messages: List[Message]) -> str:
        return "\n\n".join([f"{m.type}:\n{m.content}" for m in messages])

    @staticmethod
    def multiline_input():
        print("Enter/Paste your content. Ctrl-D or Ctrl-Z ( windows ) to save it.")
        content = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            content.append(line)
        return "\n".join(content)

    def next(
        self,
        messages: List[Message],
        prompt: Optional[str] = None,
        *,
        step_name: str,
    ) -> List[Message]:
        """
        Not yet fully supported
        """
        if prompt:
            messages.append(HumanMessage(content=prompt))

        logger.debug(f"Creating a new chat completion: {messages}")

        msgs = self.serialize_messages(messages)
        pyperclip.copy(msgs)
        Path("clipboard.txt").write_text(msgs)
        print(
            "Messages copied to clipboard and written to clipboard.txt,",
            len(msgs),
            "characters in total",
        )

        response = self.multiline_input()

        messages.append(AIMessage(content=response))
        logger.debug(f"Chat completion finished: {messages}")

        return messages
