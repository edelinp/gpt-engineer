import base64
import io
import logging
import math

from dataclasses import dataclass
from typing import List, Union

import tiktoken

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from PIL import Image

# workaround for function moved in:
# https://github.com/langchain-ai/langchain/blob/535db72607c4ae308566ede4af65295967bb33a8/libs/community/langchain_community/callbacks/openai_info.py
try:
    from langchain.callbacks.openai_info import (
        get_openai_token_cost_for_model,  # fmt: skip
    )
except ImportError:
    from langchain_community.callbacks.openai_info import (
        get_openai_token_cost_for_model,  # fmt: skip
    )


Message = Union[AIMessage, HumanMessage, SystemMessage]

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """
    Dataclass representing token usage statistics for a conversation step.

    Attributes
    ----------
    step_name : str
        The name of the conversation step.
    in_step_prompt_tokens : int
        The number of prompt tokens used in the step.
    in_step_completion_tokens : int
        The number of completion tokens used in the step.
    in_step_total_tokens : int
        The total number of tokens used in the step.
    total_prompt_tokens : int
        The cumulative number of prompt tokens used up to this step.
    total_completion_tokens : int
        The cumulative number of completion tokens used up to this step.
    total_tokens : int
        The cumulative total number of tokens used up to this step.
    """

    """
    Represents token usage statistics for a conversation step.
    """

    step_name: str
    in_step_prompt_tokens: int
    in_step_completion_tokens: int
    in_step_total_tokens: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int


class Tokenizer:
    """
    Tokenizer for counting tokens in text.
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self._tiktoken_tokenizer = (
            tiktoken.encoding_for_model(model_name)
            if "gpt-4" in model_name or "gpt-3.5" in model_name
            else tiktoken.get_encoding("cl100k_base")
        )

    def num_tokens(self, txt: str) -> int:
        """
        Get the number of tokens in a text.

        Parameters
        ----------
        txt : str
            The text to count the tokens in.

        Returns
        -------
        int
            The number of tokens in the text.
        """
        return len(self._tiktoken_tokenizer.encode(txt))

    def num_tokens_for_base64_image(
        self, image_base64: str, detail: str = "high"
    ) -> int:
        """
        Calculate the token size for a base64 encoded image based on OpenAI's token calculation rules.

        Parameters:
        - image_base64 (str): The base64 encoded string of the image.
        - detail (str): The detail level of the image, 'low' or 'high'.

        Returns:
        - int: The token size of the image.
        """

        if detail == "low":
            return 85  # Fixed cost for low detail images

        # Decode image from base64
        image_data = base64.b64decode(image_base64)

        # Convert byte data to image for size extraction
        image = Image.open(io.BytesIO(image_data))

        # Calculate the initial scale to fit within 2048 square while maintaining aspect ratio
        max_dimension = max(image.size)
        scale_factor = min(2048 / max_dimension, 1)  # Ensure we don't scale up
        new_width = int(image.size[0] * scale_factor)
        new_height = int(image.size[1] * scale_factor)

        # Scale such that the shortest side is 768px
        shortest_side = min(new_width, new_height)
        if shortest_side > 768:
            resize_factor = 768 / shortest_side
            new_width = int(new_width * resize_factor)
            new_height = int(new_height * resize_factor)

        # Calculate the number of 512px tiles needed
        width_tiles = math.ceil(new_width / 512)
        height_tiles = math.ceil(new_height / 512)
        total_tiles = width_tiles * height_tiles

        # Each tile costs 170 tokens, plus a base cost of 85 tokens for high detail
        token_cost = total_tiles * 170 + 85

        return token_cost

    def num_tokens_from_messages(self, messages: List[Message]) -> int:
        """
        Get the total number of tokens used by a list of messages, accounting for text and base64 encoded images.

        Parameters
        ----------
        messages : List[Message]
            The list of messages to count the tokens in.

        Returns
        -------
        int
            The total number of tokens used by the messages.
        """
        n_tokens = 0
        for message in messages:
            n_tokens += 4  # Account for message framing tokens

            if isinstance(message.content, str):
                # Content is a simple string
                n_tokens += self.num_tokens(message.content)
            elif isinstance(message.content, list):
                # Content is a list, potentially mixed with text and images
                for item in message.content:
                    if item.get("type") == "text":
                        n_tokens += self.num_tokens(item["text"])
                    elif item.get("type") == "image_url":
                        image_detail = item["image_url"].get("detail", "high")
                        image_base64 = item["image_url"].get("url")
                        n_tokens += self.num_tokens_for_base64_image(
                            image_base64, detail=image_detail
                        )

            n_tokens += 2  # Account for assistant's reply framing tokens

        return n_tokens


class TokenUsageLog:
    """
    A class for tracking and optimizing token usage across different models and providers.
    
    This class provides methods for tracking token usage, estimating costs, and optimizing
    token usage across different models and providers.
    
    Attributes
    ----------
    model_name : str
        The name of the model being used.
    tokenizer : Tokenizer
        The tokenizer for the model.
    usage_log : List[TokenUsage]
        A list of token usage records.
    """

    def __init__(self, model_name: str):
        """
        Initialize the TokenUsageLog.

        Parameters
        ----------
        model_name : str
            The name of the model being used.
        """
        self.model_name = model_name
        self.tokenizer = Tokenizer(model_name)
        self.usage_log: List[TokenUsage] = []
        self._optimization_threshold = 0.8  # 80% of max tokens
        self._cost_threshold = 0.1  # $0.10 per request

    def update_log(self, messages: List[Message], answer: str, step_name: str) -> None:
        """
        Update the token usage log with new usage information.

        Parameters
        ----------
        messages : List[Message]
            The messages sent to the model.
        answer : str
            The answer received from the model.
        step_name : str
            The name of the step being executed.
        """
        in_step_prompt_tokens = self.tokenizer.num_tokens_from_messages(messages)
        in_step_completion_tokens = self.tokenizer.num_tokens(answer)
        in_step_total_tokens = in_step_prompt_tokens + in_step_completion_tokens

        total_prompt_tokens = sum(usage.in_step_prompt_tokens for usage in self.usage_log) + in_step_prompt_tokens
        total_completion_tokens = sum(usage.in_step_completion_tokens for usage in self.usage_log) + in_step_completion_tokens
        total_tokens = total_prompt_tokens + total_completion_tokens

        self.usage_log.append(
            TokenUsage(
                step_name=step_name,
                in_step_prompt_tokens=in_step_prompt_tokens,
                in_step_completion_tokens=in_step_completion_tokens,
                in_step_total_tokens=in_step_total_tokens,
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
                total_tokens=total_tokens,
            )
        )

        # Log warning if approaching token limits
        if self._is_approaching_token_limit(in_step_total_tokens):
            logger.warning(f"Approaching token limit in step {step_name}")

        # Log warning if cost exceeds threshold
        if self.usage_cost() and self.usage_cost() > self._cost_threshold:
            logger.warning(f"Cost exceeds threshold in step {step_name}")

    def _is_approaching_token_limit(self, tokens: int) -> bool:
        """
        Check if the token usage is approaching the limit.

        Parameters
        ----------
        tokens : int
            The number of tokens to check.

        Returns
        -------
        bool
            True if approaching the limit, False otherwise.
        """
        max_tokens = self._get_max_tokens()
        return tokens > (max_tokens * self._optimization_threshold)

    def _get_max_tokens(self) -> int:
        """
        Get the maximum number of tokens for the model.

        Returns
        -------
        int
            The maximum number of tokens.
        """
        # Default max tokens for different models
        max_tokens_map = {
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "gpt-3.5-turbo": 16385,
            "claude-2": 100000,
            "claude-3-opus": 200000,
            "claude-3-sonnet": 200000,
        }
        return max_tokens_map.get(self.model_name, 4096)

    def optimize_messages(self, messages: List[Message]) -> List[Message]:
        """
        Optimize messages to reduce token usage.

        Parameters
        ----------
        messages : List[Message]
            The messages to optimize.

        Returns
        -------
        List[Message]
            The optimized messages.
        """
        # Remove redundant system messages
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        if len(system_messages) > 1:
            # Keep only the last system message
            messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
            messages.insert(0, system_messages[-1])

        # Truncate long messages
        max_tokens_per_message = self._get_max_tokens() // 4
        for i, msg in enumerate(messages):
            if self.tokenizer.num_tokens(msg.content) > max_tokens_per_message:
                messages[i] = self._truncate_message(msg, max_tokens_per_message)

        return messages

    def _truncate_message(self, message: Message, max_tokens: int) -> Message:
        """
        Truncate a message to fit within token limits.

        Parameters
        ----------
        message : Message
            The message to truncate.
        max_tokens : int
            The maximum number of tokens.

        Returns
        -------
        Message
            The truncated message.
        """
        content = message.content
        while self.tokenizer.num_tokens(content) > max_tokens:
            # Remove the last paragraph
            paragraphs = content.split("\n\n")
            if len(paragraphs) > 1:
                content = "\n\n".join(paragraphs[:-1])
            else:
                # If only one paragraph, remove the last sentence
                sentences = content.split(". ")
                if len(sentences) > 1:
                    content = ". ".join(sentences[:-1]) + "."
                else:
                    # If only one sentence, truncate it
                    content = content[:max_tokens * 4]  # Rough estimate of chars per token

        return type(message)(content=content)

    def log(self) -> List[TokenUsage]:
        """
        Get the token usage log.

        Returns
        -------
        List[TokenUsage]
            A log of token usage details per step in the conversation.
        """
        return self.usage_log

    def format_log(self) -> str:
        """
        Format the token usage log as a CSV string.

        Returns
        -------
        str
            The token usage log formatted as a CSV string.
        """
        result = "step_name,prompt_tokens_in_step,completion_tokens_in_step,total_tokens_in_step,total_prompt_tokens,total_completion_tokens,total_tokens\n"
        for log in self.usage_log:
            result += f"{log.step_name},{log.in_step_prompt_tokens},{log.in_step_completion_tokens},{log.in_step_total_tokens},{log.total_prompt_tokens},{log.total_completion_tokens},{log.total_tokens}\n"
        return result

    def is_openai_model(self) -> bool:
        """
        Check if the model is an OpenAI model.

        Returns
        -------
        bool
            True if the model is an OpenAI model, False otherwise.
        """
        return "gpt" in self.model_name.lower()

    def total_tokens(self) -> int:
        """
        Return the total number of tokens used in the conversation.

        Returns
        -------
        int
            The total number of tokens used in the conversation.
        """
        return sum(usage.total_tokens for usage in self.usage_log)

    def usage_cost(self) -> float | None:
        """
        Return the total cost in USD of the API usage.

        Returns
        -------
        float
            Cost in USD.
        """
        if not self.is_openai_model():
            return None

        try:
            result = 0
            for log in self.usage_log:
                result += get_openai_token_cost_for_model(
                    self.model_name, log.total_prompt_tokens, is_completion=False
                )
                result += get_openai_token_cost_for_model(
                    self.model_name, log.total_completion_tokens, is_completion=True
                )
            return result
        except Exception as e:
            print(f"Error calculating usage cost: {e}")
            return None
