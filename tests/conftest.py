"""
Pytest configuration file for gpt-engineer tests.

This file contains fixtures and configuration for pytest.
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from langchain.schema import AIMessage, HumanMessage, SystemMessage

from gpt_engineer.core.ai import AI
from gpt_engineer.core.token_usage import TokenUsageLog


@pytest.fixture
def mock_ai():
    """Create a mock AI instance with a controlled response."""
    with patch("gpt_engineer.core.ai.AI._create_chat_model") as mock_create_chat_model:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Test response")
        mock_create_chat_model.return_value = mock_llm
        
        ai = AI(model_name="gpt-4", provider="openai")
        ai.llm = mock_llm
        return ai


@pytest.fixture
def mock_token_usage_log():
    """Create a mock TokenUsageLog instance."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]  # Simulate token encoding
    
    log = TokenUsageLog(model_name="gpt-4")
    log._tokenizer = mock_tokenizer
    return log


@pytest.fixture
def sample_messages():
    """Create a sample list of messages for testing."""
    return [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?"),
        AIMessage(content="The capital of France is Paris."),
    ]


@pytest.fixture
def api_keys():
    """Get API keys from environment for testing."""
    return {
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    }


@pytest.fixture
def has_openai_key(api_keys):
    """Check if OpenAI API key is available."""
    return bool(api_keys["openai"])


@pytest.fixture
def has_anthropic_key(api_keys):
    """Check if Anthropic API key is available."""
    return bool(api_keys["anthropic"]) 