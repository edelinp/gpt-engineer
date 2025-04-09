import os
import pytest
from unittest.mock import MagicMock, patch
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models.openai import ChatOpenAI

from gpt_engineer.core.ai import AI
from gpt_engineer.core.token_usage import TokenUsageLog


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for model interactions."""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OpenAI API key not found in environment",
    )
    def test_openai_integration(self, mock_ai, sample_messages):
        """Test integration with OpenAI API."""
        ai = AI(model_name="gpt-4", provider="openai")
        messages = ai.next(sample_messages, step_name="test_openai")
        assert isinstance(messages, list)
        assert len(messages) == len(sample_messages) + 1
        assert isinstance(messages[-1], AIMessage)
        assert messages[-1].content

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Anthropic API key not found in environment",
    )
    def test_anthropic_integration(self, mock_ai, sample_messages):
        """Test integration with Anthropic API."""
        ai = AI(model_name="claude-3", provider="anthropic")
        messages = ai.next(sample_messages, step_name="test_anthropic")
        assert isinstance(messages, list)
        assert len(messages) == len(sample_messages) + 1
        assert isinstance(messages[-1], AIMessage)
        assert messages[-1].content

    def test_error_handling(self, mock_ai, sample_messages):
        """Test error handling during model interactions."""
        mock_ai.llm.invoke.side_effect = Exception("API Error")
        with pytest.raises(Exception):
            mock_ai.next(sample_messages, step_name="test_error")

    def test_backoff_strategy(self, mock_ai, sample_messages):
        """Test backoff strategy for rate limits."""
        # Create a mock chat model
        mock_chat = MagicMock(spec=ChatOpenAI)
        mock_chat.invoke.side_effect = [
            Exception("Rate limit exceeded"),
            MagicMock(content="Success after retry"),
        ]
        mock_ai.llm = mock_chat

        messages = mock_ai.next(sample_messages, step_name="test_backoff")
        assert isinstance(messages, list)
        assert len(messages) == len(sample_messages) + 1
        assert isinstance(messages[-1], AIMessage)
        assert messages[-1].content == "Success after retry"
        assert mock_chat.invoke.call_count == 2


@pytest.mark.integration
class TestTokenUsageIntegration:
    """Integration tests for token usage tracking."""

    def test_token_usage_tracking(self, mock_token_usage_log, sample_messages):
        """Test that token usage is tracked correctly during model interactions."""
        log = mock_token_usage_log
        initial_usage = log.usage_cost()

        # Simulate a conversation
        log.update_log(sample_messages, "Test response", step_name="test_token_usage")
        assert log.usage_cost() > initial_usage

        # Verify log entries
        log_csv = log.format_log()
        assert "prompt_tokens_in_step" in log_csv
        assert "completion_tokens_in_step" in log_csv
        assert "total_tokens_in_step" in log_csv 