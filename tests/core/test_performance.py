import time
import pytest
from unittest.mock import MagicMock, patch

from langchain.schema import AIMessage, HumanMessage, SystemMessage

from gpt_engineer.core.ai import AI
from gpt_engineer.core.token_usage import TokenUsageLog


@pytest.mark.performance
class TestPerformance:
    """Performance benchmark tests."""

    def test_response_time(self, mock_ai, sample_messages):
        """Benchmark response time for model interactions."""
        mock_ai.llm.invoke.side_effect = lambda x: time.sleep(0.5) or MagicMock(
            content="Test response"
        )

        start_time = time.time()
        messages = mock_ai.next(sample_messages, step_name="test_response_time")
        elapsed_time = time.time() - start_time

        assert elapsed_time >= 0.5
        assert elapsed_time < 1.0
        assert isinstance(messages, list)
        assert len(messages) == len(sample_messages) + 1
        assert isinstance(messages[-1], AIMessage)
        assert messages[-1].content == "Test response"

    def test_token_usage_performance(self, mock_token_usage_log):
        """Benchmark performance of token usage calculation."""
        # Create a large message to test performance
        large_message = "Test message " * 1000  # 20,000 characters
        messages = [
            SystemMessage(content=large_message),
            HumanMessage(content=large_message),
        ]

        start_time = time.time()
        mock_token_usage_log.update_log(messages, "Test response", step_name="test_token_usage")
        elapsed_time = time.time() - start_time

        assert elapsed_time < 1.0
        assert mock_token_usage_log.usage_cost() > 0

    def test_concurrent_requests(self, mock_ai):
        """Benchmark performance with concurrent requests."""
        mock_ai.llm.invoke.side_effect = lambda x: MagicMock(content="Test response")
        prompts = [f"Test prompt {i}" for i in range(10)]
        messages_list = [
            [SystemMessage(content="You are a helpful assistant."), HumanMessage(content=p)]
            for p in prompts
        ]

        start_time = time.time()
        responses = [
            mock_ai.next(messages, step_name=f"test_concurrent_{i}")
            for i, messages in enumerate(messages_list)
        ]
        elapsed_time = time.time() - start_time

        assert elapsed_time < 5.0
        assert len(responses) == len(prompts)
        assert all(isinstance(r, list) for r in responses)
        assert all(isinstance(r[-1], AIMessage) for r in responses)
        assert all(r[-1].content == "Test response" for r in responses)

    def test_message_serialization_performance(self, sample_messages):
        """Benchmark performance of message serialization and deserialization."""
        # Test serialization performance
        start_time = time.time()
        serialized = [msg.content for msg in sample_messages]
        serialization_time = time.time() - start_time

        assert serialization_time < 0.1
        assert len(serialized) == len(sample_messages)
        assert all(isinstance(s, str) for s in serialized)

        # Test deserialization performance
        start_time = time.time()
        deserialized = [
            SystemMessage(content=serialized[0]),
            HumanMessage(content=serialized[1]),
            AIMessage(content=serialized[2]),
        ]
        deserialization_time = time.time() - start_time

        assert deserialization_time < 0.1
        assert len(deserialized) == len(sample_messages)
        assert all(
            isinstance(msg, (SystemMessage, HumanMessage, AIMessage))
            for msg in deserialized
        ) 