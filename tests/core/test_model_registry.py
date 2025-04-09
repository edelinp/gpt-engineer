import pytest
from unittest.mock import MagicMock, patch

from langchain.chat_models.base import BaseChatModel
from langchain_community.chat_models.fake import FakeListChatModel

from gpt_engineer.core.ai import (
    AI,
    AnthropicProvider,
    ModelConfig,
    ModelProvider,
    ModelRegistry,
    OpenAIProvider,
)


@pytest.mark.unit
class TestModelConfig:
    """Test cases for the ModelConfig class."""

    def test_model_config_creation(self):
        """Test that ModelConfig can be created with default values."""
        config = ModelConfig(name="test-model", provider="openai")
        assert config.name == "test-model"
        assert config.provider == "openai"
        assert config.temperature == 0.1
        assert config.max_tokens is None
        assert config.streaming is True
        assert config.azure_endpoint is None
        assert config.vision is False

    def test_model_config_custom_values(self):
        """Test that ModelConfig can be created with custom values."""
        config = ModelConfig(
            name="test-model",
            provider="anthropic",
            temperature=0.5,
            max_tokens=1000,
            streaming=False,
            azure_endpoint="https://test-endpoint",
            vision=True,
        )
        assert config.name == "test-model"
        assert config.provider == "anthropic"
        assert config.temperature == 0.5
        assert config.max_tokens == 1000
        assert config.streaming is False
        assert config.azure_endpoint == "https://test-endpoint"
        assert config.vision is True


@pytest.mark.unit
class TestModelProvider:
    """Test cases for the ModelProvider classes."""

    @patch("gpt_engineer.core.ai.ChatOpenAI")
    def test_openai_provider(self, mock_chat_openai):
        """Test that OpenAIProvider creates a ChatOpenAI instance."""
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        provider = OpenAIProvider()
        config = ModelConfig(name="gpt-4", provider="openai", temperature=0.5)
        model = provider.create_model(config)

        mock_chat_openai.assert_called_once_with(
            model_name="gpt-4", temperature=0.5, streaming=True
        )
        assert model == mock_instance

    @patch("gpt_engineer.core.ai.AzureChatOpenAI")
    def test_openai_provider_azure(self, mock_azure_chat_openai):
        """Test that OpenAIProvider creates an AzureChatOpenAI instance when azure_endpoint is provided."""
        mock_instance = MagicMock()
        mock_azure_chat_openai.return_value = mock_instance

        provider = OpenAIProvider()
        config = ModelConfig(
            name="gpt-4",
            provider="openai",
            temperature=0.5,
            azure_endpoint="https://test-endpoint",
        )
        model = provider.create_model(config)

        mock_azure_chat_openai.assert_called_once_with(
            azure_endpoint="https://test-endpoint",
            model_name="gpt-4",
            temperature=0.5,
            streaming=True,
        )
        assert model == mock_instance

    @patch("gpt_engineer.core.ai.ChatAnthropic")
    def test_anthropic_provider(self, mock_chat_anthropic):
        """Test that AnthropicProvider creates a ChatAnthropic instance."""
        mock_instance = MagicMock()
        mock_chat_anthropic.return_value = mock_instance

        provider = AnthropicProvider()
        config = ModelConfig(name="claude-3", provider="anthropic", temperature=0.5)
        model = provider.create_model(config)

        mock_chat_anthropic.assert_called_once_with(
            model_name="claude-3", temperature=0.5, streaming=True
        )
        assert model == mock_instance


@pytest.mark.unit
class TestModelRegistry:
    """Test cases for the ModelRegistry class."""

    def test_model_registry_initialization(self):
        """Test that ModelRegistry initializes with the correct providers."""
        registry = ModelRegistry()
        assert "openai" in registry.providers
        assert "anthropic" in registry.providers
        assert isinstance(registry.providers["openai"], OpenAIProvider)
        assert isinstance(registry.providers["anthropic"], AnthropicProvider)

    def test_get_provider(self):
        """Test that get_provider returns the correct provider."""
        registry = ModelRegistry()
        provider = registry.get_provider("openai")
        assert isinstance(provider, OpenAIProvider)

    def test_get_provider_invalid(self):
        """Test that get_provider raises an error for invalid provider names."""
        registry = ModelRegistry()
        with pytest.raises(ValueError):
            registry.get_provider("invalid-provider")

    @patch("gpt_engineer.core.ai.OpenAIProvider")
    def test_create_model(self, mock_openai_provider):
        """Test that create_model creates a model with the correct provider."""
        mock_provider_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_openai_provider.return_value = mock_provider_instance
        mock_provider_instance.create_model.return_value = mock_model_instance

        registry = ModelRegistry()
        registry.providers["openai"] = mock_provider_instance

        config = ModelConfig(name="gpt-4", provider="openai", temperature=0.5)
        model = registry.create_model(config)

        mock_provider_instance.create_model.assert_called_once_with(config)
        assert model == mock_model_instance


@pytest.mark.unit
class TestAIModelRegistryIntegration:
    """Test cases for the integration of AI with ModelRegistry."""

    @patch("gpt_engineer.core.ai.ModelRegistry")
    def test_ai_initialization_with_model_registry(self, mock_model_registry):
        """Test that AI initializes with ModelRegistry and creates a model."""
        mock_registry_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_registry.return_value = mock_registry_instance
        mock_registry_instance.create_model.return_value = mock_model_instance

        ai = AI(model_name="gpt-4", provider="openai", temperature=0.5)

        mock_model_registry.assert_called_once()
        mock_registry_instance.create_model.assert_called_once()
        assert ai.llm == mock_model_instance

    @patch("gpt_engineer.core.ai.ModelRegistry")
    def test_ai_initialization_with_azure(self, mock_model_registry):
        """Test that AI initializes with ModelRegistry and creates an Azure model."""
        mock_registry_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_registry.return_value = mock_registry_instance
        mock_registry_instance.create_model.return_value = mock_model_instance

        ai = AI(
            model_name="gpt-4",
            provider="openai",
            temperature=0.5,
            azure_endpoint="https://test-endpoint",
        )

        mock_model_registry.assert_called_once()
        mock_registry_instance.create_model.assert_called_once()
        assert ai.llm == mock_model_instance 