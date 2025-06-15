"""Tests for tool discovery utilities."""

import pytest
from unittest.mock import AsyncMock

from claude_code_sdk.tools import (
    ToolDiscovery,
    ToolRegistry,
    ToolManagementClient,
    Tool,
    ToolResponse,
    ToolSearchResponse,
)


@pytest.fixture
async def mock_client():
    """Create a mock Tool Management API client."""
    return AsyncMock(spec=ToolManagementClient)


@pytest.fixture
async def discovery(mock_client):
    """Create a ToolDiscovery instance with mocked client."""
    return ToolDiscovery(mock_client)


@pytest.fixture
async def registry(mock_client):
    """Create a ToolRegistry instance with mocked client."""
    return ToolRegistry(mock_client)


class TestToolDiscovery:
    """Test tool discovery utilities."""

    async def test_search_by_capability(self, discovery, mock_client):
        """Test searching tools by capability."""
        mock_results = [
            ToolSearchResponse(
                id="tool1",
                name="data_processor",
                description="Process data",
                input_schema={},
                output_schema={},
                score=0.9,
            )
        ]
        mock_client.search_tools.return_value = mock_results

        results = await discovery.search_by_capability("data processing", limit=10)
        
        assert len(results) == 1
        assert results[0].name == "data_processor"
        mock_client.search_tools.assert_called_once_with("data processing", limit=10)

    async def test_search_by_namespace(self, discovery, mock_client):
        """Test getting tools by namespace."""
        mock_tools = [
            ToolResponse(
                id="tool1",
                name="tool1",
                namespace="test",
                description="Test tool 1",
                input_schema={},
                output_schema={},
            ),
            ToolResponse(
                id="tool2",
                name="tool2",
                namespace="test",
                description="Test tool 2",
                input_schema={},
                output_schema={},
            ),
        ]
        mock_client.get_tools.return_value = mock_tools

        results = await discovery.search_by_namespace("test")
        
        assert len(results) == 2
        assert all(isinstance(r, ToolSearchResponse) for r in results)
        assert results[0].namespace == "test"
        mock_client.get_tools.assert_called_once_with(namespace="test")

    async def test_find_similar_tools(self, discovery, mock_client):
        """Test finding similar tools."""
        mock_results = [
            ToolSearchResponse(
                id="tool1",
                name="advanced_calculator",
                description="Advanced calculator",
                input_schema={},
                output_schema={},
                score=0.95,
            )
        ]
        mock_client.retrieve_tools.return_value = mock_results

        results = await discovery.find_similar_tools("calculator", limit=5)
        
        assert len(results) == 1
        assert results[0].name == "advanced_calculator"
        mock_client.retrieve_tools.assert_called_once_with(
            "calculator", limit=5, strategy="vector_similarity"
        )

    async def test_suggest_tools_for_task(self, discovery, mock_client):
        """Test suggesting tools for a task."""
        task = "analyze sentiment in text"
        mock_results = [
            ToolSearchResponse(
                id="tool1",
                name="sentiment_analyzer",
                description="Analyze text sentiment",
                input_schema={},
                output_schema={},
                score=0.92,
            )
        ]
        mock_client.retrieve_tools.return_value = mock_results

        results = await discovery.suggest_tools_for_task(task, limit=5)
        
        assert len(results) == 1
        assert results[0].name == "sentiment_analyzer"
        mock_client.retrieve_tools.assert_called_once_with(
            task, limit=5, strategy="vector_similarity"
        )

    async def test_get_tool_by_name(self, discovery, mock_client):
        """Test getting tool by exact name match."""
        mock_results = [
            ToolSearchResponse(
                id="tool1",
                name="calculator",
                description="Basic calculator",
                input_schema={},
                output_schema={},
            ),
            ToolSearchResponse(
                id="tool2",
                name="advanced_calculator",
                description="Advanced calculator",
                input_schema={},
                output_schema={},
            ),
        ]
        mock_client.search_tools.return_value = mock_results

        result = await discovery.get_tool_by_name("calculator")
        
        assert result is not None
        assert result.name == "calculator"
        assert result.id == "tool1"

    async def test_get_tool_by_name_with_namespace(self, discovery, mock_client):
        """Test getting tool by name and namespace."""
        mock_results = [
            ToolSearchResponse(
                id="tool1",
                name="calculator",
                namespace="basic",
                description="Basic calculator",
                input_schema={},
                output_schema={},
            ),
            ToolSearchResponse(
                id="tool2",
                name="calculator",
                namespace="advanced",
                description="Advanced calculator",
                input_schema={},
                output_schema={},
            ),
        ]
        mock_client.search_tools.return_value = mock_results

        result = await discovery.get_tool_by_name("calculator", namespace="advanced")
        
        assert result is not None
        assert result.id == "tool2"
        assert result.namespace == "advanced"

    async def test_get_tool_by_name_not_found(self, discovery, mock_client):
        """Test getting tool that doesn't exist."""
        mock_client.search_tools.return_value = []

        result = await discovery.get_tool_by_name("nonexistent")
        
        assert result is None

    async def test_list_tool_categories(self, discovery, mock_client):
        """Test listing tools by category."""
        mock_tools = [
            ToolResponse(
                id="tool1",
                name="calculator",
                namespace="math",
                description="Calculator",
                input_schema={},
                output_schema={},
            ),
            ToolResponse(
                id="tool2",
                name="translator",
                namespace="language",
                description="Translator",
                input_schema={},
                output_schema={},
            ),
            ToolResponse(
                id="tool3",
                name="converter",
                namespace="math",
                description="Unit converter",
                input_schema={},
                output_schema={},
            ),
            ToolResponse(
                id="tool4",
                name="default_tool",
                namespace=None,
                description="Default tool",
                input_schema={},
                output_schema={},
            ),
        ]
        mock_client.get_tools.return_value = mock_tools

        categories = await discovery.list_tool_categories()
        
        assert "math" in categories
        assert "language" in categories
        assert "default" in categories
        assert len(categories["math"]) == 2
        assert "calculator" in categories["math"]
        assert "converter" in categories["math"]
        assert "translator" in categories["language"]
        assert "default_tool" in categories["default"]


class TestToolRegistry:
    """Test tool registry."""

    async def test_register_tool(self, registry, mock_client):
        """Test registering a new tool."""
        tool = Tool(
            name="test_tool",
            description="Test tool",
            input_schema={},
            output_schema={},
        )

        mock_response = ToolResponse(
            id="new_id",
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            output_schema=tool.output_schema,
        )
        mock_client.create_tool.return_value = mock_response

        tool_id = await registry.register_tool(tool)
        
        assert tool_id == "new_id"
        assert "test_tool" in registry._cache
        assert registry._cache["test_tool"].id == "new_id"

    async def test_get_tool_from_cache(self, registry):
        """Test getting tool from cache."""
        # Pre-populate cache
        cached_tool = ToolSearchResponse(
            id="cached_id",
            name="cached_tool",
            description="Cached tool",
            input_schema={},
            output_schema={},
        )
        registry._cache["cached_tool"] = cached_tool

        result = await registry.get_tool("cached_tool")
        
        assert result is cached_tool
        # Should not call the discovery method
        assert not hasattr(registry.discovery, "get_tool_by_name") or \
               not registry.discovery.get_tool_by_name.called

    async def test_get_tool_from_registry(self, registry, mock_client):
        """Test getting tool from registry when not cached."""
        mock_tool = ToolSearchResponse(
            id="tool1",
            name="test_tool",
            description="Test tool",
            input_schema={},
            output_schema={},
        )
        
        # Mock the discovery search
        mock_client.search_tools.return_value = [mock_tool]

        result = await registry.get_tool("test_tool")
        
        assert result is not None
        assert result.name == "test_tool"
        assert "test_tool" in registry._cache

    async def test_refresh_cache(self, registry, mock_client):
        """Test refreshing the tool cache."""
        # Pre-populate cache
        registry._cache["old_tool"] = ToolSearchResponse(
            id="old",
            name="old_tool",
            description="Old",
            input_schema={},
            output_schema={},
        )

        # Mock get_tools response
        mock_tools = [
            ToolResponse(
                id="tool1",
                name="new_tool1",
                description="New tool 1",
                input_schema={},
                output_schema={},
            ),
            ToolResponse(
                id="tool2",
                name="new_tool2",
                description="New tool 2",
                input_schema={},
                output_schema={},
            ),
        ]
        mock_client.get_tools.return_value = mock_tools

        await registry.refresh_cache()
        
        assert "old_tool" not in registry._cache
        assert "new_tool1" in registry._cache
        assert "new_tool2" in registry._cache
        assert len(registry._cache) == 2

    async def test_ensure_tools_available(self, registry, mock_client):
        """Test checking tool availability."""
        # Set up cache and mock responses
        registry._cache["cached_tool"] = ToolSearchResponse(
            id="cached",
            name="cached_tool",
            description="Cached",
            input_schema={},
            output_schema={},
        )
        
        mock_client.search_tools.return_value = [
            ToolSearchResponse(
                id="found",
                name="registry_tool",
                description="Found in registry",
                input_schema={},
                output_schema={},
            )
        ]

        tool_names = ["cached_tool", "registry_tool", "missing_tool"]
        availability = await registry.ensure_tools_available(tool_names)
        
        assert availability["cached_tool"] is True
        assert availability["registry_tool"] is True
        assert availability["missing_tool"] is False