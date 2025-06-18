"""Tests for Registry API integration."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
import anyio

from claude_code_sdk.registry_api import (
    RegistryAPIClient,
    RegistryAPIError,
    ToolExecutionResponse,
    ToolSearchStrategy,
    quick_execute_tool,
    quick_search_tools,
)


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx client."""
    return AsyncMock(spec=httpx.AsyncClient)


@pytest.fixture
def registry_client(mock_httpx_client):
    """Create a RegistryAPIClient with mocked httpx client."""
    client = RegistryAPIClient()
    client._client = mock_httpx_client
    return client


class TestRegistryAPIClient:
    """Test RegistryAPIClient methods."""

    async def test_context_manager(self):
        """Test async context manager functionality."""
        async with RegistryAPIClient() as client:
            assert client._client is not None
            assert isinstance(client._client, httpx.AsyncClient)

        # After exiting, client should be closed
        assert client._client is not None  # Still exists but closed

    async def test_ensure_client_error(self):
        """Test error when client not initialized."""
        client = RegistryAPIClient()
        with pytest.raises(RegistryAPIError, match="Client not initialized"):
            client._ensure_client()

    async def test_health_check(self, registry_client, mock_httpx_client):
        """Test health check endpoint."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {"status": "healthy", "version": "1.0.0"}
        mock_response.raise_for_status = AsyncMock()
        mock_httpx_client.get.return_value = mock_response

        result = await registry_client.health()

        assert result == {"status": "healthy", "version": "1.0.0"}
        mock_httpx_client.get.assert_called_once_with("/health")

    async def test_health_check_error(self, registry_client, mock_httpx_client):
        """Test health check error handling."""
        mock_httpx_client.get.side_effect = httpx.HTTPError("Connection failed")

        with pytest.raises(RegistryAPIError, match="Health check failed"):
            await registry_client.health()

    async def test_get_tools(self, registry_client, mock_httpx_client):
        """Test getting all tools."""
        mock_tools = [
            {
                "identifier": "tool1",
                "name": "Tool 1",
                "description": "First tool",
                "version": "1.0.0",
                "status": "active",
            },
            {
                "identifier": "tool2",
                "name": "Tool 2",
                "description": "Second tool",
                "version": "2.0.0",
                "status": "active",
            },
        ]

        mock_response = AsyncMock()
        mock_response.json.return_value = mock_tools
        mock_response.raise_for_status = AsyncMock()
        mock_httpx_client.get.return_value = mock_response

        result = await registry_client.get_tools()

        assert result == mock_tools
        mock_httpx_client.get.assert_called_once_with("/tools")

    async def test_get_tool(self, registry_client, mock_httpx_client):
        """Test getting a specific tool."""
        mock_tool = {
            "identifier": "test_tool",
            "name": "Test Tool",
            "description": "A test tool",
            "version": "1.0.0",
            "status": "active",
        }

        mock_response = AsyncMock()
        mock_response.json.return_value = mock_tool
        mock_response.raise_for_status = AsyncMock()
        mock_httpx_client.get.return_value = mock_response

        result = await registry_client.get_tool("test_tool")

        assert result == mock_tool
        mock_httpx_client.get.assert_called_once_with("/tools/test_tool")

    async def test_create_tool(self, registry_client, mock_httpx_client):
        """Test creating a new tool."""
        input_schema = {
            "type": "object",
            "properties": {"data": {"type": "string"}},
            "required": ["data"],
        }
        output_schema = {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
        }
        action = {
            "type": "python",
            "python": {
                "code": "def execute(input_data):\n    return {'result': input_data['data'].upper()}",
                "function_name": "execute",
            },
        }

        mock_tool = {
            "identifier": "new_tool",
            "name": "New Tool",
            "description": "A new tool",
            "version": "1.0.0",
            "status": "active",
            "input_schema": input_schema,
            "output_schema": output_schema,
            "action": action,
        }

        mock_response = AsyncMock()
        mock_response.json.return_value = mock_tool
        mock_response.raise_for_status = AsyncMock()
        mock_httpx_client.post.return_value = mock_response

        result = await registry_client.create_tool(
            name="New Tool",
            description="A new tool",
            input_schema=input_schema,
            output_schema=output_schema,
            action=action,
        )

        assert result == mock_tool
        mock_httpx_client.post.assert_called_once()
        call_args = mock_httpx_client.post.call_args
        assert call_args[0][0] == "/tools"
        assert "json" in call_args[1]
        assert call_args[1]["json"]["name"] == "New Tool"

    async def test_activate_tool(self, registry_client, mock_httpx_client):
        """Test activating a tool."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {"status": "activated"}
        mock_response.raise_for_status = AsyncMock()
        mock_httpx_client.post.return_value = mock_response

        result = await registry_client.activate_tool("test_tool")

        assert result == {"status": "activated"}
        mock_httpx_client.post.assert_called_once_with("/tools/test_tool/activate")

    async def test_search_tools(self, registry_client, mock_httpx_client):
        """Test searching for tools."""
        strategy: ToolSearchStrategy = {
            "type": "vector_similarity",
            "threshold": 0.8,
        }

        mock_results = [
            {"identifier": "tool1", "name": "Tool 1", "score": 0.95},
            {"identifier": "tool2", "name": "Tool 2", "score": 0.85},
        ]

        mock_response = AsyncMock()
        mock_response.json.return_value = mock_results
        mock_response.raise_for_status = AsyncMock()
        mock_httpx_client.post.return_value = mock_response

        result = await registry_client.search_tools(
            query="data processing",
            strategy=strategy,
            limit=10,
        )

        assert result == mock_results
        mock_httpx_client.post.assert_called_once()
        call_args = mock_httpx_client.post.call_args
        assert call_args[0][0] == "/tools/search"
        assert "json" in call_args[1]
        assert call_args[1]["json"]["query"] == "data processing"
        assert call_args[1]["json"]["strategy"] == strategy
        assert call_args[1]["json"]["limit"] == 10

    async def test_execute_tool(self, registry_client, mock_httpx_client):
        """Test executing a tool."""
        mock_execution_response: ToolExecutionResponse = {
            "status": "success",
            "result": {"output": "HELLO WORLD"},
            "error": None,
            "execution_time_ms": 15.5,
        }

        mock_response = AsyncMock()
        mock_response.json.return_value = mock_execution_response
        mock_response.raise_for_status = AsyncMock()
        mock_httpx_client.post.return_value = mock_response

        result = await registry_client.execute_tool(
            tool_identifier="text_processor",
            input_data={"data": "hello world"},
        )

        assert result == mock_execution_response
        mock_httpx_client.post.assert_called_once()
        call_args = mock_httpx_client.post.call_args
        assert call_args[0][0] == "/execute_tool"
        assert "json" in call_args[1]
        assert call_args[1]["json"]["tool_identifier"] == "text_processor"
        assert call_args[1]["json"]["input_data"] == {"data": "hello world"}

    async def test_execute_tool_error(self, registry_client, mock_httpx_client):
        """Test tool execution error handling."""
        mock_httpx_client.post.side_effect = httpx.HTTPError("Execution failed")

        with pytest.raises(RegistryAPIError, match="Tool execution failed"):
            await registry_client.execute_tool(
                tool_identifier="failing_tool",
                input_data={"data": "test"},
            )


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch("claude_code_sdk.registry_api.RegistryAPIClient")
    async def test_quick_execute_tool(self, mock_client_class):
        """Test quick_execute_tool function."""
        mock_instance = AsyncMock()
        mock_instance.__aenter__.return_value = mock_instance
        mock_instance.__aexit__.return_value = None
        mock_instance.execute_tool.return_value = {
            "status": "success",
            "result": {"output": "done"},
        }
        mock_client_class.return_value = mock_instance

        result = await quick_execute_tool(
            tool_identifier="quick_tool",
            input_data={"input": "test"},
        )

        assert result == {"status": "success", "result": {"output": "done"}}
        mock_instance.execute_tool.assert_called_once_with(
            "quick_tool", {"input": "test"}
        )

    @patch("claude_code_sdk.registry_api.RegistryAPIClient")
    async def test_quick_search_tools(self, mock_client_class):
        """Test quick_search_tools function."""
        mock_instance = AsyncMock()
        mock_instance.__aenter__.return_value = mock_instance
        mock_instance.__aexit__.return_value = None
        mock_instance.search_tools.return_value = [
            {"identifier": "tool1", "name": "Tool 1"},
        ]
        mock_client_class.return_value = mock_instance

        strategy: ToolSearchStrategy = {"type": "fuzzy"}
        result = await quick_search_tools("search query", strategy=strategy)

        assert result == [{"identifier": "tool1", "name": "Tool 1"}]
        mock_instance.search_tools.assert_called_once_with("search query", strategy)


class TestIntegration:
    """Integration tests with real-like scenarios."""

    async def test_full_workflow(self, registry_client, mock_httpx_client):
        """Test a complete workflow of creating and executing a tool."""
        # Mock responses for each step
        create_response = AsyncMock()
        create_response.json.return_value = {
            "identifier": "workflow_tool",
            "name": "Workflow Tool",
            "status": "inactive",
        }
        create_response.raise_for_status = AsyncMock()

        activate_response = AsyncMock()
        activate_response.json.return_value = {"status": "activated"}
        activate_response.raise_for_status = AsyncMock()

        execute_response = AsyncMock()
        execute_response.json.return_value = {
            "status": "success",
            "result": {"processed": True},
        }
        execute_response.raise_for_status = AsyncMock()

        # Set up mock responses in order
        mock_httpx_client.post.side_effect = [
            create_response,
            activate_response,
            execute_response,
        ]

        # Create tool
        tool = await registry_client.create_tool(
            name="Workflow Tool",
            description="Test workflow",
            input_schema={"type": "object", "properties": {}, "required": []},
            output_schema={"type": "object", "properties": {}, "required": []},
            action={
                "type": "python",
                "python": {"code": "pass", "function_name": "execute"},
            },
        )

        assert tool["identifier"] == "workflow_tool"

        # Activate tool
        activation = await registry_client.activate_tool("workflow_tool")
        assert activation["status"] == "activated"

        # Execute tool
        result = await registry_client.execute_tool(
            tool_identifier="workflow_tool",
            input_data={},
        )

        assert result["status"] == "success"
        assert result["result"]["processed"] is True

        # Verify all calls were made
        assert mock_httpx_client.post.call_count == 3
