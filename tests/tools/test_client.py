"""Tests for Tool Management API client."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import httpx

from claude_code_sdk.tools import (
    ToolManagementClient,
    Tool,
    ToolResponse,
    ToolSearchResponse,
    ToolExecutionResponse,
    ToolAction,
    ToolActionType,
    PythonActionConfig,
)


@pytest.fixture
async def mock_client():
    """Create a mock HTTP client."""
    mock = AsyncMock(spec=httpx.AsyncClient)
    return mock


@pytest.fixture
async def tool_client(mock_client):
    """Create a ToolManagementClient with mocked HTTP client."""
    client = ToolManagementClient()
    client._client = mock_client
    return client


class TestToolManagementClient:
    """Test Tool Management API client."""

    async def test_health_check(self, tool_client, mock_client):
        """Test health check endpoint."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "healthy"}
        mock_response.raise_for_status = Mock()
        mock_client.get.return_value = mock_response

        result = await tool_client.health_check()
        
        assert result == {"status": "healthy"}
        mock_client.get.assert_called_once_with("/health")

    async def test_get_tools(self, tool_client, mock_client):
        """Test getting all tools."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "id": "tool1",
                "name": "calculator",
                "description": "Basic calculator",
                "input_schema": {},
                "output_schema": {},
            }
        ]
        mock_response.raise_for_status = Mock()
        mock_client.get.return_value = mock_response

        tools = await tool_client.get_tools()
        
        assert len(tools) == 1
        assert isinstance(tools[0], ToolResponse)
        assert tools[0].name == "calculator"
        mock_client.get.assert_called_once_with("/tools", params={})

    async def test_get_tools_with_namespace(self, tool_client, mock_client):
        """Test getting tools filtered by namespace."""
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = Mock()
        mock_client.get.return_value = mock_response

        await tool_client.get_tools(namespace="test")
        
        mock_client.get.assert_called_once_with("/tools", params={"namespace": "test"})

    async def test_create_tool(self, tool_client, mock_client):
        """Test creating a new tool."""
        tool = Tool(
            name="test_tool",
            description="Test tool",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            action=ToolAction(
                type=ToolActionType.PYTHON,
                python=PythonActionConfig(
                    code="def test(): pass",
                    function_name="test"
                )
            )
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "new_id",
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
            "output_schema": tool.output_schema,
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response

        result = await tool_client.create_tool(tool)
        
        assert isinstance(result, ToolResponse)
        assert result.id == "new_id"
        assert result.name == tool.name

    async def test_search_tools(self, tool_client, mock_client):
        """Test searching for tools."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "id": "tool1",
                "name": "calculator",
                "description": "Calculator tool",
                "input_schema": {},
                "output_schema": {},
                "score": 0.95,
            }
        ]
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response

        results = await tool_client.search_tools("calculator", limit=5)
        
        assert len(results) == 1
        assert isinstance(results[0], ToolSearchResponse)
        assert results[0].score == 0.95
        
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/tools/search"
        assert call_args[1]["json"]["query"] == "calculator"
        assert call_args[1]["json"]["limit"] == 5

    async def test_execute_tool(self, tool_client, mock_client):
        """Test executing a tool."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "tool_id": "tool1",
            "output_data": {"result": 42}
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response

        result = await tool_client.execute_tool(
            tool_id="tool1",
            input_data={"expression": "40 + 2"}
        )
        
        assert isinstance(result, ToolExecutionResponse)
        assert result.output_data == {"result": 42}

    async def test_execute_tool_no_id_or_name(self, tool_client):
        """Test executing tool without ID or name raises error."""
        with pytest.raises(ValueError, match="Either tool_id or tool_name must be provided"):
            await tool_client.execute_tool(input_data={})

    async def test_context_manager(self):
        """Test client works as async context manager."""
        async with ToolManagementClient() as client:
            assert client._client is not None
            assert isinstance(client._client, httpx.AsyncClient)

    async def test_client_not_initialized_error(self):
        """Test error when client used without initialization."""
        client = ToolManagementClient()
        
        with pytest.raises(RuntimeError, match="Client not initialized"):
            _ = client.client