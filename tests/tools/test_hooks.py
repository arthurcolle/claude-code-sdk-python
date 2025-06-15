"""Tests for tool execution hooks."""

import pytest
from unittest.mock import Mock, AsyncMock

from claude_code_sdk.types import ToolUseBlock, ToolResultBlock
from claude_code_sdk.tools import (
    ToolExecutionHooks,
    ToolRegistryOptions,
    ToolManagementClient,
    ToolSearchResponse,
    ToolExecutionResponse,
)


class TestToolExecutionHooks:
    """Test tool execution hooks."""

    async def test_hook_callbacks(self):
        """Test that hook callbacks are called."""
        tool_use_called = False
        tool_result_called = False

        def on_tool_use(block: ToolUseBlock) -> None:
            nonlocal tool_use_called
            tool_use_called = True
            assert block.name == "test_tool"

        def on_tool_result(block: ToolResultBlock) -> None:
            nonlocal tool_result_called
            tool_result_called = True
            assert block.tool_use_id == "test_id"

        hooks = ToolExecutionHooks(
            on_tool_use=on_tool_use,
            on_tool_result=on_tool_result,
        )

        # Test tool use callback
        tool_use = ToolUseBlock(id="test_id", name="test_tool", input={})
        await hooks.handle_tool_use(tool_use)
        assert tool_use_called

        # Test tool result callback
        tool_result = ToolResultBlock(tool_use_id="test_id", content="result")
        await hooks.handle_tool_result(tool_result)
        assert tool_result_called

    async def test_remote_tool_execution(self):
        """Test remote tool execution."""
        # Mock client
        mock_client = AsyncMock(spec=ToolManagementClient)
        
        # Mock search results
        mock_tool = ToolSearchResponse(
            id="tool1",
            name="calculator",
            description="Calculator",
            input_schema={},
            output_schema={},
        )
        mock_client.search_tools.return_value = [mock_tool]
        
        # Mock execution result
        mock_result = ToolExecutionResponse(
            tool_id="tool1",
            output_data={"result": 42}
        )
        mock_client.execute_tool.return_value = mock_result

        # Create hooks with mocked client
        hooks = ToolExecutionHooks(use_remote_tools=True)
        hooks._client = mock_client

        # Execute tool
        tool_use = ToolUseBlock(
            id="use1",
            name="calculator",
            input={"expression": "40 + 2"}
        )
        
        result = await hooks.handle_tool_use(tool_use)
        
        # Verify result
        assert result is not None
        assert isinstance(result, ToolResultBlock)
        assert result.tool_use_id == "use1"
        assert "42" in result.content
        assert not result.is_error

        # Verify calls
        mock_client.search_tools.assert_called_once_with("calculator", limit=1)
        mock_client.execute_tool.assert_called_once_with(
            tool_id="tool1",
            input_data={"expression": "40 + 2"}
        )

    async def test_tool_not_found(self):
        """Test handling when tool is not found."""
        # Mock client with no results
        mock_client = AsyncMock(spec=ToolManagementClient)
        mock_client.search_tools.return_value = []

        hooks = ToolExecutionHooks(use_remote_tools=True)
        hooks._client = mock_client

        tool_use = ToolUseBlock(
            id="use1",
            name="nonexistent_tool",
            input={}
        )
        
        result = await hooks.handle_tool_use(tool_use)
        
        assert result is not None
        assert result.is_error
        assert "not found" in result.content

    async def test_execution_error(self):
        """Test handling execution errors."""
        # Mock client that raises exception
        mock_client = AsyncMock(spec=ToolManagementClient)
        mock_client.search_tools.side_effect = Exception("API Error")

        hooks = ToolExecutionHooks(use_remote_tools=True)
        hooks._client = mock_client

        tool_use = ToolUseBlock(
            id="use1",
            name="test_tool",
            input={}
        )
        
        result = await hooks.handle_tool_use(tool_use)
        
        assert result is not None
        assert result.is_error
        assert "Error executing tool" in result.content
        assert "API Error" in result.content

    async def test_context_manager(self):
        """Test hooks work as async context manager."""
        hooks = ToolExecutionHooks(
            tool_api_url="https://test.api",
            use_remote_tools=True
        )

        async with hooks:
            assert hooks._client is not None
            assert isinstance(hooks._client, ToolManagementClient)

    async def test_no_remote_execution(self):
        """Test that remote execution is skipped when disabled."""
        mock_client = AsyncMock()
        
        hooks = ToolExecutionHooks(use_remote_tools=False)
        hooks._client = mock_client

        tool_use = ToolUseBlock(id="use1", name="test", input={})
        result = await hooks.handle_tool_use(tool_use)
        
        assert result is None
        mock_client.search_tools.assert_not_called()


class TestToolRegistryOptions:
    """Test ToolRegistryOptions class."""

    def test_registry_options_init(self):
        """Test ToolRegistryOptions initialization."""
        options = ToolRegistryOptions(
            tool_api_url="https://custom.api",
            use_remote_tools=True,
            tool_namespace="test",
            allowed_tools=["Read", "Write"],
            max_thinking_tokens=10000,
        )

        assert options.tool_api_url == "https://custom.api"
        assert options.use_remote_tools is True
        assert options.tool_namespace == "test"
        assert options.allowed_tools == ["Read", "Write"]
        assert options.max_thinking_tokens == 10000

    def test_registry_options_inherits_claude_options(self):
        """Test that ToolRegistryOptions inherits from ClaudeCodeOptions."""
        from claude_code_sdk.types import ClaudeCodeOptions
        
        options = ToolRegistryOptions()
        assert isinstance(options, ClaudeCodeOptions)