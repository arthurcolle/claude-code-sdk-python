"""Comprehensive tests for error handling scenarios."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claude_code_sdk import (
    ClaudeSDKError,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ProcessError,
    query,
)
from claude_code_sdk._internal.transport.subprocess_cli import SubprocessCLITransport
from claude_code_sdk.types import ClaudeCodeOptions


class TestComprehensiveErrorHandling:
    """Test comprehensive error handling scenarios."""

    @pytest.mark.asyncio
    async def test_cli_not_found_with_suggestions(self):
        """Test CLI not found error with helpful suggestions."""
        with patch("shutil.which", return_value=None):
            with patch("sys.platform", "darwin"):
                with pytest.raises(CLINotFoundError) as exc_info:
                    async for _ in query("test"):
                        pass
                
                error_msg = str(exc_info.value)
                assert "npm install -g @anthropic-ai/claude-code" in error_msg
                assert "Node.js" in error_msg

    @pytest.mark.asyncio
    async def test_process_error_with_detailed_context(self):
        """Test process error with full context."""
        mock_transport = AsyncMock(spec=SubprocessCLITransport)
        mock_transport.send_message = AsyncMock()
        
        # Simulate process failure
        mock_transport.__aiter__.side_effect = ProcessError(
            "Claude Code process crashed",
            exit_code=127,
            stderr="node: command not found\nPlease install Node.js"
        )
        
        with patch(
            "claude_code_sdk._internal.client.SubprocessCLITransport",
            return_value=mock_transport
        ):
            with pytest.raises(ProcessError) as exc_info:
                async for _ in query("test"):
                    pass
            
            error = exc_info.value
            assert error.exit_code == 127
            assert "node: command not found" in error.stderr
            assert "Please install Node.js" in error.stderr

    @pytest.mark.asyncio
    async def test_json_decode_error_recovery(self):
        """Test handling of malformed JSON responses."""
        mock_transport = AsyncMock(spec=SubprocessCLITransport)
        
        # Mix valid and invalid JSON
        responses = [
            '{"type": "user", "content": "test"}',
            '{invalid json}',
            '{"type": "assistant", "content": [{"type": "text", "text": "Hello"}]}',
            '{"type": "result", "subtype": "success"}'
        ]
        
        async def mock_iter():
            for resp in responses:
                if "invalid" in resp:
                    try:
                        json.loads(resp)
                    except json.JSONDecodeError as e:
                        raise CLIJSONDecodeError(resp, e)
                yield json.loads(resp)
        
        mock_transport.__aiter__.return_value = mock_iter()
        
        with patch(
            "claude_code_sdk._internal.client.SubprocessCLITransport",
            return_value=mock_transport
        ):
            messages = []
            with pytest.raises(CLIJSONDecodeError) as exc_info:
                async for msg in query("test"):
                    messages.append(msg)
            
            assert len(messages) == 1  # Only first message before error
            assert "invalid json" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self):
        """Test handling of timeout scenarios."""
        mock_transport = AsyncMock(spec=SubprocessCLITransport)
        
        async def slow_response():
            await asyncio.sleep(10)  # Simulate slow response
            yield {"type": "assistant", "content": [{"type": "text", "text": "Late"}]}
        
        mock_transport.__aiter__.return_value = slow_response()
        
        with patch(
            "claude_code_sdk._internal.client.SubprocessCLITransport",
            return_value=mock_transport
        ):
            with pytest.raises(asyncio.TimeoutError):
                async with asyncio.timeout(0.1):
                    async for _ in query("test"):
                        pass

    @pytest.mark.asyncio
    async def test_connection_error_scenarios(self):
        """Test various connection error scenarios."""
        scenarios = [
            (OSError("Connection refused"), "Connection refused"),
            (OSError("No such file or directory"), "No such file"),
            (OSError("Permission denied"), "Permission denied"),
        ]
        
        for original_error, expected_msg in scenarios:
            mock_transport = AsyncMock(spec=SubprocessCLITransport)
            mock_transport.__aiter__.side_effect = CLIConnectionError(
                f"Failed to connect: {original_error}"
            )
            
            with patch(
                "claude_code_sdk._internal.client.SubprocessCLITransport",
                return_value=mock_transport
            ):
                with pytest.raises(CLIConnectionError) as exc_info:
                    async for _ in query("test"):
                        pass
                
                assert expected_msg in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_error_during_message_processing(self):
        """Test errors that occur during message processing."""
        mock_transport = AsyncMock(spec=SubprocessCLITransport)
        
        # Simulate error in the middle of conversation
        responses = [
            {"type": "user", "content": "test"},
            {"type": "assistant", "content": [{"type": "text", "text": "Processing..."}]},
            {"type": "system", "subtype": "error", "data": {"error": "Tool execution failed"}},
        ]
        
        async def mock_iter():
            for resp in responses:
                yield resp
                if resp.get("subtype") == "error":
                    raise ProcessError("Tool execution failed", exit_code=1)
        
        mock_transport.__aiter__.return_value = mock_iter()
        
        with patch(
            "claude_code_sdk._internal.client.SubprocessCLITransport",
            return_value=mock_transport
        ):
            messages = []
            with pytest.raises(ProcessError) as exc_info:
                async for msg in query("test"):
                    messages.append(msg)
            
            assert len(messages) == 3
            assert "Tool execution failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_graceful_shutdown_on_error(self):
        """Test that resources are cleaned up on error."""
        mock_transport = AsyncMock(spec=SubprocessCLITransport)
        mock_transport.close = AsyncMock()
        
        # Simulate immediate failure
        mock_transport.__aiter__.side_effect = ProcessError("Immediate failure")
        
        with patch(
            "claude_code_sdk._internal.client.SubprocessCLITransport",
            return_value=mock_transport
        ):
            with pytest.raises(ProcessError):
                async for _ in query("test"):
                    pass
            
            # Verify cleanup was called
            mock_transport.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_in_tool_result(self):
        """Test handling of tool execution errors."""
        mock_transport = AsyncMock(spec=SubprocessCLITransport)
        
        responses = [
            {"type": "user", "content": "Read non-existent file"},
            {
                "type": "assistant",
                "content": [
                    {"type": "text", "text": "I'll read that file."},
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "Read",
                        "input": {"file_path": "/non/existent/file.txt"}
                    }
                ]
            },
            {
                "type": "assistant",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_1",
                        "content": "Error: File not found: /non/existent/file.txt",
                        "is_error": True
                    }
                ]
            },
            {"type": "result", "subtype": "success"}
        ]
        
        async def mock_iter():
            for resp in responses:
                yield resp
        
        mock_transport.__aiter__.return_value = mock_iter()
        
        with patch(
            "claude_code_sdk._internal.client.SubprocessCLITransport",
            return_value=mock_transport
        ):
            tool_errors = []
            async for msg in query("Read non-existent file"):
                if hasattr(msg, 'content'):
                    for block in getattr(msg.content, '__iter__', lambda: [])():
                        if hasattr(block, 'is_error') and block.is_error:
                            tool_errors.append(block)
            
            assert len(tool_errors) == 1
            assert "File not found" in tool_errors[0].content

    @pytest.mark.asyncio
    async def test_malformed_message_type(self):
        """Test handling of unknown message types."""
        mock_transport = AsyncMock(spec=SubprocessCLITransport)
        
        responses = [
            {"type": "user", "content": "test"},
            {"type": "unknown_type", "data": "something"},  # Unknown type
            {"type": "assistant", "content": [{"type": "text", "text": "Hello"}]},
            {"type": "result", "subtype": "success"}
        ]
        
        async def mock_iter():
            for resp in responses:
                yield resp
        
        mock_transport.__aiter__.return_value = mock_iter()
        
        with patch(
            "claude_code_sdk._internal.client.SubprocessCLITransport",
            return_value=mock_transport
        ):
            messages = []
            async for msg in query("test"):
                messages.append(msg)
            
            # Should handle unknown type gracefully
            assert len(messages) == 4

    @pytest.mark.asyncio
    async def test_permission_denied_error(self):
        """Test handling of permission errors."""
        mock_transport = AsyncMock(spec=SubprocessCLITransport)
        
        # Simulate permission denied when trying to write file
        mock_transport.__aiter__.side_effect = ProcessError(
            "Permission denied",
            exit_code=1,
            stderr="Error: EACCES: permission denied, open '/protected/file.txt'"
        )
        
        with patch(
            "claude_code_sdk._internal.client.SubprocessCLITransport",
            return_value=mock_transport
        ):
            with pytest.raises(ProcessError) as exc_info:
                async for _ in query("Write to protected file"):
                    pass
            
            assert "Permission denied" in str(exc_info.value)
            assert "EACCES" in exc_info.value.stderr

    @pytest.mark.asyncio
    async def test_network_error_recovery(self):
        """Test handling of network-related errors."""
        mock_transport = AsyncMock(spec=SubprocessCLITransport)
        
        # Simulate network error
        mock_transport.__aiter__.side_effect = CLIConnectionError(
            "Network error: Unable to reach API endpoint"
        )
        
        with patch(
            "claude_code_sdk._internal.client.SubprocessCLITransport",
            return_value=mock_transport
        ):
            with pytest.raises(CLIConnectionError) as exc_info:
                async for _ in query("test"):
                    pass
            
            assert "Network error" in str(exc_info.value)
            assert "API endpoint" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_subprocess_crash_handling(self):
        """Test handling of subprocess crashes."""
        mock_transport = AsyncMock(spec=SubprocessCLITransport)
        
        # Simulate subprocess crash
        mock_transport.__aiter__.side_effect = ProcessError(
            "Subprocess terminated unexpectedly",
            exit_code=-11,  # SIGSEGV
            stderr="Segmentation fault"
        )
        
        with patch(
            "claude_code_sdk._internal.client.SubprocessCLITransport",
            return_value=mock_transport
        ):
            with pytest.raises(ProcessError) as exc_info:
                async for _ in query("test"):
                    pass
            
            assert exc_info.value.exit_code == -11
            assert "Segmentation fault" in exc_info.value.stderr

    def test_error_inheritance_chain(self):
        """Test that all errors inherit from ClaudeSDKError."""
        errors = [
            CLINotFoundError("test"),
            CLIConnectionError("test"),
            ProcessError("test", 1),
            CLIJSONDecodeError("test", json.JSONDecodeError("test", "doc", 0)),
        ]
        
        for error in errors:
            assert isinstance(error, ClaudeSDKError)
            assert isinstance(error, Exception)

    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self):
        """Test error handling with concurrent queries."""
        mock_transport = AsyncMock(spec=SubprocessCLITransport)
        
        # All queries will fail
        mock_transport.__aiter__.side_effect = ProcessError("Concurrent failure")
        
        with patch(
            "claude_code_sdk._internal.client.SubprocessCLITransport",
            return_value=mock_transport
        ):
            # Run multiple queries concurrently
            tasks = []
            for i in range(3):
                async def run_query(index):
                    try:
                        async for _ in query(f"Query {index}"):
                            pass
                    except ProcessError:
                        return f"Failed {index}"
                
                tasks.append(run_query(i))
            
            results = await asyncio.gather(*tasks)
            assert all("Failed" in r for r in results)