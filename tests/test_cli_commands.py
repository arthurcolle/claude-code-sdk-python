"""Tests for CLI command wrapper functions."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path

from claude_code_sdk import (
    update,
    mcp,
    config,
    doctor,
    version,
    CLINotFoundError,
    ProcessError,
)


@pytest.mark.anyio
async def test_update_command():
    """Test update command wrapper."""
    with patch("claude_code_sdk._run_cli_command") as mock_run:
        mock_run.return_value = "Claude Code is up to date"
        
        result = await update()
        
        mock_run.assert_called_once_with(["update"], None)
        assert result == "Claude Code is up to date"


@pytest.mark.anyio
async def test_update_command_with_cli_path():
    """Test update command with custom CLI path."""
    with patch("claude_code_sdk._run_cli_command") as mock_run:
        mock_run.return_value = "Updated to version 1.2.3"
        
        result = await update(cli_path="/custom/path/claude")
        
        mock_run.assert_called_once_with(["update"], "/custom/path/claude")
        assert result == "Updated to version 1.2.3"


@pytest.mark.anyio
async def test_mcp_list():
    """Test MCP list command."""
    with patch("claude_code_sdk._run_cli_command") as mock_run:
        mock_run.return_value = "server1\nserver2"
        
        result = await mcp(["list"])
        
        mock_run.assert_called_once_with(["mcp", "list"], None)
        assert result == "server1\nserver2"


@pytest.mark.anyio
async def test_mcp_add():
    """Test MCP add command."""
    with patch("claude_code_sdk._run_cli_command") as mock_run:
        mock_run.return_value = "Server added successfully"
        
        result = await mcp(["add", "my-server"])
        
        mock_run.assert_called_once_with(["mcp", "add", "my-server"], None)
        assert result == "Server added successfully"


@pytest.mark.anyio
async def test_mcp_no_subcommand():
    """Test MCP command without subcommand."""
    with patch("claude_code_sdk._run_cli_command") as mock_run:
        mock_run.return_value = "MCP help text"
        
        result = await mcp()
        
        mock_run.assert_called_once_with(["mcp"], None)
        assert result == "MCP help text"


@pytest.mark.anyio
async def test_config_get():
    """Test config get command."""
    with patch("claude_code_sdk._run_cli_command") as mock_run:
        mock_run.return_value = "claude-sonnet-4"
        
        result = await config(["get", "model"])
        
        mock_run.assert_called_once_with(["config", "get", "model"], None)
        assert result == "claude-sonnet-4"


@pytest.mark.anyio
async def test_config_set():
    """Test config set command."""
    with patch("claude_code_sdk._run_cli_command") as mock_run:
        mock_run.return_value = "Configuration updated"
        
        result = await config(["set", "model", "claude-opus-4"])
        
        mock_run.assert_called_once_with(["config", "set", "model", "claude-opus-4"], None)
        assert result == "Configuration updated"


@pytest.mark.anyio
async def test_doctor_command():
    """Test doctor command."""
    with patch("claude_code_sdk._run_cli_command") as mock_run:
        mock_run.return_value = "All checks passed"
        
        result = await doctor()
        
        mock_run.assert_called_once_with(["doctor"], None)
        assert result == "All checks passed"


@pytest.mark.anyio
async def test_version_command():
    """Test version command."""
    with patch("claude_code_sdk._run_cli_command") as mock_run:
        mock_run.return_value = "1.2.3"
        
        result = await version()
        
        mock_run.assert_called_once_with(["--version"], None)
        assert result == "1.2.3"


@pytest.mark.anyio
async def test_run_cli_command_not_found():
    """Test _run_cli_command when CLI is not found."""
    with patch("shutil.which", return_value=None):
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(CLINotFoundError) as exc_info:
                from claude_code_sdk import _run_cli_command
                await _run_cli_command(["test"])
            
            assert "Claude Code not found" in str(exc_info.value)


@pytest.mark.anyio
async def test_run_cli_command_process_error():
    """Test _run_cli_command when process fails."""
    mock_process = MagicMock()
    mock_process.returncode = 1
    mock_process.communicate = AsyncMock(return_value=(b"", b"Error occurred"))
    
    with patch("shutil.which", return_value="/usr/bin/claude"):
        with patch("anyio.open_process", return_value=mock_process):
            with pytest.raises(ProcessError) as exc_info:
                from claude_code_sdk import _run_cli_command
                await _run_cli_command(["test"])
            
            assert exc_info.value.exit_code == 1
            assert "Error occurred" in exc_info.value.stderr


@pytest.mark.anyio
async def test_run_cli_command_success():
    """Test _run_cli_command successful execution."""
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b"Success", b""))
    
    with patch("shutil.which", return_value="/usr/bin/claude"):
        with patch("anyio.open_process", return_value=mock_process):
            from claude_code_sdk import _run_cli_command
            result = await _run_cli_command(["test"])
            
            assert result == "Success"


@pytest.mark.anyio
async def test_run_cli_command_finds_in_locations():
    """Test _run_cli_command finds CLI in alternative locations."""
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b"Found", b""))
    
    with patch("shutil.which", return_value=None):
        # Mock path checking - only second location exists
        def mock_exists(self):
            return str(self) == str(Path.home() / ".local/bin/claude")
        
        with patch("pathlib.Path.exists", mock_exists):
            with patch("pathlib.Path.is_file", return_value=True):
                with patch("anyio.open_process", return_value=mock_process) as mock_open:
                    from claude_code_sdk import _run_cli_command
                    result = await _run_cli_command(["test"])
                    
                    # Verify it used the found path
                    called_cmd = mock_open.call_args[0][0]
                    assert str(Path.home() / ".local/bin/claude") in called_cmd[0]
                    assert result == "Found"