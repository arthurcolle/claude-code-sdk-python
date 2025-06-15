"""Tests for new CLI options in ClaudeCodeOptions."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from claude_code_sdk import ClaudeCodeOptions, query
from claude_code_sdk._internal.transport.subprocess_cli import SubprocessCLITransport


def test_claude_code_options_new_fields():
    """Test that new fields are properly initialized."""
    options = ClaudeCodeOptions()

    # Test defaults
    assert options.add_dirs == []
    assert options.dangerously_skip_permissions is False
    assert options.debug is False
    assert options.verbose is True
    assert options.output_format == "stream-json"
    assert options.input_format == "text"


def test_claude_code_options_with_values():
    """Test options with custom values."""
    options = ClaudeCodeOptions(
        add_dirs=["/path1", Path("/path2")],
        dangerously_skip_permissions=True,
        debug=True,
        verbose=False,
        output_format="json",
        input_format="stream-json",
    )

    assert options.add_dirs == ["/path1", Path("/path2")]
    assert options.dangerously_skip_permissions is True
    assert options.debug is True
    assert options.verbose is False
    assert options.output_format == "json"
    assert options.input_format == "stream-json"


def test_subprocess_cli_build_command_with_new_flags():
    """Test that new flags are properly added to CLI command."""
    options = ClaudeCodeOptions(
        add_dirs=["/dir1", "/dir2"],
        dangerously_skip_permissions=True,
        debug=True,
        verbose=True,
        output_format="json",
        input_format="stream-json",
    )

    transport = SubprocessCLITransport(
        prompt="test prompt", options=options, cli_path="/usr/bin/claude"
    )

    cmd = transport._build_command()

    # Check basic structure
    assert cmd[0] == "/usr/bin/claude"
    assert "--output-format" in cmd
    assert "json" in cmd

    # Check new flags
    assert "--verbose" in cmd
    assert "--debug" in cmd
    assert "--input-format" in cmd
    assert "stream-json" in cmd
    assert "--dangerously-skip-permissions" in cmd

    # Check add-dir flags
    add_dir_indices = [i for i, x in enumerate(cmd) if x == "--add-dir"]
    assert len(add_dir_indices) == 2
    assert cmd[add_dir_indices[0] + 1] == "/dir1"
    assert cmd[add_dir_indices[1] + 1] == "/dir2"


def test_subprocess_cli_build_command_minimal():
    """Test command building with minimal options."""
    options = ClaudeCodeOptions(
        verbose=False,  # Disable verbose
        debug=False,  # Keep debug disabled (default)
    )

    transport = SubprocessCLITransport(
        prompt="test", options=options, cli_path="/usr/bin/claude"
    )

    cmd = transport._build_command()

    # Should not have verbose or debug flags
    assert "--verbose" not in cmd
    assert "--debug" not in cmd
    assert "--dangerously-skip-permissions" not in cmd

    # Should still have required flags
    assert "--output-format" in cmd
    assert "stream-json" in cmd
    assert "--print" in cmd
    assert "test" in cmd


def test_subprocess_cli_build_command_with_paths():
    """Test that Path objects are properly converted."""
    options = ClaudeCodeOptions(
        add_dirs=[Path.home() / "dir1", "/absolute/dir2"], cwd=Path.home() / "projects"
    )

    transport = SubprocessCLITransport(
        prompt="test", options=options, cli_path="/usr/bin/claude"
    )

    cmd = transport._build_command()

    # Find add-dir arguments
    add_dir_indices = [i for i, x in enumerate(cmd) if x == "--add-dir"]
    assert len(add_dir_indices) == 2

    # Check that paths are converted to strings
    dir1 = cmd[add_dir_indices[0] + 1]
    dir2 = cmd[add_dir_indices[1] + 1]

    assert isinstance(dir1, str)
    assert isinstance(dir2, str)
    assert str(Path.home() / "dir1") == dir1
    assert dir2 == "/absolute/dir2"


def test_all_permission_modes():
    """Test all permission mode options."""
    for mode in ["default", "acceptEdits", "bypassPermissions"]:
        options = ClaudeCodeOptions(permission_mode=mode)
        transport = SubprocessCLITransport(
            prompt="test", options=options, cli_path="/usr/bin/claude"
        )

        cmd = transport._build_command()
        assert "--permission-mode" in cmd
        assert mode in cmd


def test_output_formats():
    """Test different output format options."""
    for fmt in ["text", "json", "stream-json"]:
        options = ClaudeCodeOptions(output_format=fmt)
        transport = SubprocessCLITransport(
            prompt="test", options=options, cli_path="/usr/bin/claude"
        )

        cmd = transport._build_command()
        assert "--output-format" in cmd
        assert fmt in cmd


@pytest.mark.anyio
async def test_query_with_new_options():
    """Test query function with new options."""
    options = ClaudeCodeOptions(
        add_dirs=["/test/dir"], dangerously_skip_permissions=True, debug=True
    )

    # Mock the entire client interaction
    mock_messages = [
        {"type": "user", "content": "test prompt"},
        {"type": "assistant", "content": [{"type": "text", "text": "Test response"}]},
        {
            "type": "result",
            "subtype": "success",
            "cost_usd": 0.01,
            "duration_ms": 100,
            "duration_api_ms": 50,
            "is_error": False,
            "num_turns": 1,
            "session_id": "test-session",
            "total_cost_usd": 0.01,
        },
    ]

    with patch(
        "claude_code_sdk._internal.client.InternalClient.process_query"
    ) as mock_process:

        async def mock_generator(*args, **kwargs):
            for msg in mock_messages:
                yield msg

        mock_process.return_value = mock_generator()

        messages = []
        async for message in query(prompt="test prompt", options=options):
            messages.append(message)

        # Verify the options were passed through
        mock_process.assert_called_once()
        call_args = mock_process.call_args
        assert call_args.kwargs["options"] == options
