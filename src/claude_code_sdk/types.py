"""Type definitions for Claude SDK."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict

from typing_extensions import NotRequired  # For Python < 3.11 compatibility

# Permission modes
PermissionMode = Literal["default", "acceptEdits", "bypassPermissions"]


# MCP Server config
class McpServerConfig(TypedDict):
    """MCP server configuration."""

    transport: list[str]
    env: NotRequired[dict[str, Any]]


# Content block types
@dataclass
class TextBlock:
    """Text content block."""

    text: str


@dataclass
class ToolUseBlock:
    """Tool use content block."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResultBlock:
    """Tool result content block."""

    tool_use_id: str
    content: str | list[dict[str, Any]] | None = None
    is_error: bool | None = None


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock


# Message types
@dataclass
class UserMessage:
    """User message."""

    content: str


@dataclass
class AssistantMessage:
    """Assistant message with content blocks."""

    content: list[ContentBlock]


@dataclass
class SystemMessage:
    """System message with metadata."""

    subtype: str
    data: dict[str, Any]


@dataclass
class ResultMessage:
    """Result message with cost and usage information."""

    subtype: str
    cost_usd: float
    duration_ms: int
    duration_api_ms: int
    is_error: bool
    num_turns: int
    session_id: str
    total_cost_usd: float
    usage: dict[str, Any] | None = None
    result: str | None = None


Message = UserMessage | AssistantMessage | SystemMessage | ResultMessage


@dataclass
class ClaudeCodeOptions:
    """Query options for Claude SDK."""

    allowed_tools: list[str] = field(default_factory=list)
    max_thinking_tokens: int = 8000
    system_prompt: str | None = None
    append_system_prompt: str | None = None
    mcp_tools: list[str] = field(default_factory=list)
    mcp_servers: dict[str, McpServerConfig] = field(default_factory=dict)
    permission_mode: PermissionMode | None = None
    continue_conversation: bool = False
    resume: str | None = None
    max_turns: int | None = None
    disallowed_tools: list[str] = field(default_factory=list)
    model: str | None = None
    permission_prompt_tool_name: str | None = None
    cwd: str | Path | None = None
    add_dirs: list[str | Path] = field(default_factory=list)
    dangerously_skip_permissions: bool = False
    debug: bool = False
    verbose: bool = True  # SDK always uses verbose mode by default
    output_format: Literal["text", "json", "stream-json"] = "stream-json"
    input_format: Literal["text", "stream-json"] = "text"
