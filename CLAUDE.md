# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup
```bash
# Install development dependencies
pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests with coverage
python -m pytest tests/ -v --cov=claude_code_sdk --cov-report=xml

# Run a specific test file
python -m pytest tests/test_client.py -v

# Run a specific test
python -m pytest tests/test_client.py::test_function_name -v
```

### Code Quality
```bash
# Run linting
ruff check src/ tests/

# Check formatting
ruff format --check src/ tests/

# Auto-fix formatting
ruff format src/ tests/

# Type checking
mypy src/
```

## Architecture Overview

This is a Python SDK for Claude Code that provides an async interface to the Claude Code CLI tool.

### Core Design Principles
- **Async-first**: All APIs use `anyio` for async operations
- **Type-safe**: Extensive use of dataclasses and TypedDict with strict mypy checking
- **Message-based**: Communication through typed message objects (UserMessage, AssistantMessage, etc.)
- **Transport abstraction**: Currently uses subprocess-based CLI transport, extensible for other transports

### Key Components

1. **Entry Point** (`src/claude_code_sdk/__init__.py`):
   - Single public function: `query()` - async generator for Claude interactions
   - Sets `CLAUDE_CODE_ENTRYPOINT=sdk-py` environment variable

2. **Internal Client** (`src/claude_code_sdk/_internal/client.py`):
   - `InternalClient` class manages the conversation lifecycle
   - Handles message streaming and tool execution

3. **Transport Layer** (`src/claude_code_sdk/_internal/transport/subprocess_cli.py`):
   - Manages subprocess communication with Claude Code CLI
   - Handles JSON message serialization/deserialization
   - Implements error handling for CLI failures

4. **Type System** (`src/claude_code_sdk/types.py`):
   - Comprehensive type definitions for all messages and options
   - Uses dataclasses for structured data
   - TypedDict for JSON-like structures

5. **Error Handling** (`src/claude_code_sdk/_errors.py`):
   - Hierarchical error classes for different failure modes
   - Preserves original error context (exit codes, stderr)

### Testing Strategy
- Unit tests for each component
- Integration tests for end-to-end flows
- Mock-based testing for CLI interactions
- Async test support with pytest-asyncio

### Important Notes
- Python 3.10+ required (uses modern type hints)
- Depends on Claude Code CLI being installed (`npm install -g @anthropic-ai/claude-code`)
- All public APIs are in `__init__.py` - internal modules should not be imported directly

## Available Tools and Usage Patterns

The SDK provides access to Claude Code's tool capabilities through the `allowed_tools` option:

```python
from claude_code_sdk import query, ClaudeCodeOptions

options = ClaudeCodeOptions(
    allowed_tools=["Read", "Write", "Edit", "Bash"],  # Enable specific tools
    disallowed_tools=["Delete"],  # Explicitly disable tools
)
```

## Permission Modes

Control how Claude handles tool execution permissions:

- `"default"`: CLI prompts for dangerous tools (interactive mode)
- `"acceptEdits"`: Auto-accept file edits without prompting
- `"bypassPermissions"`: Allow all tools without prompting (use with caution)

```python
options = ClaudeCodeOptions(
    permission_mode="acceptEdits"  # Auto-accept file edits
)
```

## ClaudeCodeOptions Configuration

Full configuration options available:

```python
@dataclass
class ClaudeCodeOptions:
    allowed_tools: list[str] = []              # Tools Claude can use
    max_thinking_tokens: int = 8000            # Thinking token limit
    system_prompt: str | None = None           # Override system prompt
    append_system_prompt: str | None = None    # Append to default prompt
    mcp_tools: list[str] = []                  # MCP tool names
    mcp_servers: dict[str, McpServerConfig] = {}  # MCP server configs
    permission_mode: PermissionMode | None = None  # Permission handling
    continue_conversation: bool = False        # Continue previous conversation
    resume: str | None = None                  # Resume specific session ID
    max_turns: int | None = None              # Limit conversation turns
    disallowed_tools: list[str] = []          # Tools to explicitly disable
    model: str | None = None                  # Specific model to use
    permission_prompt_tool_name: str | None = None  # Custom permission tool
    cwd: str | Path | None = None             # Working directory
```

## Message Types and Content Blocks

### Message Types

1. **UserMessage**: Simple user input
   ```python
   UserMessage(content="Hello Claude!")
   ```

2. **AssistantMessage**: Claude's response with content blocks
   ```python
   AssistantMessage(content=[
       TextBlock(text="Hello!"),
       ToolUseBlock(id="...", name="Read", input={...})
   ])
   ```

3. **SystemMessage**: System-level messages with metadata
   ```python
   SystemMessage(subtype="info", data={...})
   ```

4. **ResultMessage**: Conversation result with usage info
   ```python
   ResultMessage(
       subtype="success",
       cost_usd=0.01,
       duration_ms=1500,
       session_id="...",
       total_cost_usd=0.02,
       usage={...}
   )
   ```

### Content Block Types

1. **TextBlock**: Plain text content
2. **ToolUseBlock**: Tool invocation details
3. **ToolResultBlock**: Tool execution results

## Advanced Features

### MCP Server Integration

Configure Model Context Protocol servers:

```python
options = ClaudeCodeOptions(
    mcp_servers={
        "my-server": {
            "transport": ["node", "server.js"],
            "env": {"API_KEY": "..."}
        }
    },
    mcp_tools=["my-server:tool1", "my-server:tool2"]
)
```

### Conversation Management

Continue or resume previous conversations:

```python
# Continue the last conversation
options = ClaudeCodeOptions(continue_conversation=True)

# Resume a specific session
options = ClaudeCodeOptions(resume="session-id-123")
```

### Custom System Prompts

Override or append to the default system prompt:

```python
# Replace system prompt entirely
options = ClaudeCodeOptions(
    system_prompt="You are a Python expert assistant."
)

# Append to default prompt
options = ClaudeCodeOptions(
    append_system_prompt="Always use type hints in Python code."
)
```

### Working Directory Control

Set a custom working directory for file operations:

```python
options = ClaudeCodeOptions(cwd="/path/to/project")
```

## Error Handling

Comprehensive error hierarchy for different failure modes:

```python
from claude_code_sdk import (
    ClaudeSDKError,          # Base exception
    CLIConnectionError,      # Connection issues
    CLINotFoundError,        # CLI not installed
    ProcessError,            # CLI process failures
    CLIJSONDecodeError      # JSON parsing errors
)

try:
    async for message in query(prompt="..."):
        ...
except CLINotFoundError:
    print("Please install Claude Code CLI")
except ProcessError as e:
    print(f"Process failed: {e.exit_code}, {e.stderr}")
```

## Usage Examples

### Processing Tool Results

```python
async for message in query(prompt="Read config.json"):
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                print(f"Text: {block.text}")
            elif isinstance(block, ToolUseBlock):
                print(f"Using tool: {block.name}")
            elif isinstance(block, ToolResultBlock):
                print(f"Tool result: {block.content}")
```

### Tracking Costs and Usage

```python
total_cost = 0.0
async for message in query(prompt="..."):
    if isinstance(message, ResultMessage):
        print(f"Session: {message.session_id}")
        print(f"Cost: ${message.cost_usd:.4f}")
        print(f"Total: ${message.total_cost_usd:.4f}")
        print(f"Turns: {message.num_turns}")
        if message.usage:
            print(f"Usage: {message.usage}")
```

### Environment Variables

The SDK automatically sets `CLAUDE_CODE_ENTRYPOINT=sdk-py` to identify SDK usage to the CLI.

## Advanced Capabilities

### CLI Discovery

The SDK automatically searches for Claude Code CLI in multiple locations:
- System PATH
- `~/.npm-global/bin/claude`
- `/usr/local/bin/claude`
- `~/.local/bin/claude`
- `~/node_modules/.bin/claude`
- `~/.yarn/bin/claude`

If Node.js is not installed, the SDK provides helpful installation instructions.

### Stream Processing

The SDK handles concurrent stdout/stderr streams efficiently:
- Background stderr collection for better error diagnostics
- Graceful handling of stream closures
- Non-blocking message processing

### Tool Result Error States

Tool results can indicate error conditions:

```python
# In AssistantMessage content blocks
ToolResultBlock(
    tool_use_id="...",
    content="Error message",
    is_error=True  # Indicates tool execution failed
)
```

### Process Management

The SDK implements robust process lifecycle management:
- Graceful termination with 5-second timeout
- Forced kill if graceful termination fails
- Comprehensive error context with exit codes and stderr

### JSON Parsing Flexibility

The transport intelligently handles mixed output:
- JSON lines are parsed as messages
- Non-JSON lines are ignored (useful for debugging output)
- Clear error messages for malformed JSON

### Testing Patterns

When testing SDK integrations:

```python
# Use anyio.run() for better compatibility
import anyio

def test_my_integration():
    async def _test():
        async for message in query(prompt="..."):
            # Test assertions
            pass
    
    anyio.run(_test)
```

### Performance Optimizations

- Efficient message type dispatch using Python match statements
- Minimal memory overhead with streaming message processing
- Concurrent I/O operations for optimal throughput

### Path Flexibility

The SDK accepts both strings and `pathlib.Path` objects:

```python
from pathlib import Path

options = ClaudeCodeOptions(
    cwd=Path.home() / "projects" / "myapp"
)
```

### Rich Error Context

All errors include detailed context for debugging:

```python
try:
    async for message in query(prompt="..."):
        pass
except ProcessError as e:
    print(f"Exit code: {e.exit_code}")
    print(f"Stderr output: {e.stderr}")
    # Full error context available
```

### Message Passthrough

System messages preserve all data from the CLI:

```python
if isinstance(message, SystemMessage):
    # Access any field from the CLI response
    custom_field = message.data.get("custom_field")
    metadata = message.data.get("metadata", {})
```

## Common Patterns and Best Practices

### Handling Multiple Message Types

```python
async for message in query(prompt="Analyze and fix the code"):
    match message:
        case AssistantMessage(content=blocks):
            for block in blocks:
                match block:
                    case TextBlock(text=text):
                        print(f"Assistant: {text}")
                    case ToolUseBlock(name=name, input=input):
                        print(f"Using tool: {name}")
                    case ToolResultBlock(content=content, is_error=is_error):
                        if is_error:
                            print(f"Tool error: {content}")
        case SystemMessage(subtype=subtype, data=data):
            if subtype == "error":
                print(f"System error: {data}")
        case ResultMessage(cost_usd=cost, session_id=session):
            print(f"Session {session} cost: ${cost:.4f}")
```

### Building Interactive Applications

```python
# Create a reusable conversation session
session_id = None

async def chat(user_input: str):
    global session_id
    options = ClaudeCodeOptions(
        resume=session_id,  # Resume previous session if exists
        allowed_tools=["Read", "Write", "Edit"],
        permission_mode="acceptEdits"
    )
    
    async for message in query(prompt=user_input, options=options):
        if isinstance(message, ResultMessage):
            session_id = message.session_id  # Save for next turn
        # Process other messages...
```

### Debugging SDK Issues

Enable verbose logging to debug connection issues:

```python
import logging

# Enable debug logging for the SDK
logging.basicConfig(level=logging.DEBUG)

# The SDK will log:
# - CLI discovery process
# - Process spawning details
# - Stream communication
# - Error details with full context
```

### Handling Long-Running Operations

For operations that might take time:

```python
import asyncio

async def long_operation():
    timeout = 300  # 5 minutes
    
    try:
        async with asyncio.timeout(timeout):
            async for message in query(prompt="Run all tests and fix issues"):
                # Process messages
                if isinstance(message, AssistantMessage):
                    print(".", end="", flush=True)  # Progress indicator
    except asyncio.TimeoutError:
        print("\nOperation timed out")
```

### Tool Usage Monitoring

Track which tools Claude uses:

```python
tool_usage = {}

async for message in query(prompt="...", options=options):
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                tool_usage[block.name] = tool_usage.get(block.name, 0) + 1

print(f"Tool usage: {tool_usage}")
```

## Troubleshooting

### Common Issues

1. **CLINotFoundError**: Claude Code CLI not installed
   - Solution: `npm install -g @anthropic-ai/claude-code`

2. **ProcessError with exit code 1**: CLI command failed
   - Check stderr in the error for details
   - Verify Node.js is installed
   - Check CLI version compatibility

3. **CLIJSONDecodeError**: Malformed response
   - May indicate CLI version mismatch
   - Check for CLI error output in stderr

4. **Timeout errors**: Long operations
   - Increase timeout in your async code
   - Consider breaking into smaller operations

### Version Compatibility

- SDK requires Python 3.10+
- Claude Code CLI must be installed separately
- Use `claude --version` to check CLI version