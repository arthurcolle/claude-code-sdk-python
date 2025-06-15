# Claude Code SDK for Python

Python SDK for Claude Code with WebSocket server and multi-agent system support. See the [Claude Code SDK documentation](https://docs.anthropic.com/en/docs/claude-code/sdk) for more information.

## New Features

- **WebSocket Server**: Real-time bidirectional communication with streaming support
- **Agent System Integration**: Build multi-agent applications with specialized AI agents
- **Enhanced Error Handling**: Comprehensive error types and recovery mechanisms
- **Type Safety**: Full type hints and dataclass-based message types

## Installation

```bash
# Basic installation
pip install claude-code-sdk

# With WebSocket server support
pip install claude-code-sdk[websocket]

# With agent system (separate package)
pip install claude-code-agent-system
```

**Prerequisites:**
- Python 3.10+
- Node.js 
- Claude Code: `npm install -g @anthropic-ai/claude-code`

## Quick Start

```python
import anyio
from claude_code_sdk import query

async def main():
    async for message in query(prompt="What is 2 + 2?"):
        print(message)

anyio.run(main)
```

## Usage

### Basic Query

```python
from claude_code_sdk import query, ClaudeCodeOptions, AssistantMessage, TextBlock

# Simple query
async for message in query(prompt="Hello Claude"):
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                print(block.text)

# With options
options = ClaudeCodeOptions(
    system_prompt="You are a helpful assistant",
    max_turns=1
)

async for message in query(prompt="Tell me a joke", options=options):
    print(message)
```

### Using Tools

```python
options = ClaudeCodeOptions(
    allowed_tools=["Read", "Write", "Bash"],
    permission_mode='acceptEdits'  # auto-accept file edits
)

async for message in query(
    prompt="Create a hello.py file", 
    options=options
):
    # Process tool use and results
    pass
```

### Working Directory

```python
from pathlib import Path

options = ClaudeCodeOptions(
    cwd="/path/to/project"  # or Path("/path/to/project")
)
```

## API Reference

### `query(prompt, options=None)`

Main async function for querying Claude.

**Parameters:**
- `prompt` (str): The prompt to send to Claude
- `options` (ClaudeCodeOptions): Optional configuration

**Returns:** AsyncIterator[Message] - Stream of response messages

### Types

See [src/claude_code_sdk/types.py](src/claude_code_sdk/types.py) for complete type definitions:
- `ClaudeCodeOptions` - Configuration options
- `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage` - Message types
- `TextBlock`, `ToolUseBlock`, `ToolResultBlock` - Content blocks

## Error Handling

```python
from claude_code_sdk import (
    ClaudeSDKError,      # Base error
    CLINotFoundError,    # Claude Code not installed
    CLIConnectionError,  # Connection issues
    ProcessError,        # Process failed
    CLIJSONDecodeError,  # JSON parsing issues
)

try:
    async for message in query(prompt="Hello"):
        pass
except CLINotFoundError:
    print("Please install Claude Code")
except ProcessError as e:
    print(f"Process failed with exit code: {e.exit_code}")
except CLIJSONDecodeError as e:
    print(f"Failed to parse response: {e}")
```

See [src/claude_code_sdk/_errors.py](src/claude_code_sdk/_errors.py) for all error types.

## Available Tools

See the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/security#tools-available-to-claude) for a complete list of available tools.

## WebSocket Server

The SDK includes a WebSocket server for real-time communication:

```python
from claude_code_sdk.websocket_server import EnhancedClaudeWebSocketServer

server = EnhancedClaudeWebSocketServer()
server.run(host="0.0.0.0", port=8000)
```

Connect from JavaScript:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.send(JSON.stringify({
    type: 'query',
    prompt: 'Hello Claude!',
    options: { allowed_tools: ['Read', 'Write'] }
}));
```

See [docs/websocket-server.md](docs/websocket-server.md) for full documentation.

## Agent System Integration

Build multi-agent applications using the separate agent system package:

```python
from claude_code_sdk import query, ClaudeCodeOptions
from claude_code_agent_system import BaseAgent

class CustomAgent(BaseAgent):
    async def process_with_claude(self, prompt: str):
        options = ClaudeCodeOptions(allowed_tools=["Read", "Write"])
        async for message in query(prompt=prompt, options=options):
            # Process responses
            pass
```

See [docs/agent-system.md](docs/agent-system.md) for integration guide.

## Examples

- [examples/quick_start.py](examples/quick_start.py) - Basic SDK usage
- [examples/websocket_ui_server.py](examples/websocket_ui_server.py) - WebSocket server with UI
- [examples/agent_sdk_integration.py](examples/agent_sdk_integration.py) - Agent system integration
- [agent_system/](agent_system/) - Complete agent system implementation

## Documentation

- [WebSocket Server Guide](docs/websocket-server.md)
- [Agent System Integration](docs/agent-system.md)
- [API Reference](https://docs.anthropic.com/en/docs/claude-code/sdk)

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to our repository.

## License

MIT