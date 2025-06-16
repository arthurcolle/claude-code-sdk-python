# Claude Code SDK for Python

Python SDK for Claude Code with WebSocket server and multi-agent system support. See the [Claude Code SDK documentation](https://docs.anthropic.com/en/docs/claude-code/sdk) for more information.

## New Features

- **OAuth Authentication**: Claude Code Max users can authenticate without API keys
- **WebSocket Server**: Real-time bidirectional communication with streaming support
- **Agent System Integration**: Build multi-agent applications with specialized AI agents
- **Multi-Turn Conversations**: Maintain context across multiple interactions with session management
- **Tool Management API**: Discover, create, and execute tools from a registry
- **Enhanced Error Handling**: Comprehensive error types and recovery mechanisms
- **Type Safety**: Full type hints and dataclass-based message types
- **Jupyter Notebook Support**: Beautiful markdown rendering and interactive displays

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

## Authentication

### OAuth Authentication (Claude Code Max)

Claude Code Max plan users can authenticate using OAuth without API keys:

```bash
# Login via CLI
claude-auth login

# Check authentication status
claude-auth status
```

Use in Python:

```python
from claude_code_sdk import query_with_oauth

async for message in query_with_oauth(prompt="Hello Claude!"):
    print(message)
```

### API Key Authentication

Traditional API key authentication:

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

```python
from claude_code_sdk import query

async for message in query(prompt="Hello Claude!"):
    print(message)
```

See the [Authentication Guide](docs/authentication.md) for more details.

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

### Multi-Turn Conversations

```python
# Method 1: Using session IDs
session_id = None
async for message in query(prompt="Remember my name is Alice"):
    if isinstance(message, ResultMessage):
        session_id = message.session_id

# Continue the conversation
async for message in query(
    prompt="What's my name?",
    options=ClaudeCodeOptions(resume=session_id)
):
    print(message)

# Method 2: Auto-continue last conversation
async for message in query(
    prompt="What did we just discuss?",
    options=ClaudeCodeOptions(continue_conversation=True)
):
    print(message)
```

See the [Multi-Turn Conversations Guide](docs/multi_turn_conversations.md) for detailed examples.

### Advanced Conversation Features

```python
from claude_code_sdk.conversation_manager import ConversationManager
from claude_code_sdk.conversation_templates import TemplateManager
from claude_code_sdk.conversation_chains import create_debugging_chain

# Managed conversations with persistence
manager = ConversationManager()
session_id, messages = await manager.create_conversation(
    initial_prompt="Let's build a web app",
    tags=["project", "web"]
)

# Use pre-built templates
template_mgr = TemplateManager()
template = template_mgr.get_template("code_review")

# Automated workflows with chains
debug_chain = create_debugging_chain()
result = await debug_chain.execute(
    context_overrides={"issue_description": "Memory leak"}
)
```

See the [Advanced Conversation Features Guide](docs/advanced_conversation_features.md) for comprehensive documentation.

### Advanced Options

```python
options = ClaudeCodeOptions(
    # Add additional directories for tool access
    add_dirs=["/path/to/libs", "/path/to/data"],
    
    # Skip all permission prompts (use with caution!)
    dangerously_skip_permissions=True,
    
    # Enable debug logging in CLI
    debug=True,
    
    # Control output verbosity (default: True for SDK)
    verbose=False,
    
    # Specify output format (default: "stream-json")
    output_format="json",  # "text", "json", or "stream-json"
    
    # Specify input format
    input_format="text",  # "text" or "stream-json"
)
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

### CLI Commands

The SDK provides wrapper functions for Claude Code CLI commands:

#### `update(cli_path=None)`
Check for and install updates to Claude Code.

```python
result = await update()
print(result)
```

#### `mcp(subcommand=None, cli_path=None)`
Configure and manage Model Context Protocol (MCP) servers.

```python
# List MCP servers
servers = await mcp(["list"])

# Add MCP server
result = await mcp(["add", "my-server"])

# Remove MCP server
result = await mcp(["remove", "my-server"])
```

#### `config(subcommand=None, cli_path=None)`
Manage Claude Code configuration.

```python
# Get configuration value
model = await config(["get", "model"])

# Set configuration value
result = await config(["set", "model", "claude-opus-4"])

# List all config
all_config = await config(["list"])
```

#### `doctor(cli_path=None)`
Check health of Claude Code auto-updater.

```python
health = await doctor()
print(health)
```

#### `version(cli_path=None)`
Get Claude Code version.

```python
ver = await version()
print(f"Claude Code version: {ver}")
```

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

## Troubleshooting

### Common Issues and Solutions

#### 1. CLINotFoundError: Claude Code not found

**Problem**: The SDK cannot find the Claude Code CLI.

**Solutions**:
```bash
# Install Claude Code globally
npm install -g @anthropic-ai/claude-code

# Verify installation
claude --version
```

If Node.js is not installed:
- **macOS**: `brew install node` or download from [nodejs.org](https://nodejs.org)
- **Linux**: Use your package manager (e.g., `sudo apt install nodejs npm`)
- **Windows**: Download installer from [nodejs.org](https://nodejs.org)

#### 2. ProcessError: Exit code 1

**Problem**: The Claude Code process failed to start or crashed.

**Common causes and solutions**:

- **Missing API key**: Set your Anthropic API key:
  ```bash
  export ANTHROPIC_API_KEY="your-api-key"
  ```

- **Invalid configuration**: Check your `ClaudeCodeOptions` for typos or invalid values

- **Permission issues**: Ensure you have read/write permissions in the working directory

**Debug steps**:
```python
try:
    async for message in query(prompt="test"):
        pass
except ProcessError as e:
    print(f"Exit code: {e.exit_code}")
    print(f"Error details: {e.stderr}")
```

#### 3. CLIConnectionError: Failed to connect

**Problem**: Cannot establish connection to Claude Code CLI.

**Solutions**:
- Check if another instance is already running
- Ensure sufficient system resources (RAM, disk space)
- Try with a simple prompt first to isolate the issue

#### 4. CLIJSONDecodeError: Invalid JSON response

**Problem**: Received malformed JSON from the CLI.

**Common causes**:
- Version mismatch between SDK and CLI
- Corrupted installation

**Solutions**:
```bash
# Update both SDK and CLI
pip install --upgrade claude-code-sdk
npm update -g @anthropic-ai/claude-code
```

#### 5. TimeoutError during long operations

**Problem**: Operations taking too long and timing out.

**Solution**: Wrap your code with custom timeout:
```python
import asyncio

async def long_operation():
    async with asyncio.timeout(300):  # 5 minutes
        async for message in query(prompt="Complex task..."):
            # Process messages
            pass
```

#### 6. Permission denied errors

**Problem**: Cannot read/write files due to permissions.

**Solutions**:
- Run with appropriate user permissions
- Set working directory to a writable location:
  ```python
  options = ClaudeCodeOptions(cwd="/path/to/writable/directory")
  ```
- Use `permission_mode='bypassPermissions'` (use with caution)

#### 7. Rate limiting errors

**Problem**: Too many requests to the API.

**Solutions**:
- Implement exponential backoff:
  ```python
  import asyncio
  
  async def query_with_retry(prompt, max_retries=3):
      for attempt in range(max_retries):
          try:
              async for message in query(prompt=prompt):
                  yield message
              break
          except ProcessError as e:
              if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                  await asyncio.sleep(2 ** attempt)
              else:
                  raise
  ```

#### 8. WebSocket connection issues

**Problem**: Cannot connect to WebSocket server.

**Common issues**:
- Port already in use
- Firewall blocking connections
- CORS issues in browser

**Solutions**:
```python
# Use a different port
server.run(host="0.0.0.0", port=8080)

# Check if port is in use
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('localhost', 8000))
if result == 0:
    print("Port 8000 is already in use")
```

### Debugging Tips

#### Enable verbose logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# The SDK will now log:
# - CLI discovery process
# - Connection attempts
# - Message parsing
# - Error details
```

#### Check SDK and CLI versions

```python
import claude_code_sdk
print(f"SDK version: {claude_code_sdk.__version__}")
```

```bash
claude --version
```

#### Inspect message flow

```python
async for message in query(prompt="Debug test"):
    print(f"Message type: {type(message).__name__}")
    print(f"Message content: {message}")
    
    if hasattr(message, 'content'):
        for block in message.content:
            print(f"  Block type: {type(block).__name__}")
```

#### Test with minimal configuration

```python
# Start with the simplest possible query
async for message in query(prompt="Hello"):
    print(message)

# If that works, gradually add complexity
options = ClaudeCodeOptions(allowed_tools=["Read"])
async for message in query(prompt="Read README.md", options=options):
    print(message)
```

### Getting Help

If you're still experiencing issues:

1. Check the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code)
2. Search existing [GitHub issues](https://github.com/anthropics/claude-code/issues)
3. Create a new issue with:
   - Python version (`python --version`)
   - SDK version (`pip show claude-code-sdk`)
   - CLI version (`claude --version`)
   - Minimal code to reproduce the issue
   - Full error message and stack trace

## Best Practices

### 1. Resource Management

Always use async context managers or ensure proper cleanup:

```python
# Good: Automatic cleanup
async def process_files():
    async for message in query(prompt="Process files"):
        # SDK handles cleanup automatically
        pass

# For long-running applications
import asyncio

async def main():
    try:
        async for message in query(prompt="Long task"):
            process_message(message)
    except KeyboardInterrupt:
        # Graceful shutdown
        print("Shutting down...")
```

### 2. Error Handling Strategy

Implement comprehensive error handling:

```python
from claude_code_sdk import (
    query, 
    CLINotFoundError,
    ProcessError,
    CLIConnectionError
)

async def safe_query(prompt: str, max_retries: int = 3):
    """Query with automatic retry and error handling."""
    for attempt in range(max_retries):
        try:
            async for message in query(prompt=prompt):
                yield message
            break
        except CLINotFoundError:
            # Can't recover from this
            raise
        except (CLIConnectionError, ProcessError) as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            raise
```

### 3. Tool Usage Optimization

Be specific with tool permissions:

```python
# Good: Only request needed tools
options = ClaudeCodeOptions(
    allowed_tools=["Read", "Write"],  # Specific tools
    disallowed_tools=["Bash"],        # Explicitly disallow
)

# Better: Use appropriate permission mode
options = ClaudeCodeOptions(
    allowed_tools=["Read", "Write", "Edit"],
    permission_mode="acceptEdits",  # Auto-accept safe operations
)

# Avoid: Too permissive
# options = ClaudeCodeOptions(permission_mode="bypassPermissions")
```

### 4. Message Processing Patterns

Process messages efficiently:

```python
from claude_code_sdk import AssistantMessage, TextBlock, ToolUseBlock

async def process_claude_response(prompt: str):
    """Process different message types appropriately."""
    tool_uses = []
    text_responses = []
    
    async for message in query(prompt=prompt):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text_responses.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    tool_uses.append({
                        "tool": block.name,
                        "input": block.input
                    })
    
    return {
        "text": "\n".join(text_responses),
        "tools_used": tool_uses
    }
```

### 5. Session Management

For multi-turn conversations:

```python
# Continue previous session
options = ClaudeCodeOptions(continue_conversation=True)

# Or save and resume specific sessions
session_id = None

async def chat(user_input: str):
    global session_id
    
    options = ClaudeCodeOptions(
        resume=session_id if session_id else None,
        max_turns=10  # Prevent runaway conversations
    )
    
    async for message in query(prompt=user_input, options=options):
        if isinstance(message, ResultMessage):
            session_id = message.session_id
        yield message
```

### 6. Working Directory Best Practices

Always set explicit working directories:

```python
from pathlib import Path

# Good: Explicit, absolute path
options = ClaudeCodeOptions(
    cwd=Path("/home/user/projects/myapp").absolute()
)

# Good: Relative to script location
script_dir = Path(__file__).parent
options = ClaudeCodeOptions(
    cwd=script_dir / "workspace"
)

# Avoid: Implicit current directory
# options = ClaudeCodeOptions()  # Uses wherever script is run from
```

### 7. Performance Optimization

For better performance:

```python
# Batch operations when possible
prompt = """
Please perform these tasks:
1. Read all Python files in the src/ directory
2. Analyze the code structure
3. Generate a summary report
"""

# Instead of multiple queries
# DON'T: Multiple round trips
# for file in files:
#     async for msg in query(f"Read {file}"):
#         ...

# DO: Single comprehensive request
async for message in query(prompt=prompt, options=options):
    # Process all results at once
    pass
```

### 8. Logging and Monitoring

Implement proper logging:

```python
import logging
from claude_code_sdk import ResultMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def monitored_query(prompt: str):
    """Query with monitoring."""
    start_time = time.time()
    
    try:
        async for message in query(prompt=prompt):
            if isinstance(message, ResultMessage):
                logger.info(
                    f"Query completed - "
                    f"Cost: ${message.cost_usd:.4f}, "
                    f"Duration: {message.duration_ms}ms, "
                    f"Session: {message.session_id}"
                )
            yield message
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise
    finally:
        duration = time.time() - start_time
        logger.info(f"Total duration: {duration:.2f}s")
```

### 9. Security Considerations

Never expose sensitive information:

```python
import os

# Good: Use environment variables
options = ClaudeCodeOptions(
    system_prompt="Use the API key from environment"
)

# Set via environment
os.environ["ANTHROPIC_API_KEY"] = "your-key"

# Avoid: Hardcoding secrets
# DON'T: options = ClaudeCodeOptions(
#     system_prompt="Use API key: sk-ant-..."
# )

# Good: Sanitize file paths
safe_path = Path(user_input).resolve()
if not safe_path.is_relative_to(allowed_directory):
    raise ValueError("Access denied")
```

### 10. Testing Your Integration

Write testable code:

```python
# Make your code testable
async def analyze_code(
    file_path: str,
    query_func=query  # Injectable for testing
):
    """Analyze code with injectable query function."""
    prompt = f"Analyze the code in {file_path}"
    
    async for message in query_func(prompt=prompt):
        # Process message
        pass

# In tests
async def mock_query(prompt, options=None):
    """Mock query for testing."""
    yield AssistantMessage(content=[
        TextBlock(text="Mock analysis complete")
    ])

# Test
async def test_analyze():
    await analyze_code("test.py", query_func=mock_query)
```

### 11. Handling Large Operations

For operations that might take a long time:

```python
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def timeout_query(prompt: str, timeout_seconds: int = 300):
    """Query with timeout and cleanup."""
    task = None
    try:
        async with asyncio.timeout(timeout_seconds):
            task = asyncio.create_task(collect_messages(prompt))
            yield task
    except asyncio.TimeoutError:
        logger.warning(f"Query timed out after {timeout_seconds}s")
        raise
    finally:
        if task and not task.done():
            task.cancel()

async def collect_messages(prompt: str):
    """Collect all messages from a query."""
    messages = []
    async for message in query(prompt=prompt):
        messages.append(message)
    return messages
```

### 12. Common Pitfalls to Avoid

1. **Don't use synchronous code in async context**:
   ```python
   # Bad
   time.sleep(1)  # Blocks event loop
   
   # Good
   await asyncio.sleep(1)
   ```

2. **Don't ignore error messages**:
   ```python
   # Bad
   try:
       async for msg in query(prompt="..."):
           pass
   except Exception:
       pass  # Silent failure
   
   # Good
   except Exception as e:
       logger.error(f"Query failed: {e}")
       raise
   ```

3. **Don't leak resources**:
   ```python
   # Bad
   messages = []
   async for msg in query(prompt="..."):
       messages.append(msg)
       if len(messages) > 1000000:  # Memory leak
           break
   
   # Good
   async for msg in query(prompt="..."):
       process_message(msg)  # Process and discard
   ```

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

## Jupyter Notebook Support

The SDK includes special utilities for enhanced display in Jupyter notebooks:

```python
from claude_code_sdk.notebook_utils import display_claude_response, render_markdown

# Beautiful markdown rendering
async for message in query(prompt="Explain Python decorators"):
    display_claude_response(message)

# Or render markdown directly
render_markdown("# Hello\nThis is **bold** text with `code`")
```

Features:
- **Markdown to HTML conversion** with syntax highlighting
- **Tool use visualization** with colored formatting
- **Streaming support** for real-time display
- **Automatic notebook detection**

See [docs/notebook_utilities.md](docs/notebook_utilities.md) for full documentation and [examples/notebook_markdown_demo.ipynb](examples/notebook_markdown_demo.ipynb) for a complete demo.

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