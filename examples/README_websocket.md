# WebSocket UI for Claude Code SDK

This example demonstrates a real-time web interface for interacting with Claude using WebSockets.

## Features

- **Real-time streaming**: See Claude's responses as they're generated
- **Tool execution visualization**: Watch as Claude uses tools like Read, Write, Edit, and Bash
- **Configurable options**: Control model selection, permission modes, and allowed tools
- **Cost tracking**: Monitor usage and costs per query
- **Modern UI**: Dark theme with responsive design

## Installation

Install the WebSocket dependencies:

```bash
pip install -e ".[websocket]"
```

## Usage

1. Run the server:
   ```bash
   python examples/websocket_ui_server.py
   ```

2. Open your browser to http://localhost:8000

3. Configure options in the sidebar:
   - Select model (Default, Opus 4, Sonnet 3.5)
   - Choose permission mode
   - Enable/disable specific tools
   - Set working directory

4. Type your prompt and press Enter or click Send

## How It Works

The WebSocket server (`src/claude_code_sdk/websocket_server.py`) wraps the Claude Code SDK to provide:

- FastAPI server with WebSocket endpoint
- Real-time message streaming from Claude
- Automatic message type conversion for web display
- Session management and cost tracking

The HTML UI (`claude_ui.html`) provides:

- WebSocket client with automatic reconnection
- Message rendering with markdown support
- Tool execution visualization
- Real-time status updates

## Message Types

The WebSocket protocol uses JSON messages:

### Client to Server:
```json
{
  "type": "query",
  "prompt": "Your question here",
  "options": {
    "allowed_tools": ["Read", "Write", "Edit"],
    "permission_mode": "acceptEdits",
    "model": "claude-opus-4-20250514"
  }
}
```

### Server to Client:
- `query_start`: Query processing begun
- `assistant_message`: Claude's response with text/tool blocks
- `system_message`: System notifications
- `result_message`: Final results with cost/usage info
- `query_end`: Query completed
- `error`: Error occurred

## Customization

The UI can be customized by modifying:
- `claude_ui.html`: UI layout and styling
- `websocket_server.py`: Server behavior and message handling

## Security Notes

- The example server runs without authentication
- For production use, add proper authentication and HTTPS
- Consider rate limiting and input validation