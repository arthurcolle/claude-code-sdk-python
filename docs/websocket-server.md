# WebSocket Server Documentation

The Claude Code SDK includes an enhanced WebSocket server that provides real-time, bidirectional communication with Claude. This enables building interactive applications with features like streaming responses, query interruption, and concurrent input handling.

## Features

- **Real-time Streaming**: Receive Claude's responses as they're generated
- **Query Interruption**: Cancel ongoing queries with immediate effect
- **Concurrent Input**: Send additional input while Claude is processing (server capability dependent)
- **Session Management**: Maintain conversation context across multiple queries
- **Tool Integration**: Full support for Claude's tool use capabilities
- **Multi-client Support**: Handle multiple concurrent WebSocket connections
- **Graceful Error Handling**: Comprehensive error reporting and recovery

## Installation

The WebSocket server is included with the SDK but requires additional dependencies:

```bash
pip install claude-code-sdk[websocket]
```

Or install the required dependencies manually:

```bash
pip install fastapi uvicorn websockets
```

## Quick Start

### 1. Start the Server

```python
from claude_code_sdk.websocket_server import EnhancedClaudeWebSocketServer

# Create and run the server
server = EnhancedClaudeWebSocketServer()
server.run(host="0.0.0.0", port=8000)
```

### 2. Connect via WebSocket

```javascript
// JavaScript client example
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
    console.log('Connected to Claude WebSocket Server');
};

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    console.log('Received:', message);
};

// Send a query
ws.send(JSON.stringify({
    type: 'query',
    prompt: 'Hello, Claude!',
    options: {
        allowed_tools: ['Read', 'Write'],
        max_thinking_tokens: 8000
    }
}));
```

### 3. Python Client Example

```python
import asyncio
import websockets
import json

async def claude_client():
    async with websockets.connect('ws://localhost:8000/ws') as websocket:
        # Connection established message
        message = await websocket.recv()
        data = json.loads(message)
        print(f"Connected: {data}")
        
        # Send a query
        await websocket.send(json.dumps({
            'type': 'query',
            'prompt': 'Write a Python function to calculate factorial',
            'options': {
                'allowed_tools': ['Write'],
                'permission_mode': 'acceptEdits'
            }
        }))
        
        # Receive responses
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            if data['type'] == 'query_end':
                break
                
            print(f"Message: {data}")

asyncio.run(claude_client())
```

## Message Protocol

### Client to Server Messages

#### Query Message
```json
{
    "type": "query",
    "prompt": "Your prompt here",
    "options": {
        "allowed_tools": ["Read", "Write", "Edit"],
        "permission_mode": "default",
        "max_thinking_tokens": 8000,
        "model": "claude-3-opus-20240229",
        "cwd": "/path/to/working/directory"
    }
}
```

#### Interrupt Message
```json
{
    "type": "interrupt"
}
```

#### Ping Message
```json
{
    "type": "ping"
}
```

### Server to Client Messages

#### Connection Established
```json
{
    "type": "connection_established",
    "data": {
        "session_id": "session_12345",
        "capabilities": {
            "concurrent_input": true,
            "tool_definition": true,
            "interrupt_query": true
        }
    }
}
```

#### Assistant Message
```json
{
    "type": "assistant_message",
    "data": {
        "content": [
            {
                "type": "text",
                "text": "Here's the factorial function:"
            },
            {
                "type": "tool_use",
                "id": "tool_use_123",
                "name": "Write",
                "input": {
                    "file_path": "factorial.py",
                    "content": "def factorial(n):..."
                }
            }
        ]
    }
}
```

#### Tool Result
```json
{
    "type": "assistant_message",
    "data": {
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "tool_use_123",
                "content": "File created successfully",
                "is_error": false
            }
        ]
    }
}
```

#### Result Message
```json
{
    "type": "result_message",
    "data": {
        "subtype": "success",
        "cost_usd": 0.0125,
        "duration_ms": 3500,
        "session_id": "session_12345",
        "total_cost_usd": 0.0250,
        "num_turns": 2,
        "usage": {
            "input_tokens": 150,
            "output_tokens": 350
        }
    }
}
```

## Advanced Features

### Session Management

Sessions are automatically created and managed by the server. Each WebSocket connection gets a unique session ID that can be used for:

- Tracking conversation history
- Maintaining context across queries
- Implementing rate limiting
- Analytics and monitoring

### Query Interruption

When server capabilities include `interrupt_query`, you can cancel ongoing queries:

```javascript
// Start a long-running query
ws.send(JSON.stringify({
    type: 'query',
    prompt: 'Generate a complete web application...'
}));

// Interrupt it
ws.send(JSON.stringify({
    type: 'interrupt'
}));
```

### Concurrent Input (Future Feature)

When supported by the server, you can send additional input while a query is processing:

```javascript
// During an active query
ws.send(JSON.stringify({
    type: 'input',
    text: 'Actually, make it use TypeScript instead'
}));
```

### Custom Tool Definition

If the server has tool registry integration enabled:

```javascript
ws.send(JSON.stringify({
    type: 'define_tool',
    tool: {
        name: 'CustomFormatter',
        description: 'Formats text in a custom way',
        parameters: {
            text: { type: 'string', required: true },
            style: { type: 'string', enum: ['bold', 'italic'] }
        }
    }
}));
```

## Server Configuration

### Environment Variables

```bash
# WebSocket server configuration
WEBSOCKET_HOST=0.0.0.0
WEBSOCKET_PORT=8000

# Tool registry integration (optional)
TOOL_REGISTRY_URL=http://localhost:2016

# Session management
MAX_SESSIONS=100
SESSION_TIMEOUT_MINUTES=30
```

### Programmatic Configuration

```python
from claude_code_sdk.websocket_server import EnhancedClaudeWebSocketServer

server = EnhancedClaudeWebSocketServer(
    enable_cors=True,
    max_sessions=50,
    session_timeout_minutes=60
)

# Add custom middleware
@server.app.middleware("http")
async def add_custom_header(request, call_next):
    response = await call_next(request)
    response.headers["X-Custom-Header"] = "value"
    return response

server.run(host="0.0.0.0", port=8000, log_level="info")
```

## Integration with Frontend Frameworks

### React Example

```jsx
import { useEffect, useState } from 'react';

function ClaudeChat() {
    const [ws, setWs] = useState(null);
    const [messages, setMessages] = useState([]);
    const [connected, setConnected] = useState(false);

    useEffect(() => {
        const websocket = new WebSocket('ws://localhost:8000/ws');
        
        websocket.onopen = () => {
            setConnected(true);
        };
        
        websocket.onmessage = (event) => {
            const message = JSON.parse(event.data);
            if (message.type === 'assistant_message') {
                setMessages(prev => [...prev, message]);
            }
        };
        
        websocket.onclose = () => {
            setConnected(false);
        };
        
        setWs(websocket);
        
        return () => {
            websocket.close();
        };
    }, []);

    const sendQuery = (prompt) => {
        if (ws && connected) {
            ws.send(JSON.stringify({
                type: 'query',
                prompt: prompt,
                options: {
                    allowed_tools: ['Read', 'Write']
                }
            }));
        }
    };

    // Render UI...
}
```

### Vue.js Example

```vue
<script setup>
import { ref, onMounted, onUnmounted } from 'vue';

const ws = ref(null);
const messages = ref([]);
const connected = ref(false);

onMounted(() => {
    ws.value = new WebSocket('ws://localhost:8000/ws');
    
    ws.value.onopen = () => {
        connected.value = true;
    };
    
    ws.value.onmessage = (event) => {
        const message = JSON.parse(event.data);
        messages.value.push(message);
    };
});

onUnmounted(() => {
    if (ws.value) {
        ws.value.close();
    }
});

const sendQuery = (prompt) => {
    if (ws.value && connected.value) {
        ws.value.send(JSON.stringify({
            type: 'query',
            prompt: prompt
        }));
    }
};
</script>
```

## Error Handling

The WebSocket server provides comprehensive error handling:

```javascript
ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    
    if (message.type === 'error') {
        console.error('Server error:', message.data.error);
        // Handle error appropriately
    }
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onclose = (event) => {
    if (event.code !== 1000) {
        console.error('Abnormal closure:', event.code, event.reason);
    }
};
```

## Performance Considerations

### Connection Pooling

For high-traffic applications, consider implementing connection pooling:

```python
class WebSocketPool:
    def __init__(self, url, pool_size=10):
        self.url = url
        self.pool_size = pool_size
        self.connections = []
        self.available = asyncio.Queue()
        
    async def get_connection(self):
        if self.available.empty() and len(self.connections) < self.pool_size:
            ws = await websockets.connect(self.url)
            self.connections.append(ws)
            return ws
        return await self.available.get()
        
    async def release_connection(self, ws):
        await self.available.put(ws)
```

### Message Batching

For multiple rapid queries, consider batching:

```javascript
class MessageBatcher {
    constructor(ws, batchSize = 5, batchDelay = 100) {
        this.ws = ws;
        this.batchSize = batchSize;
        this.batchDelay = batchDelay;
        this.queue = [];
        this.timer = null;
    }
    
    send(message) {
        this.queue.push(message);
        
        if (this.queue.length >= this.batchSize) {
            this.flush();
        } else if (!this.timer) {
            this.timer = setTimeout(() => this.flush(), this.batchDelay);
        }
    }
    
    flush() {
        if (this.queue.length > 0) {
            this.ws.send(JSON.stringify({
                type: 'batch',
                messages: this.queue
            }));
            this.queue = [];
        }
        
        if (this.timer) {
            clearTimeout(this.timer);
            this.timer = null;
        }
    }
}
```

## Security Considerations

### Authentication

Implement authentication for production use:

```python
from fastapi import WebSocket, Query, HTTPException
from jose import jwt

@server.app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(...)
):
    try:
        # Verify JWT token
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("sub")
    except Exception:
        await websocket.close(code=1008, reason="Invalid authentication")
        return
    
    # Continue with authenticated connection
    await server._handle_websocket(websocket, user_id=user_id)
```

### Rate Limiting

Implement rate limiting to prevent abuse:

```python
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import WebSocketRateLimiter

@server.app.websocket("/ws")
@WebSocketRateLimiter(times=10, seconds=60)
async def websocket_endpoint(websocket: WebSocket):
    await server._handle_websocket(websocket)
```

## Monitoring and Debugging

### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Or configure specifically for the WebSocket server
logger = logging.getLogger("claude_code_sdk.websocket_server")
logger.setLevel(logging.DEBUG)
```

### Health Monitoring

The server provides health endpoints:

```bash
# Check server health
curl http://localhost:8000/health

# Get active sessions
curl http://localhost:8000/sessions
```

### Custom Metrics

Add custom metrics collection:

```python
from prometheus_client import Counter, Histogram, make_asgi_app

# Define metrics
query_counter = Counter('claude_queries_total', 'Total queries processed')
query_duration = Histogram('claude_query_duration_seconds', 'Query duration')

# Add metrics endpoint
metrics_app = make_asgi_app()
server.app.mount("/metrics", metrics_app)
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Ensure the server is running
   - Check firewall settings
   - Verify the correct host and port

2. **Messages Not Received**
   - Check WebSocket connection state
   - Verify message format
   - Enable debug logging

3. **Session Timeout**
   - Implement heartbeat/ping mechanism
   - Adjust session timeout settings
   - Handle reconnection gracefully

### Debug Mode

Enable detailed debugging:

```python
server = EnhancedClaudeWebSocketServer()
server.app.debug = True
server.run(host="0.0.0.0", port=8000, log_level="debug")
```

## Examples

Full examples are available in the `examples/` directory:

- `websocket_ui_server.py` - Complete UI server example
- `websocket_client.py` - Python client implementation
- `websocket_react_app/` - React application example

## API Reference

See the API documentation for detailed method and class references.