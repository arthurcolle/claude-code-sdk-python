"""WebSocket server wrapper for Claude Code SDK with enhanced capabilities."""

import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, Any, Optional, Set, List
from dataclasses import asdict
from pathlib import Path
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from claude_code_sdk import query, ClaudeCodeOptions
from claude_code_sdk.types import (
    UserMessage,
    AssistantMessage,
    SystemMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock
)

# Add agent_system to path if it exists
agent_system_path = Path(__file__).parent.parent.parent / "agent_system"
if agent_system_path.exists():
    sys.path.insert(0, str(agent_system_path.parent))
    try:
        from agent_system.integrations.tool_registry_client import ToolRegistryClient
        TOOL_REGISTRY_AVAILABLE = True
    except ImportError:
        TOOL_REGISTRY_AVAILABLE = False
else:
    TOOL_REGISTRY_AVAILABLE = False

logger = logging.getLogger(__name__)


class ConversationSession:
    """Manages a conversation session with Claude."""
    
    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.query_task: Optional[asyncio.Task] = None
        self.interrupt_event = asyncio.Event()
        self.is_active = True
        self.message_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.options: Optional[ClaudeCodeOptions] = None


class EnhancedClaudeWebSocketServer:
    """Enhanced WebSocket server for real-time Claude Code interactions."""
    
    def __init__(self, app: Optional[FastAPI] = None):
        self.app = app or FastAPI()
        self._setup_routes()
        self.sessions: Dict[str, ConversationSession] = {}
        self.tool_registry_client: Optional[ToolRegistryClient] = None
        
        # Initialize tool registry client if available
        if TOOL_REGISTRY_AVAILABLE:
            self.tool_registry_client = ToolRegistryClient()
        
    def _setup_routes(self):
        """Set up WebSocket and HTTP routes."""
        
        @self.app.get("/")
        async def get():
            """Serve the HTML UI."""
            ui_path = Path("claude_ui.html")
            if not ui_path.exists():
                # Try relative to the script location
                ui_path = Path(__file__).parent.parent.parent / "claude_ui.html"
            
            if ui_path.exists():
                with open(ui_path, "r") as f:
                    return HTMLResponse(content=f.read())
            else:
                return HTMLResponse(content="<h1>UI file not found</h1>")
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """Handle WebSocket connections."""
            await self._handle_websocket(websocket)
    
    async def _handle_websocket(self, websocket: WebSocket):
        """Handle a WebSocket connection."""
        await websocket.accept()
        session_id = f"session_{id(websocket)}"
        session = ConversationSession(session_id, websocket)
        self.sessions[session_id] = session
        
        # Send connection success with capabilities
        await websocket.send_json({
            "type": "connection_established",
            "data": {
                "session_id": session_id,
                "capabilities": {
                    "concurrent_input": True,
                    "tool_definition": TOOL_REGISTRY_AVAILABLE,
                    "interrupt_query": True
                }
            }
        })
        
        try:
            # Start message processor
            processor_task = asyncio.create_task(self._process_messages(session))
            
            while session.is_active:
                # Receive message from client
                try:
                    data = await websocket.receive_json()
                    await session.message_queue.put(data)
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"Error receiving message: {e}")
                    break
            
            # Cleanup
            session.is_active = False
            processor_task.cancel()
            
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            session.is_active = False
            if session.query_task and not session.query_task.done():
                session.query_task.cancel()
            del self.sessions[session_id]
            try:
                await websocket.close()
            except:
                pass
    
    async def _process_messages(self, session: ConversationSession):
        """Process messages from the client queue."""
        while session.is_active:
            try:
                # Get message with timeout to allow periodic checks
                data = await asyncio.wait_for(session.message_queue.get(), timeout=1.0)
                
                message_type = data.get("type")
                
                if message_type == "query":
                    await self._handle_query(session, data)
                elif message_type == "input":
                    await self._handle_input(session, data)
                elif message_type == "interrupt":
                    await self._handle_interrupt(session)
                elif message_type == "define_tool":
                    await self._handle_define_tool(session, data)
                elif message_type == "get_tools":
                    await self._handle_get_tools(session, data)
                elif message_type == "ping":
                    await session.websocket.send_json({"type": "pong"})
                    
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await session.websocket.send_json({
                    "type": "error",
                    "data": {"error": str(e)}
                })
    
    async def _handle_query(self, session: ConversationSession, data: Dict[str, Any]):
        """Handle a query request from the client."""
        if session.query_task and not session.query_task.done():
            await session.websocket.send_json({
                "type": "error",
                "data": {"error": "A query is already in progress. Use interrupt to stop it."}
            })
            return
        
        prompt = data.get("prompt", "")
        options_data = data.get("options", {})
        
        # Build ClaudeCodeOptions from client data
        session.options = ClaudeCodeOptions(
            allowed_tools=options_data.get("allowed_tools", []),
            permission_mode=options_data.get("permission_mode", "default"),
            max_thinking_tokens=options_data.get("max_thinking_tokens", 8000),
            model=options_data.get("model"),
            cwd=options_data.get("cwd"),
            continue_conversation=options_data.get("continue_conversation", False),
            resume=options_data.get("resume")
        )
        
        # Reset interrupt event
        session.interrupt_event.clear()
        
        # Create query task
        session.query_task = asyncio.create_task(
            self._run_query(session, prompt)
        )
    
    async def _run_query(self, session: ConversationSession, prompt: str):
        """Run a query with Claude."""
        try:
            # Send start message
            await session.websocket.send_json({
                "type": "query_start",
                "data": {"prompt": prompt}
            })
            
            # Stream responses from Claude
            async for message in query(prompt=prompt, options=session.options):
                # Check for interrupt
                if session.interrupt_event.is_set():
                    await session.websocket.send_json({
                        "type": "query_interrupted"
                    })
                    break
                
                await self._send_message(session.websocket, message)
            
            # Send end message
            await session.websocket.send_json({
                "type": "query_end"
            })
            
        except asyncio.CancelledError:
            await session.websocket.send_json({
                "type": "query_cancelled"
            })
        except Exception as e:
            logger.error(f"Query error: {e}")
            await session.websocket.send_json({
                "type": "error",
                "data": {"error": str(e)}
            })
    
    async def _handle_input(self, session: ConversationSession, data: Dict[str, Any]):
        """Handle user input during a query."""
        input_text = data.get("text", "")
        
        # For now, we'll just acknowledge the input
        # In a real implementation, this would be passed to the running query
        await session.websocket.send_json({
            "type": "input_acknowledged",
            "data": {"text": input_text}
        })
        
        # TODO: Implement actual input handling to the running query
        # This would require modifying the SDK to support interactive input
    
    async def _handle_interrupt(self, session: ConversationSession):
        """Handle query interruption."""
        if session.query_task and not session.query_task.done():
            session.interrupt_event.set()
            session.query_task.cancel()
            await session.websocket.send_json({
                "type": "interrupt_acknowledged"
            })
    
    async def _handle_define_tool(self, session: ConversationSession, data: Dict[str, Any]):
        """Handle tool definition request."""
        if not TOOL_REGISTRY_AVAILABLE:
            await session.websocket.send_json({
                "type": "error",
                "data": {"error": "Tool registry not available"}
            })
            return
        
        tool_data = data.get("tool", {})
        agent_id = data.get("agent_id", session.session_id)
        
        try:
            # Create tool in registry
            result = await self.tool_registry_client.create_tool(tool_data, agent_id)
            
            await session.websocket.send_json({
                "type": "tool_defined",
                "data": {
                    "tool_id": result.get("id"),
                    "tool_name": result.get("name"),
                    "status": "success",
                    "details": result
                }
            })
        except Exception as e:
            logger.error(f"Tool definition error: {e}")
            await session.websocket.send_json({
                "type": "error",
                "data": {"error": f"Failed to define tool: {str(e)}"}
            })
    
    async def _handle_get_tools(self, session: ConversationSession, data: Dict[str, Any]):
        """Handle request to get available tools."""
        if not TOOL_REGISTRY_AVAILABLE:
            await session.websocket.send_json({
                "type": "tools_list",
                "data": {"tools": [], "source": "default"}
            })
            return
        
        try:
            # Get tools from registry
            tools = await self.tool_registry_client.get_tools(
                name=data.get("name"),
                limit=data.get("limit", 100)
            )
            
            await session.websocket.send_json({
                "type": "tools_list",
                "data": {"tools": tools, "source": "registry"}
            })
        except Exception as e:
            logger.error(f"Get tools error: {e}")
            await session.websocket.send_json({
                "type": "error",
                "data": {"error": f"Failed to get tools: {str(e)}"}
            })
    
    async def _send_message(self, websocket: WebSocket, message):
        """Send a Claude message to the WebSocket client."""
        if isinstance(message, AssistantMessage):
            # Convert content blocks to serializable format
            content_data = []
            for block in message.content:
                if isinstance(block, TextBlock):
                    content_data.append({
                        "type": "text",
                        "text": block.text
                    })
                elif isinstance(block, ToolUseBlock):
                    content_data.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input
                    })
                elif isinstance(block, ToolResultBlock):
                    content_data.append({
                        "type": "tool_result",
                        "tool_use_id": block.tool_use_id,
                        "content": block.content,
                        "is_error": block.is_error
                    })
            
            await websocket.send_json({
                "type": "assistant_message",
                "data": {"content": content_data}
            })
            
        elif isinstance(message, SystemMessage):
            await websocket.send_json({
                "type": "system_message",
                "data": {
                    "subtype": message.subtype,
                    **message.data
                }
            })
            
        elif isinstance(message, ResultMessage):
            await websocket.send_json({
                "type": "result_message",
                "data": {
                    "subtype": message.subtype,
                    "cost_usd": message.cost_usd,
                    "duration_ms": message.duration_ms,
                    "session_id": message.session_id,
                    "total_cost_usd": message.total_cost_usd,
                    "num_turns": message.num_turns,
                    "usage": message.usage
                }
            })
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.tool_registry_client:
            await self.tool_registry_client.close()
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the WebSocket server."""
        uvicorn.run(self.app, host=host, port=port)


# Keep the old class name for backward compatibility
ClaudeWebSocketServer = EnhancedClaudeWebSocketServer


if __name__ == "__main__":
    server = EnhancedClaudeWebSocketServer()
    try:
        server.run()
    finally:
        asyncio.run(server.cleanup())