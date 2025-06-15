"""Tests for the WebSocket server functionality."""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from websockets.client import connect as websocket_connect
import httpx

from claude_code_sdk.websocket_server import EnhancedClaudeWebSocketServer, ConversationSession
from claude_code_sdk.types import AssistantMessage, TextBlock, ToolUseBlock, ResultMessage


@pytest.fixture
def websocket_server():
    """Create a WebSocket server instance for testing."""
    server = EnhancedClaudeWebSocketServer()
    return server


@pytest.fixture
def test_client(websocket_server):
    """Create a test client for the FastAPI app."""
    return TestClient(websocket_server.app)


class TestWebSocketServer:
    """Test the WebSocket server functionality."""
    
    def test_server_initialization(self, websocket_server):
        """Test server initializes correctly."""
        assert websocket_server.app is not None
        assert websocket_server.sessions == {}
        assert hasattr(websocket_server, 'tool_registry_client')
    
    def test_health_endpoint(self, test_client):
        """Test the health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "active_sessions" in data
        assert "max_sessions" in data
    
    def test_sessions_endpoint(self, test_client):
        """Test the sessions listing endpoint."""
        response = test_client.get("/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)
    
    def test_root_endpoint(self, test_client):
        """Test the root endpoint serves HTML."""
        response = test_client.get("/")
        assert response.status_code == 200
        # Should return HTML content or error message
        assert response.headers["content-type"].startswith("text/html")


@pytest.mark.asyncio
class TestWebSocketConnection:
    """Test WebSocket connection handling."""
    
    async def test_websocket_connection_established(self, websocket_server):
        """Test WebSocket connection establishment."""
        # Mock WebSocket
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.receive_json = AsyncMock(side_effect=Exception("Test disconnect"))
        mock_ws.close = AsyncMock()
        
        # Handle connection
        await websocket_server._handle_websocket(mock_ws)
        
        # Verify connection was accepted
        mock_ws.accept.assert_called_once()
        
        # Verify welcome message was sent
        mock_ws.send_json.assert_called()
        call_args = mock_ws.send_json.call_args[0][0]
        assert call_args["type"] == "connection_established"
        assert "session_id" in call_args["data"]
        assert "capabilities" in call_args["data"]
    
    async def test_message_processing(self, websocket_server):
        """Test message processing functionality."""
        session = ConversationSession("test_session", AsyncMock())
        
        # Test ping message
        await session.message_queue.put({"type": "ping"})
        
        # Process message
        process_task = asyncio.create_task(
            websocket_server._process_messages(session)
        )
        
        # Wait briefly for processing
        await asyncio.sleep(0.1)
        
        # Cancel the task
        session.is_active = False
        process_task.cancel()
        
        # Verify pong was sent
        session.websocket.send_json.assert_called_with({"type": "pong"})
    
    @patch('claude_code_sdk.query')
    async def test_query_handling(self, mock_query, websocket_server):
        """Test query message handling."""
        # Mock the query function
        async def mock_query_generator(*args, **kwargs):
            yield AssistantMessage(content=[
                TextBlock(text="Test response")
            ])
            yield ResultMessage(
                subtype="success",
                cost_usd=0.01,
                duration_ms=1000,
                session_id="test_session",
                total_cost_usd=0.01,
                num_turns=1,
                usage={}
            )
        
        mock_query.return_value = mock_query_generator()
        
        # Create session
        mock_ws = AsyncMock()
        session = ConversationSession("test_session", mock_ws)
        
        # Handle query
        await websocket_server._handle_query(session, {
            "prompt": "Test query",
            "options": {
                "allowed_tools": ["Read", "Write"],
                "permission_mode": "default"
            }
        })
        
        # Wait for query to complete
        if session.query_task:
            await session.query_task
        
        # Verify messages were sent
        assert mock_ws.send_json.call_count >= 3  # start, assistant, result, end
        
        # Check message types
        calls = mock_ws.send_json.call_args_list
        message_types = [call[0][0]["type"] for call in calls]
        assert "query_start" in message_types
        assert "assistant_message" in message_types
        assert "result_message" in message_types
        assert "query_end" in message_types
    
    async def test_interrupt_handling(self, websocket_server):
        """Test query interruption."""
        mock_ws = AsyncMock()
        session = ConversationSession("test_session", mock_ws)
        
        # Create a long-running query task
        async def long_query():
            await asyncio.sleep(10)
        
        session.query_task = asyncio.create_task(long_query())
        
        # Handle interrupt
        await websocket_server._handle_interrupt(session)
        
        # Verify task was cancelled
        assert session.query_task.cancelled()
        assert session.interrupt_event.is_set()
        
        # Verify acknowledgment was sent
        mock_ws.send_json.assert_called_with({"type": "interrupt_acknowledged"})
    
    async def test_tool_definition_without_registry(self, websocket_server):
        """Test tool definition when registry is not available."""
        websocket_server.tool_registry_client = None
        
        mock_ws = AsyncMock()
        session = ConversationSession("test_session", mock_ws)
        
        await websocket_server._handle_define_tool(session, {
            "tool": {"name": "TestTool"}
        })
        
        # Should send error
        mock_ws.send_json.assert_called_with({
            "type": "error",
            "data": {"error": "Tool registry not available"}
        })
    
    async def test_concurrent_sessions(self, websocket_server):
        """Test handling multiple concurrent sessions."""
        sessions = []
        
        # Create multiple sessions
        for i in range(3):
            mock_ws = AsyncMock()
            mock_ws.accept = AsyncMock()
            mock_ws.send_json = AsyncMock()
            mock_ws.receive_json = AsyncMock(side_effect=Exception("Disconnect"))
            mock_ws.close = AsyncMock()
            
            # Handle connection in background
            task = asyncio.create_task(
                websocket_server._handle_websocket(mock_ws)
            )
            sessions.append((mock_ws, task))
            
            # Brief delay to ensure connection is established
            await asyncio.sleep(0.1)
        
        # Should have 3 active sessions
        assert len(websocket_server.sessions) == 3
        
        # Clean up tasks
        for ws, task in sessions:
            task.cancel()
            try:
                await task
            except:
                pass
    
    async def test_message_serialization(self, websocket_server):
        """Test message serialization functionality."""
        # Test AssistantMessage
        assistant_msg = AssistantMessage(content=[
            TextBlock(text="Hello"),
            ToolUseBlock(id="123", name="Read", input={"file": "test.py"}),
            ToolResultBlock(tool_use_id="123", content="File content", is_error=False)
        ])
        
        serialized = websocket_server._serialize_message(assistant_msg)
        assert "content" in serialized
        assert len(serialized["content"]) == 3
        assert serialized["content"][0]["type"] == "text"
        assert serialized["content"][1]["type"] == "tool_use"
        assert serialized["content"][2]["type"] == "tool_result"
        
        # Test ResultMessage
        result_msg = ResultMessage(
            subtype="success",
            cost_usd=0.05,
            duration_ms=2000,
            session_id="test",
            total_cost_usd=0.10,
            num_turns=2,
            usage={"input_tokens": 100, "output_tokens": 200}
        )
        
        serialized = websocket_server._serialize_message(result_msg)
        assert serialized["subtype"] == "success"
        assert serialized["cost_usd"] == 0.05
        assert serialized["duration_ms"] == 2000
        assert serialized["usage"]["input_tokens"] == 100


@pytest.mark.asyncio
class TestWebSocketIntegration:
    """Integration tests for WebSocket functionality."""
    
    @pytest.mark.skipif(
        not hasattr(pytest, "live_server"),
        reason="Requires live server fixture"
    )
    async def test_full_websocket_flow(self, websocket_server):
        """Test complete WebSocket communication flow."""
        # This would require a running server instance
        # Placeholder for integration test
        pass


def test_conversation_session():
    """Test ConversationSession class."""
    mock_ws = MagicMock()
    session = ConversationSession("test_id", mock_ws)
    
    assert session.session_id == "test_id"
    assert session.websocket == mock_ws
    assert session.query_task is None
    assert session.is_active is True
    assert session.options is None
    assert hasattr(session, 'interrupt_event')
    assert hasattr(session, 'message_queue')