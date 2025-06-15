"""
WebSocket OAuth Authentication Example

This example demonstrates how to integrate OAuth authentication with the Claude Code SDK
websocket server. It shows how to handle OAuth flow in a websocket environment.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import webbrowser
from urllib.parse import urlparse, parse_qs

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, RedirectResponse
import uvicorn

from claude_code_sdk.auth_config import ClaudeCodeOAuthConfig
from claude_code_sdk.oauth_flow import ClaudeCodeOAuthFlow
from claude_code_sdk.auth import TokenStorage, AuthToken
from claude_code_sdk.websocket_server import EnhancedClaudeWebSocketServer

logger = logging.getLogger(__name__)


class WebSocketOAuthSession:
    """OAuth session for WebSocket connections."""
    
    def __init__(self, websocket: WebSocket, session_id: str):
        self.websocket = websocket
        self.session_id = session_id
        self.oauth_flow: Optional[ClaudeCodeOAuthFlow] = None
        self.auth_token: Optional[AuthToken] = None
        self.oauth_state: Optional[str] = None
        self.is_authenticated = False


class ClaudeWebSocketOAuthServer(EnhancedClaudeWebSocketServer):
    """Enhanced WebSocket server with OAuth authentication support."""
    
    def __init__(self, app: Optional[FastAPI] = None):
        super().__init__(app)
        self.oauth_sessions: Dict[str, WebSocketOAuthSession] = {}
        self.oauth_config = ClaudeCodeOAuthConfig.for_claude_code_max()
        self.token_storage = TokenStorage()
        self._setup_oauth_routes()
    
    def _setup_oauth_routes(self):
        """Set up OAuth-specific routes."""
        
        @self.app.get("/oauth/login/{session_id}")
        async def oauth_login(session_id: str):
            """Initiate OAuth login for a WebSocket session."""
            if session_id not in self.oauth_sessions:
                return HTMLResponse(
                    content="<h1>Invalid session</h1>",
                    status_code=400
                )
            
            oauth_session = self.oauth_sessions[session_id]
            
            # Create OAuth flow
            oauth_session.oauth_flow = ClaudeCodeOAuthFlow(
                self.oauth_config,
                self.token_storage
            )
            
            async with oauth_session.oauth_flow as flow:
                # Start callback server
                await flow.start_callback_server()
                
                # Generate auth URL
                auth_url = self.oauth_config.get_authorize_url(
                    state=flow.session.state if flow.session else None,
                    code_challenge=flow.session.challenge if flow.session and self.oauth_config.use_pkce else None
                )
                
                # Store OAuth state for later verification
                oauth_session.oauth_state = flow.session.state if flow.session else None
                
                # Notify WebSocket client
                await oauth_session.websocket.send_json({
                    "type": "oauth_login_initiated",
                    "data": {
                        "auth_url": auth_url,
                        "message": "Please complete authentication in the opened browser window"
                    }
                })
                
                return RedirectResponse(url=auth_url)
        
        @self.app.get("/oauth/callback")
        async def oauth_callback(request: Request):
            """Handle OAuth callback."""
            # Extract parameters
            code = request.query_params.get("code")
            state = request.query_params.get("state")
            error = request.query_params.get("error")
            
            if error:
                return HTMLResponse(
                    content=f"<h1>OAuth Error</h1><p>{error}</p>",
                    status_code=400
                )
            
            # Find session by state
            oauth_session = None
            for session in self.oauth_sessions.values():
                if session.oauth_state == state:
                    oauth_session = session
                    break
            
            if not oauth_session:
                return HTMLResponse(
                    content="<h1>Invalid OAuth state</h1>",
                    status_code=400
                )
            
            try:
                # Complete OAuth flow
                if oauth_session.oauth_flow:
                    async with oauth_session.oauth_flow as flow:
                        # Wait for callback to be processed
                        token = await flow._wait_for_callback()
                        
                        # Exchange code for token
                        auth_token = await flow._exchange_code_for_token(token)
                        
                        # Store token
                        oauth_session.auth_token = auth_token
                        oauth_session.is_authenticated = True
                        
                        # Save to storage
                        self.token_storage.save_token(auth_token)
                        
                        # Notify WebSocket client
                        await oauth_session.websocket.send_json({
                            "type": "oauth_success",
                            "data": {
                                "message": "Authentication successful!",
                                "token_type": auth_token.token_type,
                                "expires_at": auth_token.expires_at.isoformat() if auth_token.expires_at else None,
                                "scopes": auth_token.scope
                            }
                        })
                        
                        return HTMLResponse(content=self._generate_success_page())
                
            except Exception as e:
                logger.error(f"OAuth callback error: {e}")
                
                # Notify WebSocket client of error
                await oauth_session.websocket.send_json({
                    "type": "oauth_error",
                    "data": {
                        "error": str(e),
                        "message": "Authentication failed"
                    }
                })
                
                return HTMLResponse(
                    content=f"<h1>Authentication Failed</h1><p>{str(e)}</p>",
                    status_code=400
                )
        
        @self.app.get("/oauth/status/{session_id}")
        async def oauth_status(session_id: str):
            """Get OAuth status for a session."""
            if session_id not in self.oauth_sessions:
                return {"authenticated": False, "error": "Invalid session"}
            
            oauth_session = self.oauth_sessions[session_id]
            return {
                "authenticated": oauth_session.is_authenticated,
                "token_expires": oauth_session.auth_token.expires_at.isoformat() if oauth_session.auth_token and oauth_session.auth_token.expires_at else None
            }
    
    def _generate_success_page(self) -> str:
        """Generate OAuth success page."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Authentication Successful</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background-color: #f5f5f5;
                }
                .container {
                    text-align: center;
                    padding: 40px;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    max-width: 400px;
                }
                .success { color: #22c55e; }
                .spinner {
                    border: 3px solid #f3f3f3;
                    border-top: 3px solid #3b82f6;
                    border-radius: 50%;
                    width: 30px;
                    height: 30px;
                    animation: spin 1s linear infinite;
                    margin: 20px auto;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
            <script>
                // Auto-close after 3 seconds
                setTimeout(() => {
                    window.close();
                }, 3000);
            </script>
        </head>
        <body>
            <div class="container">
                <div style="font-size: 48px; margin-bottom: 20px;">🎉</div>
                <h1>Authentication Successful!</h1>
                <p class="success">You can now use Claude Code with OAuth authentication.</p>
                <div class="spinner"></div>
                <p>This window will close automatically...</p>
            </div>
        </body>
        </html>
        """
    
    async def _handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection with OAuth support."""
        await websocket.accept()
        session_id = f"oauth_session_{id(websocket)}"
        
        # Create OAuth session
        oauth_session = WebSocketOAuthSession(websocket, session_id)
        self.oauth_sessions[session_id] = oauth_session
        
        # Check for existing valid token
        existing_token = self.token_storage.load_token()
        if existing_token and not existing_token.is_expired():
            oauth_session.auth_token = existing_token
            oauth_session.is_authenticated = True
        
        # Send connection info with OAuth capabilities
        await websocket.send_json({
            "type": "connection_established",
            "data": {
                "session_id": session_id,
                "oauth_enabled": True,
                "authenticated": oauth_session.is_authenticated,
                "oauth_login_url": f"/oauth/login/{session_id}",
                "capabilities": {
                    "oauth_authentication": True,
                    "token_refresh": True,
                    "concurrent_input": True,
                    "interrupt_query": True
                }
            }
        })
        
        try:
            # Handle messages
            while True:
                try:
                    data = await websocket.receive_json()
                    await self._process_oauth_message(oauth_session, data)
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "data": {"error": str(e)}
                    })
        
        finally:
            # Cleanup
            if session_id in self.oauth_sessions:
                del self.oauth_sessions[session_id]
            if oauth_session.oauth_flow:
                await oauth_session.oauth_flow.cleanup()
    
    async def _process_oauth_message(self, oauth_session: WebSocketOAuthSession, data: Dict[str, Any]):
        """Process messages with OAuth context."""
        message_type = data.get("type")
        
        if message_type == "oauth_login":
            # Initiate OAuth login
            login_url = f"http://localhost:8000/oauth/login/{oauth_session.session_id}"
            await oauth_session.websocket.send_json({
                "type": "oauth_login_url",
                "data": {
                    "url": login_url,
                    "message": "Click the link or visit the URL to authenticate"
                }
            })
            
            # Optionally open browser automatically
            if data.get("auto_open", True):
                webbrowser.open(login_url)
        
        elif message_type == "oauth_logout":
            # Logout and revoke token
            if oauth_session.auth_token and oauth_session.oauth_flow:
                try:
                    async with oauth_session.oauth_flow as flow:
                        await flow.revoke_token(oauth_session.auth_token.access_token)
                except Exception as e:
                    logger.error(f"Token revocation error: {e}")
            
            # Clear token
            oauth_session.auth_token = None
            oauth_session.is_authenticated = False
            self.token_storage.clear_token()
            
            await oauth_session.websocket.send_json({
                "type": "oauth_logout_success",
                "data": {"message": "Logged out successfully"}
            })
        
        elif message_type == "oauth_refresh":
            # Refresh token
            if oauth_session.auth_token and oauth_session.auth_token.refresh_token:
                try:
                    if not oauth_session.oauth_flow:
                        oauth_session.oauth_flow = ClaudeCodeOAuthFlow(
                            self.oauth_config,
                            self.token_storage
                        )
                    
                    async with oauth_session.oauth_flow as flow:
                        new_token = await flow.refresh_token(oauth_session.auth_token.refresh_token)
                        oauth_session.auth_token = new_token
                        
                        await oauth_session.websocket.send_json({
                            "type": "oauth_refresh_success",
                            "data": {
                                "message": "Token refreshed successfully",
                                "expires_at": new_token.expires_at.isoformat() if new_token.expires_at else None
                            }
                        })
                
                except Exception as e:
                    logger.error(f"Token refresh error: {e}")
                    await oauth_session.websocket.send_json({
                        "type": "oauth_refresh_error",
                        "data": {"error": str(e)}
                    })
        
        elif message_type == "oauth_status":
            # Get authentication status
            await oauth_session.websocket.send_json({
                "type": "oauth_status",
                "data": {
                    "authenticated": oauth_session.is_authenticated,
                    "token_expires": oauth_session.auth_token.expires_at.isoformat() if oauth_session.auth_token and oauth_session.auth_token.expires_at else None,
                    "token_type": oauth_session.auth_token.token_type if oauth_session.auth_token else None,
                    "scopes": oauth_session.auth_token.scope if oauth_session.auth_token else None
                }
            })
        
        elif message_type == "query":
            # Handle query with OAuth authentication
            if not oauth_session.is_authenticated:
                await oauth_session.websocket.send_json({
                    "type": "auth_required",
                    "data": {
                        "message": "Authentication required to make queries",
                        "oauth_login_url": f"/oauth/login/{oauth_session.session_id}"
                    }
                })
                return
            
            # Check token expiry
            if oauth_session.auth_token and oauth_session.auth_token.is_expired():
                if oauth_session.auth_token.refresh_token:
                    # Try to refresh automatically
                    try:
                        if not oauth_session.oauth_flow:
                            oauth_session.oauth_flow = ClaudeCodeOAuthFlow(
                                self.oauth_config,
                                self.token_storage
                            )
                        
                        async with oauth_session.oauth_flow as flow:
                            oauth_session.auth_token = await flow.refresh_token(
                                oauth_session.auth_token.refresh_token
                            )
                    except Exception as e:
                        logger.error(f"Auto-refresh failed: {e}")
                        await oauth_session.websocket.send_json({
                            "type": "auth_expired",
                            "data": {
                                "message": "Authentication expired, please login again",
                                "oauth_login_url": f"/oauth/login/{oauth_session.session_id}"
                            }
                        })
                        return
                else:
                    await oauth_session.websocket.send_json({
                        "type": "auth_expired",
                        "data": {
                            "message": "Authentication expired, please login again",
                            "oauth_login_url": f"/oauth/login/{oauth_session.session_id}"
                        }
                    })
                    return
            
            # Process query with authentication
            # Add auth token to the query options
            options_data = data.get("options", {})
            
            # Note: This would be where you'd integrate the auth token with the Claude Code SDK
            # The SDK doesn't currently support OAuth tokens, but this shows the structure
            
            # For now, proceed with regular query handling
            prompt = data.get("prompt", "")
            
            await oauth_session.websocket.send_json({
                "type": "query_authenticated",
                "data": {
                    "message": f"Processing query with OAuth authentication: {prompt[:50]}...",
                    "token_valid": not oauth_session.auth_token.is_expired() if oauth_session.auth_token else False
                }
            })
        
        else:
            # Handle other message types (ping, etc.)
            if message_type == "ping":
                await oauth_session.websocket.send_json({"type": "pong"})


def create_oauth_ui_html() -> str:
    """Create HTML UI with OAuth support."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Claude Code WebSocket with OAuth</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .auth-section { border: 1px solid #ddd; padding: 20px; margin-bottom: 20px; border-radius: 5px; }
            .authenticated { background-color: #d4fed4; }
            .not-authenticated { background-color: #fed4d4; }
            button { padding: 10px 15px; margin: 5px; cursor: pointer; }
            #messages { height: 400px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; }
            .message { margin: 5px 0; padding: 5px; border-radius: 3px; }
            .oauth { background-color: #e6f3ff; }
            .query { background-color: #f0f0f0; }
            .error { background-color: #ffe6e6; }
        </style>
    </head>
    <body>
        <h1>Claude Code WebSocket with OAuth</h1>
        
        <div id="auth-section" class="auth-section not-authenticated">
            <h3>Authentication Status</h3>
            <p id="auth-status">Not authenticated</p>
            <button onclick="login()">Login with OAuth</button>
            <button onclick="logout()">Logout</button>
            <button onclick="refreshToken()">Refresh Token</button>
            <button onclick="checkStatus()">Check Status</button>
        </div>
        
        <div>
            <h3>Query Claude</h3>
            <input type="text" id="queryInput" placeholder="Enter your query..." style="width: 70%;">
            <button onclick="sendQuery()">Send Query</button>
        </div>
        
        <div id="messages"></div>
        
        <script>
            let ws = null;
            let sessionId = null;
            let authenticated = false;
            
            function connect() {
                ws = new WebSocket('ws://localhost:8000/ws');
                
                ws.onopen = function() {
                    addMessage('Connected to WebSocket', 'system');
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    handleMessage(data);
                };
                
                ws.onclose = function() {
                    addMessage('WebSocket connection closed', 'system');
                };
                
                ws.onerror = function(error) {
                    addMessage('WebSocket error: ' + error, 'error');
                };
            }
            
            function handleMessage(data) {
                const type = data.type;
                
                if (type === 'connection_established') {
                    sessionId = data.data.session_id;
                    authenticated = data.data.authenticated;
                    updateAuthStatus();
                    addMessage('Connection established. Session: ' + sessionId, 'system');
                    
                } else if (type === 'oauth_login_url') {
                    addMessage('OAuth login URL: ' + data.data.url, 'oauth');
                    
                } else if (type === 'oauth_success') {
                    authenticated = true;
                    updateAuthStatus();
                    addMessage('OAuth authentication successful!', 'oauth');
                    
                } else if (type === 'oauth_error') {
                    addMessage('OAuth error: ' + data.data.error, 'error');
                    
                } else if (type === 'oauth_status') {
                    authenticated = data.data.authenticated;
                    updateAuthStatus();
                    addMessage('Auth status: ' + JSON.stringify(data.data), 'oauth');
                    
                } else if (type === 'auth_required') {
                    addMessage('Authentication required: ' + data.data.message, 'error');
                    
                } else if (type === 'query_authenticated') {
                    addMessage('Query authenticated: ' + data.data.message, 'query');
                    
                } else {
                    addMessage(type + ': ' + JSON.stringify(data.data || {}), 'system');
                }
            }
            
            function updateAuthStatus() {
                const authSection = document.getElementById('auth-section');
                const authStatus = document.getElementById('auth-status');
                
                if (authenticated) {
                    authSection.className = 'auth-section authenticated';
                    authStatus.textContent = 'Authenticated ✓';
                } else {
                    authSection.className = 'auth-section not-authenticated';
                    authStatus.textContent = 'Not authenticated ✗';
                }
            }
            
            function login() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: 'oauth_login',
                        auto_open: true
                    }));
                }
            }
            
            function logout() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({type: 'oauth_logout'}));
                }
            }
            
            function refreshToken() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({type: 'oauth_refresh'}));
                }
            }
            
            function checkStatus() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({type: 'oauth_status'}));
                }
            }
            
            function sendQuery() {
                const input = document.getElementById('queryInput');
                const query = input.value.trim();
                
                if (query && ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: 'query',
                        prompt: query,
                        options: {}
                    }));
                    input.value = '';
                }
            }
            
            function addMessage(message, type) {
                const messages = document.getElementById('messages');
                const div = document.createElement('div');
                div.className = 'message ' + type;
                div.textContent = new Date().toLocaleTimeString() + ' - ' + message;
                messages.appendChild(div);
                messages.scrollTop = messages.scrollHeight;
            }
            
            // Connect on page load
            connect();
            
            // Handle Enter key in input
            document.getElementById('queryInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendQuery();
                }
            });
        </script>
    </body>
    </html>
    """


async def main():
    """Run the OAuth-enabled WebSocket server."""
    # Create the server
    server = ClaudeWebSocketOAuthServer()
    
    # Create UI file
    ui_path = Path("claude_oauth_ui.html")
    with open(ui_path, "w") as f:
        f.write(create_oauth_ui_html())
    
    print("🚀 Starting Claude Code WebSocket server with OAuth support")
    print("📱 Open http://localhost:8000 to access the UI")
    print("🔐 OAuth flow will be handled through the web interface")
    print()
    
    try:
        # Start the server
        config = uvicorn.Config(server.app, host="0.0.0.0", port=8000, log_level="info")
        server_instance = uvicorn.Server(config)
        await server_instance.serve()
    except KeyboardInterrupt:
        print("\n👋 Shutting down server")
    finally:
        await server.cleanup()


if __name__ == "__main__":
    asyncio.run(main())