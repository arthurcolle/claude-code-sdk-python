"""Enhanced OAuth 2.0 flow implementation for Claude Code.

This module implements a production-ready OAuth flow with PKCE support,
ready for when Anthropic enables OAuth for Claude Code Max users.
"""

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import parse_qs, urlparse

import httpx
from aiohttp import web

from .auth import AuthToken, TokenStorage, AuthenticationError
from .auth_config import ClaudeCodeOAuthConfig


logger = logging.getLogger(__name__)


class PKCEChallenge:
    """PKCE (Proof Key for Code Exchange) implementation for enhanced security."""
    
    @staticmethod
    def generate_verifier(length: int = 128) -> str:
        """Generate a cryptographically secure code verifier."""
        # RFC 7636 recommends 43-128 characters
        verifier = base64.urlsafe_b64encode(
            secrets.token_bytes(length)
        ).decode('utf-8').rstrip('=')
        return verifier[:128]  # Ensure max length
    
    @staticmethod
    def generate_challenge(verifier: str) -> str:
        """Generate code challenge from verifier using S256 method."""
        digest = hashlib.sha256(verifier.encode('utf-8')).digest()
        challenge = base64.urlsafe_b64encode(digest).decode('utf-8').rstrip('=')
        return challenge


class OAuthSession:
    """OAuth session management."""
    
    def __init__(self):
        """Initialize OAuth session."""
        self.state = secrets.token_urlsafe(32)
        self.verifier = PKCEChallenge.generate_verifier()
        self.challenge = PKCEChallenge.generate_challenge(self.verifier)
        self.created_at = datetime.now()
        self.code: Optional[str] = None
        self.error: Optional[str] = None
    
    def is_expired(self, timeout: int = 600) -> bool:
        """Check if session is expired."""
        return (datetime.now() - self.created_at).total_seconds() > timeout


class ClaudeCodeOAuthFlow:
    """Production-ready OAuth flow for Claude Code."""
    
    def __init__(
        self,
        config: ClaudeCodeOAuthConfig,
        storage: Optional[TokenStorage] = None,
    ):
        """Initialize OAuth flow.
        
        Args:
            config: OAuth configuration
            storage: Token storage (defaults to TokenStorage())
        """
        self.config = config
        self.storage = storage or TokenStorage()
        self.session: Optional[OAuthSession] = None
        self._server: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._http_client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self) -> "ClaudeCodeOAuthFlow":
        """Async context manager entry."""
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
        )
        return self
    
    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        await self.cleanup()
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        if self._http_client:
            await self._http_client.aclose()
    
    async def start_callback_server(self) -> None:
        """Start the OAuth callback server."""
        async def handle_callback(request: web.Request) -> web.Response:
            """Handle OAuth callback."""
            if not self.session:
                return web.Response(
                    text="No active OAuth session",
                    status=400
                )
            
            # Verify state parameter
            state = request.query.get("state")
            if state != self.session.state:
                logger.error("State mismatch in OAuth callback")
                return web.Response(
                    text="Invalid state parameter",
                    status=400
                )
            
            # Extract code or error
            if "code" in request.query:
                self.session.code = request.query["code"]
                logger.info("Received authorization code")
            elif "error" in request.query:
                self.session.error = request.query.get(
                    "error_description",
                    request.query["error"]
                )
                logger.error(f"OAuth error: {self.session.error}")
            
            # Return success page
            html = self._generate_callback_html(
                success=bool(self.session.code),
                message="Authentication successful!" if self.session.code 
                        else f"Authentication failed: {self.session.error}"
            )
            
            return web.Response(text=html, content_type="text/html")
        
        async def handle_health(request: web.Request) -> web.Response:
            """Health check endpoint."""
            return web.Response(text="OK")
        
        # Create web application
        self._server = web.Application()
        self._server.router.add_get("/callback", handle_callback)
        self._server.router.add_get("/health", handle_health)
        
        # Start server
        self._runner = web.AppRunner(self._server)
        await self._runner.setup()
        self._site = web.TCPSite(
            self._runner,
            "localhost",
            self.config.port,
        )
        await self._site.start()
        
        logger.info(f"OAuth callback server started on port {self.config.port}")
    
    def _generate_callback_html(self, success: bool, message: str) -> str:
        """Generate callback HTML page."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Claude Code Authentication</title>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background-color: #f5f5f5;
                }}
                .container {{
                    text-align: center;
                    padding: 40px;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    max-width: 400px;
                }}
                .logo {{
                    font-size: 48px;
                    margin-bottom: 20px;
                }}
                .success {{ color: #22c55e; }}
                .error {{ color: #ef4444; }}
                .message {{
                    font-size: 18px;
                    margin: 20px 0;
                    font-weight: 500;
                }}
                .note {{
                    color: #6b7280;
                    margin-top: 20px;
                }}
                .spinner {{
                    border: 3px solid #f3f3f3;
                    border-top: 3px solid #3b82f6;
                    border-radius: 50%;
                    width: 30px;
                    height: 30px;
                    animation: spin 1s linear infinite;
                    margin: 20px auto;
                }}
                @keyframes spin {{
                    0% {{ transform: rotate(0deg); }}
                    100% {{ transform: rotate(360deg); }}
                }}
            </style>
            <script>
                // Auto-close window after 3 seconds on success
                if ({str(success).lower()}) {{
                    setTimeout(() => {{
                        window.close();
                    }}, 3000);
                }}
            </script>
        </head>
        <body>
            <div class="container">
                <div class="logo">🤖</div>
                <h1>Claude Code SDK</h1>
                <p class="message {'success' if success else 'error'}">
                    {message}
                </p>
                {'<div class="spinner"></div>' if success else ''}
                <p class="note">
                    {'This window will close automatically...' if success else 'You can close this window now.'}
                </p>
            </div>
        </body>
        </html>
        """
    
    async def authenticate(self) -> AuthToken:
        """Perform the complete OAuth authentication flow.
        
        Returns:
            Authentication token
            
        Raises:
            AuthenticationError: If authentication fails
        """
        # Check for existing valid token
        existing_token = self.storage.load_token()
        if existing_token and not existing_token.is_expired():
            logger.info("Using existing valid token")
            return existing_token
        
        # Try to refresh if we have a refresh token
        if existing_token and existing_token.refresh_token:
            try:
                logger.info("Attempting to refresh token")
                return await self.refresh_token(existing_token.refresh_token)
            except AuthenticationError:
                logger.warning("Token refresh failed, starting new auth flow")
        
        # Start new OAuth flow
        logger.info("Starting new OAuth authentication flow")
        return await self._perform_oauth_flow()
    
    async def _perform_oauth_flow(self) -> AuthToken:
        """Perform the OAuth flow."""
        # Create new session
        self.session = OAuthSession()
        
        # Start callback server
        await self.start_callback_server()
        
        try:
            # Generate authorization URL
            auth_url = self.config.get_authorize_url(
                state=self.session.state,
                code_challenge=self.session.challenge if self.config.use_pkce else None
            )
            
            logger.info(f"Opening browser for authentication: {auth_url}")
            print(f"\n🔐 Opening browser for authentication...")
            print(f"If the browser doesn't open, visit: {auth_url}\n")
            
            # Open browser
            webbrowser.open(auth_url)
            
            # Wait for callback
            code = await self._wait_for_callback()
            
            # Exchange code for token
            token = await self._exchange_code_for_token(code)
            
            # Save token
            self.storage.save_token(token)
            
            print("✅ Authentication successful!")
            
            return token
            
        finally:
            # Clean up session
            self.session = None
    
    async def _wait_for_callback(self) -> str:
        """Wait for OAuth callback.
        
        Returns:
            Authorization code
            
        Raises:
            AuthenticationError: If authentication fails or times out
        """
        if not self.session:
            raise AuthenticationError("No active OAuth session")
        
        start_time = asyncio.get_event_loop().time()
        
        while True:
            # Check for timeout
            if asyncio.get_event_loop().time() - start_time > self.config.timeout:
                raise AuthenticationError("Authentication timeout")
            
            # Check session expiry
            if self.session.is_expired(self.config.timeout):
                raise AuthenticationError("OAuth session expired")
            
            # Check for code or error
            if self.session.code:
                return self.session.code
            
            if self.session.error:
                raise AuthenticationError(f"OAuth error: {self.session.error}")
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
    
    async def _exchange_code_for_token(self, code: str) -> AuthToken:
        """Exchange authorization code for access token.
        
        Args:
            code: Authorization code
            
        Returns:
            Authentication token
            
        Raises:
            AuthenticationError: If token exchange fails
        """
        if not self._http_client:
            raise RuntimeError("HTTP client not initialized")
        
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.config.redirect_uri,
            "client_id": self.config.client_id,
        }
        
        # Add PKCE verifier if used
        if self.config.use_pkce and self.session:
            data["code_verifier"] = self.session.verifier
        
        # Add client secret if available
        if self.config.client_secret:
            data["client_secret"] = self.config.client_secret
        
        try:
            response = await self._http_client.post(
                self.config.endpoints.token,
                data=data,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                }
            )
            response.raise_for_status()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Token exchange failed: {e.response.status_code} {e.response.text}")
            raise AuthenticationError(f"Token exchange failed: {e.response.status_code}")
        
        token_data = response.json()
        
        # Create token object
        expires_at = None
        if "expires_in" in token_data:
            expires_at = datetime.now() + timedelta(seconds=token_data["expires_in"])
        
        return AuthToken(
            access_token=token_data["access_token"],
            token_type=token_data.get("token_type", "Bearer"),
            expires_at=expires_at,
            refresh_token=token_data.get("refresh_token"),
            scope=token_data.get("scope"),
        )
    
    async def refresh_token(self, refresh_token: str) -> AuthToken:
        """Refresh an expired token.
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            New authentication token
            
        Raises:
            AuthenticationError: If refresh fails
        """
        if not self._http_client:
            raise RuntimeError("HTTP client not initialized")
        
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.config.client_id,
        }
        
        if self.config.client_secret:
            data["client_secret"] = self.config.client_secret
        
        try:
            response = await self._http_client.post(
                self.config.endpoints.token,
                data=data,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                }
            )
            response.raise_for_status()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Token refresh failed: {e.response.status_code}")
            raise AuthenticationError(f"Token refresh failed: {e.response.status_code}")
        
        token_data = response.json()
        
        # Create new token
        expires_at = None
        if "expires_in" in token_data:
            expires_at = datetime.now() + timedelta(seconds=token_data["expires_in"])
        
        token = AuthToken(
            access_token=token_data["access_token"],
            token_type=token_data.get("token_type", "Bearer"),
            expires_at=expires_at,
            refresh_token=token_data.get("refresh_token") or refresh_token,
            scope=token_data.get("scope"),
        )
        
        # Save updated token
        self.storage.save_token(token)
        
        return token
    
    async def revoke_token(self, token: str, token_type: str = "access_token") -> bool:
        """Revoke a token.
        
        Args:
            token: Token to revoke
            token_type: Type of token ("access_token" or "refresh_token")
            
        Returns:
            True if revocation successful
        """
        if not self._http_client:
            raise RuntimeError("HTTP client not initialized")
        
        try:
            response = await self._http_client.post(
                self.config.endpoints.revoke,
                data={
                    "token": token,
                    "token_type_hint": token_type,
                    "client_id": self.config.client_id,
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                }
            )
            response.raise_for_status()
            return True
            
        except httpx.HTTPStatusError:
            logger.error("Token revocation failed")
            return False
    
    async def get_user_info(self, access_token: str) -> dict:
        """Get user information using access token.
        
        Args:
            access_token: Valid access token
            
        Returns:
            User information
        """
        if not self._http_client:
            raise RuntimeError("HTTP client not initialized")
        
        try:
            response = await self._http_client.get(
                self.config.endpoints.userinfo,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                }
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to get user info: {e.response.status_code}")
            raise AuthenticationError(f"Failed to get user info: {e.response.status_code}")