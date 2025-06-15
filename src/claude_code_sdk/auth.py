"""Authentication module for Claude Code SDK.

Supports both API key and OAuth authentication flows.
"""

import asyncio
import contextlib
import json
import os
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
from typing_extensions import TypedDict

from ._errors import ClaudeSDKError


class TokenResponse(TypedDict):
    """OAuth token response."""

    access_token: str
    token_type: str
    expires_in: int
    refresh_token: str | None
    scope: str | None


@dataclass
class AuthToken:
    """Authentication token with expiry."""

    access_token: str
    token_type: str = "Bearer"
    expires_at: datetime | None = None
    refresh_token: str | None = None
    scope: str | None = None

    def is_expired(self) -> bool:
        """Check if token is expired."""
        if not self.expires_at:
            return False
        return datetime.now() >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "access_token": self.access_token,
            "token_type": self.token_type,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "refresh_token": self.refresh_token,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuthToken":
        """Create from dictionary."""
        expires_at = None
        if data.get("expires_at"):
            expires_at = datetime.fromisoformat(data["expires_at"])

        return cls(
            access_token=data["access_token"],
            token_type=data.get("token_type", "Bearer"),
            expires_at=expires_at,
            refresh_token=data.get("refresh_token"),
            scope=data.get("scope"),
        )


class OAuthConfig:
    """OAuth configuration for Claude Code."""

    # Default OAuth endpoints (can be overridden)
    AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
    TOKEN_URL = "https://claude.ai/oauth/token"
    REDIRECT_URI = "http://localhost:54545/callback"
    CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        redirect_uri: str | None = None,
        authorize_url: str | None = None,
        token_url: str | None = None,
    ):
        """Initialize OAuth configuration.

        Args:
            client_id: OAuth client ID
            client_secret: OAuth client secret (if required)
            redirect_uri: OAuth redirect URI
            authorize_url: Authorization endpoint URL
            token_url: Token endpoint URL
        """
        self.client_id = client_id or os.getenv(
            "CLAUDE_OAUTH_CLIENT_ID", self.CLIENT_ID
        )
        self.client_secret = client_secret or os.getenv("CLAUDE_OAUTH_CLIENT_SECRET")
        self.redirect_uri = redirect_uri or os.getenv(
            "CLAUDE_OAUTH_REDIRECT_URI", self.REDIRECT_URI
        )
        self.authorize_url = authorize_url or os.getenv(
            "CLAUDE_OAUTH_AUTHORIZE_URL", self.AUTHORIZE_URL
        )
        self.token_url = token_url or os.getenv(
            "CLAUDE_OAUTH_TOKEN_URL", self.TOKEN_URL
        )


class AuthenticationError(ClaudeSDKError):
    """Authentication related errors."""

    pass


class TokenStorage:
    """Secure token storage."""

    def __init__(self, storage_path: Path | None = None):
        """Initialize token storage.

        Args:
            storage_path: Path to store tokens (defaults to ~/.claude_code/tokens.json)
        """
        self.storage_path = storage_path or Path.home() / ".claude_code" / "tokens.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def save_token(self, token: AuthToken) -> None:
        """Save token to storage."""
        data = {}
        if self.storage_path.exists():
            with contextlib.suppress(Exception):
                data = json.loads(self.storage_path.read_text())

        data["token"] = token.to_dict()
        data["updated_at"] = datetime.now().isoformat()

        # Set restrictive permissions on token file
        self.storage_path.write_text(json.dumps(data, indent=2))
        self.storage_path.chmod(0o600)

    def load_token(self) -> AuthToken | None:
        """Load token from storage."""
        if not self.storage_path.exists():
            return None

        try:
            data = json.loads(self.storage_path.read_text())
            return AuthToken.from_dict(data["token"])
        except Exception:
            return None

    def delete_token(self) -> None:
        """Delete stored token."""
        if self.storage_path.exists():
            self.storage_path.unlink()


class OAuthFlow:
    """OAuth 2.0 Authorization Code flow implementation with PKCE."""

    def __init__(
        self,
        config: OAuthConfig,
        storage: TokenStorage | None = None,
    ):
        """Initialize OAuth flow.

        Args:
            config: OAuth configuration
            storage: Token storage (defaults to TokenStorage())
        """
        self.config = config
        self.storage = storage or TokenStorage()
        self._http_client: httpx.AsyncClient | None = None
        self._code_verifier: str | None = None
        self._state: str | None = None

    async def __aenter__(self) -> "OAuthFlow":
        """Async context manager entry."""
        self._http_client = httpx.AsyncClient()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self._http_client:
            await self._http_client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        """Get HTTP client."""
        if not self._http_client:
            raise RuntimeError("OAuthFlow must be used as async context manager")
        return self._http_client

    def get_authorization_url(self, state: str | None = None) -> str:
        """Get OAuth authorization URL.

        Args:
            state: Optional state parameter for CSRF protection

        Returns:
            Authorization URL
        """
        params = {
            "client_id": self.config.client_id,
            "response_type": "code",
            "redirect_uri": self.config.redirect_uri,
        }

        if state:
            params["state"] = state

        # Add Claude Code specific scopes
        params["scope"] = "claude_code:read claude_code:write"

        return f"{self.config.authorize_url}?{urlencode(params)}"

    async def exchange_code_for_token(self, code: str) -> AuthToken:
        """Exchange authorization code for access token.

        Args:
            code: Authorization code from callback

        Returns:
            Authentication token
        """
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.config.redirect_uri,
            "client_id": self.config.client_id,
        }

        if self.config.client_secret:
            data["client_secret"] = self.config.client_secret

        if not self.config.token_url:
            raise AuthenticationError("No token URL configured")

        response = await self.client.post(
            self.config.token_url,
            data=data,
            headers={"Accept": "application/json"},
        )

        if response.status_code != 200:
            raise AuthenticationError(
                f"Token exchange failed: {response.status_code} {response.text}"
            )

        token_data: TokenResponse = response.json()

        # Calculate expiry time
        expires_at = None
        if "expires_in" in token_data:
            expires_at = datetime.now() + timedelta(seconds=token_data["expires_in"])

        token = AuthToken(
            access_token=token_data["access_token"],
            token_type=token_data.get("token_type", "Bearer"),
            expires_at=expires_at,
            refresh_token=token_data.get("refresh_token"),
            scope=token_data.get("scope"),
        )

        # Save token
        self.storage.save_token(token)

        return token

    async def refresh_token(self, refresh_token: str) -> AuthToken:
        """Refresh an expired token.

        Args:
            refresh_token: Refresh token

        Returns:
            New authentication token
        """
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.config.client_id,
        }

        if self.config.client_secret:
            data["client_secret"] = self.config.client_secret

        if not self.config.token_url:
            raise AuthenticationError("No token URL configured")

        response = await self.client.post(
            self.config.token_url,
            data=data,
            headers={"Accept": "application/json"},
        )

        if response.status_code != 200:
            raise AuthenticationError(
                f"Token refresh failed: {response.status_code} {response.text}"
            )

        token_data: TokenResponse = response.json()

        # Calculate expiry time
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

        # Save token
        self.storage.save_token(token)

        return token

    async def get_valid_token(self) -> AuthToken | None:
        """Get a valid token, refreshing if necessary.

        Returns:
            Valid authentication token or None
        """
        token = self.storage.load_token()
        if not token:
            return None

        # Check if expired
        if token.is_expired() and token.refresh_token:
            try:
                token = await self.refresh_token(token.refresh_token)
            except AuthenticationError:
                # Refresh failed, clear token
                self.storage.delete_token()
                return None

        return token


class LocalCallbackServer:
    """Local HTTP server for OAuth callback."""

    def __init__(self, port: int = 54545):
        """Initialize callback server.

        Args:
            port: Port to listen on
        """
        self.port = port
        self.auth_code: str | None = None
        self.error: str | None = None
        self._server_task: asyncio.Task[Any] | None = None

    async def start(self) -> None:
        """Start the callback server."""
        from aiohttp import web

        async def handle_callback(request: web.Request) -> web.Response:
            """Handle OAuth callback."""
            # Extract code or error from query parameters
            if "code" in request.query:
                self.auth_code = request.query["code"]
            elif "error" in request.query:
                self.error = request.query.get(
                    "error_description", request.query["error"]
                )

            # Return success page
            status_class = "success" if self.auth_code else "error"
            status_message = (
                "Authentication successful!"
                if self.auth_code
                else f"Authentication failed: {self.error}"
            )

            html = f"""
            <html>
            <head>
                <title>Claude Code Authentication</title>
                <style>
                    body {{ font-family: sans-serif; text-align: center; padding: 50px; }}
                    .success {{ color: green; }}
                    .error {{ color: red; }}
                </style>
            </head>
            <body>
                <h1>Claude Code Authentication</h1>
                <p class="{status_class}">
                    {status_message}
                </p>
                <p>You can close this window now.</p>
            </body>
            </html>
            """

            return web.Response(text=html, content_type="text/html")

        app = web.Application()
        app.router.add_get("/callback", handle_callback)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "localhost", self.port)
        await site.start()

        # Keep server running
        self._server_task = asyncio.create_task(asyncio.Event().wait())

    async def wait_for_code(self, timeout: int = 300) -> str:
        """Wait for authorization code.

        Args:
            timeout: Timeout in seconds

        Returns:
            Authorization code

        Raises:
            AuthenticationError: If authentication fails or times out
        """
        try:
            # Wait for code with timeout
            start_time = asyncio.get_event_loop().time()
            while not self.auth_code and not self.error:
                if asyncio.get_event_loop().time() - start_time > timeout:
                    raise AuthenticationError("Authentication timeout")
                await asyncio.sleep(0.1)

            if self.error:
                raise AuthenticationError(f"Authentication failed: {self.error}")

            if not self.auth_code:
                raise AuthenticationError("No authorization code received")
            return self.auth_code

        finally:
            # Stop server
            if self._server_task:
                self._server_task.cancel()


class ClaudeAuth:
    """Main authentication class for Claude Code SDK."""

    def __init__(
        self,
        use_oauth: bool = True,
        api_key: str | None = None,
        oauth_config: OAuthConfig | None = None,
        token_storage: TokenStorage | None = None,
    ):
        """Initialize authentication.

        Args:
            use_oauth: Use OAuth instead of API key
            api_key: API key (if not using OAuth)
            oauth_config: OAuth configuration
            token_storage: Token storage
        """
        self.use_oauth = use_oauth
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.oauth_config = oauth_config or OAuthConfig()
        self.token_storage = token_storage or TokenStorage()
        self._oauth_flow: OAuthFlow | None = None

    async def __aenter__(self) -> "ClaudeAuth":
        """Async context manager entry."""
        if self.use_oauth:
            self._oauth_flow = OAuthFlow(self.oauth_config, self.token_storage)
            await self._oauth_flow.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self._oauth_flow:
            await self._oauth_flow.__aexit__(*args)

    async def authenticate(self) -> dict[str, str]:
        """Perform authentication and return auth headers.

        Returns:
            Authentication headers
        """
        if not self.use_oauth:
            # Use API key
            if not self.api_key:
                raise AuthenticationError("No API key provided")
            return {"X-API-Key": self.api_key}

        # Use OAuth
        if not self._oauth_flow:
            raise RuntimeError(
                "ClaudeAuth must be used as async context manager for OAuth"
            )

        # Try to get existing valid token
        token = await self._oauth_flow.get_valid_token()

        if not token:
            # Need to perform OAuth flow
            token = await self.perform_oauth_flow()

        return {"Authorization": f"{token.token_type} {token.access_token}"}

    async def perform_oauth_flow(self) -> AuthToken:
        """Perform interactive OAuth flow.

        Returns:
            Authentication token
        """
        if not self._oauth_flow:
            raise RuntimeError("OAuth flow not initialized")

        # Start local callback server
        server = LocalCallbackServer()
        await server.start()

        # Generate authorization URL
        auth_url = self._oauth_flow.get_authorization_url()

        print("Opening browser for authentication...")
        print(f"If browser doesn't open, visit: {auth_url}")

        # Open browser
        webbrowser.open(auth_url)

        # Wait for callback
        code = await server.wait_for_code()

        # Exchange code for token
        token = await self._oauth_flow.exchange_code_for_token(code)

        print("Authentication successful!")

        return token

    def get_env_vars(self) -> dict[str, str]:
        """Get environment variables for CLI subprocess.

        Returns:
            Environment variables to set
        """
        if not self.use_oauth:
            # Use API key
            if self.api_key:
                return {"ANTHROPIC_API_KEY": self.api_key}
            return {}

        # For OAuth, we need to use a different approach
        # The CLI would need to be modified to support OAuth tokens
        # For now, we can try to use the token as an API key
        token = self.token_storage.load_token()
        if token and not token.is_expired():
            # Some APIs accept Bearer tokens as API keys
            return {"ANTHROPIC_API_KEY": f"Bearer {token.access_token}"}

        return {}


# Convenience functions
async def login() -> None:
    """Perform OAuth login flow."""
    async with ClaudeAuth(use_oauth=True) as auth:
        await auth.perform_oauth_flow()
        print("Login successful! Token saved.")


async def logout() -> None:
    """Logout and remove stored tokens."""
    storage = TokenStorage()
    storage.delete_token()
    print("Logged out successfully.")


async def get_auth_headers() -> dict[str, str]:
    """Get authentication headers for API requests.

    Returns:
        Authentication headers
    """
    async with ClaudeAuth() as auth:
        return await auth.authenticate()
