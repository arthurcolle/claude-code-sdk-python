"""OAuth configuration for Claude Code.

This module provides OAuth configuration that will be ready when Anthropic
enables OAuth for Claude Code Max users.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class OAuthEndpoints:
    """OAuth endpoint configuration."""
    
    # These will be the actual endpoints when OAuth is enabled
    authorize: str = "https://console.anthropic.com/oauth/authorize"
    token: str = "https://api.anthropic.com/oauth/token"
    revoke: str = "https://api.anthropic.com/oauth/revoke"
    userinfo: str = "https://api.anthropic.com/oauth/userinfo"
    
    # Development/staging endpoints for testing
    dev_authorize: str = "https://dev-console.anthropic.com/oauth/authorize"
    dev_token: str = "https://dev-api.anthropic.com/oauth/token"
    
    @classmethod
    def from_environment(cls, use_dev: bool = False) -> "OAuthEndpoints":
        """Create endpoints from environment variables."""
        if use_dev:
            return cls(
                authorize=os.getenv("CLAUDE_OAUTH_DEV_AUTHORIZE_URL", cls.dev_authorize),
                token=os.getenv("CLAUDE_OAUTH_DEV_TOKEN_URL", cls.dev_token),
            )
        
        return cls(
            authorize=os.getenv("CLAUDE_OAUTH_AUTHORIZE_URL", cls().authorize),
            token=os.getenv("CLAUDE_OAUTH_TOKEN_URL", cls().token),
            revoke=os.getenv("CLAUDE_OAUTH_REVOKE_URL", cls().revoke),
            userinfo=os.getenv("CLAUDE_OAUTH_USERINFO_URL", cls().userinfo),
        )


@dataclass
class OAuthScopes:
    """OAuth scopes for Claude Code."""
    
    # Basic scopes
    PROFILE: str = "profile"
    EMAIL: str = "email"
    
    # Claude Code specific scopes
    CODE_READ: str = "claude_code:read"
    CODE_WRITE: str = "claude_code:write"
    CODE_EXECUTE: str = "claude_code:execute"
    
    # Tool-related scopes
    TOOLS_READ: str = "tools:read"
    TOOLS_WRITE: str = "tools:write"
    TOOLS_EXECUTE: str = "tools:execute"
    
    # Advanced scopes
    WORKSPACE_READ: str = "workspace:read"
    WORKSPACE_WRITE: str = "workspace:write"
    
    @classmethod
    def default(cls) -> list[str]:
        """Get default scopes for Claude Code."""
        return [
            cls.PROFILE,
            cls.CODE_READ,
            cls.CODE_WRITE,
            cls.CODE_EXECUTE,
        ]
    
    @classmethod
    def full_access(cls) -> list[str]:
        """Get all available scopes."""
        return [
            cls.PROFILE,
            cls.EMAIL,
            cls.CODE_READ,
            cls.CODE_WRITE,
            cls.CODE_EXECUTE,
            cls.TOOLS_READ,
            cls.TOOLS_WRITE,
            cls.TOOLS_EXECUTE,
            cls.WORKSPACE_READ,
            cls.WORKSPACE_WRITE,
        ]


@dataclass
class ClaudeCodeOAuthConfig:
    """Complete OAuth configuration for Claude Code."""
    
    # Client configuration
    client_id: str
    client_secret: Optional[str] = None
    redirect_uri: str = "http://localhost:54545/callback"
    
    # Endpoints
    endpoints: OAuthEndpoints = None
    
    # Scopes
    scopes: list[str] = None
    
    # Advanced settings
    use_pkce: bool = True  # Use PKCE for enhanced security
    state_length: int = 32  # Length of state parameter
    port: int = 54545  # Local server port for callback
    timeout: int = 300  # OAuth flow timeout in seconds
    
    def __post_init__(self):
        """Initialize defaults."""
        if self.endpoints is None:
            self.endpoints = OAuthEndpoints()
        
        if self.scopes is None:
            self.scopes = OAuthScopes.default()
    
    @classmethod
    def for_claude_code_max(cls) -> "ClaudeCodeOAuthConfig":
        """Create configuration for Claude Code Max users."""
        return cls(
            client_id=os.getenv(
                "CLAUDE_OAUTH_CLIENT_ID",
                "claude-code-sdk"  # This will be provided by Anthropic
            ),
            client_secret=os.getenv("CLAUDE_OAUTH_CLIENT_SECRET"),
            redirect_uri=os.getenv(
                "CLAUDE_OAUTH_REDIRECT_URI",
                "http://localhost:54545/callback"
            ),
            endpoints=OAuthEndpoints.from_environment(),
            scopes=OAuthScopes.default(),
        )
    
    @classmethod
    def for_development(cls) -> "ClaudeCodeOAuthConfig":
        """Create configuration for development/testing."""
        return cls(
            client_id="claude-code-sdk-dev",
            redirect_uri="http://localhost:8080/callback",
            endpoints=OAuthEndpoints.from_environment(use_dev=True),
            scopes=OAuthScopes.full_access(),
            port=8080,
        )
    
    def get_authorize_url(self, state: str, code_challenge: Optional[str] = None) -> str:
        """Build the authorization URL."""
        from urllib.parse import urlencode
        
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.scopes),
            "state": state,
        }
        
        if self.use_pkce and code_challenge:
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"
        
        return f"{self.endpoints.authorize}?{urlencode(params)}"