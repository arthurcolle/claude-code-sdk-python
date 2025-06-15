# OAuth Authentication Guide for Claude Code SDK

This guide explains how to use OAuth authentication with the Claude Code SDK, which will be available for Claude Code Max users.

## Overview

The Claude Code SDK supports OAuth 2.0 authentication as an alternative to API keys. This feature is designed for Claude Code Max subscribers who pay for the service and want to use their existing authentication without managing separate API keys.

> **Note**: OAuth support is coming soon. This implementation is ready for when Anthropic enables it.

## Quick Start

### 1. Basic OAuth Authentication

```python
from claude_code_sdk import query
from claude_code_sdk.auth_config import ClaudeCodeOAuthConfig
from claude_code_sdk.oauth_flow import ClaudeCodeOAuthFlow
from claude_code_sdk.auth import TokenStorage

async def authenticate_and_query():
    # Create OAuth configuration
    config = ClaudeCodeOAuthConfig.for_claude_code_max()
    
    # Create token storage
    storage = TokenStorage()
    
    # Perform OAuth flow
    async with ClaudeCodeOAuthFlow(config, storage) as flow:
        token = await flow.authenticate()
        
    # Now use the SDK with OAuth
    async for message in query("Hello, Claude!", auth_token=token.access_token):
        print(message)
```

### 2. Using the OAuth Demo

Run the included OAuth demo to see how it works:

```bash
python examples/oauth_demo.py
```

## Features

### PKCE Support

The OAuth implementation includes PKCE (Proof Key for Code Exchange) for enhanced security:

```python
from claude_code_sdk.oauth_flow import PKCEChallenge

# PKCE is automatically used when enabled in config
config = ClaudeCodeOAuthConfig(
    client_id="your-client-id",
    use_pkce=True  # Enabled by default
)
```

### Token Management

Tokens are automatically stored and refreshed:

```python
from claude_code_sdk.auth import TokenStorage

# Token storage with secure file permissions (0600)
storage = TokenStorage()

# Load existing token
token = storage.load_token()
if token and not token.is_expired():
    # Use existing valid token
    pass
```

### OAuth Scopes

The SDK supports various OAuth scopes for different capabilities:

```python
from claude_code_sdk.auth_config import OAuthScopes

# Available scopes
scopes = [
    OAuthScopes.PROFILE,           # User profile access
    OAuthScopes.EMAIL,             # Email address access
    OAuthScopes.CODE_READ,         # Read code and conversations
    OAuthScopes.CODE_WRITE,        # Create and modify code
    OAuthScopes.CODE_EXECUTE,      # Execute code and tools
    OAuthScopes.TOOLS_READ,        # Read available tools
    OAuthScopes.TOOLS_WRITE,       # Create and modify tools
    OAuthScopes.TOOLS_EXECUTE,     # Execute tools
    OAuthScopes.WORKSPACE_READ,    # Read workspace files
    OAuthScopes.WORKSPACE_WRITE,   # Modify workspace files
]

# Use default scopes
config = ClaudeCodeOAuthConfig.for_claude_code_max()  # Uses default scopes

# Or specify custom scopes
config = ClaudeCodeOAuthConfig(
    client_id="your-client-id",
    scopes=OAuthScopes.full_access()  # All available scopes
)
```

### Automatic Token Refresh

The SDK automatically refreshes expired tokens:

```python
async with ClaudeCodeOAuthFlow(config, storage) as flow:
    # This will:
    # 1. Check for existing valid token
    # 2. Refresh if expired (using refresh token)
    # 3. Perform new auth flow if needed
    token = await flow.authenticate()
```

### Error Handling

Comprehensive error handling for OAuth flows:

```python
from claude_code_sdk.auth import AuthenticationError

try:
    async with ClaudeCodeOAuthFlow(config) as flow:
        token = await flow.authenticate()
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
```

## Configuration

### Environment Variables

Configure OAuth through environment variables:

```bash
# OAuth Client Configuration
export CLAUDE_OAUTH_CLIENT_ID="your-client-id"
export CLAUDE_OAUTH_CLIENT_SECRET="your-client-secret"  # If required
export CLAUDE_OAUTH_REDIRECT_URI="http://localhost:54545/callback"

# OAuth Endpoints (defaults are provided)
export CLAUDE_OAUTH_AUTHORIZE_URL="https://console.anthropic.com/oauth/authorize"
export CLAUDE_OAUTH_TOKEN_URL="https://api.anthropic.com/oauth/token"
export CLAUDE_OAUTH_REVOKE_URL="https://api.anthropic.com/oauth/revoke"
export CLAUDE_OAUTH_USERINFO_URL="https://api.anthropic.com/oauth/userinfo"

# Enable OAuth (for demo purposes)
export CLAUDE_OAUTH_ENABLED="true"
```

### Programmatic Configuration

```python
from claude_code_sdk.auth_config import ClaudeCodeOAuthConfig, OAuthEndpoints

# Custom configuration
config = ClaudeCodeOAuthConfig(
    client_id="your-client-id",
    client_secret="your-secret",  # Optional
    redirect_uri="http://localhost:8080/callback",
    endpoints=OAuthEndpoints(
        authorize="https://custom.auth.url/authorize",
        token="https://custom.auth.url/token",
    ),
    scopes=["custom:scope1", "custom:scope2"],
    use_pkce=True,
    port=8080,  # Callback server port
    timeout=300,  # Auth timeout in seconds
)
```

## Advanced Usage

### Manual Token Management

```python
# Revoke a token
async with ClaudeCodeOAuthFlow(config) as flow:
    success = await flow.revoke_token(token.access_token)
    
# Get user information
async with ClaudeCodeOAuthFlow(config) as flow:
    user_info = await flow.get_user_info(token.access_token)
    print(f"User: {user_info.get('email')}")
```

### Custom Callback Server

The OAuth flow includes a local callback server with a nice UI:

- Automatically opens browser for authentication
- Shows success/error status
- Auto-closes on success
- Handles state verification for CSRF protection

### Integration with Claude Code CLI

When OAuth is enabled, the SDK will provide authentication to the Claude Code CLI:

```python
from claude_code_sdk import query, ClaudeCodeOptions

# OAuth token will be automatically used if available
options = ClaudeCodeOptions(
    use_oauth=True  # Future option when OAuth is enabled
)

async for message in query("Write hello world", options=options):
    print(message)
```

## Security Best Practices

1. **Token Storage**: Tokens are stored with restrictive permissions (0600)
2. **PKCE**: Always enabled by default for public clients
3. **State Parameter**: Automatically generated for CSRF protection
4. **HTTPS**: All OAuth endpoints use HTTPS
5. **Token Expiry**: Automatic handling of token expiration

## Troubleshooting

### OAuth Not Enabled

If you see "OAuth is not yet enabled for Claude Code", this means Anthropic hasn't released OAuth support yet. The implementation is ready and waiting.

### Token Refresh Fails

If token refresh fails, the SDK will automatically fall back to a new authentication flow.

### Port Already in Use

If the default callback port (54545) is in use, configure a different port:

```python
config = ClaudeCodeOAuthConfig(
    port=8080,
    redirect_uri="http://localhost:8080/callback"
)
```

## Future Enhancements

When Anthropic enables OAuth for Claude Code Max users, this implementation will support:

- Seamless authentication without API keys
- Automatic token management
- Integration with Claude Code's permission system
- Organization-level authentication
- SSO integration

## Example: Building an OAuth-Enabled Application

```python
import asyncio
from claude_code_sdk import query
from claude_code_sdk.auth_config import ClaudeCodeOAuthConfig
from claude_code_sdk.oauth_flow import ClaudeCodeOAuthFlow
from claude_code_sdk.auth import TokenStorage

class ClaudeCodeApp:
    def __init__(self):
        self.config = ClaudeCodeOAuthConfig.for_claude_code_max()
        self.storage = TokenStorage()
        
    async def ensure_authenticated(self):
        """Ensure user is authenticated."""
        async with ClaudeCodeOAuthFlow(self.config, self.storage) as flow:
            return await flow.authenticate()
            
    async def run_query(self, prompt: str):
        """Run a query with OAuth authentication."""
        # Ensure authenticated
        token = await self.ensure_authenticated()
        
        # Run query
        async for message in query(prompt, auth_token=token.access_token):
            yield message
            
    async def logout(self):
        """Logout and revoke tokens."""
        token = self.storage.load_token()
        if token:
            async with ClaudeCodeOAuthFlow(self.config, self.storage) as flow:
                # Revoke tokens
                if token.access_token:
                    await flow.revoke_token(token.access_token)
                if token.refresh_token:
                    await flow.revoke_token(token.refresh_token, "refresh_token")
                    
            # Clear storage
            self.storage.delete_token()

# Usage
async def main():
    app = ClaudeCodeApp()
    
    # Run authenticated query
    async for message in app.run_query("Help me write a Python function"):
        print(message)
        
    # Logout when done
    await app.logout()

if __name__ == "__main__":
    asyncio.run(main())
```

## Contributing

This OAuth implementation is designed to be production-ready when Anthropic enables it. If you find issues or have suggestions, please contribute to make it even better!

## Summary

The Claude Code SDK's OAuth implementation provides:

- 🔐 Secure authentication without API keys
- 🔄 Automatic token refresh
- 🛡️ PKCE support for enhanced security
- 💾 Secure token storage
- 🎨 Nice callback UI
- 🚀 Production-ready code

This will enable Claude Code Max users to authenticate seamlessly with their existing subscriptions when OAuth support is released.