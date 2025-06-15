# Authentication Guide

The Claude Code SDK supports two authentication methods:
1. **OAuth Authentication** (recommended for Claude Code Max plan users)
2. **API Key Authentication** (traditional method)

## OAuth Authentication (Claude Code Max)

Claude Code Max plan users can authenticate using OAuth, eliminating the need to manage API keys.

### Quick Start

#### 1. Login via CLI

```bash
# Install the SDK
pip install claude-code-sdk

# Login with OAuth
claude-auth login
```

This will:
- Open your browser for authentication
- Save the authentication token locally
- Allow SDK usage without API keys

#### 2. Use in Python

```python
import asyncio
from claude_code_sdk import query_with_oauth

async def main():
    # Uses OAuth authentication automatically
    async for message in query_with_oauth(prompt="Hello Claude!"):
        print(message)

asyncio.run(main())
```

### OAuth Flow Details

The SDK implements the OAuth 2.0 Authorization Code flow:

1. **Authorization**: Opens browser to Claude Console for user login
2. **Callback**: Local server receives authorization code
3. **Token Exchange**: Code is exchanged for access token
4. **Token Storage**: Tokens are securely stored locally
5. **Auto-refresh**: Expired tokens are automatically refreshed

### CLI Commands

```bash
# Login with OAuth
claude-auth login

# Check authentication status
claude-auth status

# Refresh expired token
claude-auth refresh

# Logout and clear tokens
claude-auth logout
```

### Python API

#### Basic OAuth Query

```python
from claude_code_sdk import query_with_oauth

async for message in query_with_oauth(prompt="Your prompt"):
    print(message)
```

#### Programmatic Login

```python
from claude_code_sdk import login, logout

# Perform OAuth login
await login()

# Clear authentication
await logout()
```

#### Custom OAuth Configuration

```python
from claude_code_sdk import OAuthConfig, authenticated_query

config = OAuthConfig(
    client_id="your-client-id",
    authorize_url="https://custom-auth-url",
    token_url="https://custom-token-url",
)

async for message in authenticated_query(
    prompt="Hello",
    oauth_config=config
):
    print(message)
```

### Token Management

Tokens are stored in `~/.claude_code/tokens.json` with restricted permissions (0600).

```python
from claude_code_sdk import ClaudeAuth

async with ClaudeAuth(use_oauth=True) as auth:
    # Check if authenticated
    token = auth.token_storage.load_token()
    if token:
        print(f"Authenticated: {not token.is_expired()}")
        print(f"Expires at: {token.expires_at}")
```

### Environment Variables

Configure OAuth via environment variables:

- `CLAUDE_OAUTH_CLIENT_ID`: OAuth client ID
- `CLAUDE_OAUTH_CLIENT_SECRET`: OAuth client secret (if required)
- `CLAUDE_OAUTH_REDIRECT_URI`: OAuth redirect URI (default: http://localhost:8089/callback)
- `CLAUDE_OAUTH_AUTHORIZE_URL`: Authorization endpoint
- `CLAUDE_OAUTH_TOKEN_URL`: Token endpoint

## API Key Authentication

Traditional API key authentication is still supported.

### Using Environment Variable

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

```python
from claude_code_sdk import query

async for message in query(prompt="Hello"):
    print(message)
```

### Using Code

```python
from claude_code_sdk import query_with_api_key

async for message in query_with_api_key(
    prompt="Hello",
    api_key="your-api-key"
):
    print(message)
```

## Flexible Authentication

The `authenticated_query` function supports both methods:

```python
from claude_code_sdk import authenticated_query

# Try OAuth first, fall back to API key
async for message in authenticated_query(
    prompt="Hello",
    use_oauth=True,  # Try OAuth
    api_key="fallback-key"  # Fallback to API key
):
    print(message)
```

## Security Best Practices

1. **OAuth Tokens**:
   - Stored with restricted permissions (0600)
   - Never commit token files to version control
   - Tokens expire and auto-refresh

2. **API Keys**:
   - Use environment variables
   - Never hardcode in source code
   - Rotate keys regularly

3. **Custom OAuth Apps**:
   - Keep client secrets secure
   - Use HTTPS for redirect URIs in production
   - Implement proper CSRF protection with state parameter

## Troubleshooting

### OAuth Login Issues

1. **Browser doesn't open**:
   ```bash
   # Manually visit the URL shown in terminal
   ```

2. **Token expired**:
   ```bash
   claude-auth refresh
   # Or just use the SDK - it auto-refreshes
   ```

3. **Permission denied**:
   ```bash
   # Fix token file permissions
   chmod 600 ~/.claude_code/tokens.json
   ```

### API Key Issues

1. **No API key found**:
   ```bash
   export ANTHROPIC_API_KEY="your-key"
   ```

2. **Invalid API key**:
   - Check key validity in Anthropic Console
   - Ensure no extra spaces in key

## Migration Guide

### From API Key to OAuth

1. Install latest SDK:
   ```bash
   pip install -U claude-code-sdk
   ```

2. Login with OAuth:
   ```bash
   claude-auth login
   ```

3. Update code:
   ```python
   # Old
   from claude_code_sdk import query
   
   # New (OAuth)
   from claude_code_sdk import query_with_oauth
   ```

### Backward Compatibility

The original `query()` function still works with API keys:

```python
# Still works with ANTHROPIC_API_KEY
from claude_code_sdk import query

async for message in query(prompt="Hello"):
    print(message)
```

## Advanced Usage

### Custom Authentication Middleware

```python
from claude_code_sdk import (
    AuthenticationMiddleware,
    query_with_middleware,
    get_auth_headers
)

# Create custom auth middleware
auth_middleware = AuthenticationMiddleware(
    auth_headers_provider=get_auth_headers
)

async for message in query_with_middleware(
    prompt="Hello",
    middleware=[auth_middleware]
):
    print(message)
```

### Multiple Accounts

```python
from claude_code_sdk import TokenStorage, ClaudeAuth

# Use different storage paths for different accounts
personal_storage = TokenStorage(Path.home() / ".claude_code" / "personal.json")
work_storage = TokenStorage(Path.home() / ".claude_code" / "work.json")

# Switch between accounts
async with ClaudeAuth(token_storage=personal_storage) as auth:
    # Use personal account
    pass
```

## API Reference

### Functions

- `login()`: Perform OAuth login
- `logout()`: Clear stored tokens
- `get_auth_headers()`: Get current auth headers
- `query_with_oauth()`: Query using OAuth
- `query_with_api_key()`: Query using API key
- `authenticated_query()`: Query with flexible auth

### Classes

- `ClaudeAuth`: Main authentication handler
- `OAuthConfig`: OAuth configuration
- `OAuthFlow`: OAuth flow implementation
- `AuthToken`: Authentication token
- `TokenStorage`: Token persistence
- `AuthenticationError`: Auth-specific errors

### CLI Commands

- `claude-auth login`: OAuth login
- `claude-auth logout`: Clear tokens
- `claude-auth status`: Check auth status
- `claude-auth refresh`: Refresh token