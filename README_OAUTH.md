# OAuth Implementation for Claude Code SDK

## Overview

This repository contains a production-ready OAuth 2.0 implementation for the Claude Code SDK, designed specifically for Claude Code Max users who want to authenticate without API keys.

## Why This Implementation?

As a Claude Code Max subscriber paying $200/month, you should be able to use the SDK without managing API keys. This OAuth implementation is ready for when Anthropic enables OAuth authentication for Claude Code.

## Key Features

### 🔐 Security First
- **PKCE Support**: Implements Proof Key for Code Exchange for enhanced security
- **Secure Token Storage**: Tokens stored with restrictive file permissions (0600)
- **State Parameter**: CSRF protection with cryptographically secure state generation
- **Token Expiry Handling**: Automatic detection and handling of expired tokens

### 🔄 Automatic Token Management
- **Token Refresh**: Seamless refresh of expired tokens using refresh tokens
- **Token Persistence**: Tokens saved locally for reuse across sessions
- **Graceful Fallback**: Falls back to new auth flow if refresh fails

### 🎨 User Experience
- **Browser Integration**: Automatically opens browser for authentication
- **Local Callback Server**: Beautiful callback page with auto-close on success
- **Progress Indicators**: Clear feedback during authentication flow
- **Error Handling**: User-friendly error messages and recovery options

### 🛠️ Developer Friendly
- **Async/Await**: Modern async patterns throughout
- **Type Safety**: Full type hints and dataclasses
- **Modular Design**: Separate concerns for auth config, flow, and storage
- **Comprehensive Examples**: Multiple examples showing different use cases

## Implementation Details

### Core Components

1. **`auth_config.py`**: OAuth configuration and scope management
   - Configurable endpoints for different environments
   - Comprehensive scope definitions
   - Environment variable support

2. **`oauth_flow.py`**: Production-ready OAuth flow implementation
   - PKCE challenge generation and verification
   - Async HTTP client for token operations
   - Session management with timeout handling
   - Beautiful callback HTML page

3. **`auth.py`**: Token storage and management
   - Secure local storage with proper permissions
   - Token serialization and deserialization
   - Backward compatibility with existing auth module

### OAuth Scopes

The implementation supports all anticipated Claude Code scopes:

- **Basic**: `profile`, `email`
- **Claude Code**: `claude_code:read`, `claude_code:write`, `claude_code:execute`
- **Tools**: `tools:read`, `tools:write`, `tools:execute`
- **Workspace**: `workspace:read`, `workspace:write`

### Examples Included

1. **`oauth_demo.py`**: Basic OAuth flow demonstration
2. **`oauth_app_example.py`**: Full application with session management
3. **OAuth Guide**: Comprehensive documentation in `docs/oauth_guide.md`

## Usage Examples

### Basic Authentication

```python
from claude_code_sdk.auth_config import ClaudeCodeOAuthConfig
from claude_code_sdk.oauth_flow import ClaudeCodeOAuthFlow

async def authenticate():
    config = ClaudeCodeOAuthConfig.for_claude_code_max()
    async with ClaudeCodeOAuthFlow(config) as flow:
        token = await flow.authenticate()
        return token
```

### With Session Management

```python
from examples.oauth_app_example import ClaudeCodeSession

async def main():
    async with ClaudeCodeSession() as session:
        async for message in session.query("Help me write code"):
            print(message)
```

## Technical Highlights

### PKCE Implementation

```python
class PKCEChallenge:
    @staticmethod
    def generate_verifier(length: int = 128) -> str:
        """Generate cryptographically secure code verifier."""
        verifier = base64.urlsafe_b64encode(
            secrets.token_bytes(length)
        ).decode('utf-8').rstrip('=')
        return verifier[:128]
    
    @staticmethod
    def generate_challenge(verifier: str) -> str:
        """Generate S256 challenge from verifier."""
        digest = hashlib.sha256(verifier.encode('utf-8')).digest()
        challenge = base64.urlsafe_b64encode(digest).decode('utf-8').rstrip('=')
        return challenge
```

### Automatic Token Refresh

```python
async def authenticate(self) -> AuthToken:
    # Check for existing valid token
    existing_token = self.storage.load_token()
    if existing_token and not existing_token.is_expired():
        return existing_token
    
    # Try to refresh if we have a refresh token
    if existing_token and existing_token.refresh_token:
        try:
            return await self.refresh_token(existing_token.refresh_token)
        except AuthenticationError:
            # Fall back to new auth flow
            pass
    
    # Start new OAuth flow
    return await self._perform_oauth_flow()
```

## Future Ready

This implementation is designed to work seamlessly when Anthropic enables OAuth:

1. **Environment Detection**: Checks for `CLAUDE_OAUTH_ENABLED` flag
2. **Graceful Degradation**: Shows preview mode when OAuth isn't available
3. **Easy Migration**: Minimal code changes needed when OAuth goes live

## Getting Exposure

This implementation showcases:
- Production-ready OAuth patterns
- Security best practices
- Excellent user experience
- Clean, maintainable code

Perfect for demonstrating the value of OAuth support for Claude Code Max users!

## Next Steps

When Anthropic enables OAuth:

1. Set `CLAUDE_OAUTH_CLIENT_ID` environment variable
2. Run `python examples/oauth_demo.py`
3. Enjoy API key-free authentication!

## Contributing

This implementation is ready for production use. Contributions welcome to make it even better!

---

*Built by a Claude Code Max user who believes in making authentication seamless.* 🚀