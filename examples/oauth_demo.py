"""
Claude Code SDK OAuth Demo

This demo showcases the OAuth implementation for Claude Code Max users.
Currently, this is a preview of what will be available when Anthropic
enables OAuth authentication.

To run this demo:
1. Set CLAUDE_OAUTH_CLIENT_ID environment variable (when available)
2. Run: python examples/oauth_demo.py
"""

import asyncio
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_sdk.auth_config import ClaudeCodeOAuthConfig, OAuthScopes
from claude_code_sdk.oauth_flow import ClaudeCodeOAuthFlow
from claude_code_sdk.auth import TokenStorage


async def demo_oauth_flow():
    """Demonstrate the OAuth authentication flow."""
    print("=" * 60)
    print("Claude Code SDK - OAuth Authentication Demo")
    print("=" * 60)
    print()
    
    # Check if OAuth is available (will be enabled by Anthropic)
    oauth_enabled = os.getenv("CLAUDE_OAUTH_ENABLED", "false").lower() == "true"
    
    if not oauth_enabled:
        print("⚠️  OAuth is not yet enabled for Claude Code.")
        print("This demo shows what the OAuth flow will look like when available.")
        print()
        print("For now, Claude Code uses API key authentication:")
        print("  export ANTHROPIC_API_KEY='your-api-key'")
        print()
        print("OAuth support is coming soon for Claude Code Max users!")
        print()
        
        # Show what the configuration would look like
        print("OAuth Configuration Preview:")
        print("-" * 30)
        
        config = ClaudeCodeOAuthConfig.for_claude_code_max()
        print(f"Client ID: {config.client_id}")
        print(f"Redirect URI: {config.redirect_uri}")
        print(f"Scopes: {', '.join(config.scopes)}")
        print(f"PKCE Enabled: {config.use_pkce}")
        print()
        
        return
    
    # When OAuth is enabled, this is how it will work:
    print("🚀 Starting OAuth Authentication Flow")
    print()
    
    # Create OAuth configuration
    config = ClaudeCodeOAuthConfig.for_claude_code_max()
    
    # Initialize token storage
    storage = TokenStorage()
    
    # Check for existing token
    existing_token = storage.load_token()
    if existing_token:
        print("📌 Found existing token")
        print(f"   Token Type: {existing_token.token_type}")
        print(f"   Expires At: {existing_token.expires_at}")
        print(f"   Is Expired: {existing_token.is_expired()}")
        print()
    
    # Create OAuth flow
    async with ClaudeCodeOAuthFlow(config, storage) as flow:
        try:
            # Authenticate
            print("🔐 Authenticating...")
            token = await flow.authenticate()
            
            print()
            print("✅ Authentication Successful!")
            print("-" * 30)
            print(f"Access Token: {token.access_token[:20]}...")
            print(f"Token Type: {token.token_type}")
            print(f"Expires At: {token.expires_at}")
            print(f"Scopes: {token.scope}")
            
            # Get user info (when available)
            try:
                print()
                print("👤 Fetching User Information...")
                user_info = await flow.get_user_info(token.access_token)
                print(f"   User ID: {user_info.get('id', 'N/A')}")
                print(f"   Email: {user_info.get('email', 'N/A')}")
                print(f"   Plan: {user_info.get('plan', 'N/A')}")
            except Exception as e:
                print(f"   User info not available: {e}")
            
        except Exception as e:
            print(f"\n❌ Authentication failed: {e}")
            return
    
    print()
    print("🎉 OAuth flow completed successfully!")
    print()
    print("You can now use the Claude Code SDK with your authenticated session.")


async def demo_token_management():
    """Demonstrate token management features."""
    print()
    print("=" * 60)
    print("Token Management Demo")
    print("=" * 60)
    print()
    
    storage = TokenStorage()
    token = storage.load_token()
    
    if not token:
        print("No stored token found.")
        return
    
    print("📊 Token Status:")
    print(f"   Created: {datetime.now()}")
    print(f"   Expires: {token.expires_at}")
    print(f"   Time Remaining: {token.expires_at - datetime.now() if token.expires_at else 'No expiry'}")
    print(f"   Has Refresh Token: {bool(token.refresh_token)}")
    
    # Show token refresh (when OAuth is enabled)
    if token.refresh_token and token.is_expired():
        print()
        print("🔄 Token expired, attempting refresh...")
        
        config = ClaudeCodeOAuthConfig.for_claude_code_max()
        async with ClaudeCodeOAuthFlow(config, storage) as flow:
            try:
                new_token = await flow.refresh_token(token.refresh_token)
                print("✅ Token refreshed successfully!")
                print(f"   New Expiry: {new_token.expires_at}")
            except Exception as e:
                print(f"❌ Refresh failed: {e}")


async def demo_scope_management():
    """Demonstrate OAuth scope management."""
    print()
    print("=" * 60)
    print("OAuth Scope Management Demo")
    print("=" * 60)
    print()
    
    print("📋 Available OAuth Scopes:")
    print()
    
    # Basic scopes
    print("Basic Scopes:")
    print(f"  - {OAuthScopes.PROFILE}: Access user profile information")
    print(f"  - {OAuthScopes.EMAIL}: Access user email address")
    print()
    
    # Claude Code scopes
    print("Claude Code Scopes:")
    print(f"  - {OAuthScopes.CODE_READ}: Read code and conversations")
    print(f"  - {OAuthScopes.CODE_WRITE}: Create and modify code")
    print(f"  - {OAuthScopes.CODE_EXECUTE}: Execute code and tools")
    print()
    
    # Tool scopes
    print("Tool Management Scopes:")
    print(f"  - {OAuthScopes.TOOLS_READ}: Read available tools")
    print(f"  - {OAuthScopes.TOOLS_WRITE}: Create and modify tools")
    print(f"  - {OAuthScopes.TOOLS_EXECUTE}: Execute tools")
    print()
    
    # Workspace scopes
    print("Workspace Scopes:")
    print(f"  - {OAuthScopes.WORKSPACE_READ}: Read workspace files")
    print(f"  - {OAuthScopes.WORKSPACE_WRITE}: Modify workspace files")
    print()
    
    # Show different configurations
    print("Scope Configurations:")
    print(f"  Default: {', '.join(OAuthScopes.default())}")
    print(f"  Full Access: {', '.join(OAuthScopes.full_access())}")


async def main():
    """Run all demos."""
    # OAuth flow demo
    await demo_oauth_flow()
    
    # Only run other demos if OAuth is enabled
    if os.getenv("CLAUDE_OAUTH_ENABLED", "false").lower() == "true":
        await demo_token_management()
        await demo_scope_management()
    
    print()
    print("=" * 60)
    print("Demo completed!")
    print()
    print("📚 Next Steps:")
    print("1. Wait for Anthropic to enable OAuth for Claude Code Max")
    print("2. Set your CLAUDE_OAUTH_CLIENT_ID environment variable")
    print("3. Use query_with_oauth() for authenticated requests")
    print("4. Enjoy API key-free authentication! 🎉")
    print()


if __name__ == "__main__":
    asyncio.run(main())