"""Example of using OAuth authentication with Claude Code SDK.

This example demonstrates how Claude Code Max plan users can authenticate
without needing an API key.
"""

import asyncio
from claude_code_sdk import (
    query_with_oauth,
    authenticated_query,
    login,
    logout,
    ClaudeAuth,
    OAuthConfig,
)


async def example_oauth_login():
    """Example: Perform OAuth login."""
    print("=== OAuth Login Example ===")
    
    # This will open a browser for authentication
    await login()
    print("You are now authenticated!")


async def example_oauth_query():
    """Example: Query using OAuth authentication."""
    print("\n=== OAuth Query Example ===")
    
    # Use OAuth authentication (default for Claude Code Max)
    async for message in query_with_oauth(
        prompt="What files are in the current directory?"
    ):
        print(message)


async def example_authenticated_query():
    """Example: Flexible authentication query."""
    print("\n=== Authenticated Query Example ===")
    
    # This will use OAuth if available, otherwise fall back to API key
    async for message in authenticated_query(
        prompt="Create a simple hello world Python script",
        use_oauth=True,  # Try OAuth first
    ):
        print(message)


async def example_api_key_fallback():
    """Example: Using API key authentication."""
    print("\n=== API Key Authentication Example ===")
    
    # Explicitly use API key authentication
    async for message in authenticated_query(
        prompt="What is 2 + 2?",
        use_oauth=False,
        api_key="your-api-key-here",  # Or use ANTHROPIC_API_KEY env var
    ):
        print(message)


async def example_custom_oauth_config():
    """Example: Custom OAuth configuration."""
    print("\n=== Custom OAuth Config Example ===")
    
    # Create custom OAuth configuration
    custom_config = OAuthConfig(
        client_id="your-client-id",
        client_secret="your-client-secret",
        redirect_uri="http://localhost:8089/callback",
    )
    
    # Use custom config for authentication
    async with ClaudeAuth(
        use_oauth=True,
        oauth_config=custom_config
    ) as auth:
        # Get authentication headers
        headers = await auth.authenticate()
        print(f"Auth headers: {headers}")


async def example_check_auth_status():
    """Example: Check authentication status."""
    print("\n=== Check Auth Status Example ===")
    
    async with ClaudeAuth(use_oauth=True) as auth:
        token = auth.token_storage.load_token()
        
        if token:
            print("✅ Authenticated")
            if token.is_expired():
                print("⚠️  Token is expired, will refresh automatically")
            else:
                print("✅ Token is valid")
                if token.expires_at:
                    print(f"   Expires at: {token.expires_at}")
        else:
            print("❌ Not authenticated. Run 'await login()' to authenticate.")


async def example_logout():
    """Example: Logout and clear tokens."""
    print("\n=== Logout Example ===")
    
    await logout()
    print("Logged out successfully!")


async def main():
    """Run all examples."""
    print("Claude Code SDK OAuth Authentication Examples")
    print("=" * 50)
    
    # Check current authentication status
    await example_check_auth_status()
    
    # Uncomment to run login flow
    # await example_oauth_login()
    
    # Try OAuth query (will prompt for login if not authenticated)
    try:
        await example_oauth_query()
    except Exception as e:
        print(f"OAuth query failed: {e}")
        print("Please run 'claude-auth login' or uncomment the login example")
    
    # Show other examples (without running them)
    print("\n" + "=" * 50)
    print("Other examples available in this file:")
    print("- example_authenticated_query(): Flexible auth with fallback")
    print("- example_api_key_fallback(): Use API key authentication")
    print("- example_custom_oauth_config(): Custom OAuth settings")
    print("- example_logout(): Clear authentication")
    
    print("\n" + "=" * 50)
    print("\nCLI Authentication Commands:")
    print("- claude-auth login    : Perform OAuth login")
    print("- claude-auth logout   : Clear stored tokens")
    print("- claude-auth status   : Check authentication status")
    print("- claude-auth refresh  : Refresh expired token")


if __name__ == "__main__":
    asyncio.run(main())