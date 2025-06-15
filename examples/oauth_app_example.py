"""
Advanced OAuth Application Example

This example demonstrates building a full-featured application
using the Claude Code SDK with OAuth authentication.
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import AsyncGenerator, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_sdk import query, ClaudeCodeOptions
from claude_code_sdk.types import Message, AssistantMessage, TextBlock
from claude_code_sdk.auth_config import ClaudeCodeOAuthConfig, OAuthScopes
from claude_code_sdk.oauth_flow import ClaudeCodeOAuthFlow
from claude_code_sdk.auth import TokenStorage, AuthToken, AuthenticationError


class ClaudeCodeSession:
    """Manages an authenticated Claude Code session."""
    
    def __init__(self, config: Optional[ClaudeCodeOAuthConfig] = None):
        """Initialize session with optional config."""
        self.config = config or ClaudeCodeOAuthConfig.for_claude_code_max()
        self.storage = TokenStorage()
        self.token: Optional[AuthToken] = None
        self.session_id: Optional[str] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.authenticate()
        return self
        
    async def __aexit__(self, *args):
        """Async context manager exit."""
        # Could implement cleanup here if needed
        pass
        
    async def authenticate(self) -> AuthToken:
        """Ensure user is authenticated."""
        # Check for existing valid token
        existing_token = self.storage.load_token()
        if existing_token and not existing_token.is_expired():
            self.token = existing_token
            return existing_token
            
        # Perform OAuth flow
        async with ClaudeCodeOAuthFlow(self.config, self.storage) as flow:
            self.token = await flow.authenticate()
            return self.token
            
    async def query(
        self,
        prompt: str,
        options: Optional[ClaudeCodeOptions] = None
    ) -> AsyncGenerator[Message, None]:
        """Execute a query with authentication."""
        if not self.token:
            await self.authenticate()
            
        # Merge options with session defaults
        if options is None:
            options = ClaudeCodeOptions()
            
        # Continue conversation if we have a session
        if self.session_id and not options.resume:
            options.resume = self.session_id
            
        # Execute query
        async for message in query(prompt, options=options):
            # Track session ID for conversation continuity
            if hasattr(message, 'session_id'):
                self.session_id = message.session_id
            yield message
            
    async def logout(self):
        """Logout and cleanup."""
        if self.token:
            async with ClaudeCodeOAuthFlow(self.config, self.storage) as flow:
                # Revoke tokens
                try:
                    await flow.revoke_token(self.token.access_token)
                    if self.token.refresh_token:
                        await flow.revoke_token(
                            self.token.refresh_token,
                            token_type="refresh_token"
                        )
                except Exception:
                    # Ignore revocation errors
                    pass
                    
        # Clear storage
        self.storage.delete_token()
        self.token = None
        self.session_id = None


class ClaudeCodeApplication:
    """Example application using Claude Code SDK with OAuth."""
    
    def __init__(self):
        """Initialize application."""
        self.session: Optional[ClaudeCodeSession] = None
        self.conversation_history: list[dict] = []
        
    async def start(self):
        """Start the application."""
        print("=" * 60)
        print("Claude Code OAuth Application")
        print("=" * 60)
        print()
        
        # Check if OAuth is enabled
        if not self._is_oauth_enabled():
            print("⚠️  OAuth is not yet enabled for Claude Code.")
            print("This example demonstrates what OAuth integration will look like.")
            print()
            print("For now, use API key authentication:")
            print("  export ANTHROPIC_API_KEY='your-api-key'")
            print()
            return
            
        # Create session
        self.session = ClaudeCodeSession()
        
        try:
            # Authenticate
            print("🔐 Authenticating...")
            token = await self.session.authenticate()
            print(f"✅ Authenticated successfully!")
            print(f"   Token expires: {token.expires_at}")
            print()
            
            # Run interactive loop
            await self.interactive_loop()
            
        except AuthenticationError as e:
            print(f"❌ Authentication failed: {e}")
        except KeyboardInterrupt:
            print("\n\nExiting...")
        finally:
            # Cleanup
            if self.session:
                await self.session.logout()
                print("✅ Logged out successfully")
                
    def _is_oauth_enabled(self) -> bool:
        """Check if OAuth is enabled."""
        return os.getenv("CLAUDE_OAUTH_ENABLED", "false").lower() == "true"
        
    async def interactive_loop(self):
        """Run interactive query loop."""
        print("💬 Chat with Claude Code (type 'exit' to quit)")
        print("-" * 60)
        
        while True:
            # Get user input
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
                
            if user_input.lower() in ['exit', 'quit', 'bye']:
                break
                
            if not user_input:
                continue
                
            # Special commands
            if user_input.startswith('/'):
                await self.handle_command(user_input)
                continue
                
            # Query Claude
            print("\nClaude: ", end="", flush=True)
            
            full_response = []
            async for message in self.session.query(user_input):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(block.text, end="", flush=True)
                            full_response.append(block.text)
                            
            print()  # New line after response
            
            # Save to history
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "user": user_input,
                "assistant": "".join(full_response)
            })
            
    async def handle_command(self, command: str):
        """Handle special commands."""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == '/help':
            print("""
Available commands:
  /help              - Show this help
  /history           - Show conversation history
  /clear             - Clear conversation history
  /session           - Show session info
  /tools [on|off]    - Enable/disable tools
  /scopes            - Show OAuth scopes
  /export [file]     - Export conversation to file
""")
        
        elif cmd == '/history':
            if not self.conversation_history:
                print("No conversation history yet.")
            else:
                for i, entry in enumerate(self.conversation_history, 1):
                    print(f"\n--- Exchange {i} ({entry['timestamp']}) ---")
                    print(f"You: {entry['user']}")
                    print(f"Claude: {entry['assistant'][:100]}...")
                    
        elif cmd == '/clear':
            self.conversation_history.clear()
            self.session.session_id = None
            print("Conversation history cleared.")
            
        elif cmd == '/session':
            if self.session and self.session.token:
                print(f"Session ID: {self.session.session_id or 'New session'}")
                print(f"Token Type: {self.session.token.token_type}")
                print(f"Expires: {self.session.token.expires_at}")
                print(f"Scopes: {self.session.token.scope}")
            else:
                print("No active session.")
                
        elif cmd == '/tools':
            if len(parts) > 1:
                enable = parts[1].lower() == 'on'
                # This would configure tool usage in real implementation
                print(f"Tools {'enabled' if enable else 'disabled'}.")
            else:
                print("Usage: /tools [on|off]")
                
        elif cmd == '/scopes':
            print("Available OAuth Scopes:")
            print(f"  - {OAuthScopes.PROFILE}")
            print(f"  - {OAuthScopes.EMAIL}")
            print(f"  - {OAuthScopes.CODE_READ}")
            print(f"  - {OAuthScopes.CODE_WRITE}")
            print(f"  - {OAuthScopes.CODE_EXECUTE}")
            print(f"  - {OAuthScopes.TOOLS_READ}")
            print(f"  - {OAuthScopes.TOOLS_WRITE}")
            print(f"  - {OAuthScopes.TOOLS_EXECUTE}")
            print(f"  - {OAuthScopes.WORKSPACE_READ}")
            print(f"  - {OAuthScopes.WORKSPACE_WRITE}")
            
        elif cmd == '/export':
            filename = parts[1] if len(parts) > 1 else "conversation.json"
            self.export_conversation(filename)
            
        else:
            print(f"Unknown command: {cmd}. Type /help for help.")
            
    def export_conversation(self, filename: str):
        """Export conversation history to file."""
        import json
        
        try:
            with open(filename, 'w') as f:
                json.dump({
                    "exported_at": datetime.now().isoformat(),
                    "conversation": self.conversation_history
                }, f, indent=2)
            print(f"Conversation exported to {filename}")
        except Exception as e:
            print(f"Export failed: {e}")


class AdvancedOAuthExample:
    """Advanced OAuth examples and patterns."""
    
    @staticmethod
    async def multi_session_example():
        """Example of managing multiple Claude sessions."""
        # Create sessions with different configurations
        sessions = {
            "general": ClaudeCodeSession(
                ClaudeCodeOAuthConfig(
                    scopes=[OAuthScopes.CODE_READ, OAuthScopes.CODE_WRITE]
                )
            ),
            "tools": ClaudeCodeSession(
                ClaudeCodeOAuthConfig(
                    scopes=OAuthScopes.full_access()
                )
            )
        }
        
        # Use different sessions for different purposes
        async with sessions["general"] as general:
            async for msg in general.query("Explain Python decorators"):
                print(msg)
                
        async with sessions["tools"] as tools:
            async for msg in tools.query("Create a data processing tool"):
                print(msg)
                
    @staticmethod
    async def token_management_example():
        """Example of advanced token management."""
        config = ClaudeCodeOAuthConfig.for_claude_code_max()
        storage = TokenStorage()
        
        async with ClaudeCodeOAuthFlow(config, storage) as flow:
            # Get token with automatic refresh
            token = await flow.authenticate()
            
            # Check token status
            print(f"Token valid: {not token.is_expired()}")
            print(f"Expires at: {token.expires_at}")
            
            # Get user info
            try:
                user_info = await flow.get_user_info(token.access_token)
                print(f"User: {user_info}")
            except Exception as e:
                print(f"User info not available: {e}")
                
            # Manual refresh if needed
            if token.refresh_token:
                new_token = await flow.refresh_token(token.refresh_token)
                print(f"Token refreshed, new expiry: {new_token.expires_at}")
                
    @staticmethod
    async def error_handling_example():
        """Example of comprehensive error handling."""
        session = ClaudeCodeSession()
        
        try:
            await session.authenticate()
        except AuthenticationError as e:
            # Handle auth errors
            print(f"Authentication failed: {e}")
            
            # Could implement retry logic
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    print(f"Retrying... ({retry_count + 1}/{max_retries})")
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                    await session.authenticate()
                    break
                except AuthenticationError:
                    retry_count += 1
                    
            if retry_count >= max_retries:
                print("Max retries exceeded. Please check your configuration.")


async def main():
    """Run the example application."""
    # Check command line arguments
    if len(sys.argv) > 1:
        example = sys.argv[1].lower()
        
        if example == "app":
            # Run full application
            app = ClaudeCodeApplication()
            await app.start()
            
        elif example == "multi":
            # Run multi-session example
            await AdvancedOAuthExample.multi_session_example()
            
        elif example == "token":
            # Run token management example
            await AdvancedOAuthExample.token_management_example()
            
        elif example == "error":
            # Run error handling example
            await AdvancedOAuthExample.error_handling_example()
            
        else:
            print(f"Unknown example: {example}")
            print("Available examples: app, multi, token, error")
    else:
        # Default: run the application
        app = ClaudeCodeApplication()
        await app.start()


if __name__ == "__main__":
    print("""
Claude Code OAuth Application Example

This example demonstrates building applications with OAuth authentication.

Usage:
  python oauth_app_example.py         # Run interactive application
  python oauth_app_example.py app     # Run interactive application
  python oauth_app_example.py multi   # Multi-session example
  python oauth_app_example.py token   # Token management example
  python oauth_app_example.py error   # Error handling example

Note: OAuth support is coming soon for Claude Code Max users.
""")
    
    asyncio.run(main())