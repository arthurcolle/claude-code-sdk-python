"""Authenticated query functions for Claude Code SDK."""

import os
from collections.abc import AsyncIterator

from ._internal.client import InternalClient
from .auth import ClaudeAuth, OAuthConfig
from .types import ClaudeCodeOptions, Message


async def authenticated_query(
    *,
    prompt: str,
    options: ClaudeCodeOptions | None = None,
    use_oauth: bool = True,
    api_key: str | None = None,
    oauth_config: OAuthConfig | None = None,
) -> AsyncIterator[Message]:
    """
    Query Claude Code with authentication support.

    Supports both OAuth (for Claude Code Max users) and API key authentication.

    Args:
        prompt: The prompt to send to Claude
        options: Optional configuration
        use_oauth: Use OAuth authentication (default True for Max plan)
        api_key: API key (if not using OAuth)
        oauth_config: Custom OAuth configuration

    Yields:
        Messages from the conversation

    Example:
        ```python
        # Using OAuth (Claude Code Max plan)
        async for message in authenticated_query(prompt="Hello"):
            print(message)

        # Using API key
        async for message in authenticated_query(
            prompt="Hello",
            use_oauth=False,
            api_key="your-api-key"
        ):
            print(message)
        ```
    """
    if options is None:
        options = ClaudeCodeOptions()

    # Set up authentication
    async with ClaudeAuth(
        use_oauth=use_oauth,
        api_key=api_key,
        oauth_config=oauth_config,
    ) as auth:
        # Get environment variables for CLI
        env_vars = auth.get_env_vars()

        # Update environment for subprocess
        original_env = {}
        for key, value in env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            # Ensure authenticated before starting
            if use_oauth:
                await auth.authenticate()

            # Set SDK entrypoint
            os.environ["CLAUDE_CODE_ENTRYPOINT"] = "sdk-py"

            # Use internal client
            client = InternalClient()

            async for message in client.process_query(prompt=prompt, options=options):
                yield message

        finally:
            # Restore original environment
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value


async def query_with_oauth(
    *,
    prompt: str,
    options: ClaudeCodeOptions | None = None,
    oauth_config: OAuthConfig | None = None,
) -> AsyncIterator[Message]:
    """
    Query Claude Code using OAuth authentication.

    This is the preferred method for Claude Code Max plan users.

    Args:
        prompt: The prompt to send to Claude
        options: Optional configuration
        oauth_config: Custom OAuth configuration

    Yields:
        Messages from the conversation

    Example:
        ```python
        async for message in query_with_oauth(prompt="Hello"):
            print(message)
        ```
    """
    async for message in authenticated_query(
        prompt=prompt,
        options=options,
        use_oauth=True,
        oauth_config=oauth_config,
    ):
        yield message


async def query_with_api_key(
    *,
    prompt: str,
    options: ClaudeCodeOptions | None = None,
    api_key: str | None = None,
) -> AsyncIterator[Message]:
    """
    Query Claude Code using API key authentication.

    Args:
        prompt: The prompt to send to Claude
        options: Optional configuration
        api_key: API key (uses ANTHROPIC_API_KEY env var if not provided)

    Yields:
        Messages from the conversation

    Example:
        ```python
        async for message in query_with_api_key(
            prompt="Hello",
            api_key="your-api-key"
        ):
            print(message)
        ```
    """
    async for message in authenticated_query(
        prompt=prompt,
        options=options,
        use_oauth=False,
        api_key=api_key,
    ):
        yield message
