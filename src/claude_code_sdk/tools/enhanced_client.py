"""Enhanced Claude Code client with Tool Management API integration."""

from collections.abc import AsyncIterator

from .._internal.client import InternalClient
from ..types import (
    AssistantMessage,
    ClaudeCodeOptions,
    ContentBlock,
    Message,
    ToolUseBlock,
)
from .hooks import ToolExecutionHooks, ToolRegistryOptions


class EnhancedClient(InternalClient):
    """Enhanced client with tool registry integration."""

    def __init__(self, tool_hooks: ToolExecutionHooks | None = None) -> None:
        """Initialize enhanced client.

        Args:
            tool_hooks: Optional tool execution hooks
        """
        super().__init__()
        self.tool_hooks = tool_hooks

    async def process_query(
        self, prompt: str, options: ClaudeCodeOptions
    ) -> AsyncIterator[Message]:
        """Process query with optional tool interception.

        Args:
            prompt: User prompt
            options: Query options

        Yields:
            Messages from the conversation
        """
        # Use tool hooks if provided and options support it
        if self.tool_hooks and isinstance(options, ToolRegistryOptions):
            async with self.tool_hooks:
                async for message in super().process_query(prompt, options):
                    # Intercept assistant messages to handle tool use
                    if isinstance(message, AssistantMessage):
                        modified_content: list[ContentBlock] = []

                        for block in message.content:
                            if isinstance(block, ToolUseBlock):
                                # Try to execute remotely
                                result = await self.tool_hooks.handle_tool_use(block)

                                # Add both the original tool use and any result
                                modified_content.append(block)
                                if result:
                                    modified_content.append(result)
                                    await self.tool_hooks.handle_tool_result(result)
                            else:
                                modified_content.append(block)

                        # Yield modified message
                        yield AssistantMessage(content=modified_content)
                    else:
                        yield message
        else:
            # Normal processing without hooks
            async for message in super().process_query(prompt, options):
                yield message


async def query_with_tools(
    *,
    prompt: str,
    options: ToolRegistryOptions | None = None,
    tool_api_url: str | None = None,
    use_remote_tools: bool = False,
) -> AsyncIterator[Message]:
    """Query Claude Code with Tool Management API integration.

    Args:
        prompt: The prompt to send to Claude
        options: Optional configuration with tool registry support
        tool_api_url: URL for Tool Management API
        use_remote_tools: Whether to execute tools via the API

    Yields:
        Messages from the conversation

    Example:
        ```python
        async for message in query_with_tools(
            prompt="Find and execute a calculator tool",
            use_remote_tools=True
        ):
            print(message)
        ```
    """
    if options is None:
        options = ToolRegistryOptions()

    # Set tool API options
    if tool_api_url:
        options.tool_api_url = tool_api_url
    if use_remote_tools:
        options.use_remote_tools = use_remote_tools

    # Create hooks if remote tools are enabled
    tool_hooks = None
    if options.use_remote_tools:
        tool_hooks = ToolExecutionHooks(
            tool_api_url=options.tool_api_url, use_remote_tools=True
        )

    # Use enhanced client
    client = EnhancedClient(tool_hooks=tool_hooks)

    async for message in client.process_query(prompt=prompt, options=options):
        yield message
