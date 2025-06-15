"""Tool execution hooks for integrating with Tool Management API."""

from collections.abc import Callable
from typing import Any

from ..types import ClaudeCodeOptions, ToolResultBlock, ToolUseBlock
from .client import ToolManagementClient


class ToolExecutionHooks:
    """Hooks for intercepting and handling tool execution."""

    def __init__(
        self,
        tool_api_url: str | None = None,
        on_tool_use: Callable[[ToolUseBlock], None] | None = None,
        on_tool_result: Callable[[ToolResultBlock], None] | None = None,
        use_remote_tools: bool = False,
    ):
        """Initialize tool execution hooks.

        Args:
            tool_api_url: Optional URL for Tool Management API
            on_tool_use: Optional callback when a tool is used
            on_tool_result: Optional callback when a tool returns a result
            use_remote_tools: Whether to execute tools via the API
        """
        self.tool_api_url = tool_api_url or "https://arthurcolle--registry.modal.run"
        self.on_tool_use = on_tool_use
        self.on_tool_result = on_tool_result
        self.use_remote_tools = use_remote_tools
        self._client: ToolManagementClient | None = None

    async def __aenter__(self) -> "ToolExecutionHooks":
        """Enter async context."""
        if self.use_remote_tools:
            self._client = ToolManagementClient(self.tool_api_url)
            await self._client.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context."""
        if self._client:
            await self._client.__aexit__(*args)

    async def handle_tool_use(self, tool_use: ToolUseBlock) -> ToolResultBlock | None:
        """Handle a tool use block, optionally executing it remotely.

        Args:
            tool_use: The tool use block from Claude

        Returns:
            Optional tool result if executed remotely
        """
        # Call the hook if provided
        if self.on_tool_use:
            self.on_tool_use(tool_use)

        # Execute remotely if enabled
        if self.use_remote_tools and self._client:
            try:
                # Search for the tool first
                tools = await self._client.search_tools(tool_use.name, limit=1)

                if tools:
                    # Execute the tool
                    result = await self._client.execute_tool(
                        tool_id=tools[0].id, input_data=tool_use.input
                    )

                    # Convert to tool result block
                    return ToolResultBlock(
                        tool_use_id=tool_use.id, content=str(result.output_data)
                    )
                else:
                    # Tool not found in registry
                    return ToolResultBlock(
                        tool_use_id=tool_use.id,
                        content=f"Tool '{tool_use.name}' not found in registry",
                        is_error=True,
                    )

            except Exception as e:
                # Return error result
                return ToolResultBlock(
                    tool_use_id=tool_use.id,
                    content=f"Error executing tool: {str(e)}",
                    is_error=True,
                )

        return None

    async def handle_tool_result(self, tool_result: ToolResultBlock) -> None:
        """Handle a tool result block.

        Args:
            tool_result: The tool result block
        """
        if self.on_tool_result:
            self.on_tool_result(tool_result)


class ToolRegistryOptions(ClaudeCodeOptions):
    """Extended options with tool registry support."""

    def __init__(
        self,
        *args: Any,
        tool_api_url: str | None = None,
        use_remote_tools: bool = False,
        tool_namespace: str | None = None,
        **kwargs: Any,
    ):
        """Initialize with tool registry options.

        Args:
            tool_api_url: URL for Tool Management API
            use_remote_tools: Whether to use remote tool execution
            tool_namespace: Namespace for tool lookups
            *args, **kwargs: Passed to parent class
        """
        super().__init__(*args, **kwargs)
        self.tool_api_url = tool_api_url
        self.use_remote_tools = use_remote_tools
        self.tool_namespace = tool_namespace
