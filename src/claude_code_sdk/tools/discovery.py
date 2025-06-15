"""Tool discovery and search utilities."""

from .client import ToolManagementClient
from .types import Tool, ToolSearchResponse


class ToolDiscovery:
    """Utilities for discovering and searching tools."""

    def __init__(self, client: ToolManagementClient):
        """Initialize tool discovery.

        Args:
            client: Tool Management API client
        """
        self.client = client

    async def search_by_capability(
        self, capability: str, limit: int = 10
    ) -> list[ToolSearchResponse]:
        """Search tools by capability description.

        Args:
            capability: Description of desired capability
            limit: Maximum number of results

        Returns:
            List of matching tools
        """
        return await self.client.search_tools(capability, limit=limit)

    async def search_by_namespace(self, namespace: str) -> list[ToolSearchResponse]:
        """Get all tools in a namespace.

        Args:
            namespace: Tool namespace

        Returns:
            List of tools in namespace
        """
        # Get all tools and filter by namespace
        all_tools = await self.client.get_tools(namespace=namespace)

        # Convert to search responses (without scores)
        return [
            ToolSearchResponse(
                id=tool.id,
                name=tool.name,
                namespace=tool.namespace,
                description=tool.description,
                input_schema=tool.input_schema,
                output_schema=tool.output_schema,
                action=tool.action,
                output=tool.output,
            )
            for tool in all_tools
        ]

    async def find_similar_tools(
        self, tool_name: str, limit: int = 5
    ) -> list[ToolSearchResponse]:
        """Find tools similar to a given tool.

        Args:
            tool_name: Name of reference tool
            limit: Maximum number of results

        Returns:
            List of similar tools
        """
        # Use vector similarity search
        return await self.client.retrieve_tools(
            tool_name, limit=limit, strategy="vector_similarity"
        )

    async def suggest_tools_for_task(
        self, task_description: str, limit: int = 5
    ) -> list[ToolSearchResponse]:
        """Suggest tools for a given task.

        Args:
            task_description: Description of the task
            limit: Maximum number of suggestions

        Returns:
            List of suggested tools
        """
        # Use vector similarity to find relevant tools
        return await self.client.retrieve_tools(
            task_description, limit=limit, strategy="vector_similarity"
        )

    async def get_tool_by_name(
        self, name: str, namespace: str | None = None
    ) -> ToolSearchResponse | None:
        """Get a specific tool by name.

        Args:
            name: Tool name
            namespace: Optional namespace filter

        Returns:
            Tool if found, None otherwise
        """
        # Search for exact name match
        results = await self.client.search_tools(name, limit=10)

        for tool in results:
            if tool.name == name:
                if namespace is None or tool.namespace == namespace:
                    return tool

        return None

    async def list_tool_categories(self) -> dict[str, list[str]]:
        """List tools organized by category/namespace.

        Returns:
            Dictionary mapping namespaces to tool names
        """
        all_tools = await self.client.get_tools()

        categories: dict[str, list[str]] = {}
        for tool in all_tools:
            namespace = tool.namespace or "default"
            if namespace not in categories:
                categories[namespace] = []
            categories[namespace].append(tool.name)

        return categories


class ToolRegistry:
    """Registry for managing tool availability."""

    def __init__(self, client: ToolManagementClient):
        """Initialize tool registry.

        Args:
            client: Tool Management API client
        """
        self.client = client
        self.discovery = ToolDiscovery(client)
        self._cache: dict[str, ToolSearchResponse] = {}

    async def register_tool(self, tool: Tool) -> str:
        """Register a new tool.

        Args:
            tool: Tool to register

        Returns:
            Tool ID
        """
        response = await self.client.create_tool(tool)

        # Cache the tool
        self._cache[response.name] = ToolSearchResponse(
            id=response.id,
            name=response.name,
            namespace=response.namespace,
            description=response.description,
            input_schema=response.input_schema,
            output_schema=response.output_schema,
            action=response.action,
            output=response.output,
        )

        return response.id

    async def get_tool(self, name: str) -> ToolSearchResponse | None:
        """Get a tool by name, using cache if available.

        Args:
            name: Tool name

        Returns:
            Tool if found
        """
        # Check cache first
        if name in self._cache:
            return self._cache[name]

        # Search in registry
        tool = await self.discovery.get_tool_by_name(name)
        if tool:
            self._cache[name] = tool

        return tool

    async def refresh_cache(self) -> None:
        """Refresh the tool cache."""
        self._cache.clear()

        # Get all tools and cache them
        all_tools = await self.client.get_tools()
        for tool in all_tools:
            self._cache[tool.name] = ToolSearchResponse(
                id=tool.id,
                name=tool.name,
                namespace=tool.namespace,
                description=tool.description,
                input_schema=tool.input_schema,
                output_schema=tool.output_schema,
                action=tool.action,
                output=tool.output,
            )

    async def ensure_tools_available(self, tool_names: list[str]) -> dict[str, bool]:
        """Check if tools are available in the registry.

        Args:
            tool_names: List of tool names to check

        Returns:
            Dictionary mapping tool names to availability
        """
        availability = {}

        for name in tool_names:
            tool = await self.get_tool(name)
            availability[name] = tool is not None

        return availability
