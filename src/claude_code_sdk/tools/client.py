"""Tool Management API client."""

from typing import Any, Literal, cast

import httpx

from .types import (
    EmbeddingResponse,
    ParallelToolExecutionRequest,
    SequentialToolExecutionRequest,
    Tool,
    ToolBatchRequest,
    ToolExecutionRequest,
    ToolExecutionResponse,
    ToolIdentifier,
    ToolResponse,
    ToolSearchRequest,
    ToolSearchResponse,
)


class ToolManagementClient:
    """Client for Tool Management API."""

    def __init__(self, base_url: str = "https://arthurcolle--registry.modal.run"):
        """Initialize client.

        Args:
            base_url: Base URL for the Tool Management API
        """
        self.base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ToolManagementClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(base_url=self.base_url)
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        """Get HTTP client."""
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        return self._client

    async def health_check(self) -> dict[str, Any]:
        """Check API health."""
        response = await self.client.get("/health")
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    async def get_tools(self, namespace: str | None = None) -> list[ToolResponse]:
        """Get all tools.

        Args:
            namespace: Optional namespace filter

        Returns:
            List of tools
        """
        params = {}
        if namespace:
            params["namespace"] = namespace

        response = await self.client.get("/tools", params=params)
        response.raise_for_status()

        tools_data = response.json()
        return [ToolResponse(**tool) for tool in tools_data]

    async def get_tool(self, tool_id: str) -> ToolResponse:
        """Get a specific tool.

        Args:
            tool_id: Tool ID

        Returns:
            Tool details
        """
        response = await self.client.get(f"/tools/{tool_id}")
        response.raise_for_status()
        return ToolResponse(**response.json())

    async def create_tool(self, tool: Tool) -> ToolResponse:
        """Create a new tool.

        Args:
            tool: Tool to create

        Returns:
            Created tool
        """
        response = await self.client.post(
            "/tools",
            json={
                "name": tool.name,
                "namespace": tool.namespace,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
                "code": tool.code,
                "action": {
                    "type": tool.action.type.value if tool.action else None,
                    "http": tool.action.http if tool.action else None,
                    "python": tool.action.python if tool.action else None,
                    "javascript": tool.action.javascript if tool.action else None,
                    "service_config": tool.action.service_config
                    if tool.action
                    else None,
                }
                if tool.action
                else None,
                "output": {
                    "type": tool.output.type.value if tool.output else None,
                    "content": tool.output.content if tool.output else None,
                }
                if tool.output
                else None,
                "id": tool.id,
            },
        )
        response.raise_for_status()
        return ToolResponse(**response.json())

    async def update_tool(self, tool_id: str, tool: Tool) -> ToolResponse:
        """Update an existing tool.

        Args:
            tool_id: Tool ID to update
            tool: Updated tool data

        Returns:
            Updated tool
        """
        response = await self.client.put(
            f"/tools/{tool_id}",
            json={
                "name": tool.name,
                "namespace": tool.namespace,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
                "code": tool.code,
                "action": {
                    "type": tool.action.type.value if tool.action else None,
                    "http": tool.action.http if tool.action else None,
                    "python": tool.action.python if tool.action else None,
                    "javascript": tool.action.javascript if tool.action else None,
                    "service_config": tool.action.service_config
                    if tool.action
                    else None,
                }
                if tool.action
                else None,
                "output": {
                    "type": tool.output.type.value if tool.output else None,
                    "content": tool.output.content if tool.output else None,
                }
                if tool.output
                else None,
            },
        )
        response.raise_for_status()
        return ToolResponse(**response.json())

    async def delete_tool(self, tool_id: str) -> None:
        """Delete a tool.

        Args:
            tool_id: Tool ID to delete
        """
        response = await self.client.delete(f"/tools/{tool_id}")
        response.raise_for_status()

    async def create_tools_batch(self, tools: list[Tool]) -> list[ToolResponse]:
        """Create multiple tools in batch.

        Args:
            tools: List of tools to create

        Returns:
            List of created tools
        """
        tools_data = []
        for tool in tools:
            tool_dict = {
                "name": tool.name,
                "namespace": tool.namespace,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
                "code": tool.code,
                "id": tool.id,
            }

            if tool.action:
                tool_dict["action"] = {
                    "type": tool.action.type.value,
                    "http": tool.action.http,
                    "python": tool.action.python,
                    "javascript": tool.action.javascript,
                    "service_config": tool.action.service_config,
                }

            if tool.output:
                tool_dict["output"] = {
                    "type": tool.output.type.value,
                    "content": tool.output.content,
                }

            tools_data.append(tool_dict)

        response = await self.client.post("/tools/batch", json={"tools": tools_data})
        response.raise_for_status()

        return [ToolResponse(**tool) for tool in response.json()]

    async def search_tools(
        self, query: str, limit: int = 5
    ) -> list[ToolSearchResponse]:
        """Search for tools.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching tools
        """
        request = ToolSearchRequest(query=query, limit=limit)

        response = await self.client.post(
            "/tools/search", json={"query": request.query, "limit": request.limit}
        )
        response.raise_for_status()

        return [ToolSearchResponse(**tool) for tool in response.json()]

    async def retrieve_tools(
        self,
        query: str,
        limit: int = 5,
        strategy: Literal["vector_similarity", "text"] = "vector_similarity",
    ) -> list[ToolSearchResponse]:
        """Retrieve tools using vector similarity or text search.

        Args:
            query: Search query
            limit: Maximum number of results
            strategy: Search strategy ("vector_similarity" or "text")

        Returns:
            List of matching tools with scores
        """
        request = ToolSearchRequest(query=query, limit=limit, strategy=strategy)

        response = await self.client.post(
            "/tools/retrieve",
            json={
                "query": request.query,
                "limit": request.limit,
                "strategy": request.strategy,
            },
        )
        response.raise_for_status()

        return [ToolSearchResponse(**tool) for tool in response.json()]

    async def get_tools_embeddings(self) -> list[EmbeddingResponse]:
        """Get embeddings for all tools.

        Returns:
            List of tool embeddings
        """
        response = await self.client.get("/tools/embeddings")
        response.raise_for_status()

        return [EmbeddingResponse(**emb) for emb in response.json()]

    async def execute_tool(
        self,
        tool_id: str | None = None,
        tool_name: str | None = None,
        input_data: dict[str, Any] | None = None,
        user_settings: dict[str, Any] | None = None,
        format_type: str | None = None,
    ) -> ToolExecutionResponse:
        """Execute a single tool.

        Args:
            tool_id: Tool ID
            tool_name: Tool name (alternative to ID)
            input_data: Input data for the tool
            user_settings: Optional user settings
            format_type: Optional format type

        Returns:
            Tool execution result
        """
        if not tool_id and not tool_name:
            raise ValueError("Either tool_id or tool_name must be provided")

        request = ToolExecutionRequest(
            tool_id=tool_id,
            tool_name=tool_name,
            input_data=input_data or {},
            user_settings=user_settings,
            format_type=format_type,
        )

        response = await self.client.post(
            "/execute_tool",
            json={
                "tool_id": request.tool_id,
                "tool_name": request.tool_name,
                "input_data": request.input_data,
                "user_settings": request.user_settings,
                "format_type": request.format_type,
            },
        )
        response.raise_for_status()

        return ToolExecutionResponse(**response.json())

    async def execute_tools_sequential(
        self,
        tool_ids: list[str | ToolIdentifier],
        initial_input: dict[str, Any],
    ) -> list[ToolExecutionResponse]:
        """Execute tools sequentially.

        Args:
            tool_ids: List of tool IDs to execute
            initial_input: Initial input data

        Returns:
            List of execution results
        """
        request = SequentialToolExecutionRequest(
            tool_ids=tool_ids,
            initial_input=initial_input,
        )

        response = await self.client.post(
            "/execute_tools_sequential",
            json={
                "tool_ids": request.tool_ids,
                "initial_input": request.initial_input,
            },
        )
        response.raise_for_status()

        return [ToolExecutionResponse(**result) for result in response.json()]

    async def execute_tools_parallel(
        self,
        tool_ids: list[str | ToolIdentifier],
        input_data: dict[str, dict[str, Any]],
    ) -> list[ToolExecutionResponse]:
        """Execute tools in parallel.

        Args:
            tool_ids: List of tool IDs to execute
            input_data: Mapping of tool ID to input data

        Returns:
            List of execution results
        """
        request = ParallelToolExecutionRequest(
            tool_ids=tool_ids,
            input_data=input_data,
        )

        response = await self.client.post(
            "/execute_tools_parallel",
            json={
                "tool_ids": request.tool_ids,
                "input_data": request.input_data,
            },
        )
        response.raise_for_status()

        return [ToolExecutionResponse(**result) for result in response.json()]
