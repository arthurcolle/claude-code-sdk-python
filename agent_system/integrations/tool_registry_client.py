"""
Tool Registry Client for integrating with the evalscompany tool registry on port 2016.
"""

import json
import asyncio
import httpx
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

from ..config import config


class ToolRegistryClient:
    """Client for interacting with the tool registry."""
    
    def __init__(self, registry_url: Optional[str] = None, token: Optional[str] = None):
        """Initialize the tool registry client."""
        self.registry_url = (registry_url or config.TOOL_REGISTRY_URL).rstrip('/')
        self.token = token or config.TOOL_REGISTRY_TOKEN
        self._http_client = None
        self.logger = logging.getLogger("tool_registry_client")
    
    async def _get_http_client(self):
        """Get or create the HTTP client."""
        if self._http_client is None:
            headers = {}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            headers["Content-Type"] = "application/json"
            
            self._http_client = httpx.AsyncClient(
                base_url=self.registry_url,
                headers=headers,
                timeout=30.0
            )
        return self._http_client
    
    async def close(self):
        """Close the HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
    
    async def create_tool(self, tool_data: Dict[str, Any], agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new tool in the registry."""
        client = await self._get_http_client()
        
        # Add agent_id if provided
        if agent_id:
            tool_data["agent_id"] = agent_id
            tool_data["created_by"] = "agent"
            tool_data["validation_status"] = "pending"
        
        response = await client.post("/tools", json=tool_data)
        response.raise_for_status()
        return response.json()
    
    async def get_tools(self, name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get tools from the registry."""
        client = await self._get_http_client()
        params = {}
        if name:
            params["name"] = name
        
        response = await client.get("/tools", params=params)
        response.raise_for_status()
        return response.json()
    
    async def get_tool_by_id(self, tool_id: str) -> Dict[str, Any]:
        """Get a specific tool by ID."""
        client = await self._get_http_client()
        response = await client.get(f"/tools/{tool_id}")
        response.raise_for_status()
        return response.json()
    
    async def get_tool_by_name(self, tool_name: str) -> Dict[str, Any]:
        """Get a tool by name (latest version)."""
        client = await self._get_http_client()
        response = await client.get(f"/tools/name/{tool_name}")
        response.raise_for_status()
        return response.json()
    
    async def update_tool(self, tool_id: str, tool_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing tool."""
        client = await self._get_http_client()
        response = await client.put(f"/tools/{tool_id}", json=tool_data)
        response.raise_for_status()
        return response.json()
    
    async def delete_tool(self, tool_id: str) -> bool:
        """Delete a tool from the registry."""
        client = await self._get_http_client()
        response = await client.delete(f"/tools/{tool_id}")
        response.raise_for_status()
        return True
    
    async def execute_tool(
        self, 
        tool_id: Optional[str] = None, 
        tool_name: Optional[str] = None, 
        input_data: Dict[str, Any] = None,
        format_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a tool with given input."""
        client = await self._get_http_client()
        
        request_data = {
            "input_data": input_data or {}
        }
        
        if tool_id:
            request_data["tool_id"] = tool_id
        elif tool_name:
            request_data["tool_name"] = tool_name
        else:
            raise ValueError("Either tool_id or tool_name must be provided")
        
        if format_type:
            request_data["format_type"] = format_type
        
        response = await client.post("/execute_tool", json=request_data)
        response.raise_for_status()
        result = response.json()
        return result.get("result", result)
    
    async def execute_tools_sequential(
        self,
        tool_ids: List[str],
        initial_input: Dict[str, Any],
        format_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Execute tools sequentially, passing output from one to the next."""
        client = await self._get_http_client()
        
        request_data = {
            "tool_ids": tool_ids,
            "initial_input": initial_input
        }
        
        if format_type:
            request_data["format_type"] = format_type
        
        response = await client.post("/execute_tools_sequential", json=request_data)
        response.raise_for_status()
        return response.json()
    
    async def execute_tools_parallel(
        self,
        tool_ids: List[str],
        input_data: Dict[str, Dict[str, Any]],
        format_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Execute multiple tools in parallel."""
        client = await self._get_http_client()
        
        request_data = {
            "tool_ids": tool_ids,
            "input_data": input_data
        }
        
        if format_type:
            request_data["format_type"] = format_type
        
        response = await client.post("/execute_tools_parallel", json=request_data)
        response.raise_for_status()
        return response.json()
    
    async def search_tools(
        self,
        prompt: str,
        limit: int = 5,
        for_model: str = "anthropic"
    ) -> List[Dict[str, Any]]:
        """Search for tools using vector similarity."""
        client = await self._get_http_client()
        
        request_data = {
            "prompt": prompt,
            "limit": limit,
            "for_model": for_model
        }
        
        response = await client.post("/tools/search", json=request_data)
        response.raise_for_status()
        return response.json()
    
    async def validate_tool(self, tool_id: str, action: str) -> Dict[str, Any]:
        """Validate or reject a tool created by an agent."""
        client = await self._get_http_client()
        
        request_data = {
            "action": action  # "approved" or "rejected"
        }
        
        response = await client.post(f"/tools/validate/{tool_id}", json=request_data)
        response.raise_for_status()
        return response.json()
    
    async def get_pending_tools(self) -> List[Dict[str, Any]]:
        """Get all tools pending validation."""
        client = await self._get_http_client()
        response = await client.get("/tools/pending")
        response.raise_for_status()
        result = response.json()
        # Handle both list and dict responses
        if isinstance(result, list):
            return result
        elif isinstance(result, dict):
            return result.get("tools", [])
        else:
            return []
    
    async def get_agent_tools(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get all tools created by a specific agent."""
        client = await self._get_http_client()
        response = await client.get(f"/tools/agent/{agent_id}")
        response.raise_for_status()
        return response.json()
    
    async def format_tools(self, tools: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
        """Format tools for a specific model (anthropic/openai)."""
        client = await self._get_http_client()
        
        request_data = {
            "tools": tools,
            "model": model
        }
        
        response = await client.post(f"/tools/format/{model}", json=request_data)
        response.raise_for_status()
        return response.json()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the tool registry."""
        client = await self._get_http_client()
        response = await client.get("/health")
        response.raise_for_status()
        return response.json()
    
    async def get_service_info(self) -> Dict[str, Any]:
        """Get information about the tool registry service."""
        client = await self._get_http_client()
        response = await client.get("/")
        # Parse HTML response to extract info
        html_content = response.text
        # For now, just return basic info
        return {
            "status": "connected",
            "url": self.registry_url,
            "timestamp": datetime.now().isoformat()
        }
    
    # Meta-registration methods for delayed tool activation
    
    async def meta_register_tool(self, tool_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a tool without immediate activation (meta-registration).
        The tool is stored but marked as inactive until explicitly activated.
        
        Args:
            tool_data: Tool configuration with name, description, schemas, etc.
            
        Returns:
            The created tool with meta_registered status
        """
        client = await self._get_http_client()
        response = await client.post("/tools/meta_register", json=tool_data)
        response.raise_for_status()
        return response.json()
    
    async def activate_tool(self, tool_id: str) -> Dict[str, Any]:
        """
        Activate a meta-registered tool.
        Changes the tool status from 'meta_registered' to 'active'.
        
        Args:
            tool_id: The ID of the tool to activate
            
        Returns:
            Activation result with status and message
        """
        client = await self._get_http_client()
        response = await client.post(f"/tools/{tool_id}/activate")
        response.raise_for_status()
        return response.json()
    
    async def get_meta_registered_tools(self) -> List[Dict[str, Any]]:
        """
        Get all meta-registered (inactive) tools.
        
        Returns:
            List of meta-registered tools
        """
        client = await self._get_http_client()
        response = await client.get("/tools/meta_registered")
        response.raise_for_status()
        return response.json()
    
    async def batch_activate_tools(self, tool_ids: List[str]) -> Dict[str, Any]:
        """
        Activate multiple meta-registered tools in a single operation.
        
        Args:
            tool_ids: List of tool IDs to activate
            
        Returns:
            Summary of activation results
        """
        client = await self._get_http_client()
        response = await client.post("/tools/batch_activate", json=tool_ids)
        response.raise_for_status()
        return response.json()


class MockToolRegistryClient:
    """Mock tool registry client for when tool registry is disabled."""
    
    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger("mock_tool_registry")
        self.logger.info("Using mock tool registry (tool registry disabled)")
    
    async def _get_http_client(self):
        return None
    
    async def close(self):
        pass
    
    async def create_tool(self, tool_data: Dict[str, Any], agent_id: Optional[str] = None) -> Dict[str, Any]:
        return {"id": "mock-tool-id", "name": tool_data.get("name", "mock-tool"), "status": "created"}
    
    async def get_tools(self, name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        return []
    
    async def get_tool_by_id(self, tool_id: str) -> Dict[str, Any]:
        return {"id": tool_id, "name": "mock-tool", "status": "available"}
    
    async def get_tool_by_name(self, tool_name: str) -> Dict[str, Any]:
        return {"id": "mock-id", "name": tool_name, "status": "available"}
    
    async def update_tool(self, tool_id: str, tool_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"id": tool_id, "status": "updated"}
    
    async def delete_tool(self, tool_id: str) -> bool:
        return True
    
    async def execute_tool(self, tool_id: Optional[str] = None, tool_name: Optional[str] = None, 
                          input_data: Dict[str, Any] = None, format_type: Optional[str] = None) -> Dict[str, Any]:
        return {"result": "mock execution result", "status": "success"}
    
    async def execute_tools_sequential(self, tool_ids: List[str], initial_input: Dict[str, Any],
                                     format_type: Optional[str] = None) -> List[Dict[str, Any]]:
        return [{"tool_id": tid, "result": "mock result"} for tid in tool_ids]
    
    async def execute_tools_parallel(self, tool_ids: List[str], input_data: Dict[str, Dict[str, Any]],
                                   format_type: Optional[str] = None) -> List[Dict[str, Any]]:
        return [{"tool_id": tid, "result": "mock result"} for tid in tool_ids]
    
    async def search_tools(self, prompt: str, limit: int = 5, for_model: str = "anthropic") -> List[Dict[str, Any]]:
        return []
    
    async def validate_tool(self, tool_id: str, action: str) -> Dict[str, Any]:
        return {"id": tool_id, "status": action}
    
    async def get_pending_tools(self) -> List[Dict[str, Any]]:
        return []
    
    async def get_agent_tools(self, agent_id: str) -> List[Dict[str, Any]]:
        return []
    
    async def format_tools(self, tools: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
        return tools
    
    async def health_check(self) -> Dict[str, Any]:
        return {"status": "healthy", "mock": True}
    
    async def get_service_info(self) -> Dict[str, Any]:
        return {"status": "mock", "timestamp": datetime.now().isoformat()}
    
    async def meta_register_tool(self, tool_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"id": "mock-meta-id", "name": tool_data.get("name", "mock-tool"), "status": "meta_registered"}
    
    async def activate_tool(self, tool_id: str) -> Dict[str, Any]:
        return {"id": tool_id, "status": "activated", "message": "Mock activation successful"}
    
    async def get_meta_registered_tools(self) -> List[Dict[str, Any]]:
        return []
    
    async def batch_activate_tools(self, tool_ids: List[str]) -> Dict[str, Any]:
        return {"activated": len(tool_ids), "failed": 0, "message": "Mock batch activation successful"}


def get_tool_registry_client(registry_url: Optional[str] = None, token: Optional[str] = None):
    """Get appropriate tool registry client based on configuration."""
    if config.DISABLE_TOOL_REGISTRY:
        return MockToolRegistryClient(registry_url, token)
    return ToolRegistryClient(registry_url, token)


class ToolBuilder:
    """Helper class for building tool definitions."""
    
    @staticmethod
    def create_http_tool(
        name: str,
        description: str,
        url: str,
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create an HTTP-based tool definition."""
        return {
            "name": name,
            "description": description,
            "input_schema": input_schema or {"type": "object", "properties": {}},
            "output_schema": output_schema or {"type": "object", "properties": {}},
            "action": {
                "type": "http",
                "http": {
                    "method": method,
                    "url": url,
                    "headers": headers or {}
                }
            },
            "output": {
                "type": "ai",
                "content": "Tool executed successfully"
            }
        }
    
    @staticmethod
    def create_python_tool(
        name: str,
        description: str,
        code: str,
        function_name: str,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a Python-based tool definition."""
        return {
            "name": name,
            "description": description,
            "input_schema": input_schema or {"type": "object", "properties": {}},
            "output_schema": output_schema or {"type": "object", "properties": {}},
            "action": {
                "type": "python",
                "python": {
                    "code": code,
                    "function_name": function_name
                }
            },
            "output": {
                "type": "ai",
                "content": "Tool executed successfully"
            },
            "is_adhoc": True,
            "code": code,
            "runtime": "python"
        }