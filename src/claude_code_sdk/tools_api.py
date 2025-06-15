"""Custom tool creation API for Claude SDK."""

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union, get_type_hints, get_args, get_origin, AsyncIterator
from collections.abc import Awaitable

from .types import ToolUseBlock, ToolResultBlock


@dataclass
class ToolParameter:
    """Defines a tool parameter."""
    
    name: str
    type: type
    description: str
    required: bool = True
    default: Any = None
    enum: list[Any] | None = None
    
    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON schema format."""
        schema: dict[str, Any] = {
            "description": self.description
        }
        
        # Map Python types to JSON schema types
        if self.type == str:
            schema["type"] = "string"
        elif self.type == int:
            schema["type"] = "integer"
        elif self.type == float:
            schema["type"] = "number"
        elif self.type == bool:
            schema["type"] = "boolean"
        elif self.type == list:
            schema["type"] = "array"
        elif self.type == dict:
            schema["type"] = "object"
        else:
            schema["type"] = "string"  # Default
        
        if self.enum:
            schema["enum"] = self.enum
        
        if not self.required and self.default is not None:
            schema["default"] = self.default
        
        return schema


@dataclass
class CustomTool:
    """Defines a custom tool."""
    
    name: str
    description: str
    handler: Callable[..., Union[Any, Awaitable[Any]]]
    parameters: list[ToolParameter] = field(default_factory=list)
    
    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON schema format for Claude."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }
    
    async def execute(self, **kwargs) -> Any:
        """Execute the tool handler."""
        # Validate parameters
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                raise ValueError(f"Missing required parameter: {param.name}")
            
            # Type validation could be added here
        
        # Execute handler
        if inspect.iscoroutinefunction(self.handler):
            return await self.handler(**kwargs)
        else:
            return self.handler(**kwargs)


class ToolRegistry:
    """Registry for custom tools."""
    
    def __init__(self):
        self._tools: dict[str, CustomTool] = {}
    
    def register(self, tool: CustomTool) -> None:
        """Register a custom tool."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool
    
    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        self._tools.pop(name, None)
    
    def get(self, name: str) -> Optional[CustomTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def get_schemas(self) -> list[dict[str, Any]]:
        """Get JSON schemas for all tools."""
        return [tool.to_json_schema() for tool in self._tools.values()]
    
    async def execute(self, tool_use: ToolUseBlock) -> ToolResultBlock:
        """Execute a tool use block."""
        tool = self.get(tool_use.name)
        if not tool:
            return ToolResultBlock(
                tool_use_id=tool_use.id,
                content=f"Unknown tool: {tool_use.name}",
                is_error=True
            )
        
        try:
            result = await tool.execute(**tool_use.input)
            
            # Convert result to appropriate format
            if isinstance(result, str):
                content = result
            elif isinstance(result, dict) or isinstance(result, list):
                content = json.dumps(result)
            else:
                content = str(result)
            
            return ToolResultBlock(
                tool_use_id=tool_use.id,
                content=content,
                is_error=False
            )
        except Exception as e:
            return ToolResultBlock(
                tool_use_id=tool_use.id,
                content=f"Tool execution error: {str(e)}",
                is_error=True
            )


# Decorator for easy tool creation

def create_tool(
    name: str,
    description: str,
    parameters: list[ToolParameter] | None = None
) -> Callable[[Callable], CustomTool]:
    """
    Decorator to create a custom tool.
    
    Args:
        name: Tool name
        description: Tool description
        parameters: List of tool parameters
        
    Returns:
        Decorator function
        
    Example:
        ```python
        @create_tool(
            name="weather",
            description="Get weather for a location",
            parameters=[
                ToolParameter("location", str, "City name", required=True),
                ToolParameter("units", str, "Temperature units", required=False, 
                             default="celsius", enum=["celsius", "fahrenheit"])
            ]
        )
        async def get_weather(location: str, units: str = "celsius") -> dict:
            # Implementation here
            return {"temp": 20, "units": units}
        
        # Register the tool
        registry = ToolRegistry()
        registry.register(get_weather)
        ```
    """
    def decorator(func: Callable) -> CustomTool:
        # Extract parameters from function signature if not provided
        if parameters is None:
            sig = inspect.signature(func)
            params = []
            
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                    
                param_type = str  # Default type
                if param.annotation != inspect.Parameter.empty:
                    param_type = param.annotation
                
                required = param.default == inspect.Parameter.empty
                default = None if required else param.default
                
                params.append(ToolParameter(
                    name=param_name,
                    type=param_type,
                    description=f"Parameter {param_name}",
                    required=required,
                    default=default
                ))
        else:
            params = parameters
        
        return CustomTool(
            name=name,
            description=description,
            handler=func,
            parameters=params
        )
    
    return decorator


# Global registry
_global_registry = ToolRegistry()


def get_global_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _global_registry


# Integration with query function

async def query_with_tools(
    *,
    prompt: str,
    options: Any = None,
    tools: list[CustomTool] | None = None,
    registry: Optional[ToolRegistry] = None,
    auto_execute: bool = True
) -> AsyncIterator[Any]:
    """
    Query Claude with custom tools.
    
    Args:
        prompt: The prompt to send
        options: Query options
        tools: List of custom tools to make available
        registry: Tool registry (uses global if None)
        auto_execute: Automatically execute tool calls
        
    Yields:
        Messages from conversation with tool execution
        
    Example:
        ```python
        # Define custom tools
        @create_tool("calculate", "Perform calculations")
        async def calculate(expression: str) -> float:
            return eval(expression)  # Simple example
        
        @create_tool("search", "Search for information")
        async def search(query: str) -> list[str]:
            return ["Result 1", "Result 2"]
        
        # Query with tools
        async for msg in query_with_tools(
            prompt="What is 2+2 and search for 'Python'?",
            tools=[calculate, search]
        ):
            print(msg)
        ```
    """
    from . import query
    from .types import AssistantMessage
    
    if registry is None:
        registry = get_global_tool_registry()
    
    # Register tools temporarily
    temp_tools = []
    if tools:
        for tool in tools:
            if tool.name not in registry.list_tools():
                registry.register(tool)
                temp_tools.append(tool.name)
    
    try:
        # Update options to include tool names
        if options is None:
            from .types import ClaudeCodeOptions
            options = ClaudeCodeOptions()
        
        # Add custom tool names to allowed tools
        custom_tool_names = [f"custom_{t.name}" for t in (tools or [])]
        options.allowed_tools.extend(custom_tool_names)
        
        async for message in query(prompt=prompt, options=options):
            # Auto-execute tools if enabled
            if auto_execute and isinstance(message, AssistantMessage):
                for i, block in enumerate(message.content):
                    if isinstance(block, ToolUseBlock):
                        # Check if it's a custom tool
                        tool_name = block.name
                        if tool_name.startswith("custom_"):
                            tool_name = tool_name[7:]  # Remove prefix
                        
                        if registry.get(tool_name):
                            # Execute custom tool
                            result = await registry.execute(block)
                            # Replace with result
                            message.content[i] = result
            
            yield message
            
    finally:
        # Unregister temporary tools
        for tool_name in temp_tools:
            registry.unregister(tool_name)