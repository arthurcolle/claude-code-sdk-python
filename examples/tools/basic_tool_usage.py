"""Basic example of using Tool Management API with Claude Code SDK."""

import asyncio
from claude_code_sdk.tools import (
    ToolManagementClient,
    ToolRegistry,
    query_with_tools,
    ToolRegistryOptions,
)


async def main():
    """Demonstrate basic tool usage."""
    
    # Example 1: Direct tool API usage
    print("=== Example 1: Direct Tool API Usage ===")
    
    async with ToolManagementClient() as client:
        # Health check
        health = await client.health_check()
        print(f"API Health: {health}")
        
        # Search for tools
        print("\nSearching for calculator tools...")
        calculator_tools = await client.search_tools("calculator", limit=3)
        
        for tool in calculator_tools:
            print(f"- {tool.name}: {tool.description}")
        
        # Execute a tool if found
        if calculator_tools:
            tool = calculator_tools[0]
            print(f"\nExecuting tool: {tool.name}")
            
            result = await client.execute_tool(
                tool_id=tool.id,
                input_data={"expression": "2 + 2"}
            )
            print(f"Result: {result.output_data}")
    
    # Example 2: Using query_with_tools for automatic tool execution
    print("\n\n=== Example 2: Automatic Tool Execution ===")
    
    async for message in query_with_tools(
        prompt="Calculate 10 * 25 using any available calculator tool",
        use_remote_tools=True
    ):
        print(message)
    
    # Example 3: Tool discovery and registration
    print("\n\n=== Example 3: Tool Discovery ===")
    
    async with ToolManagementClient() as client:
        registry = ToolRegistry(client)
        
        # Check tool availability
        tools_to_check = ["calculator", "weather", "translator"]
        availability = await registry.ensure_tools_available(tools_to_check)
        
        print("Tool availability:")
        for tool_name, is_available in availability.items():
            status = "✓" if is_available else "✗"
            print(f"  {status} {tool_name}")
        
        # Get tool categories
        categories = await registry.discovery.list_tool_categories()
        print("\nTool categories:")
        for category, tools in categories.items():
            print(f"  {category}: {', '.join(tools[:3])}...")


if __name__ == "__main__":
    asyncio.run(main())