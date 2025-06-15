"""Example of tool search and suggestion features."""

import asyncio
from claude_code_sdk.tools import ToolManagementClient, ToolDiscovery


async def main():
    """Demonstrate tool search and suggestion capabilities."""
    
    async with ToolManagementClient() as client:
        discovery = ToolDiscovery(client)
        
        # Example 1: Search by capability
        print("=== Searching for data processing tools ===")
        data_tools = await discovery.search_by_capability(
            "data processing transformation", 
            limit=5
        )
        
        for tool in data_tools:
            print(f"- {tool.name}: {tool.description}")
            if tool.score:
                print(f"  Relevance score: {tool.score:.3f}")
        
        # Example 2: Find similar tools
        print("\n=== Finding tools similar to 'calculator' ===")
        similar_tools = await discovery.find_similar_tools("calculator", limit=5)
        
        for tool in similar_tools:
            print(f"- {tool.name}: {tool.description}")
            if tool.score:
                print(f"  Similarity score: {tool.score:.3f}")
        
        # Example 3: Suggest tools for a task
        print("\n=== Suggesting tools for a task ===")
        task = "I need to analyze sentiment in customer reviews and generate a report"
        suggestions = await discovery.suggest_tools_for_task(task, limit=5)
        
        print(f"Task: {task}")
        print("Suggested tools:")
        for tool in suggestions:
            print(f"- {tool.name}: {tool.description}")
            if tool.score:
                print(f"  Relevance score: {tool.score:.3f}")
        
        # Example 4: Get tool by exact name
        print("\n=== Getting specific tool by name ===")
        tool_name = "calculator"
        specific_tool = await discovery.get_tool_by_name(tool_name)
        
        if specific_tool:
            print(f"Found tool: {specific_tool.name}")
            print(f"Description: {specific_tool.description}")
            print(f"Input schema: {specific_tool.input_schema}")
            print(f"Output schema: {specific_tool.output_schema}")
        else:
            print(f"Tool '{tool_name}' not found")
        
        # Example 5: List tools by namespace/category
        print("\n=== Tool categories ===")
        categories = await discovery.list_tool_categories()
        
        for namespace, tools in sorted(categories.items()):
            print(f"\n{namespace}:")
            for tool in tools[:5]:  # Show first 5 tools per category
                print(f"  - {tool}")
            if len(tools) > 5:
                print(f"  ... and {len(tools) - 5} more")


if __name__ == "__main__":
    asyncio.run(main())