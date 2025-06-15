#!/usr/bin/env python3
"""
Demo of the meta-registration functionality for tools.
This shows how to register tools without immediate activation.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_system.integrations.tool_registry_client import ToolRegistryClient, ToolBuilder


async def demo_meta_registration():
    """Demonstrate the meta-registration workflow."""
    
    # Initialize the tool registry client
    client = ToolRegistryClient()
    
    print("=== Tool Meta-Registration Demo ===\n")
    
    # 1. Create a tool definition
    tool_def = ToolBuilder.create_http_tool(
        name="weather_forecast_v2",
        description="Get weather forecast for a location (pending approval)",
        url="https://api.weather.example.com/forecast/{location}",
        method="GET",
        headers={"X-API-Key": "demo-key"},
        input_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name or coordinates"}
            },
            "required": ["location"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "temperature": {"type": "number"},
                "conditions": {"type": "string"},
                "forecast": {"type": "array", "items": {"type": "object"}}
            }
        }
    )
    
    # 2. Meta-register the tool (inactive by default)
    print("1. Meta-registering tool (inactive)...")
    try:
        meta_tool = await client.meta_register_tool(tool_def)
        tool_id = meta_tool["id"]
        print(f"   ✓ Tool meta-registered with ID: {tool_id}")
        print(f"   Status: meta_registered (inactive)")
    except Exception as e:
        print(f"   ✗ Failed to meta-register tool: {e}")
        return
    
    # 3. List all meta-registered tools
    print("\n2. Listing meta-registered tools...")
    try:
        meta_tools = await client.get_meta_registered_tools()
        print(f"   Found {len(meta_tools)} meta-registered tools:")
        for tool in meta_tools:
            print(f"   - {tool['name']} (ID: {tool['id']})")
    except Exception as e:
        print(f"   ✗ Failed to list meta-registered tools: {e}")
    
    # 4. Simulate approval process
    print("\n3. Simulating approval process...")
    await asyncio.sleep(2)  # Simulate review time
    
    # 5. Activate the tool
    print(f"\n4. Activating tool {tool_id}...")
    try:
        activation_result = await client.activate_tool(tool_id)
        print(f"   ✓ {activation_result['message']}")
        print(f"   Status: {activation_result['status']}")
    except Exception as e:
        print(f"   ✗ Failed to activate tool: {e}")
        return
    
    # 6. Verify the tool is now active
    print("\n5. Verifying tool is active...")
    try:
        active_tool = await client.get_tool(tool_id)
        print(f"   ✓ Tool '{active_tool['name']}' is now active")
    except Exception as e:
        print(f"   ✗ Failed to get tool: {e}")
    
    # 7. Demo batch activation
    print("\n6. Demo batch activation...")
    
    # Create multiple tools for batch demo
    batch_tools = []
    for i in range(3):
        batch_tool = ToolBuilder.create_code_tool(
            name=f"batch_tool_{i}",
            description=f"Batch demo tool {i}",
            code=f"def execute(input_data): return {{'result': 'Tool {i} executed'}}",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {"result": {"type": "string"}}}
        )
        
        try:
            meta_tool = await client.meta_register_tool(batch_tool)
            batch_tools.append(meta_tool["id"])
            print(f"   - Meta-registered: {batch_tool['name']} (ID: {meta_tool['id']})")
        except Exception as e:
            print(f"   ✗ Failed to meta-register batch tool {i}: {e}")
    
    if batch_tools:
        print(f"\n   Activating {len(batch_tools)} tools in batch...")
        try:
            batch_result = await client.batch_activate_tools(batch_tools)
            print(f"   ✓ Batch activation complete:")
            print(f"     - Activated: {len(batch_result.get('activated', []))}")
            print(f"     - Already active: {len(batch_result.get('already_active', []))}")
            print(f"     - Failed: {len(batch_result.get('failed', []))}")
        except Exception as e:
            print(f"   ✗ Batch activation failed: {e}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    try:
        asyncio.run(demo_meta_registration())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()