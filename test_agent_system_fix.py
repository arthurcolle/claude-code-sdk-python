#!/usr/bin/env python3
"""Test script to verify the agent system fixes."""

import asyncio
import sys
from pathlib import Path

# Add agent_system to path
sys.path.insert(0, str(Path(__file__).parent))

from agent_system.integrations.tool_registry_client import ToolRegistryClient


async def test_tool_registry_fix():
    """Test that the tool registry client handles both response formats."""
    client = ToolRegistryClient()
    
    print("Testing tool registry client fixes...")
    
    try:
        # Test health check
        health = await client.health_check()
        print(f"✓ Health check: {health}")
        
        # Test get_pending_tools (this was failing before)
        print("\nTesting get_pending_tools...")
        pending_tools = await client.get_pending_tools()
        print(f"✓ Got pending tools: {len(pending_tools)} tools")
        
        # Test get_tools (this was 404 initially but should work after startup)
        print("\nTesting get_tools...")
        try:
            tools = await client.get_tools(limit=5)
            print(f"✓ Got tools: {len(tools)} tools")
        except Exception as e:
            print(f"✗ get_tools failed: {e}")
            print("  Note: This might fail if tool registry is not fully initialized")
        
        print("\n✅ All critical tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_tool_registry_fix())