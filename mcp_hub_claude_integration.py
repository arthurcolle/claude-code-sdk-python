#!/usr/bin/env python3
"""
MCP Hub + Claude SDK Integration

This demonstrates how to integrate the distributed MCP hub with Claude Code SDK,
allowing Claude to seamlessly use tools from the distributed system.

Features:
- Claude can discover and use tools from the hub
- Hub manages tool routing and load balancing
- Support for real MCP servers via WebSocket/stdio
- Dynamic tool registration from external services
- Multi-agent workflows with shared tool access
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from distributed_mcp_hub import (
    MCPHub, SimpleMCPClient, ServiceNode, ToolSignature,
    ToolCategory, ServiceHealth
)
from datetime import datetime

# Try to import Claude SDK (may not be available in all environments)
try:
    from claude_code_sdk import query, ClaudeCodeOptions
    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_SDK_AVAILABLE = False
    print("Note: Claude SDK not available. Using simulation mode.")


@dataclass
class MCPServerConfig:
    """Configuration for connecting to an MCP server"""
    name: str
    transport: List[str]  # e.g., ["node", "server.js"] or ["python", "-m", "server"]
    env: Dict[str, str] = None
    capabilities: List[str] = None

    def __post_init__(self):
        if self.env is None:
            self.env = {}
        if self.capabilities is None:
            self.capabilities = []


class MCPHubWithClaude:
    """
    Enhanced MCP Hub that integrates with Claude SDK.

    This allows Claude to:
    1. Auto-discover all tools from the hub
    2. Execute tools via the hub's distributed system
    3. Chain tools together transparently
    4. Access external MCP servers
    """

    def __init__(self):
        self.hub = MCPHub()
        self.claude_options: Optional[ClaudeCodeOptions] = None

    async def start(self):
        """Start the hub and register external MCP servers"""
        await self.hub.start()
        await self._register_external_mcp_servers()

    async def stop(self):
        """Stop the hub"""
        await self.hub.stop()

    async def _register_external_mcp_servers(self):
        """Register external MCP servers as services"""

        # Example: Register a hypothetical weather MCP server
        weather_service = ServiceNode(
            service_id="mcp-weather-001",
            name="Weather MCP Server",
            endpoint="stdio://weather-server",
            capabilities=["weather", "forecast", "alerts"],
            tools=[
                ToolSignature(
                    name="get_weather",
                    description="Get current weather for a location",
                    category=ToolCategory.WEB,
                    parameters={
                        "location": "string",
                        "units": "string (optional, default: metric)"
                    },
                    returns={"temperature": "number", "conditions": "string"}
                ),
                ToolSignature(
                    name="get_forecast",
                    description="Get weather forecast",
                    category=ToolCategory.WEB,
                    parameters={"location": "string", "days": "integer"},
                    returns={"forecast": "array"}
                ),
            ],
            health=ServiceHealth.HEALTHY,
            last_heartbeat=datetime.now()
        )
        await self.hub.registry.register_service(weather_service)

        # Example: Register a database MCP server
        analytics_service = ServiceNode(
            service_id="mcp-analytics-001",
            name="Analytics MCP Server",
            endpoint="ws://analytics-server:8000",
            capabilities=["analytics", "metrics", "reporting"],
            tools=[
                ToolSignature(
                    name="get_metrics",
                    description="Get analytics metrics",
                    category=ToolCategory.DATA_PROCESSING,
                    parameters={"metric_name": "string", "time_range": "string"},
                    returns={"data": "array"}
                ),
                ToolSignature(
                    name="generate_report",
                    description="Generate analytics report",
                    category=ToolCategory.DATA_PROCESSING,
                    parameters={"report_type": "string", "filters": "object"},
                    returns={"report_url": "string"}
                ),
            ],
            health=ServiceHealth.HEALTHY,
            last_heartbeat=datetime.now()
        )
        await self.hub.registry.register_service(analytics_service)

    async def get_tools_for_claude(self) -> List[Dict[str, Any]]:
        """
        Get all tools in format compatible with Claude SDK.

        Returns tools in MCP tool format that Claude can understand.
        """
        tools = await self.hub.registry.get_all_tools()

        # Convert to Claude-compatible format
        claude_tools = []
        for tool in tools:
            claude_tool = {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": {
                    "type": "object",
                    "properties": tool["parameters"],
                    "required": list(tool["parameters"].keys())
                }
            }
            claude_tools.append(claude_tool)

        return claude_tools

    async def execute_tool_for_claude(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> str:
        """
        Execute a tool on behalf of Claude.

        This is called when Claude wants to use a tool from the hub.
        """
        result = await self.hub.execute(tool_name, parameters)
        return json.dumps(result, indent=2)


class ClaudeWithMCPHub:
    """
    Claude agent with full access to MCP hub functionality.

    Usage:
        agent = ClaudeWithMCPHub()
        await agent.start()
        response = await agent.chat("Get the weather in San Francisco")
    """

    def __init__(self):
        self.hub_integration = MCPHubWithClaude()
        self.conversation_history: List[Dict[str, Any]] = []

    async def start(self):
        """Start the agent and hub"""
        await self.hub_integration.start()
        print(f"✓ Started MCP Hub with {len(self.hub_integration.hub.registry.services)} services")

        # Get available tools
        tools = await self.hub_integration.get_tools_for_claude()
        print(f"✓ Discovered {len(tools)} tools for Claude")

    async def stop(self):
        """Stop the agent and hub"""
        await self.hub_integration.stop()

    async def chat(self, message: str) -> str:
        """
        Chat with Claude, who has access to all hub tools.

        This simulates Claude's tool usage. In production, you'd use
        the actual Claude SDK with custom tool execution.
        """
        print(f"\n👤 User: {message}")

        # Get available tools
        tools = await self.hub_integration.get_tools_for_claude()

        # Simulate Claude deciding which tools to use
        # In production, Claude SDK would handle this
        response = await self._simulate_claude_response(message, tools)

        print(f"🤖 Claude: {response}")
        return response

    async def _simulate_claude_response(
        self,
        message: str,
        tools: List[Dict[str, Any]]
    ) -> str:
        """
        Simulate Claude's response with tool usage.
        In production, replace with actual Claude SDK query.
        """
        message_lower = message.lower()

        # Simulate tool selection based on message
        if "weather" in message_lower:
            print("  🔧 Using tool: get_weather")
            result = await self.hub_integration.execute_tool_for_claude(
                "get_weather",
                {"location": "San Francisco", "units": "metric"}
            )
            return f"I checked the weather for you:\n{result}"

        elif "file" in message_lower or "read" in message_lower:
            print("  🔧 Using tool: read_file")
            result = await self.hub_integration.execute_tool_for_claude(
                "read_file",
                {"path": "/tmp/example.txt"}
            )
            return f"I read the file:\n{result}"

        elif "database" in message_lower or "query" in message_lower:
            print("  🔧 Using tool: sql_query")
            result = await self.hub_integration.execute_tool_for_claude(
                "sql_query",
                {"query": "SELECT COUNT(*) FROM users", "params": {}}
            )
            return f"Query results:\n{result}"

        elif "metrics" in message_lower or "analytics" in message_lower:
            print("  🔧 Using tool: get_metrics")
            result = await self.hub_integration.execute_tool_for_claude(
                "get_metrics",
                {"metric_name": "user_engagement", "time_range": "7d"}
            )
            return f"Here are the metrics:\n{result}"

        else:
            # List available tools
            tool_list = "\n".join(f"  • {t['name']}: {t['description']}" for t in tools[:5])
            return f"I have access to {len(tools)} tools including:\n{tool_list}\n\nWhat would you like me to do?"


async def demo_basic_integration():
    """Demo: Basic Claude + MCP Hub integration"""
    print("=" * 80)
    print("DEMO: Claude with MCP Hub Integration")
    print("=" * 80)
    print()

    agent = ClaudeWithMCPHub()
    await agent.start()
    print()

    # Example conversations
    await agent.chat("What tools do you have?")
    await agent.chat("Can you check the weather in San Francisco?")
    await agent.chat("Read the contents of /tmp/example.txt")
    await agent.chat("Query the database for user count")
    await agent.chat("Get the analytics metrics for the past week")

    await agent.stop()


async def demo_multi_tool_workflow():
    """Demo: Complex workflow using multiple tools"""
    print("\n" + "=" * 80)
    print("DEMO: Multi-Tool Workflow")
    print("=" * 80)
    print()

    agent = ClaudeWithMCPHub()
    await agent.start()

    print("Scenario: Analyze website performance")
    print("  1. Scrape website")
    print("  2. Run analytics")
    print("  3. Store results in database")
    print("  4. Generate report")
    print()

    # Use the hub directly for complex workflow
    hub = agent.hub_integration.hub

    print("Step 1: Scraping website...")
    scrape_result = await hub.execute("scrape_page", {
        "url": "https://example.com",
        "selector": ".metrics"
    })
    print(f"  ✓ Scraped data")

    print("\nStep 2: Getting analytics...")
    metrics_result = await hub.execute("get_metrics", {
        "metric_name": "page_performance",
        "time_range": "24h"
    })
    print(f"  ✓ Retrieved metrics")

    print("\nStep 3: Storing results...")
    db_result = await hub.execute("sql_query", {
        "query": "INSERT INTO performance_logs (data) VALUES (?)",
        "params": {"data": str(metrics_result)}
    })
    print(f"  ✓ Stored in database")

    print("\nStep 4: Generating report...")
    report_result = await hub.execute("generate_report", {
        "report_type": "performance",
        "filters": {"time_range": "24h"}
    })
    print(f"  ✓ Report generated: {report_result}")

    print("\n✓ Workflow completed!")

    await agent.stop()


async def demo_parallel_execution():
    """Demo: Parallel execution across distributed services"""
    print("\n" + "=" * 80)
    print("DEMO: Parallel Distributed Execution")
    print("=" * 80)
    print()

    agent = ClaudeWithMCPHub()
    await agent.start()

    print("Executing 10 tools in parallel across distributed services...")
    print()

    hub = agent.hub_integration.hub

    # Create diverse set of tasks
    tasks = [
        ("read_file", {"path": f"/tmp/file{i}.txt"})
        for i in range(3)
    ] + [
        ("http_get", {"url": f"https://api{i}.example.com", "headers": {}})
        for i in range(3)
    ] + [
        ("sql_query", {"query": f"SELECT * FROM table{i}", "params": {}})
        for i in range(2)
    ] + [
        ("get_metrics", {"metric_name": "cpu", "time_range": "1h"}),
        ("execute_python", {"code": "print('parallel!')", "timeout": 5})
    ]

    import time
    start = time.time()

    # Execute all in parallel
    results = await asyncio.gather(*[
        hub.execute(tool_name, params)
        for tool_name, params in tasks
    ])

    elapsed = time.time() - start

    print(f"✓ Completed {len(results)} operations in {elapsed*1000:.1f}ms")
    print(f"  Average: {elapsed*1000/len(results):.1f}ms per operation")
    print()

    # Show which services were used
    services_used = {}
    for result in results:
        service_id = result['service_id']
        services_used[service_id] = services_used.get(service_id, 0) + 1

    print("Services utilized:")
    for service_id, count in services_used.items():
        service = hub.registry.services.get(service_id)
        if service:
            print(f"  • {service.name}: {count} operations")

    await agent.stop()


async def demo_direct_client():
    """Demo: Using SimpleMCPClient directly"""
    print("\n" + "=" * 80)
    print("DEMO: Direct MCP Client Usage (No Claude)")
    print("=" * 80)
    print()

    print("A simple script can connect and use the entire ecosystem:")
    print()

    # Show minimal code example
    code_example = '''
    from mcp_hub_claude_integration import SimpleMCPClient

    async with SimpleMCPClient() as client:
        # Instantly have access to 20+ tools!

        # Read a file
        content = await client.call("read_file", {"path": "/tmp/data.txt"})

        # Query database
        rows = await client.call("sql_query", {
            "query": "SELECT * FROM users WHERE active = ?",
            "params": {"active": True}
        })

        # Get weather
        weather = await client.call("get_weather", {
            "location": "San Francisco"
        })

        # Execute in parallel
        results = await client.batch([
            ("http_get", {"url": "https://api1.com"}),
            ("http_get", {"url": "https://api2.com"}),
            ("get_metrics", {"metric_name": "traffic"})
        ])
    '''

    print(code_example)

    # Actually run it
    print("Running this code:")
    print("-" * 80)

    from distributed_mcp_hub import SimpleMCPClient

    async with SimpleMCPClient() as client:
        stats = await client.get_stats()
        print(f"\n✓ Connected! {stats['total_tools']} tools available across {stats['total_services']} services\n")

        # Show execution
        print("Executing 3 operations in parallel...")
        results = await client.batch([
            ("read_file", {"path": "/tmp/data.txt"}),
            ("sql_query", {"query": "SELECT COUNT(*) FROM users", "params": {}}),
            ("get_weather", {"location": "San Francisco", "units": "metric"})
        ])

        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['tool']}: {result['result']['status']}")

        print("\n✓ All operations completed!")


async def main():
    """Run all demos"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  MCP Hub + Claude SDK Integration                            ║
║                                                                              ║
║  Demonstrates distributed MCP architecture where:                           ║
║  • Simple clients get instant access to 20+ tools                           ║
║  • Claude can seamlessly use distributed services                           ║
║  • Tools execute across multiple service providers                          ║
║  • Everything auto-discovered and load-balanced                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    await demo_basic_integration()
    await demo_multi_tool_workflow()
    await demo_parallel_execution()
    await demo_direct_client()

    print("\n" + "=" * 80)
    print("All demonstrations completed!")
    print("=" * 80)
    print()
    print("Key takeaways:")
    print("  ✓ Single connection gives access to entire ecosystem")
    print("  ✓ Claude can use any tool transparently")
    print("  ✓ Services are auto-discovered and load-balanced")
    print("  ✓ Parallel execution across distributed infrastructure")
    print("  ✓ Zero configuration required for clients")
    print()


if __name__ == "__main__":
    asyncio.run(main())
