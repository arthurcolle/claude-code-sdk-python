"""Example of using tool execution hooks for monitoring and control."""

import asyncio
from datetime import datetime
from claude_code_sdk import query
from claude_code_sdk.tools import (
    ToolExecutionHooks,
    ToolRegistryOptions,
    EnhancedClient,
)
from claude_code_sdk.types import ToolUseBlock, ToolResultBlock


# Custom hook handlers
def on_tool_use(tool_use: ToolUseBlock) -> None:
    """Log when a tool is used."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] Tool Use Detected:")
    print(f"  Tool: {tool_use.name}")
    print(f"  ID: {tool_use.id}")
    print(f"  Input: {tool_use.input}")


def on_tool_result(tool_result: ToolResultBlock) -> None:
    """Log tool execution results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] Tool Result:")
    print(f"  Tool Use ID: {tool_result.tool_use_id}")
    print(f"  Success: {not tool_result.is_error}")
    if tool_result.content:
        print(f"  Content: {tool_result.content}")


async def example_with_monitoring():
    """Example using hooks for monitoring tool usage."""
    print("=== Example 1: Monitoring Tool Usage ===")
    
    # Create hooks with monitoring callbacks
    hooks = ToolExecutionHooks(
        on_tool_use=on_tool_use,
        on_tool_result=on_tool_result,
        use_remote_tools=False  # Just monitor, don't execute remotely
    )
    
    # Create options
    options = ToolRegistryOptions(
        allowed_tools=["Read", "Write", "Bash"],
        permission_mode="acceptEdits"
    )
    
    # Use enhanced client with hooks
    client = EnhancedClient(tool_hooks=hooks)
    
    async for message in client.process_query(
        prompt="Read the current directory listing",
        options=options
    ):
        print(f"\nMessage: {message}")


async def example_with_remote_execution():
    """Example using remote tool execution."""
    print("\n\n=== Example 2: Remote Tool Execution ===")
    
    # Create hooks for remote execution
    hooks = ToolExecutionHooks(
        tool_api_url="https://arthurcolle--registry.modal.run",
        use_remote_tools=True,
        on_tool_use=lambda t: print(f"Executing {t.name} remotely..."),
        on_tool_result=lambda r: print(f"Remote result: {r.content}")
    )
    
    # Create options with remote tools enabled
    options = ToolRegistryOptions(
        use_remote_tools=True,
        tool_namespace="examples"
    )
    
    # Use enhanced client
    client = EnhancedClient(tool_hooks=hooks)
    
    async with hooks:
        async for message in client.process_query(
            prompt="Use a calculator tool to compute 123 * 456",
            options=options
        ):
            print(f"\nMessage type: {type(message).__name__}")


async def example_with_tool_filtering():
    """Example with custom tool filtering logic."""
    print("\n\n=== Example 3: Tool Filtering ===")
    
    # Track allowed tools
    allowed_remote_tools = {"calculator", "string_reverser", "weather"}
    
    def filter_tool_use(tool_use: ToolUseBlock) -> None:
        """Only allow certain tools to be executed remotely."""
        if tool_use.name not in allowed_remote_tools:
            print(f"⚠️  Tool '{tool_use.name}' not in allowed list for remote execution")
        else:
            print(f"✓ Tool '{tool_use.name}' approved for remote execution")
    
    # Create hooks with filtering
    hooks = ToolExecutionHooks(
        on_tool_use=filter_tool_use,
        use_remote_tools=True
    )
    
    # Only certain tools will be executed remotely
    options = ToolRegistryOptions(
        allowed_tools=["Read", "calculator", "string_reverser"],
        use_remote_tools=True
    )
    
    client = EnhancedClient(tool_hooks=hooks)
    
    async with hooks:
        async for message in client.process_query(
            prompt="Read README.md and then use calculator to add 10 + 20",
            options=options
        ):
            # Process messages
            pass


async def example_with_tool_metrics():
    """Example collecting metrics about tool usage."""
    print("\n\n=== Example 4: Tool Usage Metrics ===")
    
    # Metrics collection
    tool_metrics = {
        "total_uses": 0,
        "total_errors": 0,
        "tools_used": {},
        "execution_times": []
    }
    
    start_times = {}
    
    def track_tool_use(tool_use: ToolUseBlock) -> None:
        """Track tool usage metrics."""
        tool_metrics["total_uses"] += 1
        tool_name = tool_use.name
        
        if tool_name not in tool_metrics["tools_used"]:
            tool_metrics["tools_used"][tool_name] = 0
        tool_metrics["tools_used"][tool_name] += 1
        
        # Track start time
        start_times[tool_use.id] = datetime.now()
    
    def track_tool_result(tool_result: ToolResultBlock) -> None:
        """Track tool results and timing."""
        if tool_result.is_error:
            tool_metrics["total_errors"] += 1
        
        # Calculate execution time
        if tool_result.tool_use_id in start_times:
            duration = (datetime.now() - start_times[tool_result.tool_use_id]).total_seconds()
            tool_metrics["execution_times"].append(duration)
            del start_times[tool_result.tool_use_id]
    
    # Create hooks with metrics tracking
    hooks = ToolExecutionHooks(
        on_tool_use=track_tool_use,
        on_tool_result=track_tool_result
    )
    
    options = ToolRegistryOptions(
        allowed_tools=["Read", "Write", "Bash"]
    )
    
    client = EnhancedClient(tool_hooks=hooks)
    
    # Run some operations
    async for message in client.process_query(
        prompt="List files in current directory, read README.md if it exists",
        options=options
    ):
        # Process messages
        pass
    
    # Display metrics
    print("\n📊 Tool Usage Metrics:")
    print(f"  Total tool uses: {tool_metrics['total_uses']}")
    print(f"  Total errors: {tool_metrics['total_errors']}")
    print(f"  Tools used:")
    for tool, count in tool_metrics["tools_used"].items():
        print(f"    - {tool}: {count} times")
    
    if tool_metrics["execution_times"]:
        avg_time = sum(tool_metrics["execution_times"]) / len(tool_metrics["execution_times"])
        print(f"  Average execution time: {avg_time:.3f} seconds")


async def main():
    """Run all examples."""
    await example_with_monitoring()
    await example_with_remote_execution()
    await example_with_tool_filtering()
    await example_with_tool_metrics()


if __name__ == "__main__":
    asyncio.run(main())