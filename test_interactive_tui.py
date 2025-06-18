#!/usr/bin/env python3
"""
Test script for Interactive TUI - Demonstrates agent UI control
"""

import asyncio
from multi_turn_agent_enhanced import EnhancedStatefulAgent
from multi_turn_agent_interactive_tui import (
    InteractiveAgentTUI, register_interactive_demo_tools
)

async def demo_ui_control():
    """Demonstrate the agent controlling its own UI."""
    
    # Create agent with a special system prompt
    agent = EnhancedStatefulAgent(
        system_prompt="""You are an enhanced AI assistant with full control over your terminal interface.
You can:
1. Send messages with msg_user()
2. Show notifications with notify()
3. Update panels with update_panel()
4. Create custom panels with create_custom_panel()
5. Show your thinking with show_thinking()
6. Display progress with show_progress()
7. Create visualizations with show_chart()
8. Control themes with set_theme()
9. Export views with export_view()

Be creative and proactive in using these UI capabilities to enhance our interaction!
When users ask you to demonstrate, show off multiple features.""",
        stream=False,
        num_workers=4
    )
    
    # Register demo tools
    register_interactive_demo_tools(agent)
    
    # Create TUI
    tui = InteractiveAgentTUI(agent)
    
    # Pre-load some demo interactions
    demo_prompts = [
        "Please demonstrate your UI control capabilities",
        "Create a workflow for 'Data Analysis' with steps: Load, Clean, Analyze, Visualize",
        "Show me your current thoughts about this conversation",
        "Create a custom panel showing system performance metrics",
        "Send me a success notification about task completion",
        "Change the theme to contrast mode",
        "Show me a progress update for an imaginary long-running task",
        "Export our conversation to markdown format"
    ]
    
    # Add first prompt to queue
    tui.input_queue.put(demo_prompts[0])
    
    # Run TUI
    await tui.run()

async def test_specific_features():
    """Test specific UI features."""
    agent = EnhancedStatefulAgent()
    
    # Test individual UI functions
    print("Testing UI control functions...")
    
    # Test message sending
    result = await agent.tools_registry.call("msg_user", 
                                           content="Hello from the agent!", 
                                           style="bold")
    print(f"msg_user result: {result}")
    
    # Test notification
    result = await agent.tools_registry.call("notify",
                                           message="Task completed!",
                                           type="success",
                                           duration=3.0)
    print(f"notify result: {result}")
    
    # Test panel creation
    result = await agent.tools_registry.call("create_custom_panel",
                                           panel_id="test_panel",
                                           title="Test Panel",
                                           content="This is a test panel",
                                           position="sidebar")
    print(f"create_custom_panel result: {result}")
    
    print("\nAll UI control functions working!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run tests
        asyncio.run(test_specific_features())
    else:
        # Run demo
        print("🚀 Starting Interactive TUI Demo")
        print("The agent will demonstrate its UI control capabilities")
        print("-" * 60)
        asyncio.run(demo_ui_control())