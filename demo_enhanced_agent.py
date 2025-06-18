#!/usr/bin/env python3
"""
Demo script for Enhanced Multi-turn Agent
=========================================

Shows off the key features:
1. Persistent state across sessions
2. Conversational retrieval
3. Direction management
4. Goal tracking
5. FastAPI integration
"""

import asyncio
import sys
from multi_turn_agent_enhanced import (
    StatefulMultiTurnAgent,
    ConversationDirection,
    register_stateful_tools,
    app
)
from multi_turn_agent import Environment, ToolRegistry


async def demo_basic_features():
    """Demonstrate basic enhanced features."""
    print("=== Enhanced Multi-turn Agent Demo ===\n")
    
    # Create agent with persistence
    agent = StatefulMultiTurnAgent(
        session_id="demo-session",
        system_prompt="You are a helpful AI assistant with persistent memory and advanced capabilities.",
        stream=True,
        enable_pivot=True
    )
    
    print(f"Session ID: {agent.session_id}")
    print(f"Initial direction: {agent.state.direction}\n")
    
    # Demo 1: Basic conversation with goals
    print("1. Setting and tracking goals:")
    await agent.add_goal("Learn about the enhanced agent features")
    await agent.add_goal("Test persistent storage")
    
    print("User: What are my current goals?")
    print("Assistant: ", end="")
    response = await agent.send_user("What are my current goals?")
    print(f"\n")
    
    # Demo 2: Direction change
    print("\n2. Changing conversation direction:")
    await agent.change_direction(ConversationDirection.DEBUGGING, "Demonstrating direction change")
    print(f"Direction changed to: {agent.state.direction}")
    
    print("\nUser: Let's debug the storage system")
    print("Assistant: ", end="")
    response = await agent.send_user("Let's debug the storage system")
    print(f"\n")
    
    # Demo 3: Conversational retrieval
    print("\n3. Testing conversational retrieval:")
    print("User: What did we discuss about goals earlier?")
    print("Assistant: ", end="")
    response = await agent.send_user("What did we discuss about goals earlier?", use_retrieval=True)
    print(f"\n")
    
    # Demo 4: Session summary
    print("\n4. Session summary:")
    summary = await agent.get_session_summary()
    print(f"- Turn count: {summary['turn_count']}")
    print(f"- Direction: {summary['direction']}")
    print(f"- Active goals: {summary['active_goals']}")
    print(f"- Direction changes: {summary['direction_changes']}")
    
    return agent.session_id


async def demo_persistence():
    """Demonstrate session persistence."""
    print("\n\n=== Persistence Demo ===\n")
    
    session_id = "persistence-demo"
    
    # First session
    print("Creating first agent instance...")
    agent1 = StatefulMultiTurnAgent(
        session_id=session_id,
        system_prompt="I have persistent memory across sessions."
    )
    
    await agent1.add_goal("Remember this across sessions")
    await agent1.memory.append("user", "My favorite color is blue")
    await agent1.memory.append("assistant", "I'll remember that your favorite color is blue")
    
    print(f"Added goal and conversation to session: {session_id}")
    
    # Simulate restart - create new agent with same session
    print("\nCreating second agent instance with same session ID...")
    agent2 = StatefulMultiTurnAgent(session_id=session_id)
    
    # Wait for state to load
    await asyncio.sleep(0.5)
    
    print("User: What's my favorite color?")
    print("Assistant: ", end="")
    response = await agent2.send_user("What's my favorite color?", use_retrieval=True)
    print(f"\n")
    
    summary = await agent2.get_session_summary()
    print(f"\nPersisted goals: {summary['active_goals']}")
    print(f"Conversation turns: {summary['turn_count']}")


async def demo_stateful_tools():
    """Demonstrate stateful tools."""
    print("\n\n=== Stateful Tools Demo ===\n")
    
    # Create agent with custom tools
    env = Environment()
    registry = ToolRegistry(env)
    
    # Add a custom tool
    @registry.register(description="Perform calculations")
    def calculate(expression: str) -> float:
        """Evaluate a mathematical expression."""
        return eval(expression, {"__builtins__": {}})
    
    agent = StatefulMultiTurnAgent(
        session_id="tools-demo",
        tools_registry=registry,
        stream=False  # Disable streaming for cleaner output
    )
    
    # Register stateful tools
    register_stateful_tools(registry, agent)
    
    print("Available stateful tools:")
    print("- get_session_info")
    print("- search_history")
    print("- change_focus")
    print("- manage_goals")
    print("- calculate")
    
    print("\nUser: Add a goal to master Python, then calculate 15 * 24")
    response = await agent.send_user(
        "Add a goal to master Python, then calculate 15 * 24"
    )
    print(f"Assistant: {response}")
    
    print("\nUser: Now change our focus to creating mode")
    response = await agent.send_user(
        "Now change our focus to creating mode"
    )
    print(f"Assistant: {response}")
    
    # Check final state
    summary = await agent.get_session_summary()
    print(f"\nFinal state:")
    print(f"- Direction: {summary['direction']}")
    print(f"- Goals: {summary['active_goals']}")
    print(f"- Tool usage count: {summary['tool_usage_count']}")


def demo_api_server():
    """Information about the API server."""
    print("\n\n=== FastAPI Server ===\n")
    print("To start the API server, run:")
    print("  python multi_turn_agent_enhanced.py --api")
    print("\nAvailable endpoints:")
    print("  GET  /sessions/{session_id} - Get session state")
    print("  GET  /sessions/{session_id}/messages?query=... - Search messages")
    print("  POST /sessions/{session_id}/checkpoint?name=... - Create checkpoint")
    print("  GET  /sessions/{session_id}/analytics - Get analytics")
    print("\nExample:")
    print("  curl http://localhost:8000/sessions/demo-session")


async def main():
    """Run all demos."""
    try:
        # Run demos
        session_id = await demo_basic_features()
        await demo_persistence()
        await demo_stateful_tools()
        demo_api_server()
        
        print("\n\n=== Demo Complete ===")
        print(f"\nTo continue the demo session, run:")
        print(f"  python multi_turn_agent_enhanced.py --session {session_id}")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async demo
    asyncio.run(main())