"""
Test script to demonstrate the new multi-turn agent features
"""

import asyncio
from multi_turn_agent import MultiTurnAgent, Environment, tools
from claude_max_agent import ClaudeMaxAgent


async def test_conversational_retrieval():
    """Test the new conversational retrieval capabilities."""
    print("=== Testing Conversational Retrieval ===\n")
    
    agent = MultiTurnAgent(
        system_prompt="You are a helpful assistant with perfect memory."
    )
    
    # Have a conversation
    messages = [
        "My name is Alice and I work at TechCorp",
        "I'm working on a machine learning project about sentiment analysis",
        "The project uses transformer models and has a deadline next month",
        "Let me tell you about something else - I also enjoy hiking",
        "What was my name and where do I work?",  # Test retrieval
    ]
    
    for msg in messages:
        print(f"User: {msg}")
        response = await agent.send_user(msg, use_retrieval=True)
        print(f"Assistant: {response}\n")
    
    # Test search functionality
    print("\n--- Testing Search ---")
    results = await agent.search_history("machine learning project")
    print(f"Found {len(results)} relevant messages about 'machine learning project'")
    for i, msg in enumerate(results):
        print(f"{i+1}. [{msg.get('role')}] {msg.get('content', '')[:100]}...")
    
    # Test context retrieval
    print("\n--- Testing Context Retrieval ---")
    context = agent.get_relevant_context("transformer deadline")
    print(f"Retrieved {len(context)} relevant context messages")


async def test_environment_extras():
    """Test the fixed Environment extras functionality."""
    print("\n=== Testing Environment Extras ===\n")
    
    env = Environment()
    
    # Test setting and getting extras
    env["project_name"] = "AI Assistant"
    env["version"] = "2.0"
    env["features"] = ["retrieval", "claude_max", "multi-turn"]
    
    print(f"Project: {env['project_name']}")
    print(f"Version: {env['version']}")
    print(f"Features: {env['features']}")
    
    # Test dict export
    env_dict = env.dict()
    print(f"\nEnvironment as dict: {list(env_dict.keys())}")
    print(f"Has extras: {'project_name' in env_dict}")


async def test_claude_max_integration():
    """Test Claude Max integration (requires claude CLI)."""
    print("\n=== Testing Claude Max Integration ===\n")
    
    # Check if claude CLI is available
    import subprocess
    try:
        subprocess.run(["claude", "--version"], capture_output=True, check=True)
        print("Claude CLI detected ✓")
    except:
        print("Claude CLI not found - skipping Claude Max tests")
        return
    
    agent = ClaudeMaxAgent(track_subtasks=True)
    
    # Test complexity analysis
    simple_task = "What is 2 + 2?"
    complex_task = "Explain the mathematical foundations of transformer attention mechanisms and derive the computational complexity"
    
    simple_score = await agent.analyze_complexity(simple_task)
    complex_score = await agent.analyze_complexity(complex_task)
    
    print(f"Simple task complexity: {simple_score}/5")
    print(f"Complex task complexity: {complex_score}/5")


async def test_advanced_tools():
    """Test the advanced tool functionality."""
    print("\n=== Testing Advanced Tools ===\n")
    
    # Test tool with environment injection
    @tools.register(description="Get project info")
    def get_project_info(env: Environment) -> str:
        return f"Project: {env.get('project_name', 'Unknown')}"
    
    # Set project name in environment
    env = Environment()
    env["project_name"] = "Multi-turn Agent"
    
    # Create agent with custom environment
    agent = MultiTurnAgent()
    agent.tools_registry._env["project_name"] = "Multi-turn Agent"
    
    # Test the tool
    response = await agent.send_user("What project are we working on?")
    print(f"Response: {response}")


async def main():
    """Run all tests."""
    print("Multi-turn Agent New Features Test\n")
    print("=" * 50)
    
    await test_environment_extras()
    await test_conversational_retrieval()
    await test_claude_max_integration()
    await test_advanced_tools()
    
    print("\n" + "=" * 50)
    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())