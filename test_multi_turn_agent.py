"""
Tests for Multi-turn Function Calling Agent
==========================================
"""

import asyncio
import pytest
from typing import Any
from unittest.mock import Mock, patch, AsyncMock
from multi_turn_agent import (
    Environment, ToolRegistry, ConversationMemory,
    MultiTurnAgent, tools, make_message
)


# ————————————————————————————————————————————————————————————————
# Environment Tests
# ————————————————————————————————————————————————————————————————

def test_environment_defaults():
    """Test Environment default values."""
    env = Environment()
    assert env.default_model == "gpt-4o-mini"
    assert env.temperature == 0.7
    assert env.max_context_tokens == 200_000


def test_environment_dynamic_attributes():
    """Test dynamic attribute setting."""
    env = Environment()
    
    # Set dynamic attribute
    env["custom_key"] = "custom_value"
    assert env["custom_key"] == "custom_value"
    
    # Access via getattr
    assert env.custom_key == "custom_value"
    
    # Non-existent attribute returns None
    assert env.non_existent is None


def test_environment_dict_export():
    """Test exporting environment as dict."""
    env = Environment(temperature=0.5)
    env["custom"] = "value"
    
    d = env.dict()
    assert d["temperature"] == 0.5
    assert d["custom"] == "value"


# ————————————————————————————————————————————————————————————————
# ToolRegistry Tests
# ————————————————————————————————————————————————————————————————

def test_tool_registration():
    """Test basic tool registration."""
    env = Environment()
    registry = ToolRegistry(env)
    
    @registry.register
    def test_tool(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y
    
    assert "test_tool" in registry._tools
    assert len(registry.schemas) == 1
    
    schema = registry.schemas[0]
    assert schema["function"]["name"] == "test_tool"
    assert schema["function"]["description"] == "Add two numbers."


def test_tool_with_env_injection():
    """Test tool with environment injection."""
    env = Environment(temperature=0.9)
    registry = ToolRegistry(env)
    
    @registry.register
    def get_temp(env: Environment) -> float:
        """Get current temperature."""
        return env.temperature
    
    # Should not include 'env' in parameters
    schema = registry.schemas[0]
    assert "env" not in schema["function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_tool_execution():
    """Test tool execution."""
    env = Environment()
    registry = ToolRegistry(env)
    
    @registry.register
    def multiply(x: int, y: int) -> int:
        return x * y
    
    result = await registry.call("multiply", x=3, y=4)
    assert result == 12


@pytest.mark.asyncio
async def test_tool_execution_with_env():
    """Test tool execution with environment injection."""
    env = Environment()
    env["stored_value"] = 42
    registry = ToolRegistry(env)
    
    @registry.register
    def get_stored(key: str, env: Environment) -> Any:
        return env._extras.get(key)
    
    result = await registry.call("get_stored", key="stored_value")
    assert result == 42


# ————————————————————————————————————————————————————————————————
# ConversationMemory Tests
# ————————————————————————————————————————————————————————————————

@pytest.mark.asyncio
async def test_memory_append():
    """Test appending to conversation memory."""
    memory = ConversationMemory(max_tokens=1000, threshold_words=100)
    
    await memory.append("user", "Hello")
    await memory.append("assistant", "Hi there!")
    
    assert len(memory.history) == 2
    assert memory.history[0]["role"] == "user"
    assert memory.history[0]["content"] == "Hello"


def test_word_count():
    """Test word counting in memory."""
    memory = ConversationMemory(max_tokens=1000, threshold_words=100)
    memory.history = [
        make_message("user", "Hello world"),
        make_message("assistant", "Hi there, how are you?"),
    ]
    
    assert memory._word_count() == 7  # 2 + 5 words


@pytest.mark.asyncio
async def test_memory_summarization():
    """Test automatic summarization."""
    memory = ConversationMemory(max_tokens=1000, threshold_words=10)
    
    # Mock the OpenAI client
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Summary of conversation"))]
    
    with patch.object(memory._client.chat.completions, 'create', 
                     return_value=AsyncMock(return_value=mock_response)()):
        # Add enough words to trigger summarization
        await memory.append("user", "This is a long message with many words")
        await memory.append("assistant", "This is another long response with many words")
        
        # Should trigger summarization
        assert any("Summary" in msg["content"] for msg in memory.history)


# ————————————————————————————————————————————————————————————————
# MultiTurnAgent Tests
# ————————————————————————————————————————————————————————————————

@pytest.mark.asyncio
async def test_agent_initialization():
    """Test agent initialization with system prompt."""
    agent = MultiTurnAgent(system_prompt="You are a test assistant")
    
    assert len(agent.memory.history) == 1
    assert agent.memory.history[0]["role"] == "system"
    assert agent.memory.history[0]["content"] == "You are a test assistant"


@pytest.mark.asyncio
async def test_agent_simple_conversation():
    """Test simple conversation without tools."""
    agent = MultiTurnAgent()
    
    # Mock the chat function
    async def mock_chat(*args, **kwargs):
        yield {"token": "Hello"}
        yield {"token": " there!"}
        yield {"role": "assistant", "content": "Hello there!", "index": 0}
    
    with patch('multi_turn_agent.chat', mock_chat):
        response = await agent.send_user("Hi")
        assert response == "Hello there!"
        assert len(agent.memory.history) == 2  # user + assistant


@pytest.mark.asyncio
async def test_agent_with_tool_execution():
    """Test agent executing tools."""
    # Create custom registry for testing
    env = Environment()
    test_registry = ToolRegistry(env)
    
    @test_registry.register
    def test_add(x: int, y: int) -> int:
        return x + y
    
    agent = MultiTurnAgent(tools_registry=test_registry)
    
    # Mock chat to return tool call
    async def mock_chat(*args, **kwargs):
        # First call - assistant wants to use tool
        if len(agent.memory.history) == 1:  # Just user message
            yield {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "test_add",
                        "arguments": '{"x": 2, "y": 3}'
                    }
                }],
                "index": 0
            }
        else:  # After tool execution
            yield {
                "role": "assistant",
                "content": "The result is 5",
                "index": 0
            }
    
    with patch('multi_turn_agent.chat', mock_chat):
        response = await agent.send_user("What is 2 + 3?")
        assert response == "The result is 5"
        
        # Check history includes tool result
        assert any(msg.get("role") == "tool" for msg in agent.memory.history)


@pytest.mark.asyncio
async def test_agent_tool_error_handling():
    """Test agent handling tool execution errors."""
    env = Environment()
    test_registry = ToolRegistry(env)
    
    @test_registry.register
    def failing_tool() -> None:
        raise ValueError("Tool failed!")
    
    agent = MultiTurnAgent(tools_registry=test_registry)
    
    async def mock_chat(*args, **kwargs):
        if len(agent.memory.history) == 1:
            yield {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_456",
                    "type": "function", 
                    "function": {
                        "name": "failing_tool",
                        "arguments": '{}'
                    }
                }],
                "index": 0
            }
        else:
            yield {
                "role": "assistant",
                "content": "The tool encountered an error",
                "index": 0
            }
    
    with patch('multi_turn_agent.chat', mock_chat):
        response = await agent.send_user("Run the failing tool")
        
        # Check that error was captured in history
        tool_results = [msg for msg in agent.memory.history if msg.get("role") == "tool"]
        assert len(tool_results) == 1
        assert "Error:" in tool_results[0]["content"]


@pytest.mark.asyncio  
async def test_agent_history_management():
    """Test conversation history management."""
    agent = MultiTurnAgent(system_prompt="Test system")
    
    # Add some history
    agent.memory.history.extend([
        make_message("user", "Message 1"),
        make_message("assistant", "Response 1"),
        make_message("user", "Message 2"),
        make_message("assistant", "Response 2"),
    ])
    
    # Get history
    history = await agent.get_history()
    assert len(history) == 5  # system + 4 messages
    
    # Clear history keeping system
    await agent.clear_history(keep_system=True)
    history = await agent.get_history()
    assert len(history) == 1
    assert history[0]["role"] == "system"
    
    # Clear all history
    await agent.clear_history(keep_system=False)
    history = await agent.get_history()
    assert len(history) == 0


# ————————————————————————————————————————————————————————————————
# Integration Tests
# ————————————————————————————————————————————————————————————————

@pytest.mark.asyncio
async def test_builtin_tools():
    """Test the built-in example tools."""
    # Test hello tool
    result = await tools.call("hello", name="Test")
    assert "Hello Test!" in result
    assert "gpt-4o-mini" in result
    
    # Test calculate tool  
    result = await tools.call("calculate", expression="2 + 2")
    assert result == 4.0
    
    # Test datetime tool
    result = await tools.call("get_datetime")
    assert len(result) == 19  # YYYY-MM-DD HH:MM:SS format
    
    # Test remember/recall tools
    await tools.call("remember", key="test_key", value="test_value")
    result = await tools.call("recall", key="test_key")
    assert "test_value" in result


# ————————————————————————————————————————————————————————————————
# Run Tests
# ————————————————————————————————————————————————————————————————

if __name__ == "__main__":
    # Run with pytest for better output
    pytest.main([__file__, "-v"])
    
    # Or run directly with asyncio
    # asyncio.run(test_agent_simple_conversation())