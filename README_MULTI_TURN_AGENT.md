# Multi-turn Function Calling Agent

A robust, async-friendly multi-turn conversation agent with function calling capabilities for OpenAI API.

## Features

- 🔄 **Multi-turn Conversations**: Maintains conversation history with automatic context management
- 🛠️ **Function Calling**: Register and execute Python functions as OpenAI tools
- 🌊 **Streaming Support**: Real-time token streaming for responsive interactions
- 🧠 **Memory Management**: Automatic summarization when context grows large
- ⚙️ **Dynamic Configuration**: Environment-based settings with runtime modifications
- 🔌 **Claude Max Integration**: Delegate complex tasks to claude_max CLI
- 🎯 **Specialized Agents**: Pre-configured agents for research, coding, and personal assistance
- ⚡ **Async First**: Built on asyncio for concurrent operations
- 🔧 **Extensible**: Easy to add custom tools and create specialized agents

## Installation

```bash
# Install required dependencies
pip install openai pydantic tenacity aiohttp

# For Claude Max integration (optional)
npm install -g @anthropic-ai/claude-code
```

## Quick Start

```python
import asyncio
from multi_turn_agent import MultiTurnAgent, tools

# Register a custom tool
@tools.register(description="Get the current weather")
async def get_weather(location: str) -> str:
    return f"The weather in {location} is sunny and 72°F"

# Create and use the agent
async def main():
    agent = MultiTurnAgent(
        system_prompt="You are a helpful assistant with weather access."
    )
    
    response = await agent.send_user("What's the weather in Paris?")
    print(response)

asyncio.run(main())
```

## Core Components

### 1. Environment Configuration

```python
from multi_turn_agent import Environment, env

# Configure environment
env.api_key = "your-openai-api-key"
env.default_model = "gpt-4"
env.temperature = 0.7

# Add custom configuration
env["custom_setting"] = "value"
```

### 2. Tool Registry

```python
from multi_turn_agent import tools, Environment

# Simple tool
@tools.register(description="Add two numbers")
def add(x: int, y: int) -> int:
    return x + y

# Tool with environment access
@tools.register(description="Get config value")
def get_config(key: str, env: Environment) -> str:
    return env[key]
```

### 3. Multi-turn Agent

```python
agent = MultiTurnAgent(
    system_prompt="Your custom system prompt",
    tools_registry=tools,  # Optional: custom tool registry
    stream=True  # Enable streaming
)

# Send messages
response = await agent.send_user(
    "Your message",
    auto_execute_tools=True,  # Auto-execute tool calls
    max_tool_rounds=10  # Max rounds of tool execution
)

# Get conversation history
history = await agent.get_history()

# Clear history
await agent.clear_history(keep_system=True)
```

## Advanced Usage

### Specialized Agents

```python
from advanced_agent_examples import ResearchAgent, CodeAssistant, PersonalAssistant

# Research agent for information gathering
research = ResearchAgent()
await research.send_user("Research quantum computing applications")

# Code assistant for programming help  
coder = CodeAssistant()
await coder.send_user("Write a binary search function")

# Personal assistant for daily tasks
assistant = PersonalAssistant()
await assistant.send_user("Add a reminder to review PRs")
```

### Claude Max Integration

```python
from claude_max_agent import ClaudeMaxAgent

# Create agent with Claude Max access
agent = ClaudeMaxAgent(track_subtasks=True)

# Delegate complex reasoning
await agent.send_user(
    "Analyze this architecture and suggest improvements: ..."
)

# Get subtask history
subtasks = agent.get_subtask_history()
```

### Custom Tools Examples

```python
# Async tool with external API
@tools.register(description="Fetch data from API")
async def fetch_data(endpoint: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(endpoint) as response:
            return await response.json()

# Tool with complex logic
@tools.register(description="Process dataset")
def process_data(
    data: list,
    operation: str = "sum",
    env: Environment = None
) -> float:
    if operation == "sum":
        return sum(data)
    elif operation == "mean":
        return sum(data) / len(data)
    else:
        raise ValueError(f"Unknown operation: {operation}")
```

## Built-in Tools

- `hello(name)` - Greeting with environment info
- `calculate(expression)` - Mathematical calculations
- `get_datetime()` - Current date and time
- `remember(key, value)` - Store values in environment
- `recall(key)` - Retrieve stored values

## Memory Management

The agent automatically manages conversation context:

```python
# Configure memory limits
agent = MultiTurnAgent()
agent.memory.threshold_words = 2000  # Summarize after 2000 words
agent.memory.max_tokens = 100_000    # Maximum context size
```

## Error Handling

```python
try:
    response = await agent.send_user("Your message")
except KeyError as e:
    print(f"Tool not found: {e}")
except Exception as e:
    print(f"Error: {e}")
```

## Testing

Run the test suite:

```bash
# Run all tests
python test_multi_turn_agent.py

# Or with pytest
pytest test_multi_turn_agent.py -v
```

## Examples

### Interactive Chat

```python
from multi_turn_agent import MultiTurnAgent

async def interactive_chat():
    agent = MultiTurnAgent()
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        print("AI: ", end="")
        response = await agent.send_user(user_input)
        print()

asyncio.run(interactive_chat())
```

### Multi-Agent Collaboration

```python
from advanced_agent_examples import ResearchAgent, CodeAssistant

async def collaborative_task():
    researcher = ResearchAgent()
    coder = CodeAssistant()
    
    # Research phase
    research = await researcher.send_user(
        "Research best practices for REST API design"
    )
    
    # Implementation phase
    code = await coder.send_user(
        f"Based on this research: {research}\n"
        "Generate a Python REST API example"
    )
    
    return code

asyncio.run(collaborative_task())
```

## Architecture

```
multi_turn_agent.py
├── Environment          # Dynamic configuration container
├── ToolRegistry        # Function registration and execution
├── ConversationMemory  # History with auto-summarization
├── chat()             # Low-level streaming chat function
└── MultiTurnAgent     # High-level conversation manager

advanced_agent_examples.py
├── Advanced Tools      # Weather, TODOs, code execution, etc.
├── Specialized Agents  # Research, Code, Personal assistants
└── Demo Scenarios     # Example usage patterns

claude_max_agent.py
├── Claude Max Tools   # Complex reasoning delegation
├── ClaudeMaxAgent    # Agent with subtask tracking
└── Advanced Demos    # Complex problem solving examples
```

## Contributing

Feel free to extend the agent with:
- New tool implementations
- Specialized agent types
- Additional memory strategies
- Enhanced error handling
- Performance optimizations

## License

This code is provided as-is for demonstration purposes.