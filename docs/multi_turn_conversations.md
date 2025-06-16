# Multi-Turn Conversations Guide

This guide explains how to implement multi-turn conversations using the Claude Code SDK, allowing Claude to maintain context across multiple interactions.

## Overview

Multi-turn conversations enable Claude to remember context from previous messages in a session. This is essential for:
- Building interactive applications
- Maintaining context across multiple queries
- Creating conversational interfaces
- Implementing stateful workflows

## Basic Usage

### Method 1: Using Session IDs

The most explicit way to manage multi-turn conversations is by capturing and reusing the session ID:

```python
import asyncio
from claude_code_sdk import query, ClaudeCodeOptions, ResultMessage

async def multi_turn_example():
    session_id = None
    
    # First turn
    async for message in query(prompt="Remember that I like Python"):
        if isinstance(message, ResultMessage):
            session_id = message.session_id
    
    # Second turn - resume the conversation
    async for message in query(
        prompt="What programming language do I like?",
        options=ClaudeCodeOptions(resume=session_id)
    ):
        print(message)
```

### Method 2: Auto-Continue Last Conversation

For simpler use cases, you can automatically continue the last conversation:

```python
# First query
async for message in query(prompt="I'm working on a web app"):
    pass

# Continue the same conversation
async for message in query(
    prompt="What am I working on?",
    options=ClaudeCodeOptions(continue_conversation=True)
):
    print(message)
```

## Advanced Patterns

### Building a Chat Application

Here's a pattern for building an interactive chat application:

```python
class ChatSession:
    def __init__(self):
        self.session_id = None
        self.history = []
    
    async def send_message(self, user_input: str):
        options = ClaudeCodeOptions(
            resume=self.session_id,
            allowed_tools=["Read", "Write", "Edit"],
            permission_mode="acceptEdits"
        )
        
        self.history.append({"role": "user", "content": user_input})
        
        async for message in query(prompt=user_input, options=options):
            if isinstance(message, ResultMessage):
                self.session_id = message.session_id
            # Process other message types...
```

### Managing Multiple Conversations

You can manage multiple independent conversations:

```python
conversations = {}

async def get_or_create_conversation(conv_id: str, prompt: str):
    options = ClaudeCodeOptions()
    
    if conv_id in conversations:
        options.resume = conversations[conv_id]
    
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            conversations[conv_id] = message.session_id
        yield message
```

## Configuration Options

### Key Options for Multi-Turn

- `resume`: Specific session ID to resume
- `continue_conversation`: Boolean to continue last conversation
- `max_turns`: Limit the number of turns in a conversation

```python
options = ClaudeCodeOptions(
    resume="session-abc-123",     # Resume specific session
    max_turns=10,                 # Limit to 10 turns
    allowed_tools=["Read"],       # Configure tools
    permission_mode="default"     # Set permission handling
)
```

## Best Practices

### 1. Session Management

Always store session IDs for conversations you want to continue:

```python
sessions = {}

async def chat_with_context(user_id: str, message: str):
    options = ClaudeCodeOptions(
        resume=sessions.get(user_id)
    )
    
    async for msg in query(prompt=message, options=options):
        if isinstance(msg, ResultMessage):
            sessions[user_id] = msg.session_id
        yield msg
```

### 2. Error Handling

Handle cases where sessions might expire or become invalid:

```python
try:
    async for message in query(
        prompt="Continue our discussion",
        options=ClaudeCodeOptions(resume=old_session_id)
    ):
        process_message(message)
except Exception as e:
    # Start a new conversation if session is invalid
    async for message in query(prompt="Let's start fresh..."):
        process_message(message)
```

### 3. Context Limits

Be aware that conversations have context limits. For very long conversations, consider:
- Summarizing earlier context
- Starting new sessions periodically
- Storing important information externally

### 4. Cost Tracking

Track costs across multi-turn conversations:

```python
total_cost = 0.0
turn_count = 0

async for message in query(prompt=user_input, options=options):
    if isinstance(message, ResultMessage):
        turn_count += 1
        total_cost = message.total_cost_usd
        print(f"Turn {turn_count}: ${message.cost_usd:.4f}")
        print(f"Total cost: ${total_cost:.4f}")
```

## Common Use Cases

### 1. Interactive Debugging

```python
session_id = None

# Initial code analysis
async for msg in query(prompt="Analyze this Python file: main.py"):
    if isinstance(msg, ResultMessage):
        session_id = msg.session_id

# Follow-up questions
async for msg in query(
    prompt="What are the performance bottlenecks?",
    options=ClaudeCodeOptions(resume=session_id)
):
    # Process response

# Request fixes
async for msg in query(
    prompt="Can you optimize the slow functions?",
    options=ClaudeCodeOptions(resume=session_id)
):
    # Process response
```

### 2. Iterative Development

```python
async def iterative_development():
    session_id = None
    
    steps = [
        "Create a Flask web application structure",
        "Add user authentication",
        "Implement a REST API for user management",
        "Add unit tests for the API"
    ]
    
    for step in steps:
        options = ClaudeCodeOptions(
            resume=session_id,
            allowed_tools=["Write", "Edit", "Bash"],
            permission_mode="acceptEdits"
        )
        
        async for message in query(prompt=step, options=options):
            if isinstance(message, ResultMessage):
                session_id = message.session_id
            # Process messages...
```

### 3. Code Review Assistant

```python
async def code_review_session(file_path: str):
    session_id = None
    
    # Initial review
    prompt = f"Review the code in {file_path} for best practices"
    async for msg in query(prompt=prompt):
        if isinstance(msg, ResultMessage):
            session_id = msg.session_id
    
    # Ask for specific improvements
    while True:
        user_question = input("Ask about the code (or 'done'): ")
        if user_question.lower() == 'done':
            break
        
        async for msg in query(
            prompt=user_question,
            options=ClaudeCodeOptions(resume=session_id)
        ):
            # Display response
            pass
```

## Troubleshooting

### Session Not Found

If you get errors about session not found:
1. The session may have expired
2. The session ID might be incorrect
3. Start a new conversation without the `resume` parameter

### Context Lost

If Claude doesn't remember previous context:
1. Ensure you're passing the correct session ID
2. Check that you're using `resume` or `continue_conversation`
3. Verify the session hasn't exceeded turn limits

### Performance Tips

1. **Reuse sessions** when possible to avoid re-establishing context
2. **Batch related queries** in the same session
3. **Clear context** when switching topics by starting new sessions
4. **Monitor costs** as multi-turn conversations accumulate usage

## Examples

See the following example files for complete implementations:
- `examples/multi_turn_example.py` - Basic multi-turn usage
- `examples/interactive_chat.py` - Full interactive chat application
- `examples/advanced_multi_agents.py` - Multi-agent conversations