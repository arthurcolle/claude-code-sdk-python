"""
Multi-turn Function Calling Agent
=====================================
A robust, async-friendly multi-turn agent with function calling capabilities.

Features:
- Dynamic environment configuration
- Tool registry with automatic environment injection
- Multi-turn conversation support with memory
- Streaming and non-streaming responses
- Automatic context summarization
- Retry logic with exponential backoff
- Full OpenAI function calling support
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import (
    Any, AsyncIterator, Callable, Dict, List, Literal,
    MutableMapping, Optional, Sequence, Type, Union
)

import openai
from pydantic import BaseModel, Field, PrivateAttr
from tenacity import (
    AsyncRetrying, retry_if_exception_type,
    stop_after_attempt, wait_exponential
)

# ————————————————————————————————————————————————————————————————
# Logging & Configuration
# ————————————————————————————————————————————————————————————————
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ————————————————————————————————————————————————————————————————
# Dynamic Environment & Tool Registry
# ————————————————————————————————————————————————————————————————
class Environment(BaseModel):
    """
    Dynamic container for runtime configuration.
    
    - Values pulled from constructor kwargs or env vars
    - Unknown attributes return None instead of AttributeError
    - New keys can be added at runtime via env["key"] = value
    """
    
    api_key: str | None = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    default_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
    )
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    )
    max_context_tokens: int = Field(default=200_000)
    summarize_threshold_words: int = Field(default=3_000)
    retry_attempts: int = Field(default=3)
    
    # Private field for dynamic attributes
    _extras: dict[str, Any] = PrivateAttr(default_factory=dict)
    
    def __getattr__(self, item: str) -> Any | None:
        """Fallback to extras or None for undefined attributes."""
        # Avoid recursion by accessing _extras directly from __dict__
        if item == '_extras':
            return None
        extras = self.__dict__.get('_extras', {})
        return extras.get(item)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-like assignment for dynamic attributes."""
        # Ensure _extras is initialized
        if '_extras' not in self.__dict__:
            self.__dict__['_extras'] = {}
        self.__dict__['_extras'][key] = value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access."""
        if hasattr(self, key) and key != '_extras':
            return getattr(self, key)
        extras = self.__dict__.get('_extras', {})
        return extras[key]
    
    def dict(self, **kwargs) -> dict:
        """Export all attributes including extras."""
        base = super().model_dump(**kwargs)
        extras = self.__dict__.get('_extras', {})
        return {**base, **extras}


# Global environment instance
env = Environment()


def _py_to_json_type(py_type: Any) -> str:
    """Convert Python type hints to JSON Schema types."""
    if py_type in {int, float}:
        return "number"
    elif py_type is bool:
        return "boolean"
    elif py_type in {list, List}:
        return "array"
    elif py_type in {dict, Dict}:
        return "object"
    return "string"


class ToolRegistry:
    """
    Registry for function calling tools with environment injection.
    
    Functions can declare 'env' parameter to receive the environment automatically.
    """
    
    def __init__(self, env: Environment):
        self._env = env
        self._tools: dict[str, Callable] = {}
        self._schemas: list[dict] = []
    
    def register(self, fn: Callable | None = None, **meta):
        """
        Register a function as a tool.
        
        Can be used as decorator:
            @tools.register
            def my_tool(...): ...
            
        Or imperatively:
            tools.register(my_function, description="...")
        """
        def _inner(f: Callable):
            name = meta.get("name") or f.__name__
            if name in self._tools:
                raise ValueError(f"Tool '{name}' already registered")
            
            sig = inspect.signature(f)
            params_schema: dict[str, dict] = {}
            required_params = []
            
            for param_name, param in sig.parameters.items():
                if param_name == "env":  # Skip env injection param
                    continue
                    
                param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
                param_schema = {"type": _py_to_json_type(param_type)}
                
                if param.default == inspect.Parameter.empty:
                    required_params.append(param_name)
                    
                params_schema[param_name] = param_schema
            
            self._tools[name] = f
            self._schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": meta.get("description", f.__doc__ or ""),
                    "parameters": {
                        "type": "object",
                        "properties": params_schema,
                        "required": required_params,
                    },
                },
            })
            return f
        
        if fn is None:
            return _inner
        return _inner(fn)
    
    async def call(self, tool_name: str, **kwargs) -> Any:
        """Call a registered tool, auto-injecting env if needed."""
        if tool_name not in self._tools:
            raise KeyError(f"No such tool '{tool_name}'")
        
        fn = self._tools[tool_name]
        sig = inspect.signature(fn)
        
        # Auto-inject env if function expects it
        if "env" in sig.parameters or any(
            p.annotation is Environment for p in sig.parameters.values()
        ):
            kwargs = {"env": self._env, **kwargs}
        
        result = fn(**kwargs)
        return await result if inspect.iscoroutine(result) else result
    
    @property
    def schemas(self) -> list[dict]:
        """Get OpenAI-compatible tool schemas."""
        return self._schemas


# Global tool registry
tools = ToolRegistry(env=env)


# ————————————————————————————————————————————————————————————————
# Memory Management
# ————————————————————————————————————————————————————————————————
R, C = "role", "content"
U, A, S = "user", "assistant", "system"
Message = Dict[str, Any]


def make_message(role: str, content: Any) -> Message:
    """Create a properly formatted message."""
    return {R: role, C: content}


class ConversationMemory:
    """Memory with automatic summarization when context grows too large."""
    
    def __init__(self, max_tokens: int, threshold_words: int):
        self.history: list[Message] = []
        self.max_tokens = max_tokens
        self.threshold_words = threshold_words
        self._client = openai.AsyncOpenAI(api_key=env.api_key)
        self.context_index: dict[str, list[int]] = {}  # Index for retrieval
        self.summaries: list[dict] = []  # Store summaries for retrieval
    
    async def append(self, role: str, content: Any):
        """Add message and summarize if needed."""
        self.history.append(make_message(role, content))
        if self._word_count() > self.threshold_words:
            await self._summarize()
    
    def _word_count(self) -> int:
        """Count total words in history."""
        count = 0
        for m in self.history:
            if C in m:
                count += len(str(m[C]).split())
            # For tool calls, estimate word count from the JSON structure
            if "tool_calls" in m:
                count += len(str(m["tool_calls"]).split())
        return count
    
    async def _summarize(self):
        """Summarize conversation to reduce context size."""
        logger.info(f"Context exceeded {self.threshold_words} words, summarizing...")
        
        # Preserve system message if exists
        system_msgs = [m for m in self.history if m[R] == S]
        other_msgs = [m for m in self.history if m[R] != S]
        
        # Build prompt, handling messages with and without content
        prompt_parts = []
        for m in other_msgs:
            if C in m:
                prompt_parts.append(f"[{m[R]}] {m[C]}")
            elif "tool_calls" in m:
                prompt_parts.append(f"[{m[R]}] [Tool calls: {len(m['tool_calls'])} function(s) called]")
        prompt = "\n".join(prompt_parts)
        
        summary_resp = await self._client.chat.completions.create(
            model=env.default_model,
            messages=[
                make_message(S, "Summarize the following conversation concisely:"),
                make_message(U, prompt),
            ],
            temperature=0,
            stream=False,
        )
        
        summary_text = summary_resp.choices[0].message.content
        
        # Store summary for retrieval
        self.summaries.append({
            "summary": summary_text,
            "messages": other_msgs.copy(),
            "timestamp": datetime.now().isoformat()
        })
        
        self.history = system_msgs + [make_message(S, f"Previous conversation summary: {summary_text}")]
    
    async def search(self, query: str, top_k: int = 3) -> list[Message]:
        """Search conversation history for relevant messages."""
        query_lower = query.lower()
        scores = []
        
        # Score current history
        for i, msg in enumerate(self.history):
            content = str(msg.get(C, "")).lower()
            score = sum(1 for word in query_lower.split() if word in content)
            if score > 0:
                scores.append((score, i, msg))
        
        # Score summaries
        for summary_data in self.summaries:
            summary_text = summary_data["summary"].lower()
            score = sum(1 for word in query_lower.split() if word in summary_text)
            if score > 0:
                for msg in summary_data["messages"]:
                    scores.append((score * 0.5, -1, msg))  # Lower weight for summarized messages
        
        # Sort by score and return top messages
        scores.sort(key=lambda x: x[0], reverse=True)
        return [msg for _, _, msg in scores[:top_k]]
    
    def build_index(self):
        """Build an index for efficient retrieval."""
        self.context_index.clear()
        
        # Index current messages
        for i, msg in enumerate(self.history):
            content = str(msg.get(C, "")).lower()
            words = content.split()
            for word in set(words):
                if word not in self.context_index:
                    self.context_index[word] = []
                self.context_index[word].append(i)
    
    def get_relevant_context(self, query: str, max_messages: int = 5) -> list[Message]:
        """Get relevant historical context for a query."""
        query_words = query.lower().split()
        message_scores = defaultdict(int)
        
        # Score messages based on word matches
        for word in query_words:
            if word in self.context_index:
                for msg_idx in self.context_index[word]:
                    message_scores[msg_idx] += 1
        
        # Get top scoring messages
        sorted_indices = sorted(message_scores.items(), key=lambda x: x[1], reverse=True)
        relevant_messages = []
        
        for idx, _ in sorted_indices[:max_messages]:
            if idx < len(self.history):
                relevant_messages.append(self.history[idx])
        
        return relevant_messages


# ————————————————————————————————————————————————————————————————
# Core Chat Functions
# ————————————————————————————————————————————————————————————————
async def chat(
    messages: Sequence[Message],
    *,
    model: str | None = None,
    stream: bool = True,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools_param: Sequence[dict[str, Any]] | None = None,
    tool_choice: Union[str, dict[str, Any], None] = None,
    **kwargs: Any,
) -> AsyncIterator[Message]:
    """
    Low-level chat function with streaming support and tool calling.
    
    Yields token events during streaming, then full messages.
    """
    client = openai.AsyncOpenAI(api_key=env.api_key)
    
    request_body = {
        "model": model or env.default_model,
        "messages": list(messages),
        "stream": stream,
        "temperature": temperature if temperature is not None else env.temperature,
        "max_tokens": max_tokens,
        "tools": tools_param,
        "tool_choice": tool_choice,
        **kwargs,
    }
    
    # Remove None values
    request_body = {k: v for k, v in request_body.items() if v is not None}
    
    async def _send_once() -> AsyncIterator[Message]:
        if stream:
            # Streaming mode
            stream_response = await client.chat.completions.create(**request_body)
            
            buffers: MutableMapping[int, str] = defaultdict(str)
            tool_calls_buffers: MutableMapping[int, list] = defaultdict(list)
            
            async for chunk in stream_response:
                for choice in chunk.choices:
                    idx = choice.index
                    
                    # Handle text content
                    if hasattr(choice.delta, 'content') and choice.delta.content:
                        buffers[idx] += choice.delta.content
                        yield {"index": idx, "token": choice.delta.content}
                    
                    # Handle tool calls
                    if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                        for tool_call in choice.delta.tool_calls:
                            if len(tool_calls_buffers[idx]) <= tool_call.index:
                                tool_calls_buffers[idx].append({
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                })
                            
                            tc_buffer = tool_calls_buffers[idx][tool_call.index]
                            if tool_call.id:
                                tc_buffer["id"] = tool_call.id
                            if tool_call.function:
                                if tool_call.function.name:
                                    tc_buffer["function"]["name"] = tool_call.function.name
                                if tool_call.function.arguments:
                                    tc_buffer["function"]["arguments"] += tool_call.function.arguments
            
            # Emit complete messages
            for idx in buffers.keys() | tool_calls_buffers.keys():
                msg: Message = {R: A}
                if idx in buffers and buffers[idx]:
                    msg[C] = buffers[idx]
                if idx in tool_calls_buffers and tool_calls_buffers[idx]:
                    msg["tool_calls"] = tool_calls_buffers[idx]
                msg["index"] = idx
                yield msg
        else:
            # Non-streaming mode
            completion = await client.chat.completions.create(**request_body)
            
            for choice in completion.choices:
                msg: Message = {
                    R: choice.message.role,
                    "index": choice.index
                }
                if choice.message.content:
                    msg[C] = choice.message.content
                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in choice.message.tool_calls
                    ]
                yield msg
    
    # Retry logic
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(env.retry_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    ):
        with attempt:
            async for msg in _send_once():
                yield msg


# ————————————————————————————————————————————————————————————————
# Multi-turn Agent
# ————————————————————————————————————————————————————————————————
class MultiTurnAgent:
    """
    Stateful multi-turn conversation agent with function calling.
    
    Features:
    - Maintains conversation history
    - Automatic tool execution
    - Streaming support
    - Memory management with summarization
    """
    
    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        tools_registry: ToolRegistry | None = None,
        stream: bool = True,
    ):
        self.memory = ConversationMemory(
            max_tokens=env.max_context_tokens,
            threshold_words=env.summarize_threshold_words,
        )
        self.tools_registry = tools_registry or tools
        self.stream = stream
        
        if system_prompt:
            self.memory.history.append(make_message(S, system_prompt))
    
    async def send_user(
        self,
        content: str,
        *,
        auto_execute_tools: bool = True,
        max_tool_rounds: int = 10,
        use_retrieval: bool = True,
        **chat_kwargs: Any,
    ) -> str:
        """
        Send user message and get response, handling tool calls automatically.
        
        Returns the final assistant response text.
        """
        await self.memory.append(U, content)
        
        # Build index for retrieval if enabled
        if use_retrieval:
            self.memory.build_index()
        
        # Combine registered tools with any user-provided tools
        user_tools = chat_kwargs.pop("tools_param", None) or []
        tools_combined = [*user_tools, *self.tools_registry.schemas] if (
            user_tools or self.tools_registry.schemas
        ) else None
        
        rounds = 0
        final_response = ""
        
        while rounds < max_tool_rounds:
            rounds += 1
            assistant_content = ""
            tool_calls = []
            
            # Get assistant response
            async for msg in chat(
                self.memory.history,
                tools_param=tools_combined,
                tool_choice="auto" if tools_combined and rounds == 1 else None,
                stream=self.stream,
                **chat_kwargs,
            ):
                if "token" in msg:  # Streaming token
                    print(msg["token"], end="", flush=True)
                else:  # Complete message
                    if C in msg:
                        assistant_content = msg[C]
                    if "tool_calls" in msg:
                        tool_calls = msg["tool_calls"]
            
            # Add assistant message to history
            assistant_msg = {R: A}
            if assistant_content:
                assistant_msg[C] = assistant_content
                final_response = assistant_content
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            
            # Append assistant message to memory with proper format
            if assistant_content:
                await self.memory.append(A, assistant_content)
            elif tool_calls:
                # For tool calls without content, we need to append the full message structure
                self.memory.history.append(assistant_msg)
            
            # Execute tool calls if any
            if tool_calls and auto_execute_tools:
                print()  # New line after streaming
                
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args = json.loads(tool_call["function"]["arguments"])
                    
                    logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                    
                    try:
                        result = await self.tools_registry.call(tool_name, **tool_args)
                        result_str = str(result)
                    except Exception as e:
                        result_str = f"Error: {str(e)}"
                        logger.error(f"Tool execution failed: {e}")
                    
                    # Add tool result to history
                    tool_msg = {
                        R: "tool",
                        "tool_call_id": tool_call["id"],
                        C: result_str,
                    }
                    self.memory.history.append(tool_msg)
                
                # Continue conversation after tool execution
                continue
            else:
                # No tool calls, conversation complete
                break
        
        return final_response
    
    async def get_history(self) -> list[Message]:
        """Get current conversation history."""
        return self.memory.history.copy()
    
    async def clear_history(self, keep_system: bool = True):
        """Clear conversation history, optionally keeping system message."""
        if keep_system:
            system_msgs = [m for m in self.memory.history if m[R] == S]
            self.memory.history = system_msgs
        else:
            self.memory.history = []
        # Clear retrieval indices
        self.memory.context_index.clear()
        self.memory.summaries.clear()
    
    async def search_history(self, query: str, top_k: int = 3) -> list[Message]:
        """Search conversation history for relevant messages."""
        return await self.memory.search(query, top_k)
    
    def get_relevant_context(self, query: str, max_messages: int = 5) -> list[Message]:
        """Get relevant historical context for a query."""
        return self.memory.get_relevant_context(query, max_messages)


# ————————————————————————————————————————————————————————————————
# Example Tools
# ————————————————————————————————————————————————————————————————
@tools.register(description="Greet someone using current config")
def hello(name: str, env: Environment) -> str:
    """Say hello with environment info."""
    return (
        f"Hello {name}! I'm powered by {env.default_model} "
        f"at temperature {env.temperature}."
    )


@tools.register(description="Perform arithmetic calculations")
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    # Safe evaluation of math expressions
    import math
    allowed_names = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'pow': pow,
        'pi': math.pi,
        'e': math.e,
    }
    
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return float(result)
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


@tools.register(description="Get current date and time")
def get_datetime() -> str:
    """Get current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tools.register(description="Store a value in memory")
def remember(key: str, value: str, env: Environment) -> str:
    """Store a key-value pair in the environment."""
    env[key] = value
    return f"Remembered: {key} = {value}"


@tools.register(description="Recall a value from memory")
def recall(key: str, env: Environment) -> str:
    """Recall a stored value from the environment."""
    value = env._extras.get(key)
    if value is None:
        return f"No memory found for key: {key}"
    return f"Recalled: {key} = {value}"


# ————————————————————————————————————————————————————————————————
# Demo & Testing
# ————————————————————————————————————————————————————————————————
async def demo():
    """Demonstrate multi-turn agent capabilities."""
    print("Multi-turn Function Calling Agent Demo")
    print("=" * 50)
    
    # Create agent
    agent = MultiTurnAgent(
        system_prompt="You are a helpful AI assistant with access to various tools. "
                     "Use them when appropriate to help answer questions.",
        stream=True,
    )
    
    # Example conversations
    conversations = [
        "Hello! What's your name and what model are you using?",
        "What's the square root of 997?",
        "What's the current date and time?",
        "Remember that my favorite color is blue",
        "What was my favorite color again?",
        "Calculate (2^10 + 3^5) / 7",
    ]
    
    for user_input in conversations:
        print(f"\nUser: {user_input}")
        print("Assistant: ", end="")
        
        response = await agent.send_user(user_input)
        print()  # New line after response
    
    # Show conversation history
    print("\n" + "=" * 50)
    print("Conversation History:")
    history = await agent.get_history()
    for i, msg in enumerate(history):
        role = msg.get(R, "unknown")
        content = msg.get(C, "")
        
        if role == "tool":
            print(f"{i}: [TOOL RESULT] {content[:100]}...")
        elif "tool_calls" in msg:
            tool_names = [tc["function"]["name"] for tc in msg["tool_calls"]]
            print(f"{i}: [{role.upper()}] Calling tools: {tool_names}")
            if content:
                print(f"    Content: {content}")
        else:
            print(f"{i}: [{role.upper()}] {content[:100]}...")


async def interactive_demo():
    """Interactive conversation with the agent."""
    print("Multi-turn Agent - Interactive Mode")
    print("Type 'quit' to exit, 'history' to see conversation, 'clear' to reset")
    print("=" * 50)
    
    agent = MultiTurnAgent(
        system_prompt="You are a helpful AI assistant with tool access. "
                     "Be concise but friendly in your responses.",
        stream=True,
    )
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'history':
                history = await agent.get_history()
                print("\n--- Conversation History ---")
                for msg in history:
                    role = msg.get(R, "unknown")
                    content = msg.get(C, "")[:200]
                    print(f"[{role}] {content}...")
                continue
            elif user_input.lower() == 'clear':
                await agent.clear_history()
                print("History cleared!")
                continue
            
            print("Assistant: ", end="")
            response = await agent.send_user(user_input)
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo())
    
    # Uncomment for interactive mode
    # asyncio.run(interactive_demo())