#!/usr/bin/env python3
"""Example demonstrating multi-turn conversations with Claude Code SDK."""

import asyncio
from claude_code_sdk import query, ClaudeCodeOptions, ResultMessage


async def multi_turn_conversation():
    """Demonstrate multi-turn conversation capability."""
    
    # Store session ID to continue the conversation
    session_id = None
    
    # First turn
    print("=== Turn 1 ===")
    async for message in query(
        prompt="Hello! Can you remember that my favorite color is blue?",
        options=ClaudeCodeOptions()
    ):
        if isinstance(message, ResultMessage):
            session_id = message.session_id
            print(f"Session ID: {session_id}")
        else:
            print(message)
    
    # Second turn - continue the conversation
    print("\n=== Turn 2 ===")
    async for message in query(
        prompt="What's my favorite color?",
        options=ClaudeCodeOptions(resume=session_id)
    ):
        print(message)
    
    # Third turn - continue again
    print("\n=== Turn 3 ===")
    async for message in query(
        prompt="Now remember that I also like programming in Python.",
        options=ClaudeCodeOptions(resume=session_id)
    ):
        print(message)
    
    # Fourth turn - test memory
    print("\n=== Turn 4 ===")
    async for message in query(
        prompt="What are the two things you know about me?",
        options=ClaudeCodeOptions(resume=session_id)
    ):
        print(message)


async def auto_continue_conversation():
    """Demonstrate using continue_conversation flag."""
    
    # First conversation
    print("=== First Conversation ===")
    async for message in query(
        prompt="Remember that I'm working on a weather app.",
    ):
        print(message)
    
    # Continue the last conversation automatically
    print("\n=== Continuing Last Conversation ===")
    async for message in query(
        prompt="What project am I working on?",
        options=ClaudeCodeOptions(continue_conversation=True)
    ):
        print(message)


async def main():
    """Run the examples."""
    print("Multi-turn Conversation Example\n")
    
    # Example 1: Manual session management
    print("Example 1: Manual session management with resume")
    print("-" * 50)
    await multi_turn_conversation()
    
    print("\n" * 2)
    
    # Example 2: Auto-continue last conversation
    print("Example 2: Auto-continue last conversation")
    print("-" * 50)
    await auto_continue_conversation()


if __name__ == "__main__":
    asyncio.run(main())