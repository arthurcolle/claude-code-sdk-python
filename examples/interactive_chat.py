#!/usr/bin/env python3
"""Interactive chat application with multi-turn conversation support."""

import asyncio
import sys
from claude_code_sdk import (
    query, 
    ClaudeCodeOptions, 
    ResultMessage, 
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock
)


class InteractiveChat:
    """Interactive chat session with Claude."""
    
    def __init__(self):
        self.session_id = None
        self.turn_count = 0
        self.total_cost = 0.0
    
    async def chat_turn(self, user_input: str):
        """Process a single chat turn."""
        self.turn_count += 1
        
        # Configure options for multi-turn
        options = ClaudeCodeOptions(
            resume=self.session_id,  # Resume previous session if exists
            allowed_tools=["Read", "Write", "Edit", "Bash"],  # Enable tools
            permission_mode="acceptEdits",  # Auto-accept edits for smoother flow
        )
        
        print(f"\n[Turn {self.turn_count}]")
        print("-" * 40)
        
        async for message in query(prompt=user_input, options=options):
            if isinstance(message, AssistantMessage):
                # Process assistant response
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text}")
                    elif isinstance(block, ToolUseBlock):
                        print(f"🔧 Using tool: {block.name}")
                    elif isinstance(block, ToolResultBlock):
                        if block.is_error:
                            print(f"❌ Tool error: {block.content}")
                        else:
                            print(f"✅ Tool result received")
            
            elif isinstance(message, ResultMessage):
                # Update session info
                self.session_id = message.session_id
                self.total_cost = message.total_cost_usd
                print(f"\n💰 Session cost: ${message.cost_usd:.4f} (Total: ${self.total_cost:.4f})")
    
    async def run(self):
        """Run the interactive chat loop."""
        print("🤖 Claude Interactive Chat")
        print("=" * 50)
        print("Type 'exit' to quit, 'new' to start a new conversation")
        print("Multi-turn conversation is enabled - Claude will remember context!")
        print("=" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                # Handle special commands
                if user_input.lower() == 'exit':
                    print("\nGoodbye! 👋")
                    break
                elif user_input.lower() == 'new':
                    self.session_id = None
                    self.turn_count = 0
                    print("\n🔄 Started new conversation")
                    continue
                elif not user_input:
                    continue
                
                # Process the chat turn
                await self.chat_turn(user_input)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("You can continue chatting or type 'new' to start fresh.")


async def main():
    """Run the interactive chat application."""
    chat = InteractiveChat()
    await chat.run()


if __name__ == "__main__":
    # Check if running in interactive mode
    if sys.stdin.isatty():
        asyncio.run(main())
    else:
        print("This script requires an interactive terminal.")
        print("Run it directly: python interactive_chat.py")