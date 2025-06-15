"""Examples of session continuation and resumption with Claude SDK."""

import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any

import anyio

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ResultMessage,
    TextBlock,
    query,
)


class SessionManager:
    """Manage Claude conversation sessions."""
    
    def __init__(self, storage_path: Path = Path(".claude_sessions")):
        self.storage_path = storage_path
        self.storage_path.mkdir(exist_ok=True)
        self.current_session_id: Optional[str] = None
        self.session_history: list[Dict[str, Any]] = []
        
    def save_session(self, session_id: str, metadata: Dict[str, Any]):
        """Save session metadata to disk."""
        session_file = self.storage_path / f"{session_id}.json"
        with open(session_file, 'w') as f:
            json.dump({
                'session_id': session_id,
                'metadata': metadata,
                'history': self.session_history
            }, f, indent=2)
            
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session metadata from disk."""
        session_file = self.storage_path / f"{session_id}.json"
        if session_file.exists():
            with open(session_file, 'r') as f:
                return json.load(f)
        return None
        
    def list_sessions(self) -> list[str]:
        """List all available sessions."""
        return [f.stem for f in self.storage_path.glob("*.json")]


# Example 1: Basic session continuation
async def basic_session_continuation():
    """Demonstrate basic session continuation."""
    print("\n=== Example 1: Basic Session Continuation ===")
    
    # First conversation turn
    print("\nFirst turn:")
    session_id = None
    
    async for message in query(prompt="My name is Alice. Remember this."):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"Claude: {block.text}")
        elif isinstance(message, ResultMessage):
            session_id = message.session_id
            print(f"Session ID: {session_id}")
            
    # Continue the conversation
    print("\nSecond turn (continuing session):")
    options = ClaudeCodeOptions(resume=session_id)
    
    async for message in query(prompt="What's my name?", options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"Claude: {block.text}")


# Example 2: Session manager with persistence
async def session_with_persistence():
    """Demonstrate session management with persistence."""
    print("\n=== Example 2: Session Management with Persistence ===")
    
    manager = SessionManager()
    
    # Create a new session
    print("\nStarting new session...")
    async for message in query(prompt="Let's work on a Python project together"):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"Claude: {block.text[:100]}...")
        elif isinstance(message, ResultMessage):
            manager.current_session_id = message.session_id
            manager.save_session(message.session_id, {
                'project': 'python-example',
                'created_at': str(asyncio.get_event_loop().time()),
                'cost_usd': message.cost_usd
            })
            print(f"Session saved: {message.session_id}")
            
    # List available sessions
    print("\nAvailable sessions:")
    for session in manager.list_sessions():
        print(f"  - {session}")


# Example 3: Multi-turn conversation
async def multi_turn_conversation():
    """Demonstrate multi-turn conversation with context."""
    print("\n=== Example 3: Multi-Turn Conversation ===")
    
    session_id = None
    turns = [
        "I want to create a simple web server in Python",
        "What libraries should I use?",
        "Can you show me a basic example?",
        "How do I add routing to handle different URLs?"
    ]
    
    for i, prompt in enumerate(turns):
        print(f"\nTurn {i+1}: {prompt}")
        
        options = ClaudeCodeOptions(
            resume=session_id if session_id else None,
            max_turns=10  # Prevent runaway conversations
        )
        
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        # Print first 200 chars for brevity
                        print(f"Claude: {block.text[:200]}...")
            elif isinstance(message, ResultMessage):
                session_id = message.session_id
                print(f"  [Session: {session_id}, Cost: ${message.cost_usd:.4f}]")


# Example 4: Branching conversations
async def branching_conversations():
    """Demonstrate branching from a saved conversation point."""
    print("\n=== Example 4: Branching Conversations ===")
    
    # Start base conversation
    print("\nBase conversation:")
    base_session_id = None
    
    async for message in query(prompt="I'm building a data analysis tool. It needs to process CSV files."):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"Claude: {block.text[:150]}...")
        elif isinstance(message, ResultMessage):
            base_session_id = message.session_id
            
    # Branch 1: Focus on performance
    print("\n\nBranch 1 - Performance focus:")
    options1 = ClaudeCodeOptions(resume=base_session_id)
    
    async for message in query(
        prompt="What are the best practices for handling very large CSV files efficiently?",
        options=options1
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"Claude: {block.text[:150]}...")
                    
    # Branch 2: Focus on visualization
    print("\n\nBranch 2 - Visualization focus:")
    options2 = ClaudeCodeOptions(resume=base_session_id)
    
    async for message in query(
        prompt="How can I add data visualization capabilities to show charts and graphs?",
        options=options2
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"Claude: {block.text[:150]}...")


# Example 5: Session with file context
async def session_with_file_context():
    """Demonstrate maintaining file context across sessions."""
    print("\n=== Example 5: Session with File Context ===")
    
    session_id = None
    
    # First: Create a file
    print("\nCreating a file in first turn:")
    options = ClaudeCodeOptions(
        allowed_tools=["Write"],
        permission_mode="acceptEdits"
    )
    
    prompt1 = """Create a file called 'calculator.py' with a simple Calculator class 
    that has add, subtract, multiply, and divide methods."""
    
    async for message in query(prompt=prompt1, options=options):
        if isinstance(message, AssistantMessage):
            print("Claude is creating the file...")
        elif isinstance(message, ResultMessage):
            session_id = message.session_id
            print(f"File created. Session: {session_id}")
            
    # Continue: Add more features
    print("\nAdding features in second turn:")
    options.resume = session_id
    
    prompt2 = "Add a method called 'power' to the Calculator class that calculates x^y"
    
    async for message in query(prompt=prompt2, options=options):
        if isinstance(message, AssistantMessage):
            print("Claude is updating the file...")
        elif isinstance(message, ResultMessage):
            print(f"File updated. Total cost: ${message.total_cost_usd:.4f}")


# Example 6: Conversation state recovery
class ConversationState:
    """Track conversation state and context."""
    
    def __init__(self):
        self.topics_discussed: list[str] = []
        self.files_created: list[str] = []
        self.commands_run: list[str] = []
        self.session_id: Optional[str] = None
        
    async def continue_conversation(self, prompt: str):
        """Continue conversation with state tracking."""
        options = ClaudeCodeOptions(
            resume=self.session_id if self.session_id else None,
            allowed_tools=["Read", "Write", "Bash"],
            permission_mode="acceptEdits"
        )
        
        # Track what we're discussing
        self.topics_discussed.append(prompt[:50])
        
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                # Track tool usage
                for block in message.content:
                    if hasattr(block, 'name'):
                        if block.name == "Write":
                            self.files_created.append(block.input.get('file_path', 'unknown'))
                        elif block.name == "Bash":
                            self.commands_run.append(block.input.get('command', 'unknown'))
                            
            elif isinstance(message, ResultMessage):
                self.session_id = message.session_id
                
            yield message
            
    def get_summary(self) -> Dict[str, Any]:
        """Get conversation summary."""
        return {
            'session_id': self.session_id,
            'topics': self.topics_discussed,
            'files_created': self.files_created,
            'commands_run': self.commands_run
        }


async def stateful_conversation_example():
    """Demonstrate stateful conversation tracking."""
    print("\n=== Example 6: Stateful Conversation ===")
    
    state = ConversationState()
    
    prompts = [
        "Create a simple TODO list application in Python",
        "Add a function to mark tasks as completed",
        "Create a test file for the TODO application"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        async for message in state.continue_conversation(prompt):
            if isinstance(message, AssistantMessage):
                print("Claude is working...")
            elif isinstance(message, ResultMessage):
                print(f"Turn completed. Session: {message.session_id}")
                
    # Show conversation summary
    print("\n=== Conversation Summary ===")
    summary = state.get_summary()
    print(f"Session ID: {summary['session_id']}")
    print(f"Topics discussed: {len(summary['topics'])}")
    print(f"Files created: {summary['files_created']}")
    print(f"Commands run: {summary['commands_run']}")


# Example 7: Session templates
class SessionTemplate:
    """Pre-configured session templates for common workflows."""
    
    @staticmethod
    def create_python_project_session() -> ClaudeCodeOptions:
        """Create options for Python project development."""
        return ClaudeCodeOptions(
            allowed_tools=["Read", "Write", "Edit", "Bash", "Grep"],
            permission_mode="acceptEdits",
            system_prompt="You are helping develop a Python project. Follow PEP 8 style guidelines.",
            max_turns=20
        )
        
    @staticmethod
    def create_code_review_session() -> ClaudeCodeOptions:
        """Create options for code review."""
        return ClaudeCodeOptions(
            allowed_tools=["Read", "Grep"],
            permission_mode="default",
            system_prompt="You are performing a code review. Focus on finding bugs, security issues, and suggesting improvements.",
            max_turns=10
        )
        
    @staticmethod
    def create_debugging_session() -> ClaudeCodeOptions:
        """Create options for debugging."""
        return ClaudeCodeOptions(
            allowed_tools=["Read", "Edit", "Bash"],
            permission_mode="acceptEdits",
            system_prompt="You are helping debug code. Be systematic and thorough in finding the root cause.",
            max_turns=15
        )


async def session_templates_example():
    """Demonstrate using session templates."""
    print("\n=== Example 7: Session Templates ===")
    
    # Code review session
    print("\nStarting code review session:")
    review_options = SessionTemplate.create_code_review_session()
    
    async for message in query(
        prompt="Review the calculator.py file we created earlier",
        options=review_options
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"Review: {block.text[:200]}...")
        elif isinstance(message, ResultMessage):
            print(f"Review complete. Session: {message.session_id}")


# Main execution
async def main():
    """Run all session management examples."""
    examples = [
        ("Basic Session Continuation", basic_session_continuation),
        ("Session with Persistence", session_with_persistence),
        ("Multi-Turn Conversation", multi_turn_conversation),
        ("Branching Conversations", branching_conversations),
        ("Session with File Context", session_with_file_context),
        ("Stateful Conversation", stateful_conversation_example),
        ("Session Templates", session_templates_example),
    ]
    
    print("Claude SDK Session Management Examples")
    print("======================================")
    
    for name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"\nExample '{name}' failed: {e}")
            
        # Small delay between examples
        await asyncio.sleep(1)
        
    print("\n\nAll examples completed!")


if __name__ == "__main__":
    anyio.run(main)