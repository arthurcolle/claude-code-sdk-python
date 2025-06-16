"""Advanced conversation management for multi-turn interactions."""

import json
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncIterator
from collections import defaultdict

from .types import (
    ClaudeCodeOptions,
    Message,
    ResultMessage,
    AssistantMessage,
    UserMessage,
)
from . import query


@dataclass
class ConversationContext:
    """Context information for a conversation."""
    session_id: str
    created_at: datetime
    last_updated: datetime
    turn_count: int = 0
    total_cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    parent_session_id: Optional[str] = None  # For branched conversations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "turn_count": self.turn_count,
            "total_cost": self.total_cost,
            "metadata": self.metadata,
            "tags": self.tags,
            "parent_session_id": self.parent_session_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationContext":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            turn_count=data.get("turn_count", 0),
            total_cost=data.get("total_cost", 0.0),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            parent_session_id=data.get("parent_session_id"),
        )


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    prompt: str
    response: List[Message]
    timestamp: datetime
    cost: float
    turn_number: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prompt": self.prompt,
            "timestamp": self.timestamp.isoformat(),
            "cost": self.cost,
            "turn_number": self.turn_number,
            # Note: Response messages are not serialized here for simplicity
        }


class ConversationManager:
    """Manages multiple conversations with persistence and advanced features."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize the conversation manager.
        
        Args:
            storage_path: Optional path for persisting conversations
        """
        self.storage_path = storage_path or Path.home() / ".claude_code_conversations"
        self.storage_path.mkdir(exist_ok=True)
        
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.conversation_history: Dict[str, List[ConversationTurn]] = defaultdict(list)
        self._load_conversations()
    
    def _load_conversations(self):
        """Load persisted conversations from storage."""
        index_file = self.storage_path / "index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                data = json.load(f)
                for conv_data in data.get("conversations", []):
                    context = ConversationContext.from_dict(conv_data)
                    self.active_conversations[context.session_id] = context
    
    def _save_conversations(self):
        """Save conversations to storage."""
        index_file = self.storage_path / "index.json"
        data = {
            "conversations": [
                conv.to_dict() for conv in self.active_conversations.values()
            ]
        }
        with open(index_file, "w") as f:
            json.dump(data, f, indent=2)
    
    async def create_conversation(
        self,
        initial_prompt: str,
        options: Optional[ClaudeCodeOptions] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> tuple[str, AsyncIterator[Message]]:
        """Create a new conversation.
        
        Args:
            initial_prompt: The first prompt
            options: Claude options
            metadata: Optional metadata to attach
            tags: Optional tags for categorization
            
        Returns:
            Tuple of (session_id, message_iterator)
        """
        if options is None:
            options = ClaudeCodeOptions()
        
        # Track the session ID when we get the result
        session_id = None
        messages = []
        turn_cost = 0.0
        
        async def wrapped_query():
            nonlocal session_id, turn_cost
            async for message in query(prompt=initial_prompt, options=options):
                messages.append(message)
                if isinstance(message, ResultMessage):
                    session_id = message.session_id
                    turn_cost = message.cost_usd
                    
                    # Create context
                    context = ConversationContext(
                        session_id=session_id,
                        created_at=datetime.now(),
                        last_updated=datetime.now(),
                        turn_count=1,
                        total_cost=turn_cost,
                        metadata=metadata or {},
                        tags=tags or [],
                    )
                    self.active_conversations[session_id] = context
                    
                    # Record turn
                    turn = ConversationTurn(
                        prompt=initial_prompt,
                        response=messages,
                        timestamp=datetime.now(),
                        cost=turn_cost,
                        turn_number=1,
                    )
                    self.conversation_history[session_id].append(turn)
                    
                    self._save_conversations()
                
                yield message
        
        # Collect all messages to ensure we get the session ID
        all_messages = []
        async for message in wrapped_query():
            all_messages.append(message)
        
        # Create an async iterator that yields the collected messages
        async def yield_all():
            for msg in all_messages:
                yield msg
        
        # session_id should be set now from ResultMessage
        if session_id is None:
            raise RuntimeError("No session ID received from Claude Code")
        
        return session_id, yield_all()
    
    async def continue_conversation(
        self,
        session_id: str,
        prompt: str,
        options: Optional[ClaudeCodeOptions] = None,
    ) -> AsyncIterator[Message]:
        """Continue an existing conversation.
        
        Args:
            session_id: The session to continue
            prompt: The new prompt
            options: Claude options (will override resume)
            
        Yields:
            Messages from the conversation
        """
        if session_id not in self.active_conversations:
            raise ValueError(f"Unknown session ID: {session_id}")
        
        context = self.active_conversations[session_id]
        
        if options is None:
            options = ClaudeCodeOptions()
        options.resume = session_id
        
        messages = []
        turn_cost = 0.0
        
        async for message in query(prompt=prompt, options=options):
            messages.append(message)
            if isinstance(message, ResultMessage):
                turn_cost = message.cost_usd
                context.turn_count = message.num_turns
                context.total_cost = message.total_cost_usd
                context.last_updated = datetime.now()
                
                # Record turn
                turn = ConversationTurn(
                    prompt=prompt,
                    response=messages,
                    timestamp=datetime.now(),
                    cost=turn_cost,
                    turn_number=context.turn_count,
                )
                self.conversation_history[session_id].append(turn)
                
                self._save_conversations()
            
            yield message
    
    async def branch_conversation(
        self,
        parent_session_id: str,
        prompt: str,
        options: Optional[ClaudeCodeOptions] = None,
    ) -> tuple[str, AsyncIterator[Message]]:
        """Create a new branch from an existing conversation.
        
        This continues from the parent's context but creates a new session.
        
        Args:
            parent_session_id: The session to branch from
            prompt: The prompt for the new branch
            options: Claude options
            
        Returns:
            Tuple of (new_session_id, message_iterator)
        """
        if parent_session_id not in self.active_conversations:
            raise ValueError(f"Unknown parent session ID: {parent_session_id}")
        
        parent_context = self.active_conversations[parent_session_id]
        
        # Create new conversation continuing from parent
        if options is None:
            options = ClaudeCodeOptions()
        options.resume = parent_session_id  # Start from parent's context
        
        # Get new session ID from the response
        new_session_id, messages = await self.create_conversation(
            prompt, options, 
            metadata={"branched_from": parent_session_id},
            tags=parent_context.tags + ["branched"]
        )
        
        # Update the context to mark it as a branch
        if new_session_id in self.active_conversations:
            self.active_conversations[new_session_id].parent_session_id = parent_session_id
            self._save_conversations()
        
        return new_session_id, messages
    
    def get_conversation_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get context for a conversation."""
        return self.active_conversations.get(session_id)
    
    def get_conversation_history(self, session_id: str) -> List[ConversationTurn]:
        """Get full history of a conversation."""
        return self.conversation_history.get(session_id, [])
    
    def list_conversations(
        self,
        tags: Optional[List[str]] = None,
        since: Optional[datetime] = None,
    ) -> List[ConversationContext]:
        """List conversations with optional filtering.
        
        Args:
            tags: Filter by tags (conversations must have all specified tags)
            since: Only return conversations updated since this time
            
        Returns:
            List of conversation contexts
        """
        conversations = list(self.active_conversations.values())
        
        if tags:
            conversations = [
                c for c in conversations
                if all(tag in c.tags for tag in tags)
            ]
        
        if since:
            conversations = [
                c for c in conversations
                if c.last_updated >= since
            ]
        
        return sorted(conversations, key=lambda c: c.last_updated, reverse=True)
    
    def tag_conversation(self, session_id: str, tags: List[str]):
        """Add tags to a conversation."""
        if session_id in self.active_conversations:
            context = self.active_conversations[session_id]
            context.tags.extend(tags)
            context.tags = list(set(context.tags))  # Remove duplicates
            self._save_conversations()
    
    def add_metadata(self, session_id: str, metadata: Dict[str, Any]):
        """Add metadata to a conversation."""
        if session_id in self.active_conversations:
            context = self.active_conversations[session_id]
            context.metadata.update(metadata)
            self._save_conversations()
    
    def export_conversation(self, session_id: str, output_path: Path):
        """Export a conversation with its full history."""
        if session_id not in self.active_conversations:
            raise ValueError(f"Unknown session ID: {session_id}")
        
        context = self.active_conversations[session_id]
        history = self.conversation_history.get(session_id, [])
        
        export_data = {
            "context": context.to_dict(),
            "history": [turn.to_dict() for turn in history],
            "exported_at": datetime.now().isoformat(),
        }
        
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)
    
    def get_total_cost(self) -> float:
        """Get total cost across all conversations."""
        return sum(c.total_cost for c in self.active_conversations.values())
    
    def get_conversation_tree(self, root_session_id: str) -> Dict[str, Any]:
        """Get conversation tree starting from a root session.
        
        Returns:
            Dictionary representing the conversation tree
        """
        def build_tree(session_id: str) -> Dict[str, Any]:
            context = self.active_conversations.get(session_id)
            if not context:
                return {}
            
            # Find children
            children = [
                c.session_id for c in self.active_conversations.values()
                if c.parent_session_id == session_id
            ]
            
            return {
                "session_id": session_id,
                "created_at": context.created_at.isoformat(),
                "turn_count": context.turn_count,
                "tags": context.tags,
                "children": [build_tree(child_id) for child_id in children],
            }
        
        return build_tree(root_session_id)