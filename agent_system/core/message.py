"""
Message system for inter-agent communication.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field
import uuid


class MessageType(str, Enum):
    """Types of messages in the agent system."""
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    TOOL_REQUEST = "tool_request"
    TOOL_RESPONSE = "tool_response"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    BROADCAST = "broadcast"
    QUERY = "query"
    RESPONSE = "response"


class MessagePriority(str, Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class Message(BaseModel):
    """Message structure for agent communication."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType
    priority: MessagePriority = MessagePriority.NORMAL
    sender_id: str
    recipient_id: Optional[str] = None  # None for broadcasts
    timestamp: datetime = Field(default_factory=datetime.now)
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None  # For request-response patterns
    reply_to: Optional[str] = None  # ID of message being replied to
    ttl: Optional[int] = None  # Time to live in seconds
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def is_expired(self) -> bool:
        """Check if message has expired based on TTL."""
        if self.ttl is None:
            return False
        elapsed = (datetime.now() - self.timestamp).total_seconds()
        return elapsed > self.ttl
    
    def create_reply(self, sender_id: str, payload: Dict[str, Any]) -> "Message":
        """Create a reply to this message."""
        return Message(
            type=MessageType.RESPONSE,
            priority=self.priority,
            sender_id=sender_id,
            recipient_id=self.sender_id,
            payload=payload,
            correlation_id=self.correlation_id or self.id,
            reply_to=self.id
        )


class MessageQueue:
    """Simple in-memory message queue for agents."""
    
    def __init__(self):
        self._queues: Dict[str, List[Message]] = {}
        self._broadcast_queue: List[Message] = []
    
    def send(self, message: Message):
        """Send a message to a specific agent or broadcast."""
        if message.recipient_id:
            if message.recipient_id not in self._queues:
                self._queues[message.recipient_id] = []
            self._queues[message.recipient_id].append(message)
        else:
            self._broadcast_queue.append(message)
    
    def receive(self, agent_id: str, include_broadcasts: bool = True) -> List[Message]:
        """Receive messages for a specific agent."""
        messages = []
        
        # Get direct messages
        if agent_id in self._queues:
            messages.extend(self._queues[agent_id])
            self._queues[agent_id] = []
        
        # Get broadcast messages
        if include_broadcasts:
            messages.extend(self._broadcast_queue)
        
        # Filter out expired messages
        return [msg for msg in messages if not msg.is_expired()]
    
    def clear_broadcasts(self):
        """Clear broadcast queue (typically after all agents have read)."""
        self._broadcast_queue = []