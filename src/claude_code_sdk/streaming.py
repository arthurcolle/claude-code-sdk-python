"""Streaming control features for Claude SDK."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from collections.abc import AsyncIterator
from enum import Enum

from .types import Message


class StreamState(Enum):
    """Streaming states."""
    IDLE = "idle"
    STREAMING = "streaming"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


@dataclass
class StreamController:
    """Controls streaming operations with pause/resume/cancel functionality."""
    
    _state: StreamState = field(default=StreamState.IDLE, init=False)
    _pause_event: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _message_queue: asyncio.Queue[Message | None] = field(
        default_factory=lambda: asyncio.Queue(maxsize=100), init=False
    )
    _on_state_change: Optional[Callable[[StreamState], None]] = None
    
    def __post_init__(self):
        self._pause_event.set()  # Start unpaused
    
    @property
    def state(self) -> StreamState:
        """Get current stream state."""
        return self._state
    
    def _set_state(self, state: StreamState) -> None:
        """Set state and notify callback."""
        self._state = state
        if self._on_state_change:
            self._on_state_change(state)
    
    async def pause(self) -> None:
        """Pause the stream."""
        if self._state == StreamState.STREAMING:
            self._set_state(StreamState.PAUSED)
            self._pause_event.clear()
    
    async def resume(self) -> None:
        """Resume the stream."""
        if self._state == StreamState.PAUSED:
            self._set_state(StreamState.STREAMING)
            self._pause_event.set()
    
    async def cancel(self) -> None:
        """Cancel the stream."""
        if self._state in (StreamState.STREAMING, StreamState.PAUSED):
            self._set_state(StreamState.CANCELLED)
            self._cancel_event.set()
            self._pause_event.set()  # Unblock if paused
    
    async def wait_if_paused(self) -> bool:
        """Wait if paused. Returns False if cancelled."""
        await self._pause_event.wait()
        return not self._cancel_event.is_set()
    
    def is_cancelled(self) -> bool:
        """Check if stream is cancelled."""
        return self._cancel_event.is_set()
    
    async def emit(self, message: Message) -> None:
        """Emit a message to the queue."""
        if not self.is_cancelled():
            await self._message_queue.put(message)
    
    async def complete(self) -> None:
        """Mark stream as completed."""
        self._set_state(StreamState.COMPLETED)
        await self._message_queue.put(None)  # Sentinel
    
    def on_state_change(self, callback: Callable[[StreamState], None]) -> None:
        """Register state change callback."""
        self._on_state_change = callback


class ControlledStream:
    """Wrapper for controlled streaming."""
    
    def __init__(
        self,
        stream_generator: AsyncIterator[Message],
        controller: StreamController
    ):
        self.stream_generator = stream_generator
        self.controller = controller
        self._task: Optional[asyncio.Task] = None
    
    async def _process_stream(self) -> None:
        """Process the underlying stream with control."""
        try:
            self.controller._set_state(StreamState.STREAMING)
            async for message in self.stream_generator:
                if not await self.controller.wait_if_paused():
                    break  # Cancelled
                await self.controller.emit(message)
        finally:
            await self.controller.complete()
    
    def start(self) -> None:
        """Start processing the stream."""
        if self._task is None:
            self._task = asyncio.create_task(self._process_stream())
    
    async def stop(self) -> None:
        """Stop the stream processing."""
        if self._task:
            await self.controller.cancel()
            await self._task
            self._task = None
    
    async def __aiter__(self) -> AsyncIterator[Message]:
        """Iterate over controlled messages."""
        self.start()
        while True:
            message = await self.controller._message_queue.get()
            if message is None:  # Sentinel
                break
            yield message


def create_controlled_stream(
    stream_generator: AsyncIterator[Message],
    on_state_change: Optional[Callable[[StreamState], None]] = None
) -> tuple[ControlledStream, StreamController]:
    """
    Create a controlled stream with pause/resume/cancel functionality.
    
    Args:
        stream_generator: The underlying message stream
        on_state_change: Optional callback for state changes
        
    Returns:
        Tuple of (controlled_stream, controller)
        
    Example:
        ```python
        stream, controller = create_controlled_stream(
            query(prompt="Hello"),
            on_state_change=lambda state: print(f"State: {state}")
        )
        
        # In another coroutine
        await controller.pause()
        await controller.resume()
        await controller.cancel()
        
        # Iterate with control
        async for message in stream:
            print(message)
        ```
    """
    controller = StreamController()
    if on_state_change:
        controller.on_state_change(on_state_change)
    
    controlled_stream = ControlledStream(stream_generator, controller)
    return controlled_stream, controller