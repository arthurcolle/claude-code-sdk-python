"""Progress tracking with callbacks for Claude SDK."""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from collections.abc import AsyncIterator
from datetime import datetime
from enum import Enum

from .types import Message, AssistantMessage, ResultMessage, ToolUseBlock


class ProgressEvent(Enum):
    """Types of progress events."""
    QUERY_START = "query_start"
    QUERY_END = "query_end"
    MESSAGE_RECEIVED = "message_received"
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_END = "tool_use_end"
    TOKEN_UPDATE = "token_update"
    COST_UPDATE = "cost_update"
    ERROR = "error"


@dataclass
class ProgressInfo:
    """Information about query progress."""
    
    event: ProgressEvent
    timestamp: datetime = field(default_factory=datetime.now)
    data: dict[str, Any] = field(default_factory=dict)
    
    # Cumulative stats
    message_count: int = 0
    tool_use_count: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    elapsed_ms: int = 0


@dataclass
class ProgressTracker:
    """Tracks progress of Claude queries."""
    
    on_progress: Optional[Callable[[ProgressInfo], None]] = None
    track_tokens: bool = True
    track_costs: bool = True
    track_tools: bool = True
    
    _start_time: Optional[datetime] = field(default=None, init=False)
    _message_count: int = field(default=0, init=False)
    _tool_use_count: int = field(default=0, init=False)
    _total_tokens: int = field(default=0, init=False)
    _total_cost: float = field(default=0.0, init=False)
    _active_tools: dict[str, datetime] = field(default_factory=dict, init=False)
    
    def _emit(self, event: ProgressEvent, data: dict[str, Any] | None = None) -> None:
        """Emit a progress event."""
        if self.on_progress:
            elapsed_ms = 0
            if self._start_time:
                elapsed_ms = int((datetime.now() - self._start_time).total_seconds() * 1000)
            
            info = ProgressInfo(
                event=event,
                data=data or {},
                message_count=self._message_count,
                tool_use_count=self._tool_use_count,
                total_tokens=self._total_tokens,
                total_cost_usd=self._total_cost,
                elapsed_ms=elapsed_ms,
            )
            self.on_progress(info)
    
    def start(self) -> None:
        """Mark query start."""
        self._start_time = datetime.now()
        self._emit(ProgressEvent.QUERY_START)
    
    def end(self) -> None:
        """Mark query end."""
        self._emit(ProgressEvent.QUERY_END)
    
    def track_message(self, message: Message) -> None:
        """Track a message."""
        self._message_count += 1
        data = {"message_type": type(message).__name__}
        
        if isinstance(message, AssistantMessage):
            # Track tool uses
            if self.track_tools:
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        self._tool_use_count += 1
                        self._active_tools[block.id] = datetime.now()
                        self._emit(
                            ProgressEvent.TOOL_USE_START,
                            {"tool_name": block.name, "tool_id": block.id}
                        )
        
        elif isinstance(message, ResultMessage):
            # Track costs and tokens
            if self.track_costs and message.cost_usd:
                self._total_cost = message.total_cost_usd
                self._emit(
                    ProgressEvent.COST_UPDATE,
                    {
                        "session_cost": message.cost_usd,
                        "total_cost": message.total_cost_usd,
                    }
                )
            
            if self.track_tokens and message.usage:
                total_tokens = message.usage.get("total_tokens", 0)
                self._total_tokens = total_tokens
                self._emit(
                    ProgressEvent.TOKEN_UPDATE,
                    {"usage": message.usage}
                )
        
        self._emit(ProgressEvent.MESSAGE_RECEIVED, data)
    
    def track_error(self, error: Exception) -> None:
        """Track an error."""
        self._emit(
            ProgressEvent.ERROR,
            {"error_type": type(error).__name__, "error_message": str(error)}
        )


class ProgressReporter:
    """Reports progress with customizable formatting."""
    
    def __init__(
        self,
        format_func: Optional[Callable[[ProgressInfo], str]] = None,
        output_func: Optional[Callable[[str], None]] = None,
        report_interval_ms: int = 100,
    ):
        self.format_func = format_func or self._default_format
        self.output_func = output_func or print
        self.report_interval_ms = report_interval_ms
        self._last_report_time: Optional[datetime] = None
    
    def _default_format(self, info: ProgressInfo) -> str:
        """Default progress formatting."""
        if info.event == ProgressEvent.QUERY_START:
            return "🚀 Starting query..."
        elif info.event == ProgressEvent.QUERY_END:
            return f"✅ Query completed in {info.elapsed_ms}ms"
        elif info.event == ProgressEvent.MESSAGE_RECEIVED:
            return f"📩 Message {info.message_count} received"
        elif info.event == ProgressEvent.TOOL_USE_START:
            tool_name = info.data.get("tool_name", "unknown")
            return f"🔧 Using tool: {tool_name}"
        elif info.event == ProgressEvent.COST_UPDATE:
            return f"💰 Cost: ${info.total_cost_usd:.4f}"
        elif info.event == ProgressEvent.ERROR:
            return f"❌ Error: {info.data.get('error_message', 'Unknown')}"
        else:
            return f"📊 {info.event.value}: {info.data}"
    
    def should_report(self, info: ProgressInfo) -> bool:
        """Check if should report based on interval."""
        # Always report start/end/error events
        if info.event in (ProgressEvent.QUERY_START, ProgressEvent.QUERY_END, ProgressEvent.ERROR):
            return True
        
        # Rate limit other events
        now = datetime.now()
        if self._last_report_time is None:
            self._last_report_time = now
            return True
        
        elapsed = (now - self._last_report_time).total_seconds() * 1000
        if elapsed >= self.report_interval_ms:
            self._last_report_time = now
            return True
        
        return False
    
    def report(self, info: ProgressInfo) -> None:
        """Report progress if appropriate."""
        if self.should_report(info):
            message = self.format_func(info)
            self.output_func(message)


async def query_with_progress(
    *,
    prompt: str,
    options: Any = None,
    on_progress: Optional[Callable[[ProgressInfo], None]] = None,
    reporter: Optional[ProgressReporter] = None,
    **tracker_kwargs
) -> AsyncIterator[Message]:
    """
    Query Claude with progress tracking.
    
    Args:
        prompt: The prompt to send
        options: Query options
        on_progress: Progress callback function
        reporter: Progress reporter (creates default if None)
        **tracker_kwargs: Additional ProgressTracker arguments
        
    Yields:
        Messages from the conversation
        
    Example:
        ```python
        # Simple progress tracking
        async for message in query_with_progress(
            prompt="Analyze this data",
            on_progress=lambda info: print(f"{info.event}: {info.elapsed_ms}ms")
        ):
            print(message)
        
        # With custom reporter
        reporter = ProgressReporter(
            format_func=lambda info: f"[{info.elapsed_ms}ms] {info.event}",
            output_func=logger.info
        )
        
        async for message in query_with_progress(
            prompt="Hello",
            reporter=reporter
        ):
            process(message)
        ```
    """
    from . import query
    
    # Create tracker
    tracker = ProgressTracker(on_progress=on_progress, **tracker_kwargs)
    
    # Set up reporter
    if reporter is None and on_progress is None:
        reporter = ProgressReporter()
    
    if reporter:
        tracker.on_progress = reporter.report
    
    try:
        tracker.start()
        
        async for message in query(prompt=prompt, options=options):
            tracker.track_message(message)
            yield message
            
        tracker.end()
        
    except Exception as e:
        tracker.track_error(e)
        raise


@dataclass
class ProgressBar:
    """Simple progress bar for terminal output."""
    
    total: int = 100
    width: int = 40
    filled_char: str = "█"
    empty_char: str = "░"
    
    def render(self, current: int, prefix: str = "", suffix: str = "") -> str:
        """Render progress bar."""
        percent = min(100, int(100 * current / self.total))
        filled_width = int(self.width * percent / 100)
        empty_width = self.width - filled_width
        
        bar = self.filled_char * filled_width + self.empty_char * empty_width
        return f"{prefix}[{bar}] {percent}% {suffix}"


class ProgressBarReporter(ProgressReporter):
    """Reporter that shows a progress bar."""
    
    def __init__(self, estimated_messages: int = 10):
        self.progress_bar = ProgressBar(total=estimated_messages)
        self.estimated_messages = estimated_messages
        super().__init__(
            format_func=self._format_with_bar,
            output_func=self._update_line
        )
        self._last_line = ""
    
    def _format_with_bar(self, info: ProgressInfo) -> str:
        """Format with progress bar."""
        if info.event == ProgressEvent.QUERY_END:
            return self.progress_bar.render(
                info.message_count,
                prefix="Progress: ",
                suffix=f" Complete! ({info.elapsed_ms}ms)"
            )
        else:
            return self.progress_bar.render(
                info.message_count,
                prefix="Progress: ",
                suffix=f" {info.event.value}"
            )
    
    def _update_line(self, message: str) -> None:
        """Update the same line."""
        # Clear previous line
        print(f"\r{' ' * len(self._last_line)}", end="")
        # Print new message
        print(f"\r{message}", end="", flush=True)
        self._last_line = message
        
        # Add newline on completion
        if "Complete!" in message:
            print()  # New line