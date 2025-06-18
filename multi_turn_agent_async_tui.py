#!/usr/bin/env python3
"""
Async-Safe Interactive TUI with Proper Message Queue Management
==============================================================
Handles concurrent user input properly without message buildup.
"""

import asyncio
import sys
from datetime import datetime
from typing import List, Optional, Dict, Any, Callable
from enum import Enum
from collections import deque
import json

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich import box
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.validation import Validator, ValidationError

# Import enhanced agent
from multi_turn_agent_enhanced import (
    EnhancedStatefulAgent, TaskStatus, TaskPriority,
    register_enhanced_demo_tools
)

# ————————————————————————————————————————————————————————————————
# UI State Management
# ————————————————————————————————————————————————————————————————

class AgentState(Enum):
    """Agent processing states."""
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING_CONFIRMATION = "waiting_confirmation"
    ERROR = "error"
    BUSY = "busy"

class InputMode(Enum):
    """Input handling modes."""
    NORMAL = "normal"
    DISABLED = "disabled"
    CONFIRMATION = "confirmation"
    COMMAND = "command"

class MessageQueue:
    """Thread-safe message queue with overflow handling."""
    
    def __init__(self, max_size: int = 10):
        self.queue: deque = deque(maxlen=max_size)
        self.processing_lock = asyncio.Lock()
        self.overflow_count = 0
        
    async def add(self, message: str) -> bool:
        """Add message to queue. Returns False if queue is full."""
        if len(self.queue) >= self.queue.maxlen:
            self.overflow_count += 1
            return False
        self.queue.append(message)
        return True
    
    async def get_next(self) -> Optional[str]:
        """Get next message if available."""
        async with self.processing_lock:
            if self.queue:
                return self.queue.popleft()
            return None
    
    def clear(self):
        """Clear the queue."""
        self.queue.clear()
        self.overflow_count = 0
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self.queue) == 0
    
    def size(self) -> int:
        """Get current queue size."""
        return len(self.queue)

# ————————————————————————————————————————————————————————————————
# Async-Safe Interactive TUI
# ————————————————————————————————————————————————————————————————

class AsyncSafeAgentTUI:
    """TUI with proper async handling and message queue management."""
    
    def __init__(self, agent: EnhancedStatefulAgent):
        self.agent = agent
        self.console = Console()
        self.layout = self._create_layout()
        
        # State management
        self.agent_state = AgentState.IDLE
        self.input_mode = InputMode.NORMAL
        self.message_queue = MessageQueue(max_size=5)
        self.current_processing_message = None
        
        # UI elements
        self.chat_history: List[Dict[str, Any]] = []
        self.notifications: List[Dict[str, Any]] = []
        self.status_message = "Ready"
        
        # Control flags
        self.running = True
        self.processing_task: Optional[asyncio.Task] = None
        
        # Confirmation handling
        self.pending_confirmation = None
        self.confirmation_callback = None
        
        # Input session
        self.prompt_session = None
        self.input_validator = None
        
        # Register UI tools
        self._register_ui_tools()
        
        # Stats
        self.stats = {
            "messages_processed": 0,
            "messages_dropped": 0,
            "avg_response_time": 0,
            "total_response_time": 0
        }
    
    def _create_layout(self) -> Layout:
        """Create the main layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="status", size=3),
            Layout(name="input", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="chat", ratio=2),
            Layout(name="sidebar", ratio=1)
        )
        
        layout["sidebar"].split_column(
            Layout(name="queue", size=8),
            Layout(name="metrics", size=10),
            Layout(name="agent_state", size=6)
        )
        
        return layout
    
    def _register_ui_tools(self):
        """Register UI control tools with the agent."""
        
        @self.agent.tools_registry.register(description="Send message to user")
        async def msg_user(content: str, style: str = "default") -> str:
            """Send a message to the user."""
            self.add_assistant_message(content, style)
            return "Message sent"
        
        @self.agent.tools_registry.register(description="Update UI status")
        async def update_status(message: str) -> str:
            """Update the status bar."""
            self.status_message = message
            return "Status updated"
        
        @self.agent.tools_registry.register(description="Show notification")
        async def notify(message: str, type: str = "info", duration: float = 3.0) -> str:
            """Show a notification."""
            self.add_notification(message, type, duration)
            return "Notification shown"
        
        @self.agent.tools_registry.register(description="Request user confirmation")
        async def request_confirmation(
            question: str,
            options: List[str] = ["Yes", "No"],
            timeout: float = 30.0
        ) -> str:
            """Request confirmation from user."""
            result = await self._request_confirmation(question, options, timeout)
            return f"User selected: {result}"
        
        @self.agent.tools_registry.register(description="Clear message queue")
        async def clear_queue() -> str:
            """Clear the message queue."""
            count = self.message_queue.size()
            self.message_queue.clear()
            return f"Cleared {count} queued messages"
        
        @self.agent.tools_registry.register(description="Set processing state")
        async def set_busy(busy: bool = True, message: str = "Processing...") -> str:
            """Set agent busy state."""
            if busy:
                self.agent_state = AgentState.BUSY
                self.status_message = message
            else:
                self.agent_state = AgentState.IDLE
                self.status_message = "Ready"
            return f"Busy state: {busy}"
    
    def update_header(self) -> Panel:
        """Update header panel."""
        # Title and session info
        title = Text()
        title.append("🤖 Async-Safe Agent TUI", style="bold cyan")
        title.append(" | Session: ", style="dim")
        title.append(self.agent.session_id[:8], style="yellow")
        title.append(" | ", style="dim")
        
        # Agent state indicator
        state_styles = {
            AgentState.IDLE: ("✅ Ready", "green"),
            AgentState.PROCESSING: ("🔄 Processing", "blue"),
            AgentState.WAITING_CONFIRMATION: ("❓ Waiting", "yellow"),
            AgentState.ERROR: ("❌ Error", "red"),
            AgentState.BUSY: ("🔒 Busy", "orange")
        }
        
        state_text, state_style = state_styles.get(
            self.agent_state, ("Unknown", "white")
        )
        title.append(state_text, style=f"bold {state_style}")
        
        return Panel(
            Align.center(title),
            box=box.DOUBLE,
            style="on blue"
        )
    
    def update_queue_panel(self) -> Panel:
        """Update message queue panel."""
        content = []
        
        # Queue status
        queue_size = self.message_queue.size()
        max_size = self.message_queue.queue.maxlen
        
        # Visual queue indicator
        queue_bar = "█" * queue_size + "░" * (max_size - queue_size)
        color = "red" if queue_size >= max_size - 1 else "yellow" if queue_size > 2 else "green"
        
        content.append(Text(f"Queue: [{queue_bar}]", style=color))
        content.append(Text(f"Size: {queue_size}/{max_size}", style="dim"))
        
        if self.message_queue.overflow_count > 0:
            content.append(Text(f"Dropped: {self.message_queue.overflow_count}", style="red"))
        
        # Show queued messages
        if queue_size > 0:
            content.append(Text("\nQueued:", style="bold"))
            for i, msg in enumerate(list(self.message_queue.queue)[:3]):
                preview = msg[:30] + "..." if len(msg) > 30 else msg
                content.append(Text(f"{i+1}. {preview}", style="dim"))
        
        return Panel(
            Group(*content),
            title="📬 Message Queue",
            box=box.ROUNDED,
            border_style="green" if queue_size < 3 else "yellow"
        )
    
    def update_agent_state_panel(self) -> Panel:
        """Update agent state panel."""
        content = []
        
        # Current state
        content.append(Text("State: ", style="bold"))
        state_color = {
            AgentState.IDLE: "green",
            AgentState.PROCESSING: "blue",
            AgentState.WAITING_CONFIRMATION: "yellow",
            AgentState.ERROR: "red",
            AgentState.BUSY: "orange"
        }.get(self.agent_state, "white")
        
        content.append(Text(self.agent_state.value, style=f"bold {state_color}"))
        
        # Input mode
        content.append(Text("\nInput: ", style="bold"))
        mode_color = {
            InputMode.NORMAL: "green",
            InputMode.DISABLED: "red",
            InputMode.CONFIRMATION: "yellow",
            InputMode.COMMAND: "blue"
        }.get(self.input_mode, "white")
        
        content.append(Text(self.input_mode.value, style=mode_color))
        
        # Current processing
        if self.current_processing_message:
            preview = self.current_processing_message[:25] + "..."
            content.append(Text(f"\nProcessing:\n{preview}", style="dim italic"))
        
        return Panel(
            Group(*content),
            title="🎛️ Agent State",
            box=box.ROUNDED
        )
    
    def update_chat(self) -> Panel:
        """Update chat panel."""
        messages = []
        
        for msg in self.chat_history[-25:]:  # Last 25 messages
            role = msg["role"]
            content = msg["content"]
            timestamp = msg.get("timestamp", "")
            
            if role == "user":
                header = Text(f"[{timestamp}] You:", style="bold cyan")
                # Add queue indicator if this message was queued
                if msg.get("was_queued"):
                    header.append(" [queued]", style="dim yellow")
                messages.append(header)
                messages.append(Text(f"  {content}\n", style="cyan"))
                
            elif role == "assistant":
                header = Text(f"[{timestamp}] Agent:", style="bold green")
                messages.append(header)
                
                # Handle markdown/code
                if "```" in content:
                    parts = content.split("```")
                    for i, part in enumerate(parts):
                        if i % 2 == 0 and part.strip():
                            messages.append(Text(f"  {part.strip()}", style="green"))
                        elif i % 2 == 1:
                            lines = part.split('\n')
                            lang = lines[0] if lines else "text"
                            code = '\n'.join(lines[1:]) if len(lines) > 1 else part
                            messages.append(Syntax(code, lang, theme="monokai", 
                                                 line_numbers=True, indent_guides=True))
                else:
                    messages.append(Text(f"  {content}\n", style="green"))
                    
            elif role == "system":
                messages.append(Text(f"[{timestamp}] System: {content}\n", 
                                   style="yellow italic"))
                                   
            elif role == "notification":
                style_map = {
                    "info": "blue",
                    "success": "green",
                    "warning": "yellow",
                    "error": "red"
                }
                color = style_map.get(msg.get("type", "info"), "white")
                messages.append(Text(f"[{timestamp}] 📢 {content}\n", 
                                   style=f"bold {color}"))
        
        if not messages:
            messages = [Text("Welcome! Type your message below.\n", style="dim italic")]
            messages.append(Text("Messages sent while I'm processing will be queued.\n", 
                               style="dim"))
        
        return Panel(
            Group(*messages),
            title="💬 Chat",
            box=box.ROUNDED,
            padding=(1, 2)
        )
    
    def update_metrics(self) -> Panel:
        """Update metrics panel."""
        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        # Session metrics
        duration = (datetime.now() - self.agent.state.created_at).total_seconds() / 60
        table.add_row("Session", f"{duration:.1f}m")
        table.add_row("Messages", str(self.stats["messages_processed"]))
        table.add_row("Dropped", str(self.stats["messages_dropped"]))
        
        # Response time
        if self.stats["messages_processed"] > 0:
            avg_time = self.stats["avg_response_time"]
            table.add_row("Avg Response", f"{avg_time:.1f}s")
        
        # Agent metrics
        table.add_row("Turns", str(self.agent.metrics["turn_count"]))
        table.add_row("Tools", str(self.agent.metrics["tool_usage_count"]))
        
        # Worker status
        worker_status = self.agent.worker_pool.get_status()
        active = worker_status["active_workers"]
        total = worker_status["num_workers"]
        table.add_row("Workers", f"{active}/{total}")
        
        return Panel(table, title="📊 Metrics", box=box.ROUNDED)
    
    def update_status(self) -> Panel:
        """Update status bar."""
        status = Text()
        
        # Status message
        status.append(self.status_message, style="bold")
        
        # Queue warning
        if self.message_queue.size() > 0:
            status.append(" | ", style="dim")
            status.append(f"{self.message_queue.size()} messages queued", 
                         style="yellow")
        
        # Confirmation prompt
        if self.pending_confirmation:
            status.append("\n", style="")
            status.append(self.pending_confirmation["question"], style="bold yellow")
            status.append(" [", style="dim")
            for i, option in enumerate(self.pending_confirmation["options"]):
                if i > 0:
                    status.append("/", style="dim")
                status.append(f"{i+1}:{option}", style="cyan")
            status.append("]", style="dim")
        
        return Panel(status, box=box.ROUNDED, style="on grey23")
    
    def update_input(self) -> Panel:
        """Update input panel."""
        if self.input_mode == InputMode.DISABLED:
            prompt = Text("Input disabled while processing...", style="dim red")
        elif self.input_mode == InputMode.CONFIRMATION:
            prompt = Text("Enter option number: ", style="bold yellow")
        else:
            prompt = Text("You: ", style="bold cyan")
            if self.agent_state != AgentState.IDLE:
                prompt.append("(message will be queued)", style="dim yellow")
        
        return Panel(prompt, box=box.ROUNDED, height=3)
    
    def add_user_message(self, content: str, was_queued: bool = False):
        """Add user message to chat."""
        self.chat_history.append({
            "role": "user",
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "was_queued": was_queued
        })
    
    def add_assistant_message(self, content: str, style: str = "default"):
        """Add assistant message to chat."""
        self.chat_history.append({
            "role": "assistant",
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "style": style
        })
    
    def add_system_message(self, content: str):
        """Add system message to chat."""
        self.chat_history.append({
            "role": "system",
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    
    def add_notification(self, message: str, type: str, duration: float):
        """Add notification to chat."""
        self.chat_history.append({
            "role": "notification",
            "content": message,
            "type": type,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "duration": duration
        })
    
    async def _request_confirmation(
        self, 
        question: str, 
        options: List[str], 
        timeout: float
    ) -> str:
        """Request confirmation from user."""
        self.pending_confirmation = {
            "question": question,
            "options": options,
            "timeout": timeout
        }
        self.agent_state = AgentState.WAITING_CONFIRMATION
        self.input_mode = InputMode.CONFIRMATION
        
        # Wait for response
        future = asyncio.Future()
        self.confirmation_callback = future
        
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            return "Timeout"
        finally:
            self.pending_confirmation = None
            self.confirmation_callback = None
            self.agent_state = AgentState.IDLE
            self.input_mode = InputMode.NORMAL
    
    def update_display(self):
        """Update all display panels."""
        self.layout["header"].update(self.update_header())
        self.layout["chat"].update(self.update_chat())
        self.layout["queue"].update(self.update_queue_panel())
        self.layout["metrics"].update(self.update_metrics())
        self.layout["agent_state"].update(self.update_agent_state_panel())
        self.layout["status"].update(self.update_status())
        self.layout["input"].update(self.update_input())
    
    async def process_message(self, message: str, was_queued: bool = False):
        """Process a single message."""
        if self.agent_state != AgentState.IDLE:
            # Queue the message instead
            added = await self.message_queue.add(message)
            if added:
                self.add_system_message(f"Message queued (queue size: {self.message_queue.size()})")
            else:
                self.stats["messages_dropped"] += 1
                self.add_system_message("Message dropped - queue full!")
            return
        
        # Set processing state
        self.agent_state = AgentState.PROCESSING
        self.current_processing_message = message
        self.status_message = "Processing your message..."
        
        # Add to chat
        self.add_user_message(message, was_queued)
        
        # Track timing
        start_time = datetime.now()
        
        try:
            # Process with agent
            response = await self.agent.send_user(message, stream=False)
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            self.stats["messages_processed"] += 1
            self.stats["total_response_time"] += response_time
            self.stats["avg_response_time"] = (
                self.stats["total_response_time"] / self.stats["messages_processed"]
            )
            
            # Agent will use msg_user to respond
            
        except Exception as e:
            self.agent_state = AgentState.ERROR
            self.add_system_message(f"Error: {str(e)}")
            self.status_message = "Error occurred"
            await asyncio.sleep(2)
        
        finally:
            self.current_processing_message = None
            self.agent_state = AgentState.IDLE
            self.status_message = "Ready"
            
            # Process next queued message if any
            if not self.message_queue.is_empty():
                next_message = await self.message_queue.get_next()
                if next_message:
                    # Process next message after a short delay
                    await asyncio.sleep(0.5)
                    await self.process_message(next_message, was_queued=True)
    
    async def handle_input(self, user_input: str):
        """Handle user input based on current mode."""
        if self.input_mode == InputMode.CONFIRMATION:
            # Handle confirmation
            try:
                choice = int(user_input) - 1
                if 0 <= choice < len(self.pending_confirmation["options"]):
                    selected = self.pending_confirmation["options"][choice]
                    if self.confirmation_callback:
                        self.confirmation_callback.set_result(selected)
                else:
                    self.add_system_message("Invalid option number")
            except ValueError:
                self.add_system_message("Please enter a number")
        
        elif user_input.startswith("/"):
            # Handle commands
            await self.handle_command(user_input)
        
        else:
            # Regular message
            await self.process_message(user_input)
    
    async def handle_command(self, command: str):
        """Handle slash commands."""
        cmd = command.lower().strip()
        
        if cmd == "/help":
            help_text = """
Commands:
/help - Show this help
/clear - Clear chat
/queue - Show queue status
/cancel - Cancel current operation
/stats - Show statistics
/quit - Exit

The TUI automatically queues messages when the agent is busy.
Messages are processed in order. Queue has a maximum size of 5.
"""
            self.add_system_message(help_text.strip())
            
        elif cmd == "/clear":
            self.chat_history = []
            self.add_system_message("Chat cleared")
            
        elif cmd == "/queue":
            status = f"Queue: {self.message_queue.size()}/{self.message_queue.queue.maxlen}"
            if self.message_queue.overflow_count > 0:
                status += f" (dropped: {self.message_queue.overflow_count})"
            self.add_system_message(status)
            
        elif cmd == "/cancel":
            if self.processing_task and not self.processing_task.done():
                self.processing_task.cancel()
                self.add_system_message("Operation cancelled")
            else:
                self.add_system_message("No operation to cancel")
                
        elif cmd == "/stats":
            stats_text = f"""
Statistics:
- Messages processed: {self.stats['messages_processed']}
- Messages dropped: {self.stats['messages_dropped']}
- Average response time: {self.stats['avg_response_time']:.1f}s
- Queue size: {self.message_queue.size()}
"""
            self.add_system_message(stats_text.strip())
            
        elif cmd == "/quit":
            self.running = False
            
        else:
            self.add_system_message(f"Unknown command: {command}")
    
    async def run(self):
        """Run the async-safe TUI."""
        # Welcome
        self.add_system_message("🚀 Async-Safe Agent TUI Started!")
        self.add_assistant_message(
            "Hello! I'm your enhanced agent. Messages sent while I'm processing "
            "will be automatically queued. Type /help for commands."
        )
        
        # Create prompt session
        self.prompt_session = PromptSession(
            history=FileHistory('.agent_async_history'),
            auto_suggest=AutoSuggestFromHistory(),
            multiline=False,
            complete_while_typing=True
        )
        
        # Main loop
        with Live(self.layout, console=self.console, refresh_per_second=4) as live:
            while self.running:
                self.update_display()
                
                # Use prompt_toolkit with patch_stdout to work with Live
                try:
                    with patch_stdout():
                        # Adjust prompt based on state
                        if self.input_mode == InputMode.CONFIRMATION:
                            prompt_text = "Option: "
                        elif self.agent_state != AgentState.IDLE:
                            prompt_text = "You (queued): "
                        else:
                            prompt_text = "You: "
                        
                        # Get input asynchronously
                        user_input = await self.prompt_session.prompt_async(
                            prompt_text,
                            enable_history_search=True
                        )
                    
                    if user_input.strip():
                        await self.handle_input(user_input.strip())
                        
                except (EOFError, KeyboardInterrupt):
                    self.running = False
                    break
                except Exception as e:
                    self.add_system_message(f"Input error: {str(e)}")
        
        # Cleanup
        await self.agent.worker_pool.stop()
        self.console.print("\n[bold green]👋 TUI shutdown complete[/bold green]")

# ————————————————————————————————————————————————————————————————
# Main Entry Point
# ————————————————————————————————————————————————————————————————

async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Async-Safe Agent TUI")
    parser.add_argument("--session", help="Resume session ID")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--queue-size", type=int, default=5, help="Max queue size")
    args = parser.parse_args()
    
    # Create agent
    agent = EnhancedStatefulAgent(
        session_id=args.session,
        stream=False,
        num_workers=args.workers
    )
    
    # Register demo tools
    register_enhanced_demo_tools(agent)
    
    # Create and run TUI
    tui = AsyncSafeAgentTUI(agent)
    
    # Adjust queue size if specified
    if args.queue_size:
        tui.message_queue = MessageQueue(max_size=args.queue_size)
    
    await tui.run()

if __name__ == "__main__":
    # Check dependencies
    try:
        import rich
        import prompt_toolkit
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "prompt_toolkit"])
        sys.exit(0)
    
    asyncio.run(main())