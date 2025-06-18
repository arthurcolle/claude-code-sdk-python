#!/usr/bin/env python3
"""
Interactive TUI (Terminal User Interface) for Enhanced Multi-turn Agent
=======================================================================
A modern, responsive terminal interface with real-time updates, multiple panels,
and rich visualizations for the enhanced agent.

Features:
- Split-pane layout with chat, status, and task views
- Real-time updates of agent metrics
- Syntax highlighting for code
- Mouse support
- Keyboard shortcuts
- Rich formatting and colors
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.syntax import Syntax
from rich.align import Align
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.markdown import Markdown
from rich.tree import Tree
from rich import box
from rich.style import Style
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Input, Static, Button, Label, 
    ListView, ListItem, ProgressBar, Sparkline, DataTable
)
from textual.reactive import reactive
from textual.message import Message
from textual import events
from textual.binding import Binding
from textual.timer import Timer

# Import our enhanced agent
from multi_turn_agent_enhanced import (
    EnhancedStatefulAgent, TaskStatus, TaskPriority,
    visualize_agent_status, register_enhanced_demo_tools
)

# ————————————————————————————————————————————————————————————————
# Custom Widgets
# ————————————————————————————————————————————————————————————————

class ChatMessage(Static):
    """A single chat message widget."""
    
    def __init__(self, role: str, content: str, timestamp: Optional[datetime] = None):
        super().__init__()
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        
    def compose(self) -> ComposeResult:
        role_styles = {
            "user": "bold cyan",
            "assistant": "bold green",
            "system": "dim yellow",
            "tool": "bold magenta"
        }
        role_emojis = {
            "user": "👤",
            "assistant": "🤖",
            "system": "⚙️",
            "tool": "🔧"
        }
        
        style = role_styles.get(self.role, "white")
        emoji = role_emojis.get(self.role, "❓")
        
        # Format timestamp
        time_str = self.timestamp.strftime("%H:%M:%S")
        
        # Create message text
        header = Text(f"{emoji} {self.role.title()} ", style=style)
        header.append(f"[{time_str}]", style="dim")
        
        # Handle code blocks and markdown
        if "```" in self.content:
            # Extract code blocks
            parts = self.content.split("```")
            content_widgets = []
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    # Regular text
                    if part.strip():
                        content_widgets.append(Markdown(part.strip()))
                else:
                    # Code block
                    lines = part.strip().split('\n')
                    lang = lines[0] if lines else "python"
                    code = '\n'.join(lines[1:]) if len(lines) > 1 else part
                    content_widgets.append(
                        Panel(
                            Syntax(code, lang, theme="monokai", line_numbers=True),
                            border_style="dim",
                            box=box.ROUNDED
                        )
                    )
            
            yield Static(header)
            for widget in content_widgets:
                yield Static(widget)
        else:
            # Regular message
            yield Static(Group(header, Markdown(self.content)))

class TaskCard(Static):
    """A card widget for displaying task information."""
    
    def __init__(self, task):
        super().__init__()
        self.task = task
        
    def render(self) -> str:
        status_colors = {
            TaskStatus.PENDING: "yellow",
            TaskStatus.QUEUED: "cyan",
            TaskStatus.RUNNING: "blue",
            TaskStatus.COMPLETED: "green",
            TaskStatus.FAILED: "red",
            TaskStatus.CANCELLED: "dim"
        }
        
        status_emojis = {
            TaskStatus.PENDING: "📍",
            TaskStatus.QUEUED: "⏳",
            TaskStatus.RUNNING: "🔄",
            TaskStatus.COMPLETED: "✅",
            TaskStatus.FAILED: "❌",
            TaskStatus.CANCELLED: "🚫"
        }
        
        color = status_colors.get(self.task.status, "white")
        emoji = status_emojis.get(self.task.status, "❓")
        
        # Calculate duration
        duration = ""
        if self.task.started_at and self.task.completed_at:
            dur = (self.task.completed_at - self.task.started_at).total_seconds()
            duration = f" ({dur:.1f}s)"
        elif self.task.started_at:
            dur = (datetime.now() - self.task.started_at).total_seconds()
            duration = f" ({dur:.1f}s elapsed)"
        
        priority_badge = f"[{self.task.priority.name}]" if self.task.priority else ""
        
        return f"[{color}]{emoji} {self.task.name[:40]}...{duration} {priority_badge}[/{color}]"

class MetricsPanel(Static):
    """A panel for displaying agent metrics."""
    
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        
    def render(self) -> Table:
        table = Table(box=box.SIMPLE, show_header=False, padding=0)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        # Session info
        duration = (datetime.now() - self.agent.state.created_at).total_seconds() / 60
        table.add_row("Session", f"{self.agent.session_id[:8]}")
        table.add_row("Duration", f"{duration:.1f} min")
        table.add_row("Direction", str(self.agent.state.direction))
        
        # Metrics
        table.add_row("Turns", str(self.agent.metrics["turn_count"]))
        table.add_row("Tools Used", str(self.agent.metrics["tool_usage_count"]))
        table.add_row("Tasks Created", str(self.agent.metrics["tasks_created"]))
        table.add_row("Reflections", str(self.agent.metrics["reflections_generated"]))
        
        # Queue status
        queue_status = self.agent.task_queue.get_status()
        table.add_row("Tasks Total", str(queue_status["total_tasks"]))
        table.add_row("Tasks Queued", str(queue_status["queued"]))
        
        # Worker status
        worker_status = self.agent.worker_pool.get_status()
        active = worker_status["active_workers"]
        total = worker_status["num_workers"]
        table.add_row("Workers", f"{active}/{total} active")
        
        return table

class EnvironmentMonitor(Static):
    """Widget for environment monitoring."""
    
    def __init__(self, env_monitor):
        super().__init__()
        self.env_monitor = env_monitor
        
    def render(self) -> Panel:
        summary = self.env_monitor.get_summary()
        state = summary.get("current_state", {})
        
        if not state:
            return Panel("No environment data available", title="🖥️ Environment")
        
        system = state.get("system", {})
        process = state.get("process", {})
        
        # Create progress bars
        mem_percent = system.get("memory", {}).get("percent", 0)
        disk_percent = system.get("disk", {}).get("percent", 0)
        cpu_percent = process.get("cpu_percent", 0)
        
        content = f"""
[bold]System:[/bold] {system.get('platform', 'Unknown')} | Python {system.get('python_version', 'Unknown')}

[bold cyan]Memory:[/bold cyan] {self._progress_bar(mem_percent, 100)} {mem_percent:.1f}%
[bold yellow]Disk:[/bold yellow]   {self._progress_bar(disk_percent, 100)} {disk_percent:.1f}%
[bold green]CPU:[/bold green]    {self._progress_bar(cpu_percent, 100)} {cpu_percent:.1f}%

[bold]Process:[/bold] PID {process.get('pid', 0)} | {process.get('memory_mb', 0):.1f}MB
"""
        
        # Add alerts if any
        alerts = summary.get("alerts", [])
        if alerts:
            content += "\n[bold red]⚠️ Alerts:[/bold red]\n"
            for alert in alerts:
                content += f"  • {alert}\n"
        
        return Panel(content.strip(), title="🖥️ Environment", box=box.ROUNDED)
    
    def _progress_bar(self, value: float, max_value: float, width: int = 20) -> str:
        """Create a simple text progress bar."""
        filled = int((value / max_value) * width)
        bar = "█" * filled + "░" * (width - filled)
        
        # Color based on value
        if value > 90:
            color = "red"
        elif value > 70:
            color = "yellow"
        else:
            color = "green"
        
        return f"[{color}]{bar}[/{color}]"

# ————————————————————————————————————————————————————————————————
# Main TUI Application
# ————————————————————————————————————————————————————————————————

class AgentTUI(App):
    """Enhanced Multi-turn Agent TUI Application."""
    
    CSS = """
    Screen {
        layout: grid;
        grid-size: 3 3;
        grid-columns: 2fr 1fr;
        grid-rows: 1fr 8fr 1fr;
    }
    
    #header {
        column-span: 2;
        height: 3;
        background: $primary;
    }
    
    #chat-container {
        border: solid $primary;
        overflow-y: scroll;
        padding: 1;
    }
    
    #sidebar {
        layout: vertical;
    }
    
    #metrics-panel {
        height: 12;
        border: solid $secondary;
        padding: 1;
    }
    
    #tasks-panel {
        border: solid $secondary;
        overflow-y: scroll;
        padding: 1;
    }
    
    #environment-panel {
        height: 12;
        border: solid $secondary;
        padding: 1;
    }
    
    #input-container {
        column-span: 2;
        height: 3;
        layout: horizontal;
    }
    
    #input {
        width: 100%;
        dock: left;
    }
    
    #send-button {
        width: 10;
        dock: right;
    }
    
    ChatMessage {
        margin: 1 0;
    }
    
    TaskCard {
        margin: 0 0 1 0;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+s", "show_status", "Status"),
        Binding("ctrl+t", "show_tasks", "Tasks"),
        Binding("ctrl+r", "show_reflections", "Reflections"),
        Binding("ctrl+h", "show_help", "Help"),
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("f1", "toggle_theme", "Theme"),
        Binding("f2", "export_chat", "Export"),
    ]
    
    def __init__(self, agent: EnhancedStatefulAgent):
        super().__init__()
        self.agent = agent
        self.chat_messages: List[ChatMessage] = []
        self.update_timer: Optional[Timer] = None
        
    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield Header(show_clock=True)
        
        # Main chat area
        with ScrollableContainer(id="chat-container"):
            yield Container(id="chat-messages")
        
        # Sidebar
        with Container(id="sidebar"):
            yield MetricsPanel(self.agent).data_bind(id="metrics-panel")
            
            with ScrollableContainer(id="tasks-panel"):
                yield Static("📋 [bold]Task Queue[/bold]")
                yield Container(id="task-list")
            
            yield EnvironmentMonitor(self.agent.env_monitor).data_bind(id="environment-panel")
        
        # Input area
        with Horizontal(id="input-container"):
            yield Input(placeholder="Type your message here...", id="input")
            yield Button("Send", variant="primary", id="send-button")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize the app when mounted."""
        self.title = f"🤖 Enhanced Agent TUI - Session {self.agent.session_id[:8]}"
        
        # Add welcome message
        self.add_message("system", "Welcome to the Enhanced Multi-turn Agent TUI!")
        self.add_message("system", "Type 'help' for available commands or use Ctrl+H")
        
        # Focus on input
        self.query_one("#input").focus()
        
        # Start update timer
        self.update_timer = self.set_interval(1.0, self.update_displays)
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the chat."""
        msg = ChatMessage(role, content)
        self.chat_messages.append(msg)
        
        container = self.query_one("#chat-messages")
        container.mount(msg)
        
        # Scroll to bottom
        self.query_one("#chat-container").scroll_end()
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        message = event.value.strip()
        if not message:
            return
        
        # Clear input
        event.input.value = ""
        
        # Add user message
        self.add_message("user", message)
        
        # Handle special commands
        if message.lower() == "help":
            self.show_help()
            return
        elif message.lower() == "status":
            await self.show_full_status()
            return
        elif message.lower() == "clear":
            self.clear_chat()
            return
        elif message.lower().startswith("export"):
            self.export_chat()
            return
        
        # Process with agent
        await self.process_message(message)
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "send-button":
            input_widget = self.query_one("#input", Input)
            if input_widget.value.strip():
                # Trigger submission
                input_widget.action_submit()
    
    async def process_message(self, message: str) -> None:
        """Process a message with the agent."""
        # Show typing indicator
        self.add_message("system", "🤔 Agent is thinking...")
        
        try:
            # Send to agent (in background to keep UI responsive)
            response = await asyncio.create_task(
                self.agent.send_user(message, stream=False)
            )
            
            # Remove typing indicator
            container = self.query_one("#chat-messages")
            if container.children:
                await container.children[-1].remove()
            
            # Add assistant response
            self.add_message("assistant", response)
            
        except Exception as e:
            self.add_message("system", f"❌ Error: {str(e)}")
    
    def update_displays(self) -> None:
        """Update all display panels."""
        # Update metrics
        metrics_panel = self.query_one("#metrics-panel", MetricsPanel)
        metrics_panel.refresh()
        
        # Update environment
        env_panel = self.query_one("#environment-panel", EnvironmentMonitor)
        env_panel.refresh()
        
        # Update task list
        self.update_task_list()
    
    def update_task_list(self) -> None:
        """Update the task list display."""
        container = self.query_one("#task-list")
        container.remove_children()
        
        # Get all tasks
        all_tasks = []
        
        # Active tasks
        for task_id in self.agent.worker_pool.active_tasks.values():
            if task_id in self.agent.task_queue.tasks:
                task = self.agent.task_queue.tasks[task_id]
                all_tasks.append((task, True))
        
        # Queued tasks
        queued_tasks = []
        temp_queue = []
        while not self.agent.task_queue.queue.empty():
            try:
                task = self.agent.task_queue.queue.get_nowait()
                queued_tasks.append(task)
                temp_queue.append(task)
            except:
                break
        
        # Put tasks back
        for task in temp_queue:
            self.agent.task_queue.queue.put_nowait(task)
        
        for task in queued_tasks[:5]:  # Show top 5 queued
            all_tasks.append((task, False))
        
        # Recent completed tasks
        for task in list(self.agent.task_queue.completed_tasks)[-3:]:
            all_tasks.append((task, False))
        
        # Display tasks
        if all_tasks:
            for task, is_active in all_tasks:
                card = TaskCard(task)
                if is_active:
                    card.styles.border = ("solid", "green")
                container.mount(card)
        else:
            container.mount(Static("[dim]No tasks[/dim]"))
    
    def show_help(self) -> None:
        """Show help information."""
        help_text = """
[bold cyan]Available Commands:[/bold cyan]
• help     - Show this help message
• status   - Show full agent status
• clear    - Clear chat history
• export   - Export chat to file

[bold cyan]Keyboard Shortcuts:[/bold cyan]
• Ctrl+S   - Show status panel
• Ctrl+T   - Focus on tasks
• Ctrl+R   - Show reflections
• Ctrl+H   - Show help
• Ctrl+L   - Clear chat
• Ctrl+C   - Quit
• F1       - Toggle theme
• F2       - Export chat

[bold cyan]Agent Capabilities:[/bold cyan]
• create_task - Create a new task
• get_task_status - Check task status
• get_insights - View reflection insights
• get_environment - Check system status
• search_history - Search conversation
• analyze_performance - Run analysis
• create_workflow - Create task workflow
"""
        self.add_message("system", help_text.strip())
    
    async def show_full_status(self) -> None:
        """Show full agent status."""
        from io import StringIO
        import sys
        
        # Capture status output
        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()
        
        visualize_agent_status(self.agent)
        
        sys.stdout = old_stdout
        status_text = buffer.getvalue()
        
        # Add to chat
        self.add_message("system", f"```\n{status_text}\n```")
    
    def clear_chat(self) -> None:
        """Clear chat messages."""
        container = self.query_one("#chat-messages")
        container.remove_children()
        self.chat_messages.clear()
        self.add_message("system", "Chat cleared. History preserved in agent memory.")
    
    def export_chat(self) -> None:
        """Export chat to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_export_{self.agent.session_id[:8]}_{timestamp}.md"
        
        with open(filename, "w") as f:
            f.write(f"# Chat Export - Session {self.agent.session_id}\n")
            f.write(f"## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for msg in self.chat_messages:
                f.write(f"### {msg.role.title()} [{msg.timestamp.strftime('%H:%M:%S')}]\n")
                f.write(f"{msg.content}\n\n")
        
        self.add_message("system", f"✅ Chat exported to {filename}")
    
    async def action_show_status(self) -> None:
        """Action to show status."""
        await self.show_full_status()
    
    def action_show_tasks(self) -> None:
        """Action to focus on tasks panel."""
        self.query_one("#tasks-panel").focus()
    
    def action_show_reflections(self) -> None:
        """Action to show reflections."""
        insights = self.agent.reflection_system.get_insights()
        
        text = f"""
[bold cyan]Reflection Insights:[/bold cyan]
• Total Reflections: {insights['total_reflections']}
• Average Confidence: {insights['avg_confidence']:.2f}
• Performance Trend: {insights['performance_trends']['avg_task_duration']:.2f}s avg

[bold cyan]Common Errors:[/bold cyan]
"""
        for error, count in list(insights['common_errors'].items())[:3]:
            text += f"• {error}: {count} times\n"
        
        text += "\n[bold cyan]Recent Improvements:[/bold cyan]\n"
        for imp in insights['recent_improvements'][:3]:
            text += f"• {imp[:60]}...\n"
        
        self.add_message("system", text.strip())
    
    def action_clear_chat(self) -> None:
        """Action to clear chat."""
        self.clear_chat()
    
    def action_toggle_theme(self) -> None:
        """Toggle between light and dark themes."""
        self.dark = not self.dark
        theme = "dark" if self.dark else "light"
        self.add_message("system", f"Switched to {theme} theme")
    
    def action_export_chat(self) -> None:
        """Action to export chat."""
        self.export_chat()

# ————————————————————————————————————————————————————————————————
# Launch Function
# ————————————————————————————————————————————————————————————————

async def run_tui():
    """Run the TUI application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Agent TUI")
    parser.add_argument("--session", help="Resume session ID")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    args = parser.parse_args()
    
    # Create agent
    print("🚀 Starting Enhanced Agent TUI...")
    agent = EnhancedStatefulAgent(
        session_id=args.session,
        stream=False,  # TUI handles display
        num_workers=args.workers
    )
    
    # Register demo tools
    register_enhanced_demo_tools(agent)
    
    # Create and run TUI
    app = AgentTUI(agent)
    await app.run_async()
    
    # Cleanup
    await agent.worker_pool.stop()
    print("\n👋 Agent TUI shutdown complete")

if __name__ == "__main__":
    # Install required packages if needed
    try:
        import textual
        import rich
    except ImportError:
        print("Installing required TUI packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "textual", "rich"])
        print("Packages installed! Please run again.")
        sys.exit(0)
    
    # Run the TUI
    asyncio.run(run_tui())