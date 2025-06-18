#!/usr/bin/env python3
"""
Interactive TUI for Enhanced Multi-turn Agent with Agent-Controlled UI
=====================================================================
The agent has full control over the TUI through exposed functions,
allowing it to send messages, update displays, and manage the interface.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Callable
import threading
from queue import Queue
import json
from enum import Enum

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
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings

# Import our enhanced agent
from multi_turn_agent_enhanced import (
    EnhancedStatefulAgent, TaskStatus, TaskPriority,
    register_enhanced_demo_tools, Task
)

# ————————————————————————————————————————————————————————————————
# TUI State Management
# ————————————————————————————————————————————————————————————————

class UIPanel(Enum):
    """Available UI panels."""
    CHAT = "chat"
    TASKS = "tasks"
    METRICS = "metrics"
    ENVIRONMENT = "environment"
    INSIGHTS = "insights"
    CUSTOM = "custom"

class NotificationType(Enum):
    """Notification types for UI alerts."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

# ————————————————————————————————————————————————————————————————
# Interactive Agent TUI
# ————————————————————————————————————————————————————————————————

class InteractiveAgentTUI:
    """TUI where the agent has control over the interface."""
    
    def __init__(self, agent: EnhancedStatefulAgent):
        self.agent = agent
        self.console = Console()
        self.layout = self._create_layout()
        
        # UI State
        self.chat_history: List[Dict[str, Any]] = []
        self.notifications: List[Dict[str, Any]] = []
        self.custom_panels: Dict[str, Any] = {}
        self.ui_state = {
            "focused_panel": UIPanel.CHAT,
            "theme": "dark",
            "auto_scroll": True,
            "show_timestamps": True,
            "notification_duration": 5.0
        }
        
        # Control flags
        self.running = True
        self.input_queue = Queue()
        self.update_queue = Queue()
        
        # Register TUI control tools with the agent
        self._register_tui_tools()
        
        # Keyboard bindings
        self.kb = KeyBindings()
        self._setup_keybindings()
    
    def _create_layout(self) -> Layout:
        """Create the main layout."""
        layout = Layout()
        
        # Main structure
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=4)
        )
        
        # Body split
        layout["body"].split_row(
            Layout(name="main", ratio=2),
            Layout(name="sidebar", ratio=1)
        )
        
        # Main area split
        layout["main"].split_column(
            Layout(name="notifications", size=3),
            Layout(name="chat")
        )
        
        # Sidebar split
        layout["sidebar"].split_column(
            Layout(name="metrics", size=10),
            Layout(name="tasks", ratio=1),
            Layout(name="insights", size=8)
        )
        
        return layout
    
    def _setup_keybindings(self):
        """Setup keyboard shortcuts."""
        @self.kb.add('c-q')
        def _(event):
            """Quit."""
            self.running = False
            event.app.exit()
        
        @self.kb.add('c-l')
        def _(event):
            """Clear chat."""
            self.input_queue.put("/agent clear_chat")
            event.app.exit()
        
        @self.kb.add('tab')
        def _(event):
            """Cycle focus between panels."""
            self.input_queue.put("/agent focus_next_panel")
            event.app.exit()
    
    def _register_tui_tools(self):
        """Register TUI control functions with the agent."""
        
        @self.agent.tools_registry.register(description="Send a message to the user")
        async def msg_user(
            content: str,
            style: str = "default",
            markdown: bool = True
        ) -> str:
            """Send a message to the user in the chat panel."""
            self.add_message("assistant", content, {
                "style": style,
                "markdown": markdown
            })
            return "Message sent to user"
        
        @self.agent.tools_registry.register(description="Show a notification")
        async def notify(
            message: str,
            type: str = "info",
            duration: float = 5.0
        ) -> str:
            """Show a notification banner."""
            try:
                notif_type = NotificationType(type)
            except:
                notif_type = NotificationType.INFO
            
            self.add_notification(message, notif_type, duration)
            return f"Notification shown: {message}"
        
        @self.agent.tools_registry.register(description="Update a UI panel")
        async def update_panel(
            panel: str,
            content: str,
            title: Optional[str] = None
        ) -> str:
            """Update content of a specific UI panel."""
            self.update_queue.put({
                "action": "update_panel",
                "panel": panel,
                "content": content,
                "title": title
            })
            return f"Panel {panel} updated"
        
        @self.agent.tools_registry.register(description="Create a custom panel")
        async def create_custom_panel(
            panel_id: str,
            title: str,
            content: str,
            position: str = "sidebar"
        ) -> str:
            """Create a custom panel in the UI."""
            self.custom_panels[panel_id] = {
                "title": title,
                "content": content,
                "position": position,
                "created": datetime.now()
            }
            return f"Custom panel '{panel_id}' created"
        
        @self.agent.tools_registry.register(description="Clear the chat display")
        async def clear_chat() -> str:
            """Clear the chat display (history is preserved)."""
            self.chat_history = []
            self.add_message("system", "Chat display cleared by agent")
            return "Chat cleared"
        
        @self.agent.tools_registry.register(description="Focus on a specific panel")
        async def focus_panel(panel: str) -> str:
            """Set focus to a specific UI panel."""
            try:
                self.ui_state["focused_panel"] = UIPanel(panel)
                return f"Focused on {panel} panel"
            except:
                return f"Unknown panel: {panel}"
        
        @self.agent.tools_registry.register(description="Show progress for a task")
        async def show_progress(
            task_id: str,
            progress: float,
            message: str = ""
        ) -> str:
            """Show progress for a specific task."""
            self.update_queue.put({
                "action": "task_progress",
                "task_id": task_id,
                "progress": progress,
                "message": message
            })
            return f"Progress updated for task {task_id[:8]}"
        
        @self.agent.tools_registry.register(description="Display data visualization")
        async def show_chart(
            chart_type: str,
            data: Dict[str, Any],
            title: str = "Chart"
        ) -> str:
            """Display a chart or visualization."""
            self.update_queue.put({
                "action": "show_chart",
                "type": chart_type,
                "data": data,
                "title": title
            })
            return f"Chart '{title}' displayed"
        
        @self.agent.tools_registry.register(description="Update UI theme")
        async def set_theme(theme: str) -> str:
            """Change the UI theme."""
            if theme in ["dark", "light", "contrast"]:
                self.ui_state["theme"] = theme
                self.apply_theme(theme)
                return f"Theme changed to {theme}"
            return "Invalid theme"
        
        @self.agent.tools_registry.register(description="Export current view")
        async def export_view(
            format: str = "markdown",
            filename: Optional[str] = None
        ) -> str:
            """Export the current view to a file."""
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"tui_export_{timestamp}.{format}"
            
            content = self._generate_export(format)
            with open(filename, "w") as f:
                f.write(content)
            
            return f"View exported to {filename}"
        
        @self.agent.tools_registry.register(description="Highlight important information")
        async def highlight(
            text: str,
            color: str = "yellow",
            panel: str = "chat"
        ) -> str:
            """Highlight important text in a panel."""
            self.update_queue.put({
                "action": "highlight",
                "text": text,
                "color": color,
                "panel": panel
            })
            return f"Highlighted text in {panel}"
        
        @self.agent.tools_registry.register(description="Show agent thinking process")
        async def show_thinking(
            thought: str,
            confidence: float = 0.5
        ) -> str:
            """Display the agent's thinking process."""
            self.update_queue.put({
                "action": "thinking",
                "thought": thought,
                "confidence": confidence
            })
            return "Thinking displayed"
        
        @self.agent.tools_registry.register(description="Request user confirmation")
        async def request_confirmation(
            question: str,
            options: List[str] = ["Yes", "No"]
        ) -> str:
            """Request confirmation from the user."""
            self.update_queue.put({
                "action": "confirmation",
                "question": question,
                "options": options
            })
            # This would normally wait for user response
            return "Confirmation requested"
        
        @self.agent.tools_registry.register(description="Display task workflow")
        async def show_workflow(
            workflow_name: str,
            tasks: List[Dict[str, Any]]
        ) -> str:
            """Display a visual workflow of tasks."""
            self.update_queue.put({
                "action": "workflow",
                "name": workflow_name,
                "tasks": tasks
            })
            return f"Workflow '{workflow_name}' displayed"
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to chat history."""
        self.chat_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        })
    
    def add_notification(self, message: str, type: NotificationType, duration: float):
        """Add a notification."""
        self.notifications.append({
            "message": message,
            "type": type,
            "timestamp": datetime.now(),
            "duration": duration
        })
    
    def update_header(self) -> Panel:
        """Update header panel."""
        # Session info
        info = Text()
        info.append("🤖 Interactive Agent TUI", style="bold cyan")
        info.append(" | Session: ", style="dim")
        info.append(self.agent.session_id[:8], style="yellow")
        info.append(" | ", style="dim")
        info.append(str(self.ui_state["focused_panel"].value), style="green")
        info.append(" | ", style="dim")
        info.append(datetime.now().strftime("%H:%M:%S"), style="dim")
        
        return Panel(
            Align.center(info),
            box=box.HEAVY,
            style="bold on blue"
        )
    
    def update_notifications(self) -> Panel:
        """Update notifications panel."""
        current_time = datetime.now()
        active_notifs = []
        
        # Filter active notifications
        self.notifications = [
            n for n in self.notifications
            if (current_time - n["timestamp"]).total_seconds() < n["duration"]
        ]
        
        if not self.notifications:
            return Panel("", box=box.NONE, height=3)
        
        # Display latest notification
        latest = self.notifications[-1]
        styles = {
            NotificationType.INFO: "cyan",
            NotificationType.SUCCESS: "green",
            NotificationType.WARNING: "yellow",
            NotificationType.ERROR: "red",
            NotificationType.CRITICAL: "bold red on white"
        }
        
        style = styles.get(latest["type"], "white")
        icon = {
            NotificationType.INFO: "ℹ️",
            NotificationType.SUCCESS: "✅",
            NotificationType.WARNING: "⚠️",
            NotificationType.ERROR: "❌",
            NotificationType.CRITICAL: "🚨"
        }.get(latest["type"], "📢")
        
        return Panel(
            Text(f"{icon} {latest['message']}", style=style),
            box=box.ROUNDED,
            style=style,
            height=3
        )
    
    def update_chat(self) -> Panel:
        """Update chat panel."""
        messages = []
        
        for msg in self.chat_history[-30:]:  # Last 30 messages
            role = msg["role"]
            content = msg["content"]
            timestamp = msg["timestamp"].strftime("%H:%M:%S") if self.ui_state["show_timestamps"] else ""
            metadata = msg.get("metadata", {})
            
            # Format message
            if role == "user":
                header = Text(f"{'[' + timestamp + '] ' if timestamp else ''}You:", style="bold cyan")
                messages.append(header)
                messages.append(Text(f"  {content}\n", style="cyan"))
            
            elif role == "assistant":
                header = Text(f"{'[' + timestamp + '] ' if timestamp else ''}Agent:", style="bold green")
                messages.append(header)
                
                # Handle markdown if enabled
                if metadata.get("markdown", True) and "```" not in content:
                    messages.append(Markdown(content, inline_code_theme="monokai"))
                else:
                    # Handle code blocks
                    if "```" in content:
                        parts = content.split("```")
                        for i, part in enumerate(parts):
                            if i % 2 == 0:
                                messages.append(Text(part, style="green"))
                            else:
                                lines = part.split('\n')
                                lang = lines[0] if lines else "python"
                                code = '\n'.join(lines[1:]) if len(lines) > 1 else part
                                messages.append(Syntax(code, lang, theme="monokai"))
                    else:
                        messages.append(Text(f"  {content}\n", style="green"))
            
            elif role == "system":
                messages.append(Text(f"{'[' + timestamp + '] ' if timestamp else ''}System: {content}\n", 
                                   style="yellow italic"))
            
            elif role == "thinking":
                # Special role for agent's thinking
                confidence = msg.get("metadata", {}).get("confidence", 0.5)
                conf_color = "green" if confidence > 0.8 else "yellow" if confidence > 0.5 else "red"
                messages.append(Text(f"💭 Thinking [{confidence:.2f}]: {content}\n", 
                                   style=f"dim {conf_color}"))
        
        if not messages:
            messages = [Text("Welcome! I'm your interactive agent. I can control this interface!\n", 
                           style="dim italic")]
            messages.append(Text("Try: 'Show me a notification' or 'Create a custom panel'\n", 
                               style="dim"))
        
        # Add focused indicator
        border_style = "bold green" if self.ui_state["focused_panel"] == UIPanel.CHAT else "dim"
        
        return Panel(
            Group(*messages),
            title="💬 Chat",
            box=box.ROUNDED,
            border_style=border_style,
            padding=(1, 2)
        )
    
    def update_metrics(self) -> Panel:
        """Update metrics panel."""
        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        # Agent metrics
        duration = (datetime.now() - self.agent.state.created_at).total_seconds() / 60
        table.add_row("Session", f"{duration:.1f}m")
        table.add_row("Turns", str(self.agent.metrics["turn_count"]))
        table.add_row("Tools", str(self.agent.metrics["tool_usage_count"]))
        table.add_row("Tasks", str(self.agent.metrics["tasks_created"]))
        
        # Worker status with visual indicator
        worker_status = self.agent.worker_pool.get_status()
        active = worker_status["active_workers"]
        total = worker_status["num_workers"]
        worker_bar = "█" * active + "░" * (total - active)
        table.add_row("Workers", f"{worker_bar} {active}/{total}")
        
        # Queue status
        queue_status = self.agent.task_queue.get_status()
        table.add_row("Queued", str(queue_status["queued"]))
        
        border_style = "bold green" if self.ui_state["focused_panel"] == UIPanel.METRICS else "dim"
        
        return Panel(table, title="📊 Metrics", box=box.ROUNDED, border_style=border_style)
    
    def update_tasks(self) -> Panel:
        """Update tasks panel with visual workflow."""
        lines = []
        
        # Active tasks with progress
        active_tasks = []
        for worker, task_id in self.agent.worker_pool.active_tasks.items():
            if task_id in self.agent.task_queue.tasks:
                task = self.agent.task_queue.tasks[task_id]
                active_tasks.append(task)
        
        if active_tasks:
            lines.append(Text("🔄 Active Tasks:", style="bold blue"))
            for task in active_tasks:
                # Calculate progress
                if task.started_at:
                    elapsed = (datetime.now() - task.started_at).total_seconds()
                    # Estimate progress (simple time-based)
                    progress = min(elapsed / 10.0, 0.95)  # Assume 10s average
                    bar_width = 15
                    filled = int(progress * bar_width)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    
                    lines.append(Text(f"  {task.name[:25]}...", style="blue"))
                    lines.append(Text(f"  [{bar}] {progress*100:.0f}%", style="dim blue"))
        
        # Task dependencies visualization
        pending_with_deps = [
            t for t in self.agent.task_queue.tasks.values()
            if t.status == TaskStatus.PENDING and t.dependencies
        ]
        
        if pending_with_deps:
            lines.append(Text("\n⏳ Waiting on Dependencies:", style="bold yellow"))
            for task in pending_with_deps[:3]:
                deps_status = []
                for dep_id in task.dependencies:
                    if dep_id in self.agent.task_queue.tasks:
                        dep_task = self.agent.task_queue.tasks[dep_id]
                        status_icon = "✅" if dep_task.status == TaskStatus.COMPLETED else "⏳"
                        deps_status.append(status_icon)
                
                lines.append(Text(f"  {task.name[:25]}... [{' '.join(deps_status)}]", style="yellow"))
        
        # Recent completed
        completed = list(self.agent.task_queue.completed_tasks)[-3:]
        if completed:
            lines.append(Text("\n✅ Recently Completed:", style="bold green"))
            for task in completed:
                duration = ""
                if task.started_at and task.completed_at:
                    dur = (task.completed_at - task.started_at).total_seconds()
                    duration = f" ({dur:.1f}s)"
                lines.append(Text(f"  ✓ {task.name[:25]}...{duration}", style="green"))
        
        if not lines:
            lines = [Text("No tasks yet", style="dim italic")]
        
        border_style = "bold green" if self.ui_state["focused_panel"] == UIPanel.TASKS else "dim"
        
        return Panel(Group(*lines), title="📋 Tasks & Workflows", box=box.ROUNDED, 
                    border_style=border_style)
    
    def update_insights(self) -> Panel:
        """Update insights panel with agent's reflections."""
        insights = self.agent.reflection_system.get_insights()
        
        lines = []
        
        # Confidence meter
        conf = insights["avg_confidence"]
        conf_bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
        conf_color = "green" if conf > 0.8 else "yellow" if conf > 0.5 else "red"
        lines.append(Text(f"Confidence: [{conf_bar}] {conf:.2f}", style=conf_color))
        
        # Learning insights
        if insights["recent_improvements"]:
            lines.append(Text("\n💡 Learning:", style="bold"))
            for imp in insights["recent_improvements"][:2]:
                lines.append(Text(f"• {imp[:40]}...", style="dim"))
        
        # Performance trend
        avg_duration = insights["performance_trends"]["avg_task_duration"]
        if avg_duration > 0:
            lines.append(Text(f"\n⚡ Avg task time: {avg_duration:.1f}s", style="cyan"))
        
        border_style = "bold green" if self.ui_state["focused_panel"] == UIPanel.INSIGHTS else "dim"
        
        return Panel(Group(*lines), title="🤔 Agent Insights", box=box.ROUNDED,
                    border_style=border_style)
    
    def update_footer(self) -> Panel:
        """Update footer with dynamic help."""
        # Context-sensitive help based on focused panel
        help_text = {
            UIPanel.CHAT: "Type to chat | Tab: Switch panels | Ctrl+Q: Quit",
            UIPanel.TASKS: "Viewing tasks | Tab: Switch | Enter: Details",
            UIPanel.METRICS: "Live metrics | Tab: Switch | R: Refresh",
            UIPanel.INSIGHTS: "Agent learning | Tab: Switch",
        }.get(self.ui_state["focused_panel"], "Tab: Switch panels | Ctrl+Q: Quit")
        
        # Add agent status
        status_line = Text()
        if self.agent.worker_pool.active_tasks:
            status_line.append("🔄 Working", style="blue")
        else:
            status_line.append("✅ Ready", style="green")
        
        status_line.append(" | ", style="dim")
        status_line.append(help_text, style="dim")
        
        return Panel(
            Align.center(status_line),
            box=box.ROUNDED,
            style="dim"
        )
    
    def apply_theme(self, theme: str):
        """Apply a theme to the UI."""
        # This would update console styles
        # For now, just log the change
        self.add_message("system", f"Theme changed to {theme}")
    
    def _generate_export(self, format: str) -> str:
        """Generate export content."""
        if format == "markdown":
            content = f"# Agent TUI Export\n"
            content += f"## Session: {self.agent.session_id}\n"
            content += f"## Date: {datetime.now()}\n\n"
            
            content += "### Chat History\n"
            for msg in self.chat_history:
                content += f"**{msg['role']}** [{msg['timestamp']}]: {msg['content']}\n\n"
            
            return content
        
        elif format == "json":
            return json.dumps({
                "session": self.agent.session_id,
                "timestamp": datetime.now().isoformat(),
                "chat": self.chat_history,
                "metrics": self.agent.metrics,
                "state": {
                    "direction": str(self.agent.state.direction),
                    "goals": self.agent.state.active_goals
                }
            }, indent=2, default=str)
        
        return "Unsupported format"
    
    def update_display(self):
        """Update all display panels."""
        self.layout["header"].update(self.update_header())
        self.layout["notifications"].update(self.update_notifications())
        self.layout["chat"].update(self.update_chat())
        self.layout["metrics"].update(self.update_metrics())
        self.layout["tasks"].update(self.update_tasks())
        self.layout["insights"].update(self.update_insights())
        self.layout["footer"].update(self.update_footer())
        
        # Process update queue
        while not self.update_queue.empty():
            try:
                update = self.update_queue.get_nowait()
                self._process_update(update)
            except:
                pass
    
    def _process_update(self, update: Dict[str, Any]):
        """Process UI update from agent."""
        action = update.get("action")
        
        if action == "thinking":
            # Add thinking to chat
            self.chat_history.append({
                "role": "thinking",
                "content": update["thought"],
                "timestamp": datetime.now(),
                "metadata": {"confidence": update["confidence"]}
            })
        
        elif action == "highlight":
            # Would implement highlighting logic
            pass
        
        elif action == "workflow":
            # Would create workflow visualization
            self.add_message("system", 
                           f"Workflow '{update['name']}' with {len(update['tasks'])} tasks")
    
    async def process_input(self, user_input: str):
        """Process user input."""
        if user_input.startswith("/agent "):
            # Direct agent command
            cmd = user_input[7:]
            self.add_message("system", f"Agent command: {cmd}")
            
            # Let agent handle it
            response = await self.agent.send_user(f"Execute TUI command: {cmd}")
            
        elif user_input.startswith("/"):
            # System command
            await self.handle_command(user_input)
            
        else:
            # Regular chat
            self.add_message("user", user_input)
            
            # Send to agent
            try:
                response = await self.agent.send_user(user_input, stream=False)
                # Agent will use msg_user to respond
            except Exception as e:
                self.add_message("system", f"Error: {str(e)}")
    
    async def handle_command(self, command: str):
        """Handle system commands."""
        if command == "/help":
            help_text = """
System Commands:
/help - Show this help
/quit - Exit TUI

Agent Commands (use /agent <command>):
- msg_user "message" - Send a message
- notify "message" "type" - Show notification  
- clear_chat - Clear the chat
- focus_panel "panel" - Focus a panel
- show_thinking "thought" - Show thinking
- create_custom_panel "id" "title" "content"
- export_view "format" - Export current view

The agent can control the entire UI!
"""
            self.add_message("system", help_text.strip())
        
        elif command == "/quit":
            self.running = False
    
    async def run(self):
        """Run the interactive TUI."""
        # Welcome
        self.add_message("system", "🚀 Interactive Agent TUI Started!")
        self.add_message("assistant", 
                        "Hello! I'm your enhanced agent with UI control. "
                        "I can update this interface, show notifications, "
                        "create panels, and more! Try asking me to 'show a notification' "
                        "or 'create a workflow visualization'.")
        
        # Input thread
        input_thread = threading.Thread(target=self.input_loop, daemon=True)
        input_thread.start()
        
        # Main display loop
        with Live(self.layout, console=self.console, refresh_per_second=4) as live:
            while self.running:
                self.update_display()
                
                # Process input
                if not self.input_queue.empty():
                    user_input = self.input_queue.get()
                    await self.process_input(user_input)
                
                await asyncio.sleep(0.1)
        
        # Cleanup
        await self.agent.worker_pool.stop()
        self.console.print("\n[bold green]👋 Interactive TUI shutdown[/bold green]")
    
    def input_loop(self):
        """Input loop in separate thread."""
        history = FileHistory('.agent_interactive_history')
        
        while self.running:
            try:
                user_input = prompt(
                    "You: ",
                    history=history,
                    auto_suggest=AutoSuggestFromHistory(),
                    key_bindings=self.kb
                )
                
                if user_input.strip():
                    self.input_queue.put(user_input.strip())
                    
            except (EOFError, KeyboardInterrupt):
                self.running = False
                break

# ————————————————————————————————————————————————————————————————
# Enhanced Demo Tools
# ————————————————————————————————————————————————————————————————

def register_interactive_demo_tools(agent: EnhancedStatefulAgent):
    """Register demo tools that showcase UI control."""
    
    @agent.tools_registry.register(description="Demo UI capabilities")
    async def demo_ui_features() -> str:
        """Demonstrate various UI control features."""
        # This would be called by the agent to show off UI features
        demos = [
            "msg_user('Let me show you what I can do!', style='bold')",
            "notify('This is an info notification', type='info')",
            "notify('Task completed successfully!', type='success')",
            "show_thinking('I can display my reasoning process...', confidence=0.85)",
            "create_custom_panel('demo', 'Demo Panel', 'Custom content here!')",
            "show_progress('task-123', 0.75, 'Processing data...')"
        ]
        
        return f"UI demo ready. I can execute: {', '.join(demos)}"
    
    @agent.tools_registry.register(description="Create an interactive workflow")
    async def create_interactive_workflow(
        name: str,
        steps: List[str]
    ) -> str:
        """Create a workflow with UI visualization."""
        # Create tasks
        workflow_tasks = []
        for i, step in enumerate(steps):
            task = Task(
                name=f"{name} - {step}",
                description=step,
                priority=TaskPriority.HIGH if i == 0 else TaskPriority.MEDIUM
            )
            workflow_tasks.append({
                "id": task.id,
                "name": task.name,
                "status": "pending",
                "dependencies": [workflow_tasks[-1]["id"]] if workflow_tasks else []
            })
        
        # Show in UI
        await agent.tools_registry.call("show_workflow", 
                                      workflow_name=name, 
                                      tasks=workflow_tasks)
        
        return f"Interactive workflow '{name}' created with {len(steps)} steps"

# ————————————————————————————————————————————————————————————————
# Main Entry Point  
# ————————————————————————————————————————————————————————————————

async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Agent TUI")
    parser.add_argument("--session", help="Resume session ID")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--demo", action="store_true", help="Run UI demo on start")
    args = parser.parse_args()
    
    # Create agent
    agent = EnhancedStatefulAgent(
        session_id=args.session,
        stream=False,
        num_workers=args.workers
    )
    
    # Register demo tools
    register_enhanced_demo_tools(agent)
    register_interactive_demo_tools(agent)
    
    # Create and run TUI
    tui = InteractiveAgentTUI(agent)
    
    # If demo mode, queue a demo command
    if args.demo:
        tui.input_queue.put("Show me all your UI capabilities with examples")
    
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