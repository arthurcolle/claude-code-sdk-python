#!/usr/bin/env python3
"""
Rich Console TUI for Enhanced Multi-turn Agent
==============================================
A simpler but feature-rich terminal interface using Rich's Live display.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import threading
from queue import Queue
import signal

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich import box
from rich.style import Style
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style as PromptStyle

# Import our enhanced agent
from multi_turn_agent_enhanced import (
    EnhancedStatefulAgent, TaskStatus, TaskPriority,
    register_enhanced_demo_tools
)

# ————————————————————————————————————————————————————————————————
# Rich TUI Components
# ————————————————————————————————————————————————————————————————

class RichAgentTUI:
    """Rich console-based TUI for the enhanced agent."""
    
    def __init__(self, agent: EnhancedStatefulAgent):
        self.agent = agent
        self.console = Console()
        self.chat_history: List[Dict[str, Any]] = []
        self.running = True
        self.input_queue = Queue()
        self.layout = self._create_layout()
        self.last_update = datetime.now()
        
        # Prompt toolkit setup
        self.kb = KeyBindings()
        self._setup_keybindings()
        
        # Styling
        self.prompt_style = PromptStyle.from_dict({
            'prompt': '#00aa00 bold',
            'prompt.prefix': '#888888',
        })
    
    def _create_layout(self) -> Layout:
        """Create the main layout."""
        layout = Layout()
        
        # Split into header, body, and footer
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # Split body into chat and sidebar
        layout["body"].split_row(
            Layout(name="chat", ratio=2),
            Layout(name="sidebar", ratio=1)
        )
        
        # Split sidebar into multiple panels
        layout["sidebar"].split_column(
            Layout(name="metrics", size=10),
            Layout(name="tasks", ratio=2),
            Layout(name="environment", size=10),
            Layout(name="insights", ratio=1)
        )
        
        return layout
    
    def _setup_keybindings(self):
        """Setup keyboard shortcuts."""
        @self.kb.add('c-s')
        def _(event):
            """Show status."""
            self.input_queue.put("/status")
            event.app.exit()
        
        @self.kb.add('c-h')
        def _(event):
            """Show help."""
            self.input_queue.put("/help")
            event.app.exit()
        
        @self.kb.add('c-l')
        def _(event):
            """Clear screen."""
            self.input_queue.put("/clear")
            event.app.exit()
        
        @self.kb.add('c-q')
        def _(event):
            """Quit."""
            self.input_queue.put("/quit")
            event.app.exit()
    
    def update_header(self) -> Panel:
        """Update header panel."""
        session_info = Text()
        session_info.append("🤖 Enhanced Multi-turn Agent TUI", style="bold cyan")
        session_info.append(" | ")
        session_info.append(f"Session: {self.agent.session_id[:8]}", style="yellow")
        session_info.append(" | ")
        session_info.append(f"Direction: {self.agent.state.direction}", style="green")
        session_info.append(" | ")
        session_info.append(datetime.now().strftime("%H:%M:%S"), style="dim")
        
        return Panel(
            Align.center(session_info),
            box=box.DOUBLE,
            style="bold on dark_blue"
        )
    
    def update_chat(self) -> Panel:
        """Update chat panel."""
        chat_content = []
        
        # Show last N messages
        for msg in self.chat_history[-20:]:  # Last 20 messages
            role = msg["role"]
            content = msg["content"]
            timestamp = msg.get("timestamp", "")
            
            # Format based on role
            if role == "user":
                chat_content.append(Text(f"[{timestamp}] You:", style="bold cyan"))
                chat_content.append(Text(f"  {content}\n", style="cyan"))
            elif role == "assistant":
                chat_content.append(Text(f"[{timestamp}] Assistant:", style="bold green"))
                # Handle code blocks
                if "```" in content:
                    parts = content.split("```")
                    for i, part in enumerate(parts):
                        if i % 2 == 0:
                            chat_content.append(Text(f"  {part.strip()}", style="green"))
                        else:
                            lines = part.strip().split('\n')
                            lang = lines[0] if lines else "python"
                            code = '\n'.join(lines[1:]) if len(lines) > 1 else part
                            chat_content.append(Syntax(code, lang, theme="monokai", line_numbers=True))
                else:
                    chat_content.append(Text(f"  {content}\n", style="green"))
            elif role == "system":
                chat_content.append(Text(f"[{timestamp}] System: {content}\n", style="yellow dim"))
            elif role == "tool":
                chat_content.append(Text(f"[{timestamp}] Tool Result:", style="bold magenta"))
                chat_content.append(Text(f"  {content}\n", style="magenta"))
        
        if not chat_content:
            chat_content = [Text("Welcome! Type your message or /help for commands.", style="dim")]
        
        return Panel(
            Group(*chat_content),
            title="💬 Chat",
            box=box.ROUNDED,
            padding=(1, 2)
        )
    
    def update_metrics(self) -> Panel:
        """Update metrics panel."""
        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="white")
        
        # Calculate session duration
        duration = (datetime.now() - self.agent.state.created_at).total_seconds() / 60
        
        # Add metrics
        table.add_row("Duration", f"{duration:.1f} min")
        table.add_row("Turns", str(self.agent.metrics["turn_count"]))
        table.add_row("Tools", str(self.agent.metrics["tool_usage_count"]))
        table.add_row("Tasks", str(self.agent.metrics["tasks_created"]))
        table.add_row("Reflections", str(self.agent.metrics["reflections_generated"]))
        
        # Worker status
        worker_status = self.agent.worker_pool.get_status()
        active = worker_status["active_workers"]
        total = worker_status["num_workers"]
        worker_bar = self._create_progress_bar(active, total, 10)
        table.add_row("Workers", f"{worker_bar} {active}/{total}")
        
        return Panel(table, title="📊 Metrics", box=box.ROUNDED)
    
    def update_tasks(self) -> Panel:
        """Update tasks panel."""
        task_lines = []
        
        # Task queue summary
        queue_status = self.agent.task_queue.get_status()
        task_lines.append(Text(f"Total: {queue_status['total_tasks']} | Queued: {queue_status['queued']}", style="bold"))
        task_lines.append(Text())
        
        # Active tasks
        if self.agent.worker_pool.active_tasks:
            task_lines.append(Text("🔄 Active:", style="bold blue"))
            for worker, task_id in self.agent.worker_pool.active_tasks.items():
                if task_id in self.agent.task_queue.tasks:
                    task = self.agent.task_queue.tasks[task_id]
                    elapsed = ""
                    if task.started_at:
                        elapsed = f" ({(datetime.now() - task.started_at).total_seconds():.1f}s)"
                    task_lines.append(Text(f"  • {task.name[:30]}...{elapsed}", style="blue"))
            task_lines.append(Text())
        
        # Recent completed
        completed = list(self.agent.task_queue.completed_tasks)[-5:]
        if completed:
            task_lines.append(Text("✅ Recent:", style="bold green"))
            for task in completed:
                status_icon = "✅" if task.status == TaskStatus.COMPLETED else "❌"
                duration = ""
                if task.started_at and task.completed_at:
                    duration = f" ({(task.completed_at - task.started_at).total_seconds():.1f}s)"
                style = "green" if task.status == TaskStatus.COMPLETED else "red"
                task_lines.append(Text(f"  {status_icon} {task.name[:30]}...{duration}", style=style))
        
        if not task_lines[2:]:  # If only header
            task_lines.append(Text("No tasks yet", style="dim"))
        
        return Panel(Group(*task_lines), title="📋 Tasks", box=box.ROUNDED)
    
    def update_environment(self) -> Panel:
        """Update environment panel."""
        env_summary = self.agent.environment_monitor.get_summary()
        state = env_summary.get("current_state", {})
        
        if not state:
            return Panel("No data", title="🖥️ Environment", box=box.ROUNDED)
        
        system = state.get("system", {})
        process = state.get("process", {})
        
        # Create content
        lines = []
        
        # Memory
        mem = system.get("memory", {}).get("percent", 0)
        mem_bar = self._create_progress_bar(mem, 100, 15)
        mem_color = "red" if mem > 90 else "yellow" if mem > 70 else "green"
        lines.append(Text.from_markup(f"Mem: [{mem_color}]{mem_bar}[/] {mem:.1f}%"))
        
        # Disk
        disk = system.get("disk", {}).get("percent", 0)
        disk_bar = self._create_progress_bar(disk, 100, 15)
        disk_color = "red" if disk > 90 else "yellow" if disk > 70 else "green"
        lines.append(Text.from_markup(f"Disk: [{disk_color}]{disk_bar}[/] {disk:.1f}%"))
        
        # CPU
        cpu = process.get("cpu_percent", 0)
        cpu_bar = self._create_progress_bar(cpu, 100, 15)
        cpu_color = "red" if cpu > 80 else "yellow" if cpu > 60 else "green"
        lines.append(Text.from_markup(f"CPU: [{cpu_color}]{cpu_bar}[/] {cpu:.1f}%"))
        
        # Alerts
        alerts = env_summary.get("alerts", [])
        if alerts:
            lines.append(Text())
            lines.append(Text("⚠️ Alerts:", style="bold red"))
            for alert in alerts[:2]:
                lines.append(Text(f"  • {alert}", style="red"))
        
        return Panel(Group(*lines), title="🖥️ Environment", box=box.ROUNDED)
    
    def update_insights(self) -> Panel:
        """Update insights panel."""
        insights = self.agent.reflection_system.get_insights()
        
        lines = []
        lines.append(Text(f"Reflections: {insights['total_reflections']}", style="cyan"))
        lines.append(Text(f"Confidence: {insights['avg_confidence']:.2f}", style="green"))
        
        if insights["recent_improvements"]:
            lines.append(Text())
            lines.append(Text("💡 Improvements:", style="bold"))
            for imp in insights["recent_improvements"][:2]:
                lines.append(Text(f"• {imp[:40]}...", style="dim"))
        
        return Panel(Group(*lines), title="🤔 Insights", box=box.ROUNDED)
    
    def update_footer(self) -> Panel:
        """Update footer panel."""
        shortcuts = [
            "[bold]Commands:[/] /help | /status | /clear | /export | /quit",
            "[bold]Shortcuts:[/] Ctrl+S (status) | Ctrl+H (help) | Ctrl+L (clear) | Ctrl+Q (quit)"
        ]
        
        return Panel(
            "\n".join(shortcuts),
            box=box.ROUNDED,
            style="dim"
        )
    
    def _create_progress_bar(self, value: float, max_value: float, width: int = 20) -> str:
        """Create a text progress bar."""
        filled = int((value / max_value) * width)
        return "█" * filled + "░" * (width - filled)
    
    def update_display(self):
        """Update all display panels."""
        self.layout["header"].update(self.update_header())
        self.layout["chat"].update(self.update_chat())
        self.layout["metrics"].update(self.update_metrics())
        self.layout["tasks"].update(self.update_tasks())
        self.layout["environment"].update(self.update_environment())
        self.layout["insights"].update(self.update_insights())
        self.layout["footer"].update(self.update_footer())
    
    def add_message(self, role: str, content: str):
        """Add a message to chat history."""
        self.chat_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    
    async def process_input(self, user_input: str):
        """Process user input."""
        # Handle commands
        if user_input.startswith("/"):
            await self.handle_command(user_input)
        else:
            # Regular message
            self.add_message("user", user_input)
            
            # Process with agent
            self.add_message("system", "🤔 Processing...")
            
            try:
                # Get response
                response = await self.agent.send_user(user_input, stream=False)
                
                # Remove processing message
                self.chat_history.pop()
                
                # Add response
                self.add_message("assistant", response)
                
            except Exception as e:
                self.chat_history.pop()  # Remove processing
                self.add_message("system", f"❌ Error: {str(e)}")
    
    async def handle_command(self, command: str):
        """Handle special commands."""
        cmd = command.lower().strip()
        
        if cmd == "/help":
            help_text = """
Available commands:
• /help     - Show this help
• /status   - Show full agent status
• /clear    - Clear chat display
• /export   - Export chat to file
• /tasks    - Show all tasks
• /insights - Show detailed insights
• /quit     - Exit the TUI

Agent tools:
• create_task - Create a new task
• get_task_status - Check task status
• get_insights - View reflections
• get_environment - System status
• search_history - Search chat
• analyze_performance - Run analysis
• create_workflow - Task workflow
"""
            self.add_message("system", help_text.strip())
            
        elif cmd == "/status":
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = buffer = StringIO()
            
            from multi_turn_agent_enhanced import visualize_agent_status
            visualize_agent_status(self.agent)
            
            sys.stdout = old_stdout
            status = buffer.getvalue()
            self.add_message("system", f"```\n{status}\n```")
            
        elif cmd == "/clear":
            self.chat_history = []
            self.add_message("system", "Chat display cleared (history preserved)")
            
        elif cmd == "/export":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_export_{self.agent.session_id[:8]}_{timestamp}.md"
            
            with open(filename, "w") as f:
                f.write(f"# Chat Export - Session {self.agent.session_id}\n\n")
                for msg in self.chat_history:
                    f.write(f"## {msg['role'].title()} [{msg['timestamp']}]\n")
                    f.write(f"{msg['content']}\n\n")
            
            self.add_message("system", f"✅ Exported to {filename}")
            
        elif cmd == "/tasks":
            queue_status = self.agent.task_queue.get_status()
            text = f"Task Queue Status:\n"
            text += f"Total: {queue_status['total_tasks']}\n"
            text += f"By status: {queue_status['by_status']}\n"
            self.add_message("system", text)
            
        elif cmd == "/insights":
            insights = self.agent.reflection_system.get_insights()
            text = f"Reflection Insights:\n"
            text += f"Total: {insights['total_reflections']}\n"
            text += f"Confidence: {insights['avg_confidence']:.2f}\n"
            text += f"Avg task duration: {insights['performance_trends']['avg_task_duration']:.2f}s\n"
            if insights['common_errors']:
                text += f"Common errors: {list(insights['common_errors'].items())[:3]}\n"
            self.add_message("system", text)
            
        elif cmd == "/quit":
            self.running = False
            
        else:
            self.add_message("system", f"Unknown command: {command}")
    
    async def run(self):
        """Run the TUI."""
        # Welcome message
        self.add_message("system", "🚀 Enhanced Agent TUI Started!")
        self.add_message("system", "Type /help for commands or just start chatting")
        
        # Create input thread
        input_thread = threading.Thread(target=self.input_loop, daemon=True)
        input_thread.start()
        
        # Main display loop
        with Live(self.layout, console=self.console, refresh_per_second=2) as live:
            while self.running:
                # Update display
                self.update_display()
                
                # Check for input
                try:
                    if not self.input_queue.empty():
                        user_input = self.input_queue.get_nowait()
                        await self.process_input(user_input)
                except:
                    pass
                
                # Small delay
                await asyncio.sleep(0.1)
        
        # Cleanup
        await self.agent.worker_pool.stop()
        self.console.print("\n[bold green]👋 Agent TUI shutdown complete[/bold green]")
    
    def input_loop(self):
        """Input loop in separate thread."""
        history = FileHistory('.agent_tui_history')
        
        while self.running:
            try:
                # Get input with prompt toolkit
                user_input = prompt(
                    HTML('<ansigreen><b>You:</b></ansigreen> '),
                    history=history,
                    auto_suggest=AutoSuggestFromHistory(),
                    key_bindings=self.kb,
                    style=self.prompt_style,
                    multiline=False,
                )
                
                if user_input.strip():
                    self.input_queue.put(user_input.strip())
                    
            except (EOFError, KeyboardInterrupt):
                self.input_queue.put("/quit")
                break

# ————————————————————————————————————————————————————————————————
# Main Entry Point
# ————————————————————————————————————————————————————————————————

async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Rich TUI for Enhanced Agent")
    parser.add_argument("--session", help="Resume session ID")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
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
    tui = RichAgentTUI(agent)
    
    # Handle signals
    def signal_handler(sig, frame):
        tui.running = False
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run
    await tui.run()

if __name__ == "__main__":
    # Install required packages if needed
    try:
        import rich
        import prompt_toolkit
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "prompt_toolkit"])
        print("Packages installed! Please run again.")
        sys.exit(0)
    
    # Run
    asyncio.run(main())