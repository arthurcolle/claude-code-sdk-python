#!/usr/bin/env python3
"""
NEURAL INTERFACE v2.0 - Cyberpunk Agent TUI
==========================================
A futuristic, visually stunning terminal interface with animations,
effects, and an immersive cyberpunk aesthetic.
"""

import asyncio
import random
import time
import sys
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from collections import deque
import math
import json

from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich import box
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn, TimeElapsedColumn
from rich.columns import Columns
from rich.rule import Rule
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

# Import enhanced agent
from multi_turn_agent_enhanced import (
    EnhancedStatefulAgent, TaskStatus, TaskPriority,
    register_enhanced_demo_tools
)

# ————————————————————————————————————————————————————————————————
# Cyberpunk Theme & Effects
# ————————————————————————————————————————————————————————————————

NEON_COLORS = ["cyan", "magenta", "green", "yellow", "blue", "red"]
MATRIX_CHARS = "ﾊﾐﾋｰｳｼﾅﾓﾆｻﾜﾂｵﾘｱﾎﾃﾏｹﾒｴｶｷﾑﾕﾗｾﾈｽﾀﾇﾍ012345789Z"

class CyberpunkEffects:
    """Visual effects for cyberpunk theme."""
    
    @staticmethod
    def glitch_text(text: str, intensity: float = 0.1) -> str:
        """Apply glitch effect to text."""
        if random.random() > intensity:
            return text
        
        glitch_chars = "▓▒░█▄▌▐▀"
        result = list(text)
        for _ in range(int(len(text) * intensity)):
            pos = random.randint(0, len(result) - 1)
            result[pos] = random.choice(glitch_chars)
        return ''.join(result)
    
    @staticmethod
    def neon_text(text: str, color: Optional[str] = None) -> Text:
        """Create neon glowing text effect."""
        if not color:
            color = random.choice(NEON_COLORS)
        
        neon = Text()
        # Add glow layers
        neon.append("░", style=f"dim {color}")
        neon.append(text, style=f"bold {color}")
        neon.append("░", style=f"dim {color}")
        return neon
    
    @staticmethod
    def matrix_rain(width: int, height: int) -> List[str]:
        """Generate matrix rain effect."""
        lines = []
        for _ in range(height):
            line = ""
            for _ in range(width):
                if random.random() < 0.1:
                    char = random.choice(MATRIX_CHARS)
                    brightness = random.choice(["dim", "normal", "bold"])
                    line += f"[{brightness} green]{char}[/]"
                else:
                    line += " "
            lines.append(line)
        return lines
    
    @staticmethod
    def cyber_border(title: str) -> str:
        """Create cyberpunk style border."""
        border = "◢▓▓▓◣" + "▓" * (len(title) + 2) + "◢▓▓▓◣"
        return border
    
    @staticmethod
    def pulse_animation(frame: int) -> str:
        """Create pulsing animation character."""
        pulse_chars = ["◐", "◓", "◑", "◒"]
        return pulse_chars[frame % len(pulse_chars)]

class SystemStatus(Enum):
    """Neural system status levels."""
    OPTIMAL = ("OPTIMAL", "green", "▰▰▰▰▰")
    NOMINAL = ("NOMINAL", "cyan", "▰▰▰▰▱")
    DEGRADED = ("DEGRADED", "yellow", "▰▰▰▱▱")
    WARNING = ("WARNING", "magenta", "▰▰▱▱▱")
    CRITICAL = ("CRITICAL", "red", "▰▱▱▱▱")

# ————————————————————————————————————————————————————————————————
# Neural Queue System
# ————————————————————————————————————————————————————————————————

class NeuralQueue:
    """Cyberpunk-themed message queue with visual effects."""
    
    def __init__(self, max_size: int = 7):
        self.queue: deque = deque(maxlen=max_size)
        self.priority_queue: deque = deque(maxlen=3)
        self.dropped_packets = 0
        self.processed_packets = 0
        self.bandwidth_used = 0
        self.encryption_level = "AES-256"
        
    async def add(self, message: str, priority: bool = False) -> Tuple[bool, str]:
        """Add message to neural queue."""
        packet_size = len(message.encode('utf-8'))
        self.bandwidth_used += packet_size
        
        if priority and len(self.priority_queue) < self.priority_queue.maxlen:
            self.priority_queue.append({
                "data": message,
                "timestamp": time.time(),
                "size": packet_size,
                "encrypted": True,
                "priority": "HIGH"
            })
            return True, "PRIORITY_QUEUED"
        
        if len(self.queue) >= self.queue.maxlen:
            self.dropped_packets += 1
            return False, "BUFFER_OVERFLOW"
            
        self.queue.append({
            "data": message,
            "timestamp": time.time(),
            "size": packet_size,
            "encrypted": True,
            "priority": "NORMAL"
        })
        return True, "QUEUED"
    
    async def get_next(self) -> Optional[Dict[str, Any]]:
        """Get next packet from queue."""
        # Priority queue first
        if self.priority_queue:
            packet = self.priority_queue.popleft()
            self.processed_packets += 1
            return packet
            
        if self.queue:
            packet = self.queue.popleft()
            self.processed_packets += 1
            return packet
            
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get queue metrics."""
        total_size = len(self.queue) + len(self.priority_queue)
        capacity = self.queue.maxlen + self.priority_queue.maxlen
        
        return {
            "utilization": total_size / capacity,
            "normal_queue": len(self.queue),
            "priority_queue": len(self.priority_queue),
            "dropped": self.dropped_packets,
            "processed": self.processed_packets,
            "bandwidth_mb": self.bandwidth_used / 1024 / 1024,
            "encryption": self.encryption_level
        }

# ————————————————————————————————————————————————————————————————
# Neural Interface Components
# ————————————————————————————————————————————————————————————————

class NeuralInterface:
    """The main cyberpunk neural interface."""
    
    def __init__(self, agent: EnhancedStatefulAgent):
        self.agent = agent
        self.console = Console()
        self.layout = self._create_layout()
        
        # Neural state
        self.neural_queue = NeuralQueue()
        self.system_status = SystemStatus.OPTIMAL
        self.connection_strength = 1.0
        self.neural_sync = 0.95
        self.frame_counter = 0
        
        # Interface state
        self.chat_log: List[Dict[str, Any]] = []
        self.alerts: deque = deque(maxlen=5)
        self.running = True
        self.processing = False
        self.hack_mode = False
        
        # Visual effects
        self.matrix_positions = {}
        self.glitch_intensity = 0.0
        self.last_activity = time.time()
        
        # Stats
        self.stats = {
            "packets_sent": 0,
            "packets_received": 0,
            "neural_cycles": 0,
            "glitches": 0,
            "uptime": time.time()
        }
        
        # Register neural tools
        self._register_neural_tools()
        
        # ASCII art logos
        self.neural_logo = """
╔═╗┬ ┬┌┐ ┌─┐┬─┐  ╔╗╔┌─┐┬ ┬┬─┐┌─┐┬  
║  └┬┘├┴┐├┤ ├┬┘  ║║║├┤ │ │├┬┘├─┤│  
╚═╝ ┴ └─┘└─┘┴└─  ╝╚╝└─┘└─┘┴└─┴ ┴┴─┘"""
        
    def _create_layout(self) -> Layout:
        """Create cyberpunk layout."""
        layout = Layout()
        
        # Main structure
        layout.split_column(
            Layout(name="header", size=6),
            Layout(name="main"),
            Layout(name="footer", size=4)
        )
        
        # Main area
        layout["main"].split_row(
            Layout(name="left_panel", ratio=1),
            Layout(name="neural_core", ratio=2),
            Layout(name="right_panel", ratio=1)
        )
        
        # Left panel
        layout["left_panel"].split_column(
            Layout(name="system_status", size=12),
            Layout(name="neural_metrics", size=10),
            Layout(name="queue_viz")
        )
        
        # Right panel
        layout["right_panel"].split_column(
            Layout(name="matrix_rain", size=15),
            Layout(name="data_stream"),
            Layout(name="alerts", size=8)
        )
        
        return layout
    
    def _register_neural_tools(self):
        """Register neural interface tools."""
        
        @self.agent.tools_registry.register(description="Send neural transmission")
        async def neural_transmit(
            data: str,
            encryption: str = "AES-256",
            priority: bool = False
        ) -> str:
            """Transmit data through neural link."""
            self.add_neural_message(data, encryption, priority)
            self.stats["packets_sent"] += 1
            return f"TRANSMISSION_CONFIRMED: {len(data)} bytes"
        
        @self.agent.tools_registry.register(description="Activate hack mode")
        async def hack_mode(activate: bool = True) -> str:
            """Toggle hack mode for enhanced access."""
            self.hack_mode = activate
            self.glitch_intensity = 0.3 if activate else 0.0
            return f"HACK_MODE: {'ACTIVATED' if activate else 'DEACTIVATED'}"
        
        @self.agent.tools_registry.register(description="Neural scan")
        async def neural_scan(target: str = "environment") -> Dict[str, Any]:
            """Perform neural scan of target."""
            scan_data = {
                "target": target,
                "timestamp": datetime.now().isoformat(),
                "neural_patterns": random.randint(1000, 9999),
                "anomalies": random.randint(0, 10),
                "encryption_layers": random.randint(1, 5)
            }
            
            # Add visual effect
            self.alerts.append({
                "type": "SCAN",
                "message": f"Neural scan initiated: {target}",
                "timestamp": time.time()
            })
            
            return scan_data
        
        @self.agent.tools_registry.register(description="System diagnostic")
        async def run_diagnostic() -> str:
            """Run full system diagnostic."""
            self.system_status = random.choice(list(SystemStatus))
            diagnostic = f"""
DIAGNOSTIC_REPORT:
- Neural Sync: {self.neural_sync:.2%}
- Connection: {self.connection_strength:.2%}
- Queue Status: {self.neural_queue.get_status()['utilization']:.1%}
- System: {self.system_status.value[0]}
"""
            return diagnostic
    
    def create_header(self) -> Panel:
        """Create cyberpunk header."""
        # Animated neural logo with glitch
        logo_lines = []
        for line in self.neural_logo.split('\n'):
            if self.glitch_intensity > 0:
                line = CyberpunkEffects.glitch_text(line, self.glitch_intensity)
            logo_lines.append(Text(line, style="bold cyan"))
        
        # Connection status
        conn_bar = self._create_connection_bar()
        
        # Neural sync indicator
        sync_indicator = self._create_sync_indicator()
        
        header_content = Group(
            Align.center(Group(*logo_lines)),
            Rule(style="cyan"),
            Columns([conn_bar, sync_indicator], expand=True, equal=True)
        )
        
        return Panel(
            header_content,
            box=box.DOUBLE,
            border_style="cyan",
            style="on black"
        )
    
    def _create_connection_bar(self) -> Text:
        """Create connection strength bar."""
        bar_width = 20
        filled = int(self.connection_strength * bar_width)
        
        bar = Text("CONNECTION: [", style="dim cyan")
        bar.append("█" * filled, style="bold green")
        bar.append("░" * (bar_width - filled), style="dim red")
        bar.append(f"] {self.connection_strength:.0%}", style="cyan")
        
        return bar
    
    def _create_sync_indicator(self) -> Text:
        """Create neural sync indicator."""
        sync = Text("NEURAL SYNC: ", style="dim cyan")
        
        # Animated sync level
        pulse = CyberpunkEffects.pulse_animation(self.frame_counter)
        
        if self.neural_sync > 0.9:
            sync.append(f"{pulse} {self.neural_sync:.1%}", style="bold green")
        elif self.neural_sync > 0.7:
            sync.append(f"{pulse} {self.neural_sync:.1%}", style="yellow")
        else:
            sync.append(f"{pulse} {self.neural_sync:.1%}", style="red blink")
            
        return sync
    
    def create_system_status(self) -> Panel:
        """Create system status panel."""
        status = self.system_status.value
        
        # Status indicator with color
        status_text = Text("SYSTEM STATUS\n", style="bold")
        status_text.append("━" * 15 + "\n", style="dim")
        status_text.append(f"{status[2]}\n", style=status[1])
        status_text.append(status[0], style=f"bold {status[1]}")
        
        # Additional metrics
        metrics = Text("\n\nMETRICS:\n", style="bold")
        metrics.append(f"CPU: {random.randint(20, 80)}%\n", style="cyan")
        metrics.append(f"RAM: {random.randint(40, 90)}%\n", style="magenta")
        metrics.append(f"GPU: {random.randint(30, 70)}%\n", style="green")
        
        return Panel(
            Group(status_text, metrics),
            title="[SYSTEM]",
            box=box.HEAVY,
            border_style=status[1]
        )
    
    def create_neural_metrics(self) -> Panel:
        """Create neural metrics panel."""
        table = Table(box=None, show_header=False, padding=0)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        
        # Calculate uptime
        uptime = time.time() - self.stats["uptime"]
        uptime_str = f"{int(uptime // 60)}:{int(uptime % 60):02d}"
        
        table.add_row("Uptime", uptime_str)
        table.add_row("Packets TX", str(self.stats["packets_sent"]))
        table.add_row("Packets RX", str(self.stats["packets_received"]))
        table.add_row("Neural Cycles", str(self.stats["neural_cycles"]))
        table.add_row("Glitches", str(self.stats["glitches"]))
        
        return Panel(
            table,
            title="[NEURAL METRICS]",
            box=box.HEAVY,
            border_style="magenta"
        )
    
    def create_queue_visualization(self) -> Panel:
        """Create visual queue representation."""
        queue_status = self.neural_queue.get_status()
        
        content = []
        
        # Queue bars
        normal_bar = self._create_queue_bar(
            "NORMAL", 
            queue_status["normal_queue"], 
            self.neural_queue.queue.maxlen,
            "cyan"
        )
        priority_bar = self._create_queue_bar(
            "PRIORITY", 
            queue_status["priority_queue"], 
            self.neural_queue.priority_queue.maxlen,
            "magenta"
        )
        
        content.append(normal_bar)
        content.append(priority_bar)
        
        # Stats
        content.append(Text("\n━━━━━━━━━━━━━━━", style="dim"))
        content.append(Text(f"Dropped: {queue_status['dropped']}", style="red"))
        content.append(Text(f"Processed: {queue_status['processed']}", style="green"))
        content.append(Text(f"Bandwidth: {queue_status['bandwidth_mb']:.2f}MB", style="yellow"))
        
        return Panel(
            Group(*content),
            title="[NEURAL QUEUE]",
            box=box.HEAVY,
            border_style="cyan"
        )
    
    def _create_queue_bar(self, label: str, current: int, max_val: int, color: str) -> Text:
        """Create a visual queue bar."""
        bar_width = 12
        filled = int((current / max_val) * bar_width) if max_val > 0 else 0
        
        bar = Text(f"{label:8} [", style=f"dim {color}")
        
        # Animated fill based on level
        if filled > bar_width * 0.8:
            bar.append("█" * filled, style=f"bold {color} blink")
        else:
            bar.append("█" * filled, style=f"bold {color}")
            
        bar.append("░" * (bar_width - filled), style="dim white")
        bar.append(f"] {current}/{max_val}", style=color)
        
        return bar
    
    def create_neural_core(self) -> Panel:
        """Create main neural core display."""
        messages = []
        
        # Display chat messages with effects
        for msg in self.chat_log[-20:]:
            role = msg["role"]
            content = msg["content"]
            timestamp = msg.get("timestamp", "")
            
            if role == "user":
                header = Text(f"\n[{timestamp}] ", style="dim")
                header.append("USER://", style="bold cyan")
                messages.append(header)
                
                # Apply encryption effect
                if msg.get("encrypted"):
                    messages.append(Text(f"  [ENCRYPTED] {content}", style="cyan"))
                else:
                    messages.append(Text(f"  {content}", style="cyan"))
                    
            elif role == "neural":
                header = Text(f"\n[{timestamp}] ", style="dim")
                header.append("NEURAL://", style="bold magenta")
                messages.append(header)
                
                # Add neural transmission effect
                if self.processing:
                    content = CyberpunkEffects.glitch_text(content, 0.1)
                    
                messages.append(Text(f"  {content}", style="magenta"))
                
            elif role == "system":
                messages.append(Text(f"\n[SYSTEM] {content}", style="yellow italic"))
        
        if not messages:
            welcome = Text("NEURAL INTERFACE INITIALIZED\n", style="bold green")
            welcome.append("Establishing neural link...\n", style="dim green")
            welcome.append("Type to begin transmission.", style="dim")
            messages = [welcome]
        
        # Add scan lines effect
        if self.hack_mode:
            scan_line = Text("=" * 50, style="dim green")
            messages.insert(0, scan_line)
            messages.append(scan_line)
        
        return Panel(
            Group(*messages),
            title="[NEURAL CORE]",
            box=box.DOUBLE,
            border_style="magenta" if self.processing else "cyan",
            padding=(1, 2)
        )
    
    def create_matrix_rain(self) -> Panel:
        """Create matrix rain effect."""
        width = 30
        height = 13
        
        # Update matrix positions
        if self.frame_counter % 2 == 0:
            for x in range(width):
                if x not in self.matrix_positions:
                    if random.random() < 0.1:
                        self.matrix_positions[x] = 0
                else:
                    self.matrix_positions[x] += 1
                    if self.matrix_positions[x] > height:
                        del self.matrix_positions[x]
        
        # Generate matrix
        lines = []
        for y in range(height):
            line = ""
            for x in range(width):
                if x in self.matrix_positions:
                    pos = self.matrix_positions[x]
                    if pos == y:
                        char = random.choice(MATRIX_CHARS)
                        line += f"[bold green]{char}[/]"
                    elif pos - 1 == y:
                        char = random.choice(MATRIX_CHARS)
                        line += f"[green]{char}[/]"
                    elif pos - 2 == y:
                        char = random.choice(MATRIX_CHARS)
                        line += f"[dim green]{char}[/]"
                    else:
                        line += " "
                else:
                    line += " "
            lines.append(Text.from_markup(line))
        
        return Panel(
            Group(*lines),
            title="[MATRIX FEED]",
            box=box.HEAVY,
            border_style="green",
            style="on black"
        )
    
    def create_data_stream(self) -> Panel:
        """Create data stream visualization."""
        stream_lines = []
        
        # Recent agent activity
        if self.agent.worker_pool.active_tasks:
            stream_lines.append(Text("ACTIVE PROCESSES:", style="bold green"))
            for worker, task_id in list(self.agent.worker_pool.active_tasks.items())[:3]:
                stream_lines.append(Text(f"  ├─ {worker}: TASK_{task_id[:8]}", style="green"))
        
        # Queue activity
        if self.neural_queue.queue:
            stream_lines.append(Text("\nQUEUED TRANSMISSIONS:", style="bold cyan"))
            for i, packet in enumerate(list(self.neural_queue.queue)[:3]):
                preview = packet["data"][:20] + "..."
                stream_lines.append(Text(f"  ├─ PKT_{i}: {preview}", style="cyan"))
        
        # Random data streams
        if random.random() < 0.3:
            stream_lines.append(Text(f"\n[DATA] {random.randbytes(16).hex()}", style="dim"))
        
        return Panel(
            Group(*stream_lines) if stream_lines else Text("NO ACTIVE STREAMS", style="dim"),
            title="[DATA STREAM]",
            box=box.HEAVY,
            border_style="cyan"
        )
    
    def create_alerts(self) -> Panel:
        """Create alerts panel."""
        alert_lines = []
        
        current_time = time.time()
        for alert in reversed(list(self.alerts)):
            age = current_time - alert["timestamp"]
            if age < 10:  # Show alerts for 10 seconds
                opacity = 1.0 - (age / 10)
                style = f"{'bold' if opacity > 0.5 else 'dim'} yellow"
                alert_lines.append(Text(f"⚠ {alert['message']}", style=style))
        
        if not alert_lines:
            alert_lines = [Text("NO ALERTS", style="dim green")]
        
        return Panel(
            Group(*alert_lines),
            title="[ALERTS]",
            box=box.HEAVY,
            border_style="yellow" if alert_lines[0].plain != "NO ALERTS" else "green"
        )
    
    def create_footer(self) -> Panel:
        """Create footer with input indicator."""
        # Input mode indicator
        if self.processing:
            mode = Text("PROCESSING", style="yellow blink")
            hint = "Neural transmission in progress..."
        elif self.hack_mode:
            mode = Text("HACK MODE", style="red bold")
            hint = "Enhanced access granted. Use with caution."
        else:
            mode = Text("READY", style="green")
            hint = "Type message or /help for commands"
        
        # Create footer content
        footer = Columns([
            Text(f"MODE: ", style="dim") + mode,
            Text(f"QUEUE: {self.neural_queue.get_status()['utilization']:.0%}", style="cyan"),
            Text(f"SYNC: {self.neural_sync:.0%}", style="magenta"),
            Text(hint, style="dim", justify="right")
        ], expand=True)
        
        return Panel(
            footer,
            box=box.HEAVY,
            border_style="cyan",
            height=4
        )
    
    def add_neural_message(self, content: str, encryption: str = "AES-256", priority: bool = False):
        """Add message to neural core."""
        self.chat_log.append({
            "role": "neural",
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "encrypted": encryption is not None,
            "priority": priority
        })
        self.stats["packets_received"] += 1
    
    def add_user_message(self, content: str):
        """Add user message."""
        self.chat_log.append({
            "role": "user",
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "encrypted": True
        })
        self.stats["packets_sent"] += 1
    
    def add_system_message(self, content: str):
        """Add system message."""
        self.chat_log.append({
            "role": "system",
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    
    def update_display(self):
        """Update all panels."""
        self.frame_counter += 1
        self.stats["neural_cycles"] += 1
        
        # Update connection based on activity
        time_since_activity = time.time() - self.last_activity
        self.connection_strength = max(0.5, 1.0 - (time_since_activity / 60))
        
        # Random neural sync fluctuation
        self.neural_sync += random.uniform(-0.02, 0.02)
        self.neural_sync = max(0.5, min(1.0, self.neural_sync))
        
        # Update layout
        self.layout["header"].update(self.create_header())
        self.layout["system_status"].update(self.create_system_status())
        self.layout["neural_metrics"].update(self.create_neural_metrics())
        self.layout["queue_viz"].update(self.create_queue_visualization())
        self.layout["neural_core"].update(self.create_neural_core())
        self.layout["matrix_rain"].update(self.create_matrix_rain())
        self.layout["data_stream"].update(self.create_data_stream())
        self.layout["alerts"].update(self.create_alerts())
        self.layout["footer"].update(self.create_footer())
    
    async def process_transmission(self, message: str):
        """Process neural transmission."""
        self.processing = True
        self.last_activity = time.time()
        
        # Add to neural log
        self.add_user_message(message)
        
        # Simulate neural processing
        self.alerts.append({
            "type": "TRANSMISSION",
            "message": "Neural transmission initiated",
            "timestamp": time.time()
        })
        
        try:
            # Send to agent
            response = await self.agent.send_user(message, stream=False)
            
            # Agent will use neural_transmit to respond
            
        except Exception as e:
            self.stats["glitches"] += 1
            self.glitch_intensity = 0.5
            self.add_system_message(f"TRANSMISSION ERROR: {str(e)}")
            
            # Add error alert
            self.alerts.append({
                "type": "ERROR",
                "message": f"Neural glitch detected: {type(e).__name__}",
                "timestamp": time.time()
            })
        
        finally:
            self.processing = False
            self.glitch_intensity = 0.0
    
    async def handle_command(self, command: str):
        """Handle neural commands."""
        cmd = command.lower().strip()
        
        if cmd == "/help":
            help_text = """
NEURAL INTERFACE COMMANDS:
/help      - Show commands
/scan      - Perform neural scan
/hack      - Toggle hack mode
/status    - System diagnostic
/clear     - Clear neural log
/matrix    - Toggle matrix view
/quit      - Disconnect

TRANSMISSION: Type message and press ENTER
"""
            self.add_system_message(help_text.strip())
            
        elif cmd == "/scan":
            self.add_system_message("Initiating neural scan...")
            await self.agent.tools_registry.call("neural_scan", target="full_spectrum")
            
        elif cmd == "/hack":
            await self.agent.tools_registry.call("hack_mode", activate=not self.hack_mode)
            
        elif cmd == "/status":
            result = await self.agent.tools_registry.call("run_diagnostic")
            self.add_system_message(result)
            
        elif cmd == "/clear":
            self.chat_log = []
            self.add_system_message("Neural log cleared")
            
        elif cmd == "/quit":
            self.running = False
            
        else:
            self.add_system_message(f"Unknown command: {command}")
    
    async def run(self):
        """Run the neural interface."""
        # Initialize
        self.add_system_message("NEURAL INTERFACE v2.0 INITIALIZED")
        await asyncio.sleep(0.5)
        self.add_neural_message("Neural link established. Synchronization at 95%.")
        
        # Create prompt session
        prompt_session = PromptSession(
            history=FileHistory('.neural_history'),
            auto_suggest=AutoSuggestFromHistory()
        )
        
        # Main loop
        with Live(
            self.layout, 
            console=self.console, 
            refresh_per_second=10,  # Higher refresh for animations
            transient=False
        ) as live:
            while self.running:
                self.update_display()
                
                # Check for queued messages
                if not self.processing and not self.neural_queue.queue.empty():
                    packet = await self.neural_queue.get_next()
                    if packet:
                        await self.process_transmission(packet["data"])
                        continue
                
                # Get input
                try:
                    with patch_stdout():
                        user_input = await prompt_session.prompt_async(
                            "NEURAL> ",
                            enable_history_search=True
                        )
                    
                    if user_input.strip():
                        if user_input.startswith("/"):
                            await self.handle_command(user_input)
                        else:
                            # Check if we should queue
                            if self.processing:
                                success, status = await self.neural_queue.add(user_input)
                                if success:
                                    self.add_system_message(f"Transmission queued: {status}")
                                else:
                                    self.add_system_message(f"Queue overflow: {status}")
                            else:
                                await self.process_transmission(user_input)
                                
                except (EOFError, KeyboardInterrupt):
                    self.running = False
                    break
                except Exception as e:
                    self.stats["glitches"] += 1
                    self.add_system_message(f"INPUT ERROR: {str(e)}")
        
        # Shutdown sequence
        self.add_system_message("Terminating neural link...")
        await self.agent.worker_pool.stop()
        self.console.print("\n[bold green]NEURAL INTERFACE DISCONNECTED[/bold green]")

# ————————————————————————————————————————————————————————————————
# Main Entry Point
# ————————————————————————————————————————————————————————————————

async def main():
    """Initialize and run the neural interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cyberpunk Neural Interface")
    parser.add_argument("--session", help="Resume neural session")
    parser.add_argument("--workers", type=int, default=4, help="Neural processors")
    parser.add_argument("--hack", action="store_true", help="Start in hack mode")
    args = parser.parse_args()
    
    # ASCII art intro
    intro = """
    ░▒▓███████▓▒░ ░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓█▓▒░      
    ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      
    ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      
    ░▒▓█▓▒░░▒▓█▓▒░▒▓██████▓▒░ ░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░░▒▓████████▓▒░▒▓█▓▒░      
    ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      
    ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      
    ░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░
                                                                                     
                        I N T E R F A C E   v 2 . 0                                 
    """
    
    print(intro)
    await asyncio.sleep(1)
    
    # Create agent
    print("[INITIALIZING] Neural processors...")
    agent = EnhancedStatefulAgent(
        session_id=args.session,
        stream=False,
        num_workers=args.workers
    )
    
    # Register tools
    register_enhanced_demo_tools(agent)
    
    # Create interface
    print("[ESTABLISHING] Neural link...")
    interface = NeuralInterface(agent)
    
    if args.hack:
        interface.hack_mode = True
        interface.glitch_intensity = 0.3
    
    # Run
    print("[CONNECTED] Entering neural interface...\n")
    await asyncio.sleep(1)
    
    await interface.run()

if __name__ == "__main__":
    # Check dependencies
    try:
        import rich
        import prompt_toolkit
    except ImportError:
        print("[ERROR] Missing dependencies. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "prompt_toolkit"])
        sys.exit(0)
    
    asyncio.run(main())