#!/usr/bin/env python3
"""
🌌 COSMIC NEURAL INTERFACE - Space Station AI Terminal
=====================================================
A beautiful space-themed TUI with orbital mechanics, star maps,
and cosmic visualizations.
"""

import asyncio
import random
import math
import time
import sys
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
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
from rich.columns import Columns
from rich.progress import Progress, BarColumn, TextColumn
from rich import box
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
# Cosmic Theme & Visuals
# ————————————————————————————————————————————————————————————————

STAR_CHARS = ["✦", "✧", "⋆", ".", "·", "•", "◦", "○", "◯"]
PLANET_CHARS = ["🪐", "🌍", "🌎", "🌏", "🔴", "🟠", "🟡", "🔵"]
SHIP_FRAMES = ["◁═", "◁▪", "◁◦", "◁○", "◁◯"]

class CosmicElements:
    """Space-themed visual elements."""
    
    @staticmethod
    def generate_starfield(width: int, height: int, density: float = 0.1) -> List[Text]:
        """Generate a starfield background."""
        lines = []
        for y in range(height):
            line = Text()
            for x in range(width):
                if random.random() < density:
                    star = random.choice(STAR_CHARS)
                    brightness = random.choice(["dim white", "white", "bright_white"])
                    line.append(star, style=brightness)
                else:
                    line.append(" ")
            lines.append(line)
        return lines
    
    @staticmethod
    def create_orbit(center_x: int, center_y: int, radius: int, angle: float) -> Tuple[int, int]:
        """Calculate orbital position."""
        x = center_x + int(radius * math.cos(angle))
        y = center_y + int(radius * math.sin(angle) * 0.5)  # Elliptical
        return x, y
    
    @staticmethod
    def solar_wind(width: int, intensity: float = 0.3) -> str:
        """Create solar wind effect."""
        wind = ""
        for _ in range(width):
            if random.random() < intensity:
                wind += random.choice(["╱", "╲", "│", "─"])
            else:
                wind += " "
        return wind
    
    @staticmethod
    def constellation_map() -> List[str]:
        """Create a mini constellation map."""
        return [
            "    ✦           ✧    ",
            "  ·   \\       /   ·  ",
            "✦------✦-----✦------✦",
            "  ·     \\   /     ·  ",
            "    ·    \\ /    ·    ",
            "      ·   ✦   ·      ",
            "        · | ·        ",
            "          ✦          "
        ]

class MissionStatus(Enum):
    """Space mission status levels."""
    OPTIMAL = ("OPTIMAL", "green", "🟢")
    STABLE = ("STABLE", "cyan", "🔵")
    CAUTION = ("CAUTION", "yellow", "🟡")
    WARNING = ("WARNING", "orange", "🟠")
    CRITICAL = ("CRITICAL", "red", "🔴")

# ————————————————————————————————————————————————————————————————
# Orbital Communication System
# ————————————————————————————————————————————————————————————————

class OrbitalComms:
    """Space station communication system."""
    
    def __init__(self, max_buffer: int = 10):
        self.transmission_queue = deque(maxlen=max_buffer)
        self.priority_channel = deque(maxlen=3)
        self.signal_strength = 1.0
        self.orbital_delay = 0.0
        self.packets_lost = 0
        self.total_transmissions = 0
        
    async def transmit(self, message: str, priority: bool = False) -> Tuple[bool, str]:
        """Transmit message through orbital relay."""
        self.total_transmissions += 1
        
        # Simulate signal degradation based on orbital position
        if self.signal_strength < 0.3:
            self.packets_lost += 1
            return False, "SIGNAL_LOST"
        
        packet = {
            "data": message,
            "timestamp": time.time(),
            "signal_strength": self.signal_strength,
            "priority": priority
        }
        
        if priority and len(self.priority_channel) < self.priority_channel.maxlen:
            self.priority_channel.append(packet)
            return True, "PRIORITY_RELAY"
        elif len(self.transmission_queue) < self.transmission_queue.maxlen:
            self.transmission_queue.append(packet)
            return True, "QUEUED_FOR_TRANSMISSION"
        else:
            self.packets_lost += 1
            return False, "BUFFER_FULL"
    
    async def receive(self) -> Optional[Dict[str, Any]]:
        """Receive next transmission."""
        if self.priority_channel:
            return self.priority_channel.popleft()
        elif self.transmission_queue:
            return self.transmission_queue.popleft()
        return None
    
    def update_orbital_position(self, angle: float):
        """Update signal based on orbital position."""
        # Signal varies with orbital position (simulate satellite relay)
        self.signal_strength = 0.5 + 0.5 * abs(math.cos(angle))
        self.orbital_delay = 0.1 + 0.2 * (1 - self.signal_strength)

# ————————————————————————————————————————————————————————————————
# Cosmic Interface
# ————————————————————————————————————————————————————————————————

class CosmicInterface:
    """Space station AI terminal interface."""
    
    def __init__(self, agent: EnhancedStatefulAgent):
        self.agent = agent
        self.console = Console()
        self.layout = self._create_layout()
        
        # Space station state
        self.station_name = "COSMOS-7"
        self.orbital_angle = 0.0
        self.mission_status = MissionStatus.OPTIMAL
        self.oxygen_level = 0.98
        self.power_level = 0.95
        self.hull_integrity = 1.0
        
        # Communication system
        self.comms = OrbitalComms()
        self.transmission_log: List[Dict[str, Any]] = []
        
        # Visual state
        self.frame_count = 0
        self.ship_position = 0
        self.asteroid_positions = []
        self.star_positions = self._generate_star_positions()
        
        # Mission data
        self.mission_elapsed = 0
        self.discoveries = []
        self.alerts = deque(maxlen=5)
        self.running = True
        self.processing = False
        
        # Stats
        self.stats = {
            "transmissions_sent": 0,
            "transmissions_received": 0,
            "discoveries_made": 0,
            "anomalies_detected": 0,
            "mission_start": time.time()
        }
        
        # Register space tools
        self._register_space_tools()
        
    def _create_layout(self) -> Layout:
        """Create space station layout."""
        layout = Layout()
        
        # Main structure
        layout.split_column(
            Layout(name="header", size=7),
            Layout(name="main"),
            Layout(name="systems", size=5),
            Layout(name="input", size=3)
        )
        
        # Main area
        layout["main"].split_row(
            Layout(name="left_panel", ratio=1),
            Layout(name="viewport", ratio=2),
            Layout(name="right_panel", ratio=1)
        )
        
        # Left panel
        layout["left_panel"].split_column(
            Layout(name="orbital_display", size=10),
            Layout(name="comms_status", size=12),
            Layout(name="mission_log")
        )
        
        # Right panel
        layout["right_panel"].split_column(
            Layout(name="star_map", size=10),
            Layout(name="sensors", size=8),
            Layout(name="discoveries")
        )
        
        return layout
    
    def _generate_star_positions(self) -> List[Tuple[int, int, str]]:
        """Generate fixed star positions."""
        stars = []
        for _ in range(50):
            x = random.randint(0, 60)
            y = random.randint(0, 20)
            star = random.choice(STAR_CHARS[:3])
            stars.append((x, y, star))
        return stars
    
    def _register_space_tools(self):
        """Register space station tools."""
        
        @self.agent.tools_registry.register(description="Send space transmission")
        async def space_transmit(
            message: str,
            frequency: str = "STANDARD",
            priority: bool = False
        ) -> str:
            """Transmit message through space relay."""
            self.add_transmission("AI", message, frequency)
            self.stats["transmissions_sent"] += 1
            return f"TRANSMISSION_SENT: {frequency} CHANNEL"
        
        @self.agent.tools_registry.register(description="Scan space sector")
        async def scan_sector(
            sector: str = "ALPHA",
            range_km: int = 1000
        ) -> Dict[str, Any]:
            """Perform deep space scan."""
            scan_result = {
                "sector": sector,
                "range": range_km,
                "objects_detected": random.randint(0, 15),
                "anomalies": random.randint(0, 3),
                "radiation_level": random.uniform(0.1, 0.9)
            }
            
            self.stats["anomalies_detected"] += scan_result["anomalies"]
            
            if scan_result["anomalies"] > 0:
                self.alerts.append({
                    "type": "ANOMALY",
                    "message": f"Anomaly detected in sector {sector}",
                    "timestamp": time.time()
                })
            
            return scan_result
        
        @self.agent.tools_registry.register(description="Adjust orbital trajectory")
        async def adjust_orbit(
            delta_v: float = 0.0,
            inclination: float = 0.0
        ) -> str:
            """Adjust station orbital parameters."""
            self.orbital_angle += delta_v * 0.1
            return f"ORBITAL_ADJUSTMENT: Δv={delta_v}m/s, i={inclination}°"
        
        @self.agent.tools_registry.register(description="Deploy probe")
        async def deploy_probe(
            target: str,
            mission_type: str = "SURVEY"
        ) -> str:
            """Deploy exploration probe."""
            self.ship_position = 0
            self.discoveries.append({
                "type": "PROBE_DEPLOYED",
                "target": target,
                "mission": mission_type,
                "timestamp": datetime.now()
            })
            self.stats["discoveries_made"] += 1
            return f"PROBE_DEPLOYED: {mission_type} mission to {target}"
        
        @self.agent.tools_registry.register(description="Station diagnostics")
        async def run_diagnostics() -> Dict[str, Any]:
            """Run full station diagnostics."""
            return {
                "hull_integrity": f"{self.hull_integrity:.1%}",
                "oxygen_level": f"{self.oxygen_level:.1%}",
                "power_level": f"{self.power_level:.1%}",
                "signal_strength": f"{self.comms.signal_strength:.1%}",
                "mission_status": self.mission_status.value[0]
            }
    
    def create_header(self) -> Panel:
        """Create space station header."""
        # Station ASCII art
        station_art = [
            "     ╔═══╗ ┌─────┐ ╔═══╗     ",
            "  ═══╬───╫─┤COSMOS├─╫───╬═══  ",
            "     ╚═══╝ └─────┘ ╚═══╝     "
        ]
        
        # Mission timer
        elapsed = time.time() - self.stats["mission_start"]
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        # Create header content
        art_lines = [Text(line, style="bold cyan") for line in station_art]
        
        status_line = Columns([
            Text(f"STATION: {self.station_name}", style="bold white"),
            Text(f"MISSION TIME: {hours:03d}:{minutes:02d}:{seconds:02d}", style="green"),
            Text(f"STATUS: {self.mission_status.value[2]} {self.mission_status.value[0]}", 
                 style=self.mission_status.value[1])
        ], expand=True, equal=True)
        
        return Panel(
            Group(
                Align.center(Group(*art_lines)),
                Rule(style="cyan"),
                status_line
            ),
            box=box.DOUBLE,
            border_style="cyan",
            style="on rgb(0,0,20)"
        )
    
    def create_orbital_display(self) -> Panel:
        """Create orbital position display."""
        width, height = 25, 8
        center_x, center_y = width // 2, height // 2
        
        # Create display
        display = [[" " for _ in range(width)] for _ in range(height)]
        
        # Draw orbit
        for angle in range(0, 360, 10):
            x, y = CosmicElements.create_orbit(
                center_x, center_y, 8, math.radians(angle)
            )
            if 0 <= x < width and 0 <= y < height:
                display[y][x] = "·"
        
        # Draw planet
        display[center_y][center_x] = "🌍"
        
        # Draw station
        station_x, station_y = CosmicElements.create_orbit(
            center_x, center_y, 8, self.orbital_angle
        )
        if 0 <= station_x < width and 0 <= station_y < height:
            display[station_y][station_x] = "◈"
        
        # Convert to text
        lines = []
        for row in display:
            lines.append(Text("".join(row), style="cyan"))
        
        # Add orbital data
        lines.append(Text())
        lines.append(Text(f"Altitude: {348 + 50 * math.sin(self.orbital_angle):.0f} km", 
                         style="green"))
        lines.append(Text(f"Velocity: 7.66 km/s", style="cyan"))
        
        return Panel(
            Group(*lines),
            title="[ORBITAL POSITION]",
            box=box.ROUNDED,
            border_style="cyan"
        )
    
    def create_comms_status(self) -> Panel:
        """Create communications status panel."""
        # Signal strength bar
        signal_bar = self._create_signal_bar()
        
        # Queue status
        queue_lines = []
        queue_lines.append(Text("TRANSMISSION QUEUE:", style="bold"))
        queue_lines.append(Text())
        
        normal_count = len(self.comms.transmission_queue)
        priority_count = len(self.comms.priority_channel)
        
        # Visual queue
        queue_visual = Text("STD  [", style="cyan")
        queue_visual.append("█" * normal_count, style="cyan")
        queue_visual.append("░" * (10 - normal_count), style="dim")
        queue_visual.append(f"] {normal_count}/10", style="cyan")
        
        priority_visual = Text("PRIO [", style="magenta")
        priority_visual.append("█" * priority_count, style="magenta")
        priority_visual.append("░" * (3 - priority_count), style="dim")
        priority_visual.append(f"] {priority_count}/3", style="magenta")
        
        queue_lines.append(queue_visual)
        queue_lines.append(priority_visual)
        
        # Stats
        queue_lines.append(Text())
        queue_lines.append(Text(f"Lost: {self.comms.packets_lost}", style="red"))
        queue_lines.append(Text(f"Total: {self.comms.total_transmissions}", style="green"))
        
        return Panel(
            Group(signal_bar, Text(), *queue_lines),
            title="[COMMS STATUS]",
            box=box.ROUNDED,
            border_style="green" if self.comms.signal_strength > 0.7 else "yellow"
        )
    
    def _create_signal_bar(self) -> Text:
        """Create signal strength indicator."""
        strength = self.comms.signal_strength
        bars = int(strength * 5)
        
        signal = Text("SIGNAL: ", style="bold")
        signal.append("▁▃▅▇█"[:bars], style="green" if strength > 0.7 else "yellow")
        signal.append("     "[bars:], style="dim red")
        signal.append(f" {strength:.0%}", style="white")
        
        if strength < 0.5:
            signal.append(" ⚠", style="yellow blink")
        
        return signal
    
    def create_viewport(self) -> Panel:
        """Create main viewport with transmissions."""
        messages = []
        
        # Show transmission log
        for trans in self.transmission_log[-15:]:
            role = trans["role"]
            content = trans["content"]
            timestamp = trans["timestamp"]
            freq = trans.get("frequency", "STANDARD")
            
            if role == "OPERATOR":
                header = Text(f"\n[{timestamp}] ", style="dim")
                header.append(f"OPERATOR@{freq}> ", style="bold cyan")
                messages.append(header)
                messages.append(Text(f"  {content}", style="cyan"))
                
            elif role == "AI":
                header = Text(f"\n[{timestamp}] ", style="dim")
                header.append(f"AI@{self.station_name}> ", style="bold green")
                messages.append(header)
                
                # Add transmission effect
                if self.processing:
                    messages.append(Text(f"  ▶ TRANSMITTING... ◀", style="green blink"))
                messages.append(Text(f"  {content}", style="green"))
                
            elif role == "SYSTEM":
                messages.append(Text(f"\n[SYSTEM] {content}", style="yellow italic"))
        
        if not messages:
            welcome = Text("SPACE STATION AI TERMINAL\n", style="bold")
            welcome.append(f"Connected to {self.station_name}\n", style="green")
            welcome.append("Signal strength optimal. Begin transmission.", style="dim")
            messages = [welcome]
        
        # Add space effect at bottom
        if self.frame_count % 10 < 5:
            messages.append(Text("\n" + CosmicElements.solar_wind(50), style="dim blue"))
        
        return Panel(
            Group(*messages),
            title="[VIEWPORT]",
            box=box.DOUBLE,
            border_style="green" if not self.processing else "yellow",
            padding=(1, 2),
            style="on rgb(0,0,30)"
        )
    
    def create_star_map(self) -> Panel:
        """Create animated star map."""
        width, height = 30, 8
        
        # Create star field
        field = [[" " for _ in range(width)] for _ in range(height)]
        
        # Place stars
        for x, y, star in self.star_positions:
            if 0 <= x < width and 0 <= y < height:
                # Twinkle effect
                if random.random() < 0.9:
                    field[y][x] = star
        
        # Add constellation
        if self.frame_count % 30 < 20:
            constellation = CosmicElements.constellation_map()
            start_x, start_y = 5, 0
            for dy, line in enumerate(constellation[:height]):
                for dx, char in enumerate(line[:width - start_x]):
                    if char != " " and start_y + dy < height:
                        field[start_y + dy][start_x + dx] = char
        
        # Convert to text
        lines = []
        for row in field:
            line = ""
            for char in row:
                if char in STAR_CHARS[:3]:
                    line += f"[bright_white]{char}[/]"
                elif char in ["\\", "/", "-", "|"]:
                    line += f"[dim cyan]{char}[/]"
                else:
                    line += char
            lines.append(Text.from_markup(line))
        
        return Panel(
            Group(*lines),
            title="[STAR MAP]",
            box=box.ROUNDED,
            border_style="blue",
            style="on rgb(0,0,15)"
        )
    
    def create_sensors(self) -> Panel:
        """Create sensor readout panel."""
        table = Table(box=None, show_header=False, padding=0)
        table.add_column("Sensor", style="cyan")
        table.add_column("Value", style="green", justify="right")
        
        # Add sensor readings with dynamic values
        table.add_row("Radiation", f"{random.uniform(0.1, 0.5):.2f} mSv/h")
        table.add_row("Gravity", f"{random.uniform(0.95, 1.05):.2f} g")
        table.add_row("Temperature", f"{random.randint(-150, -100)}°C")
        table.add_row("Pressure", "0.0 kPa")
        table.add_row("Solar Wind", f"{random.randint(300, 500)} km/s")
        
        # Anomaly detection
        if random.random() < 0.1:
            table.add_row("Anomaly", "DETECTED", style="yellow blink")
            self.stats["anomalies_detected"] += 1
        
        return Panel(
            table,
            title="[SENSORS]",
            box=box.ROUNDED,
            border_style="green"
        )
    
    def create_discoveries(self) -> Panel:
        """Create discoveries panel."""
        disc_lines = []
        
        if self.discoveries:
            for disc in self.discoveries[-5:]:
                disc_lines.append(Text(f"◆ {disc['type']}", style="bold cyan"))
                disc_lines.append(Text(f"  {disc['target']}", style="dim"))
        else:
            disc_lines.append(Text("No discoveries yet", style="dim"))
            disc_lines.append(Text("Deploy probes to explore", style="dim italic"))
        
        disc_lines.append(Text())
        disc_lines.append(Text(f"Total: {self.stats['discoveries_made']}", style="green"))
        
        return Panel(
            Group(*disc_lines),
            title="[DISCOVERIES]",
            box=box.ROUNDED,
            border_style="cyan"
        )
    
    def create_mission_log(self) -> Panel:
        """Create mission log panel."""
        log_lines = []
        
        # Recent alerts
        current_time = time.time()
        for alert in reversed(list(self.alerts)):
            age = current_time - alert["timestamp"]
            if age < 30:  # Show for 30 seconds
                style = "yellow" if age < 10 else "dim yellow"
                log_lines.append(Text(f"▸ {alert['message']}", style=style))
        
        if not log_lines:
            log_lines.append(Text("All systems nominal", style="green"))
        
        return Panel(
            Group(*log_lines),
            title="[MISSION LOG]",
            box=box.ROUNDED,
            border_style="yellow" if log_lines[0].plain != "All systems nominal" else "green"
        )
    
    def create_systems_panel(self) -> Panel:
        """Create systems status panel."""
        # Life support
        life_support = Columns([
            self._create_system_gauge("O₂", self.oxygen_level, "cyan"),
            self._create_system_gauge("PWR", self.power_level, "yellow"),
            self._create_system_gauge("HULL", self.hull_integrity, "green"),
        ], expand=True, equal=True)
        
        # Command hint
        hint = Text("CMD: ", style="dim")
        hint.append("Type message or /help for commands", style="dim italic")
        
        return Panel(
            Group(life_support, Text(), hint),
            box=box.HEAVY,
            border_style="cyan",
            height=5
        )
    
    def _create_system_gauge(self, label: str, value: float, color: str) -> Text:
        """Create a system gauge."""
        gauge = Text(f"{label}: ", style="bold")
        
        # Visual bar
        bar_width = 10
        filled = int(value * bar_width)
        
        if value < 0.3:
            gauge.append("█" * filled, style=f"{color} blink")
            gauge.append("░" * (bar_width - filled), style="red")
        else:
            gauge.append("█" * filled, style=color)
            gauge.append("░" * (bar_width - filled), style="dim")
        
        gauge.append(f" {value:.0%}", style=color if value > 0.3 else "red")
        
        return gauge
    
    def add_transmission(self, role: str, content: str, frequency: str = "STANDARD"):
        """Add transmission to log."""
        self.transmission_log.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "frequency": frequency
        })
        
        if role == "AI":
            self.stats["transmissions_sent"] += 1
        else:
            self.stats["transmissions_received"] += 1
    
    def update_display(self):
        """Update all panels."""
        self.frame_count += 1
        
        # Update orbital position
        self.orbital_angle += 0.02
        if self.orbital_angle > 2 * math.pi:
            self.orbital_angle = 0
        
        # Update communications
        self.comms.update_orbital_position(self.orbital_angle)
        
        # Update ship animation
        if self.ship_position < 40:
            self.ship_position += 1
        
        # Random system fluctuations
        self.oxygen_level += random.uniform(-0.001, 0.001)
        self.oxygen_level = max(0.8, min(1.0, self.oxygen_level))
        
        self.power_level += random.uniform(-0.002, 0.001)
        self.power_level = max(0.7, min(1.0, self.power_level))
        
        # Update status based on systems
        if self.oxygen_level < 0.85 or self.power_level < 0.8:
            self.mission_status = MissionStatus.WARNING
        elif self.hull_integrity < 0.9:
            self.mission_status = MissionStatus.CAUTION
        else:
            self.mission_status = MissionStatus.OPTIMAL
        
        # Update layout
        self.layout["header"].update(self.create_header())
        self.layout["orbital_display"].update(self.create_orbital_display())
        self.layout["comms_status"].update(self.create_comms_status())
        self.layout["mission_log"].update(self.create_mission_log())
        self.layout["viewport"].update(self.create_viewport())
        self.layout["star_map"].update(self.create_star_map())
        self.layout["sensors"].update(self.create_sensors())
        self.layout["discoveries"].update(self.create_discoveries())
        self.layout["systems"].update(self.create_systems_panel())
    
    async def process_transmission(self, message: str):
        """Process operator transmission."""
        self.processing = True
        
        # Add to log
        self.add_transmission("OPERATOR", message)
        
        # Add processing alert
        self.alerts.append({
            "type": "TRANSMISSION",
            "message": "Processing transmission...",
            "timestamp": time.time()
        })
        
        try:
            # Send to AI
            response = await self.agent.send_user(message, stream=False)
            
            # AI will use space_transmit to respond
            
        except Exception as e:
            self.add_transmission("SYSTEM", f"TRANSMISSION ERROR: {str(e)}")
            self.alerts.append({
                "type": "ERROR",
                "message": f"Communication error: {type(e).__name__}",
                "timestamp": time.time()
            })
        
        finally:
            self.processing = False
    
    async def handle_command(self, command: str):
        """Handle space station commands."""
        cmd = command.lower().strip()
        
        if cmd == "/help":
            help_text = """
SPACE STATION COMMANDS:
/help     - Show commands
/scan     - Scan current sector
/deploy   - Deploy exploration probe
/orbit    - Adjust orbital parameters
/status   - Full system diagnostics
/clear    - Clear transmission log
/quit     - End session

TRANSMISSION: Type message and press ENTER
"""
            self.add_transmission("SYSTEM", help_text.strip())
            
        elif cmd == "/scan":
            self.add_transmission("SYSTEM", "Initiating sector scan...")
            result = await self.agent.tools_registry.call("scan_sector", 
                                                         sector="CURRENT", 
                                                         range_km=5000)
            
        elif cmd.startswith("/deploy"):
            parts = cmd.split()
            target = parts[1] if len(parts) > 1 else "UNKNOWN"
            await self.agent.tools_registry.call("deploy_probe", target=target)
            
        elif cmd == "/status":
            result = await self.agent.tools_registry.call("run_diagnostics")
            self.add_transmission("SYSTEM", str(result))
            
        elif cmd == "/clear":
            self.transmission_log = []
            self.add_transmission("SYSTEM", "Transmission log cleared")
            
        elif cmd == "/quit":
            self.running = False
            
        else:
            self.add_transmission("SYSTEM", f"Unknown command: {command}")
    
    async def run(self):
        """Run the cosmic interface."""
        # Initialize
        self.add_transmission("SYSTEM", f"COSMIC INTERFACE INITIALIZED")
        self.add_transmission("SYSTEM", f"Connected to {self.station_name}")
        await asyncio.sleep(0.5)
        self.add_transmission("AI", "Space station AI online. All systems nominal.")
        
        # Create prompt session
        prompt_session = PromptSession(
            history=FileHistory('.cosmic_history'),
            auto_suggest=AutoSuggestFromHistory()
        )
        
        # Main loop
        with Live(
            self.layout,
            console=self.console,
            refresh_per_second=5,
            transient=False
        ) as live:
            while self.running:
                self.update_display()
                
                # Check for queued transmissions
                if not self.processing:
                    packet = await self.comms.receive()
                    if packet:
                        await self.process_transmission(packet["data"])
                        continue
                
                # Get input
                try:
                    with patch_stdout():
                        user_input = await prompt_session.prompt_async(
                            "OPERATOR> ",
                            enable_history_search=True
                        )
                    
                    if user_input.strip():
                        if user_input.startswith("/"):
                            await self.handle_command(user_input)
                        else:
                            # Check signal and queue if needed
                            if self.processing or self.comms.signal_strength < 0.3:
                                success, status = await self.comms.transmit(user_input)
                                if success:
                                    self.add_transmission("SYSTEM", 
                                                        f"Transmission queued: {status}")
                                else:
                                    self.add_transmission("SYSTEM", 
                                                        f"Transmission failed: {status}")
                            else:
                                await self.process_transmission(user_input)
                                
                except (EOFError, KeyboardInterrupt):
                    self.running = False
                    break
                except Exception as e:
                    self.add_transmission("SYSTEM", f"INPUT ERROR: {str(e)}")
        
        # Shutdown
        self.add_transmission("SYSTEM", "Closing connection...")
        await self.agent.worker_pool.stop()
        self.console.print("\n[bold]TRANSMISSION ENDED[/bold]")

# ————————————————————————————————————————————————————————————————
# Main Entry Point
# ————————————————————————————————————————————————————————————————

async def main():
    """Launch the cosmic interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cosmic Neural Interface")
    parser.add_argument("--session", help="Resume mission session")
    parser.add_argument("--station", default="COSMOS-7", help="Station designation")
    args = parser.parse_args()
    
    # Mission briefing
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    SPACE STATION AI TERMINAL                  ║
    ║                                                               ║
    ║  Mission: Deep Space Communication & Exploration              ║
    ║  Station: """ + args.station.ljust(52) + """║
    ║  Status:  All systems operational                             ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    await asyncio.sleep(1)
    
    # Create agent
    print("[INITIALIZING] Neural systems...")
    agent = EnhancedStatefulAgent(
        session_id=args.session,
        stream=False
    )
    
    # Register tools
    register_enhanced_demo_tools(agent)
    
    # Create interface
    print("[ESTABLISHING] Communication link...")
    interface = CosmicInterface(agent)
    interface.station_name = args.station
    
    # Launch
    print("[CONNECTED] Entering cosmic interface...\n")
    await asyncio.sleep(1)
    
    await interface.run()

if __name__ == "__main__":
    asyncio.run(main())