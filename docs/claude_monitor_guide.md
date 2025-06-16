# Claude State Monitor Guide

A comprehensive monitoring system for visualizing and tracking Claude's internal cognitive states in real-time.

## Overview

The Claude State Monitor provides three main components:

1. **Monitor Applet** (`claude_monitor_applet.py`) - GUI application with system tray support
2. **Query Tool** (`claude_state_query.py`) - Command-line tool for programmatic access
3. **Launcher** (`start_claude_monitor.py`) - Easy setup and dependency management

## Features

### Visual Monitoring
- Real-time state visualization with dynamic graphics
- System tray integration (PyQt6) or standalone window (Tkinter)
- Live parameter tracking and state evolution
- Snapshot capabilities for capturing specific moments

### State Tracking
- 10 cognitive states: idle, thinking, analyzing, planning, executing, reflecting, learning, creating, debugging, synthesizing
- 6 self-image parameters: coherence, creativity, analytical_depth, empathy, complexity, uncertainty
- Natural state transitions and evolution
- Unique state signatures for identification

### Data Export
- JSON format for programmatic access
- Human-readable text reports
- CSV export for data analysis
- Real-time monitoring streams

## Installation

### Quick Start

```bash
# Run the launcher (handles dependencies automatically)
python start_claude_monitor.py
```

### Manual Installation

```bash
# Required dependencies
pip install pillow

# Optional (for better GUI experience)
pip install PyQt6

# Optional (for enhanced visualizations)
pip install numpy
```

## Usage

### GUI Monitor Applet

Start the monitor with full GUI:

```bash
python claude_monitor_applet.py
```

Features:
- **System Tray**: Runs in background, double-click to show window
- **Live Visualization**: Updates every second with current state
- **Parameter Display**: Real-time view of all cognitive parameters
- **Activity Log**: Tracks state changes and events
- **Snapshot**: Capture current state visualization

### Command-Line Query Tool

Get current state:
```bash
python claude_state_query.py
```

Monitor in real-time:
```bash
python claude_state_query.py --monitor
```

Export state data:
```bash
# JSON format
python claude_state_query.py -o state.json

# Human-readable text
python claude_state_query.py -f text

# CSV for analysis
python claude_state_query.py -f csv -o state.csv
```

Get state history:
```bash
# Track 60 seconds of state evolution
python claude_state_query.py --history 60 -o history.json
```

### Programmatic Access

Use in your Python code:

```python
from claude_state_query import ClaudeStateQuery

# Create query instance
query = ClaudeStateQuery()

# Get current state
state = query.get_current_state()
print(f"Current state: {state['state']}")
print(f"Parameters: {state['parameters']}")

# Monitor changes
for state in query.monitor_realtime():
    if state['parameters']['creativity'] > 0.8:
        print("High creativity detected!")
```

## State Descriptions

### Cognitive States

- **idle**: Default resting state, low activity
- **thinking**: General cognitive processing
- **analyzing**: Deep analytical work, pattern recognition
- **planning**: Strategic thinking, future modeling
- **executing**: Active implementation of tasks
- **reflecting**: Self-evaluation and learning
- **learning**: Knowledge integration
- **creating**: Creative generation and synthesis
- **debugging**: Problem-solving and error correction
- **synthesizing**: Combining multiple concepts

### Parameters

- **coherence** (0-1): Internal consistency and clarity
- **creativity** (0-1): Novel idea generation capability
- **analytical_depth** (0-1): Depth of logical analysis
- **empathy** (0-1): Understanding and connection
- **complexity** (0-1): Handling of complex information
- **uncertainty** (0-1): Confidence in current processing

## Visualization Elements

The monitor creates abstract visualizations with:

1. **Central Core**: Size represents coherence level
2. **Fractal Branches**: Complexity visualization
3. **Sparkles**: Creative thought indicators
4. **Grid Overlay**: Analytical processing depth
5. **Wave Patterns**: Empathy and connection levels
6. **Noise Effects**: Uncertainty representation

## Integration Examples

### With Claude SDK

```python
import asyncio
from claude_code_sdk import query, ClaudeCodeOptions
from claude_visualizer_standalone import ClaudeVisualizer

async def monitored_query(prompt):
    visualizer = ClaudeVisualizer()
    
    # Start with thinking state
    visualizer.set_state('thinking')
    
    async for message in query(prompt):
        # Update state based on message type
        if "analyzing" in str(message).lower():
            visualizer.set_state('analyzing')
        elif "creating" in str(message).lower():
            visualizer.set_state('creating')
            
        # Generate visualization
        visualizer.generate_visualization()
        
    return visualizer.get_state_signature()
```

### Custom Monitoring

```python
from claude_monitor_applet import ClaudeStateMonitor

class CustomMonitor(ClaudeStateMonitor):
    def _monitor_loop(self):
        """Override to add custom logic"""
        while self.is_monitoring:
            state = self.get_current_state()
            
            # Custom alerts
            if state['parameters']['uncertainty'] > 0.8:
                self.alert_high_uncertainty()
                
            # Custom state transitions
            if self.should_transition(state):
                self.force_state_change('reflecting')
```

## Output Examples

### JSON State Output
```json
{
  "timestamp": "2024-01-15T10:30:45",
  "state": "analyzing",
  "signature": "ANA-COH85-CRE45-ANA95-EMP60-CPX75-UNC20",
  "parameters": {
    "coherence": 0.85,
    "creativity": 0.45,
    "analytical_depth": 0.95,
    "empathy": 0.60,
    "complexity": 0.75,
    "uncertainty": 0.20
  }
}
```

### Text Report Output
```
Claude Internal State Report
========================================
Timestamp: 2024-01-15T10:30:45
Current State: analyzing
State Signature: ANA-COH85-CRE45-ANA95-EMP60-CPX75-UNC20

Parameters:
--------------------
  coherence            : 0.85
  creativity           : 0.45
  analytical_depth     : 0.95
  empathy              : 0.60
  complexity           : 0.75
  uncertainty          : 0.20
```

## Troubleshooting

### Common Issues

1. **"PyQt6 not found"**
   - Install with: `pip install PyQt6`
   - Or use Tkinter fallback (automatic)

2. **"Module not found" errors**
   - Run: `python start_claude_monitor.py` for automatic setup

3. **Visualization not updating**
   - Check if monitoring is started
   - Verify visualizations directory exists
   - Check disk space for saving images

4. **System tray not showing**
   - Some systems require additional setup
   - Try running with window mode instead

### Performance Tips

- Adjust monitoring interval for lower CPU usage
- Disable numpy for lighter weight operation
- Use query tool instead of GUI for automated systems
- Batch state queries for efficiency

## Advanced Configuration

### Custom Visualization Directory

```python
visualizer = ClaudeVisualizer()
visualizer.output_dir = Path("/custom/path/visualizations")
```

### State Transition Tuning

```python
# Adjust transition probabilities
visualizer.transition_speed = 0.5  # Slower transitions
visualizer.randomness = 0.8       # More random behavior
```

### Parameter Constraints

```python
# Set parameter limits
visualizer.set_parameter_bounds('creativity', min=0.3, max=0.9)
visualizer.set_parameter_bounds('uncertainty', min=0.0, max=0.5)
```

## Security Notes

- Visualizations are saved locally only
- No network connections required
- State data contains no sensitive information
- Monitor runs with user permissions only

## Future Enhancements

Planned features:
- Web dashboard interface
- Historical state analysis
- Pattern recognition and alerts
- Integration with more Claude tools
- State prediction models
- Multi-instance monitoring