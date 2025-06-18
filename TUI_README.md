# Enhanced Agent Terminal User Interfaces (TUI)

This directory contains two sophisticated Terminal User Interface implementations for the Enhanced Multi-turn Agent with reflection, task management, and environment monitoring capabilities.

## 🚀 Features

### Core Agent Features
- **Self-Reflection System**: Analyzes performance and learns from interactions
- **Task Queue**: Priority-based task execution with dependency management
- **Worker Pool**: 4 concurrent workers for parallel task execution
- **Environment Monitoring**: Real-time system resource tracking
- **Persistent State**: SQLite/DuckDB storage for conversation history
- **Semantic Search**: Search through conversation history

### TUI Features
- **Real-time Updates**: Live updating displays of all agent metrics
- **Multi-panel Layout**: Chat, tasks, metrics, and environment views
- **Rich Formatting**: Syntax highlighting, markdown rendering, progress bars
- **Keyboard Shortcuts**: Quick access to common commands
- **Mouse Support**: Click and scroll in supported terminals
- **Export Functionality**: Save chat history to markdown files

## 📦 Installation

```bash
# Install base requirements
pip install -r requirements_enhanced.txt

# Install TUI-specific requirements
pip install -r requirements_tui.txt
```

## 🎨 Available TUI Versions

### 1. Textual TUI (`multi_turn_agent_tui.py`)
A full-featured TUI using the Textual framework with multiple panels and widgets.

**Features:**
- Split-pane layout with dedicated areas
- Mouse support and clickable elements
- Scrollable chat and task lists
- Real-time sparkline charts
- Theme switching (F1)

**Launch:**
```bash
python multi_turn_agent_tui.py [--session SESSION_ID] [--workers N]
```

**Screenshots:**
```
┌─────────────────────────────────────────┬──────────────────────┐
│ 🤖 Enhanced Agent TUI - Session abc123  │ 📊 Metrics          │
├─────────────────────────────────────────┤ Duration: 5.2 min   │
│ 💬 Chat                                 │ Turns: 12           │
│                                         │ Tools: 8            │
│ 👤 User [10:23:45]                     │ Tasks: 5            │
│   Create a performance analysis task    │ Workers: 2/4 active │
│                                         ├─────────────────────┤
│ 🤖 Assistant [10:23:47]                │ 📋 Task Queue       │
│   I'll create a performance analysis    │ Total: 5 | Queued: 2│
│   task for you.                         │                     │
│                                         │ 🔄 Active:          │
│   Created task abc12345: Performance    │ • Performance anal...│
│   analysis: system (priority: medium)   │ • Memory optimiz... │
│                                         │                     │
│ 🔧 Tool Result [10:23:48]              │ ✅ Recent:          │
│   Started performance analysis task     │ ✅ Task cleanup (1s) │
│                                         │ ✅ Data export (3s)  │
├─────────────────────────────────────────┼─────────────────────┤
│ 💬 You: _                               │ 🖥️ Environment      │
│ [Send]                                  │ Mem: ████░░ 78.5%   │
└─────────────────────────────────────────┴─────────────────────┘
```

### 2. Rich Console TUI (`multi_turn_agent_rich_tui.py`)
A simpler but powerful TUI using Rich's Live display with prompt_toolkit input.

**Features:**
- Live updating console display
- Clean, responsive layout
- Advanced prompt with history and auto-suggestions
- Minimal dependencies
- Works in more terminal environments

**Launch:**
```bash
python multi_turn_agent_rich_tui.py [--session SESSION_ID] [--workers N]
```

## ⌨️ Keyboard Shortcuts

### Common Shortcuts (Both TUIs)
- `Ctrl+H` - Show help
- `Ctrl+S` - Show full status
- `Ctrl+L` - Clear chat display
- `Ctrl+Q` - Quit application
- `Ctrl+C` - Emergency exit

### Textual TUI Additional
- `F1` - Toggle light/dark theme
- `F2` - Export chat to file
- `Tab` - Navigate between panels
- `Arrow Keys` - Navigate within panels

## 💬 Available Commands

### Chat Commands
- `/help` - Show available commands
- `/status` - Display full agent status
- `/clear` - Clear chat display (preserves history)
- `/export` - Export chat to markdown file
- `/tasks` - Show detailed task information
- `/insights` - Display reflection insights
- `/quit` - Exit the application

### Agent Tools (Use in Chat)
```
create_task("Task name", priority="high")
get_task_status()
get_insights()
get_environment()
search_history("query")
analyze_performance("system", duration=5)
create_workflow("My Workflow", ["Step 1", "Step 2", "Step 3"])
```

## 🎯 Usage Examples

### Basic Conversation
```
You: Hello! What can you help me with today?
Assistant: I'm an enhanced AI assistant with advanced capabilities including 
self-reflection, task management, concurrent execution, and environment monitoring.
I can help you with various tasks while learning and improving from our interactions.

You: Create a task to analyze system performance
Assistant: I'll create a performance analysis task for you.
[Creates task with ID abc123]

You: Show me the current status
Assistant: [Displays comprehensive status including tasks, metrics, and insights]
```

### Advanced Workflow
```
You: Create a workflow for data processing with 3 steps: fetch, transform, analyze
Assistant: Creating workflow 'data processing' with 3 dependent steps...
[Creates 3 linked tasks that execute in sequence]

You: get_insights()
Assistant: Here are my current insights:
- Total reflections: 15
- Average confidence: 0.85
- Recent improvements: 
  • Optimized task scheduling for better throughput
  • Added retry logic for network-related failures
```

## 🔧 Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_MODEL="claude-3-sonnet"  # Optional
export AGENT_WORKERS=8                    # Number of workers (default: 4)
```

### Session Management
```bash
# Start new session
python multi_turn_agent_tui.py

# Resume existing session
python multi_turn_agent_tui.py --session abc123def456

# Custom worker count
python multi_turn_agent_tui.py --workers 8
```

## 🐛 Troubleshooting

### Common Issues

1. **Terminal Compatibility**
   - Textual TUI requires a modern terminal with Unicode support
   - Rich TUI works in most terminals but may have reduced features in basic ones
   - Try setting `export TERM=xterm-256color`

2. **Performance Issues**
   - Reduce update frequency if TUI is laggy
   - Use fewer workers if system resources are limited
   - Consider using Rich TUI for lower overhead

3. **Display Problems**
   - Ensure terminal window is large enough (minimum 80x24)
   - Check terminal font supports Unicode characters
   - Try different terminal emulators (iTerm2, Windows Terminal, etc.)

## 🎨 Customization

### Modifying Layouts
Both TUIs can be customized by editing the layout definitions:

```python
# In Textual TUI
layout.split_column(
    Layout(name="header", size=3),
    Layout(name="body", ratio=3),    # Adjust ratios
    Layout(name="footer", size=4)     # Adjust sizes
)

# In Rich TUI
self.layout["body"].split_row(
    Layout(name="chat", ratio=3),     # Make chat wider
    Layout(name="sidebar", ratio=1)
)
```

### Adding Custom Panels
Extend the TUI with your own panels:

```python
def update_custom_panel(self) -> Panel:
    """Your custom panel logic."""
    content = Text("Custom content here")
    return Panel(content, title="🎯 Custom", box=box.ROUNDED)
```

## 📊 Performance Considerations

- **Update Frequency**: Both TUIs update 2-4 times per second by default
- **Message History**: Displays last 20-50 messages to maintain performance
- **Task Display**: Shows active tasks and last 5 completed tasks
- **Worker Pool**: 4 workers by default, adjustable via `--workers`

## 🚀 Future Enhancements

Planned features:
- [ ] Graph visualizations for metrics
- [ ] Task dependency diagrams
- [ ] Voice input/output support
- [ ] Web-based TUI option
- [ ] Plugin system for custom tools
- [ ] Multi-agent collaboration view

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

---

Happy chatting with your enhanced agent! 🤖✨