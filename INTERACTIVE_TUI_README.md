# Interactive Agent TUI - Agent-Controlled Interface

A revolutionary approach to AI interfaces where the agent has full control over its own terminal UI, creating a truly interactive and dynamic experience.

## 🎯 Key Concept

Unlike traditional chatbots where the UI is separate from the agent, this Interactive TUI exposes UI control functions directly to the agent as tools. This means the agent can:

- Send formatted messages
- Display notifications
- Create and update panels
- Show its thinking process
- Visualize data and workflows
- Request user confirmation
- Export conversations
- Control the entire interface

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements_tui.txt

# Run the interactive TUI
python multi_turn_agent_interactive_tui.py

# Run with demo mode
python multi_turn_agent_interactive_tui.py --demo

# Test UI functions
python test_interactive_tui.py test
```

## 🛠️ Agent UI Tools

### Core UI Functions

| Tool | Description | Example |
|------|-------------|---------|
| `msg_user` | Send formatted messages | `msg_user("Hello!", style="bold", markdown=True)` |
| `notify` | Show notifications | `notify("Task complete", type="success")` |
| `update_panel` | Update UI panels | `update_panel("tasks", "New content")` |
| `create_custom_panel` | Create new panels | `create_custom_panel("stats", "Statistics", "...")` |
| `clear_chat` | Clear chat display | `clear_chat()` |
| `focus_panel` | Change focus | `focus_panel("metrics")` |
| `show_progress` | Display progress bars | `show_progress("task-1", 0.75, "Processing...")` |
| `show_chart` | Display visualizations | `show_chart("bar", data, "Title")` |
| `set_theme` | Change UI theme | `set_theme("dark")` |
| `export_view` | Export current view | `export_view("markdown", "export.md")` |
| `highlight` | Highlight text | `highlight("important", color="yellow")` |
| `show_thinking` | Display reasoning | `show_thinking("Analyzing...", confidence=0.8)` |
| `request_confirmation` | Get user confirmation | `request_confirmation("Proceed?", ["Yes", "No"])` |
| `show_workflow` | Display workflows | `show_workflow("Pipeline", tasks)` |

## 📸 UI Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ 🤖 Interactive Agent TUI | Session: abc123 | chat | 14:32:10   │
├─────────────────────────────────────┬───────────────────────────┤
│ ℹ️ Agent is analyzing your request  │ 📊 Metrics               │
├─────────────────────────────────────┤ Session    15.3m         │
│ 💬 Chat                             │ Turns      42            │
│                                     │ Tools      18            │
│ [14:30:15] You:                    │ Tasks      7             │
│   Show me system performance        │ Workers    ████░ 3/4     │
│                                     │ Queued     2             │
│ [14:30:17] Agent:                   ├──────────────────────────┤
│   I'll analyze system performance   │ 📋 Tasks & Workflows     │
│   and create a dashboard for you.   │                          │
│                                     │ 🔄 Active Tasks:         │
│   [Creating dashboard...]           │   Performance analys...  │
│                                     │   [███████░░░] 70%       │
│ 💭 Thinking [0.85]:                 │                          │
│   System metrics show high memory   │ ⏳ Waiting on Deps:      │
│   usage. I should investigate...    │   Report generation...   │
│                                     │                          │
│ [14:30:23] System:                  │ ✅ Recently Completed:   │
│   Dashboard created successfully    │   ✓ Data fetch (2.1s)    │
│                                     │   ✓ Analysis (5.3s)      │
├─────────────────────────────────────┼──────────────────────────┤
│ You: _                              │ 🤔 Agent Insights        │
│                                     │ Confidence: [████████] 0.82 │
│ ✅ Ready | Type to chat | Tab: Switch panels | Ctrl+Q: Quit    │
└─────────────────────────────────────────────────────────────────┘
```

## 🎨 Features

### 1. **Agent-Initiated UI Updates**
The agent can proactively update the interface based on context:

```python
# Agent detects a long-running task
await show_progress("analysis", 0.0, "Starting analysis...")
await notify("This might take a while", type="info")

# Agent shows its reasoning
await show_thinking("User seems interested in performance, I should create visualizations", 0.9)

# Agent creates a custom dashboard
await create_custom_panel("perf_dashboard", "Performance Dashboard", metrics_content)
```

### 2. **Rich Messaging**
Messages support various formats and styles:

```python
# Formatted text
await msg_user("**Important:** System requires attention!", markdown=True)

# Code blocks (automatically syntax highlighted)
await msg_user("""
```python
def optimize_query():
    return db.query().filter().limit(100)
```
""")

# Custom styles
await msg_user("✅ Task completed successfully!", style="bold green")
```

### 3. **Interactive Workflows**
The agent can create visual workflow representations:

```python
await show_workflow("Deployment Pipeline", [
    {"id": "1", "name": "Tests", "status": "completed"},
    {"id": "2", "name": "Build", "status": "running"},
    {"id": "3", "name": "Deploy", "status": "pending", "dependencies": ["2"]}
])
```

### 4. **Dynamic Panels**
Create custom panels for specific purposes:

```python
# Real-time metrics panel
await create_custom_panel("metrics", "Live Metrics", """
CPU:    ████████░░ 80%
Memory: ██████░░░░ 60%
Disk:   ███░░░░░░░ 30%
""")

# Code editor panel
await create_custom_panel("editor", "Code Editor", code_content)
await highlight("def main():", color="yellow", panel="editor")
```

### 5. **Notification System**
Different notification types for various situations:

- `info` - General information (blue)
- `success` - Successful operations (green)
- `warning` - Warnings (yellow)
- `error` - Errors (red)
- `critical` - Critical alerts (red on white)

## 🧠 Agent Behaviors

### Proactive UI Management
The agent can:
- Detect when to show progress for long operations
- Create panels when complex data needs visualization
- Show thinking process during complex reasoning
- Highlight important information automatically
- Export results when conversations contain valuable insights

### Context-Aware Responses
```python
# User asks about performance
Agent: creates performance dashboard, shows metrics, highlights issues

# User requests data analysis
Agent: shows progress, creates visualization panels, exports results

# User needs help debugging
Agent: creates code panels, highlights issues, shows thinking process
```

## 🔧 Customization

### Adding New UI Tools

```python
@agent.tools_registry.register(description="Custom UI function")
async def my_custom_ui_function(param1: str, param2: int) -> str:
    """Your custom UI function."""
    # Implementation
    return "Result"
```

### Extending the TUI

```python
class MyCustomTUI(InteractiveAgentTUI):
    def __init__(self, agent):
        super().__init__(agent)
        # Add custom initialization
    
    def update_custom_panel(self) -> Panel:
        """Add your custom panel."""
        return Panel("Custom content", title="My Panel")
```

## 📝 Example Conversations

### Example 1: Performance Analysis
```
User: Analyze the system performance and show me any issues

Agent:
1. notify("Starting system performance analysis", type="info")
2. show_thinking("I'll check CPU, memory, disk, and network metrics", 0.85)
3. create_custom_panel("perf_metrics", "Performance Metrics", "Loading...")
4. show_progress("analysis", 0.3, "Collecting CPU metrics")
5. show_progress("analysis", 0.6, "Analyzing memory usage")
6. update_panel("perf_metrics", detailed_metrics)
7. highlight("Memory: 92%", color="red", panel="perf_metrics")
8. notify("⚠️ High memory usage detected", type="warning")
9. msg_user("I've found a memory issue. Here's my analysis and recommendations...")
```

### Example 2: Interactive Tutorial
```
User: Teach me about async programming in Python

Agent:
1. msg_user("I'll create an interactive tutorial on async programming!")
2. create_custom_panel("code_example", "Code Example", initial_async_code)
3. create_custom_panel("explanation", "Explanation", "")
4. show_thinking("Starting with basic concepts would be best", 0.9)
5. update_panel("explanation", "Async allows concurrent execution...")
6. highlight("async def", color="yellow", panel="code_example")
7. request_confirmation("Ready for the next example?", ["Yes", "Show me more about this"])
8. (Updates panels based on user response)
```

## 🚦 UI State Management

The TUI maintains state for:
- Current focused panel
- Theme preferences
- Notification queue
- Custom panels
- Chat history
- UI configuration

## 🔌 Integration

### With Task System
```python
# Agent creates task and shows progress
task_id = await create_task("Data processing")
await show_progress(task_id, 0.0, "Starting...")
# Progress updates automatically as task executes
```

### With Reflection System
```python
# Agent shows its learning
insights = await get_insights()
await show_thinking(f"Based on {insights['total_reflections']} past interactions...", 
                   confidence=insights['avg_confidence'])
```

## 🎯 Best Practices

1. **Use appropriate UI elements** - Notifications for alerts, panels for persistent info
2. **Show progress** - Always indicate progress for operations > 2 seconds
3. **Be transparent** - Use show_thinking() to explain reasoning
4. **Confirm dangerous actions** - Always use request_confirmation()
5. **Export valuable content** - Offer to save important results
6. **Adapt to user** - Learn preferences and adjust UI behavior

## 🐛 Troubleshooting

### Common Issues

1. **UI not updating**: Ensure the update queue is being processed
2. **Notifications not showing**: Check notification duration settings
3. **Panels not appearing**: Verify panel IDs are unique
4. **Progress stuck**: Ensure progress updates are being sent

### Debug Mode

Run with debug logging:
```bash
python multi_turn_agent_interactive_tui.py --debug
```

## 🚀 Future Enhancements

- [ ] Mouse support for clickable elements
- [ ] Drag-and-drop panel rearrangement
- [ ] Real-time collaborative editing
- [ ] Voice input/output integration
- [ ] Web-based terminal emulator
- [ ] Plugin system for custom UI components
- [ ] Recorded session playback
- [ ] Multi-agent UI coordination

## 📄 License

MIT License - See LICENSE file

---

Experience the future of AI interfaces where the agent is in control! 🤖✨