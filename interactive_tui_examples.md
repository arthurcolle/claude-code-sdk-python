# Interactive TUI Examples

These examples demonstrate how the agent can control its own terminal interface.

## Basic UI Control

### 1. Sending Messages
```
You: Send me a formatted message about the weather

Agent might respond with:
- msg_user("☀️ Today's weather looks great!", style="bold yellow")
- msg_user("Temperature: 72°F with clear skies", markdown=True)
```

### 2. Notifications
```
You: Alert me when the analysis is complete

Agent might use:
- notify("Analysis started", type="info")
- (after processing)
- notify("✅ Analysis complete! 150 data points processed", type="success", duration=10)
```

### 3. Showing Thinking Process
```
You: Solve this complex problem: optimize our database queries

Agent might use:
- show_thinking("Analyzing current query patterns...", confidence=0.7)
- show_thinking("Found N+1 query issue in user loader", confidence=0.9)
- show_thinking("Considering batch loading strategy", confidence=0.85)
- msg_user("I've identified several optimization opportunities...")
```

## Advanced UI Features

### 4. Creating Custom Panels
```
You: Show me a dashboard of current system status

Agent might execute:
- create_custom_panel("sys_dashboard", "System Dashboard", """
  CPU Usage: ████░░░░░░ 40%
  Memory:    ███████░░░ 70%
  Disk:      ██░░░░░░░░ 20%
  Network:   ████████░░ 80%
  """)
- update_panel("sys_dashboard", updated_content)
```

### 5. Progress Tracking
```
You: Process this large dataset

Agent might show:
- show_progress("dataset-processing", 0.0, "Starting processing...")
- show_progress("dataset-processing", 0.25, "Loaded 1000/4000 records")
- show_progress("dataset-processing", 0.5, "Data validation complete")
- show_progress("dataset-processing", 0.75, "Applying transformations")
- show_progress("dataset-processing", 1.0, "Processing complete!")
```

### 6. Workflow Visualization
```
You: Create a deployment workflow

Agent might use:
- show_workflow("Deployment Pipeline", [
    {"id": "1", "name": "Run Tests", "status": "completed"},
    {"id": "2", "name": "Build Docker Image", "status": "running"},
    {"id": "3", "name": "Push to Registry", "status": "pending"},
    {"id": "4", "name": "Deploy to Staging", "status": "pending"},
    {"id": "5", "name": "Run Smoke Tests", "status": "pending"}
  ])
```

### 7. Interactive Confirmations
```
You: Delete all temporary files

Agent might use:
- request_confirmation("Are you sure you want to delete 47 temporary files?", 
                     options=["Yes, delete all", "No, keep them", "Show me the list first"])
```

### 8. Data Visualization
```
You: Show me the performance metrics

Agent might use:
- show_chart("bar", {
    "title": "Response Times (ms)",
    "data": {
      "API Calls": 45,
      "Database": 120,
      "Cache": 5,
      "External Services": 200
    }
  }, "Performance Breakdown")
```

## Complex Interactions

### 9. Multi-Step Process with UI Updates
```
You: Analyze and optimize this codebase

Agent orchestrates:
1. notify("Starting codebase analysis", type="info")
2. create_custom_panel("analysis_stats", "Analysis Progress", "Scanning...")
3. show_thinking("Detecting code patterns and potential issues", confidence=0.8)
4. show_progress("analysis", 0.3, "Found 1,247 files to analyze")
5. update_panel("analysis_stats", "Files: 1,247\nIssues: 23\nSuggestions: 45")
6. show_progress("analysis", 0.6, "Running static analysis")
7. notify("⚠️ Found 3 critical issues", type="warning")
8. show_progress("analysis", 1.0, "Analysis complete")
9. msg_user("## Analysis Complete\n\nHere's what I found...", markdown=True)
10. export_view("markdown", "codebase_analysis_report.md")
```

### 10. Environment-Aware Responses
```
You: Monitor system resources and alert me to issues

Agent sets up:
1. create_custom_panel("resource_monitor", "Resource Monitor", "Initializing...")
2. (Periodically updates panel with live data)
3. When memory > 90%:
   - notify("⚠️ High memory usage detected!", type="warning")
   - show_thinking("Memory spike detected, analyzing cause...", confidence=0.9)
   - msg_user("I've detected high memory usage. Here are my recommendations...")
```

## UI Control Commands

The agent can also respond to direct UI commands:

```
You: /agent clear_chat
You: /agent focus_panel tasks
You: /agent set_theme dark
You: /agent export_view json my_session.json
```

## Creative Uses

### 11. Interactive Tutorials
```
You: Teach me about async programming

Agent creates an interactive tutorial:
1. create_custom_panel("code_example", "Example Code", initial_code)
2. msg_user("Let's start with the basics of async/await...")
3. highlight("async def", color="yellow", panel="code_example")
4. show_thinking("The user might be confused about event loops", confidence=0.7)
5. update_panel("code_example", updated_code_with_comments)
6. request_confirmation("Ready for the next example?", ["Yes", "Explain this again"])
```

### 12. Real-time Collaboration
```
You: Let's work on this problem together

Agent enables collaboration:
1. create_custom_panel("workspace", "Shared Workspace", "")
2. create_custom_panel("agent_notes", "My Notes", "")
3. msg_user("I've set up a shared workspace. You tell me your ideas, and I'll add my analysis!")
4. (As conversation progresses, updates both panels)
5. show_thinking("The user's approach is interesting, let me build on it", confidence=0.85)
```

## Best Practices for Agent UI Control

1. **Be Contextual**: Use appropriate UI elements for the task
2. **Show Progress**: For long operations, always show progress
3. **Think Out Loud**: Use show_thinking() to make reasoning transparent
4. **Confirm Dangerous Actions**: Always use request_confirmation() for destructive operations
5. **Visualize Complex Data**: Use charts and custom panels for better understanding
6. **Export Important Results**: Offer to export significant findings or conversations
7. **Adapt to User Preferences**: Remember if users prefer certain themes or styles

## Example Conversation Flow

```
User: Help me debug this performance issue

Agent:
1. msg_user("I'll help you debug the performance issue. Let me set up a debugging workspace.")
2. create_custom_panel("metrics", "Performance Metrics", "Collecting...")
3. notify("Starting performance analysis", type="info")
4. show_thinking("I should first check for common bottlenecks", confidence=0.8)
5. show_progress("perf_analysis", 0.2, "Analyzing database queries")
6. update_panel("metrics", "Slow Queries: 5\nN+1 Issues: 2\nMissing Indexes: 3")
7. highlight("SELECT * FROM users", color="red", panel="metrics")
8. msg_user("I've found several performance bottlenecks. The most critical is...")
9. show_workflow("Optimization Plan", [optimization_steps])
10. request_confirmation("Shall I generate the optimization code?", ["Yes", "Show me details first"])
```

This creates a rich, interactive debugging session where the agent actively uses the UI to communicate findings and guide the process.