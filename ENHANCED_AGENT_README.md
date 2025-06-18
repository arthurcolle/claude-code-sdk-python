# Enhanced Multi-turn Agent with Persistent State

An advanced conversational agent built on top of the original multi-turn agent, adding persistent state management, conversational retrieval, and dynamic direction control.

## Key Features

### 1. **Persistent State Management**
- **SQLite** for conversation metadata and message storage
- **DuckDB** for analytics and complex queries
- Session persistence across restarts
- Checkpoint and recovery system

### 2. **Advanced Conversational Retrieval**
- Semantic search using sentence embeddings (all-MiniLM-L6-v2)
- Context-aware message retrieval
- Automatic relevant context injection
- Indexed search for performance

### 3. **Dynamic Direction Control**
- Multiple conversation modes:
  - `EXPLORING` - Open-ended discovery
  - `FOCUSED` - Task-oriented execution
  - `DEBUGGING` - Problem-solving mode
  - `CREATING` - Creative/generative tasks
  - `REVIEWING` - Analysis and refinement
  - `PIVOTING` - Transitioning between modes
- Automatic pivot detection with `--pivot` flag
- Manual direction changes via commands

### 4. **Goal Management**
- Track active and completed goals
- Goal-oriented conversation flow
- Progress tracking and reporting

### 5. **FastAPI State Management API**
- RESTful endpoints for state inspection
- Message search API
- Analytics dashboard
- Checkpoint management

## Installation

```bash
# Install dependencies
pip install openai fastapi uvicorn sentence-transformers duckdb numpy

# For development
pip install pytest pytest-asyncio
```

## Usage

### Basic Interactive Mode

```bash
python multi_turn_agent_enhanced.py
```

### Resume Previous Session

```bash
python multi_turn_agent_enhanced.py --session <session-id>
```

### Enable Automatic Pivoting

```bash
python multi_turn_agent_enhanced.py --pivot
```

### Start API Server

```bash
python multi_turn_agent_enhanced.py --api
```

## Interactive Commands

While in interactive mode:

- `quit` - Exit the session
- `history` - Show recent conversation history
- `goals` - List active and completed goals
  - `goals add <goal>` - Add a new goal
  - `goals complete <goal>` - Mark goal as completed
- `summary` - Show session summary with metrics
- `pivot <direction>` - Change conversation direction
- `clear` - Clear conversation history (preserves state)

## API Endpoints

When running with `--api`:

### Get Session State
```http
GET /sessions/{session_id}
```

### Search Messages
```http
GET /sessions/{session_id}/messages?query=<search-query>&top_k=5&role=user
```

### Create Checkpoint
```http
POST /sessions/{session_id}/checkpoint?name=<checkpoint-name>
```

### Get Analytics
```http
GET /sessions/{session_id}/analytics
```

## Programmatic Usage

```python
from multi_turn_agent_enhanced import StatefulMultiTurnAgent, ConversationDirection

# Create agent
agent = StatefulMultiTurnAgent(
    session_id="my-session",
    system_prompt="You are a helpful coding assistant",
    enable_pivot=True
)

# Send messages
response = await agent.send_user(
    "Help me build a web scraper",
    use_retrieval=True,  # Enable context retrieval
    pivot_on_request=True  # Check for pivot requests
)

# Manage goals
await agent.add_goal("Build web scraper")
await agent.complete_goal("Build web scraper")

# Change direction
await agent.change_direction(
    ConversationDirection.DEBUGGING,
    "User reported issues with scraper"
)

# Get session summary
summary = await agent.get_session_summary()
```

## Stateful Tools

The enhanced agent includes tools that interact with the session state:

```python
from multi_turn_agent_enhanced import register_stateful_tools

# Register state-aware tools
register_stateful_tools(tools_registry, agent)

# Available tools:
# - get_session_info() - Get current session information
# - search_history(query, max_results) - Search conversation history
# - change_focus(new_direction, reason) - Change conversation direction
# - manage_goals(action, goal) - Add/complete/list goals
```

## Database Schema

### SQLite Tables

1. **sessions** - Agent state and metadata
   - session_id (PRIMARY KEY)
   - direction, context_summary, goals, etc.

2. **messages** - Conversation history with embeddings
   - id, session_id, role, content, embedding

3. **checkpoints** - State snapshots for recovery
   - id, session_id, checkpoint_name, state_data

### DuckDB Tables

1. **conversation_analytics** - Performance metrics
   - session_id, turn_count, tool_usage_count, etc.

## Key Differences from Original Agent

1. **Persistence**: All state and messages are saved to disk
2. **Retrieval**: Semantic search for relevant context
3. **Direction**: Explicit conversation mode management
4. **Goals**: Built-in goal tracking system
5. **Analytics**: Detailed metrics and performance tracking
6. **API**: RESTful interface for external integration
7. **Recovery**: Checkpoint system for state recovery

## Testing

Run the enhanced test suite:

```bash
pytest test_multi_turn_agent_enhanced.py -v
```

## Architecture

```
┌─────────────────────┐
│   User Interface    │
│  (CLI/API/Program)  │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ StatefulMultiTurnAgent │
│  - Direction Control │
│  - Goal Management  │
│  - Metrics Tracking │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ PersistentMemory    │
│  - Message Storage  │
│  - Semantic Search  │
│  - Summarization    │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ PersistentStorage   │
│  - SQLite (Meta)    │
│  - DuckDB (Analytics)│
│  - Embeddings       │
└─────────────────────┘
```

## Future Enhancements

1. **Multi-agent collaboration** - Multiple agents sharing state
2. **Advanced analytics** - ML-based conversation insights
3. **Plugin system** - Extensible tool ecosystem
4. **Vector database** - Scalable semantic search
5. **Streaming state updates** - Real-time state synchronization

## Performance Considerations

- Embeddings are computed once and cached
- Indexes on session_id and timestamp for fast queries
- DuckDB for analytical queries without blocking main DB
- Checkpoint system prevents data loss
- Automatic summarization keeps context size manageable

## Migration from Original Agent

The enhanced agent is backward compatible. To migrate:

1. Sessions start fresh (no automatic migration of old conversations)
2. Tools from original agent work without modification
3. Add `storage_path` parameter to enable persistence
4. Use `enable_pivot` for direction change detection