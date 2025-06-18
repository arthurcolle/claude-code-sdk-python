# Dynamic Multi-Agent System

A flexible, runtime-configurable multi-agent framework for the Claude Code SDK that enables dynamic agent creation, task distribution, and intelligent orchestration.

## Overview

The Dynamic Multi-Agent System provides:

- **Dynamic Agent Spawning**: Create agents on-demand based on task requirements
- **Capability-Based Task Routing**: Automatically match tasks to agents with appropriate skills
- **Agent Registry**: Pre-defined and custom agent profiles
- **Inter-Agent Communication**: Message bus for agent coordination
- **Lifecycle Management**: Spawn, monitor, and terminate agents as needed
- **Adaptive Scaling**: Automatically scale agent pool based on workload

## Quick Start

```python
from claude_code_sdk.dynamic_multi_agent import (
    DynamicMultiAgentCoordinator,
    AgentCapability,
    create_development_team
)

# Create coordinator
coordinator = DynamicMultiAgentCoordinator()
await coordinator.start()

# Spawn a development team
team = await create_development_team(coordinator)

# Submit a task
task_id = await coordinator.create_and_submit_task(
    description="Implement user authentication",
    required_capabilities={AgentCapability.CODE_GENERATION},
    priority=8
)

# Check status
status = await coordinator.get_system_status()
print(f"Active agents: {len(status['agents'])}")
print(f"Completed tasks: {status['completed_tasks']}")
```

## Core Components

### Agent Capabilities

Pre-defined capabilities that agents can possess:

- `CODE_GENERATION` - Writing code
- `CODE_REVIEW` - Reviewing code quality
- `TESTING` - Writing and running tests
- `DOCUMENTATION` - Creating documentation
- `DEBUGGING` - Finding and fixing bugs
- `ARCHITECTURE` - System design
- `DATA_ANALYSIS` - Analyzing data
- `RESEARCH` - Researching solutions
- `PLANNING` - Project planning
- `COORDINATION` - Task coordination
- `SECURITY` - Security analysis
- `PERFORMANCE` - Performance optimization
- `UI_UX` - User interface design
- `DEVOPS` - Deployment and operations
- `DATABASE` - Database design

### Agent Profiles

Pre-configured agent templates:

```python
# Available profiles
profiles = coordinator.registry.list_profiles()
# Includes: developer, reviewer, tester, architect, 
# documenter, devops, coordinator
```

### Custom Agent Profiles

Create specialized agents:

```python
from claude_code_sdk.dynamic_multi_agent import AgentProfile

custom_profile = AgentProfile(
    name="Full-Stack Developer",
    capabilities={
        AgentCapability.CODE_GENERATION,
        AgentCapability.UI_UX,
        AgentCapability.DATABASE
    },
    tools=["Read", "Write", "Edit", "Bash"],
    max_concurrent_tasks=4,
    priority=8,
    system_prompt="You are a full-stack developer..."
)

agent_id = await coordinator.spawn_agent(profile=custom_profile)
```

## Task Management

### Creating Tasks

```python
from claude_code_sdk.dynamic_multi_agent import DynamicTask

# Simple task creation
task_id = await coordinator.create_and_submit_task(
    description="Write unit tests for auth module",
    required_capabilities={AgentCapability.TESTING},
    priority=7,
    context={"module": "authentication", "coverage_target": 90}
)

# Advanced task with dependencies
task = DynamicTask(
    id="task-123",
    description="Complex feature implementation",
    required_capabilities={AgentCapability.CODE_GENERATION, AgentCapability.TESTING},
    priority=9,
    dependencies=["task-100", "task-101"],  # Wait for these tasks
    deadline=datetime.now() + timedelta(hours=2)
)
await coordinator.submit_task(task)
```

### Task States

- `pending` - Waiting for assignment
- `in_progress` - Being processed by an agent
- `completed` - Successfully finished

## Agent Communication

### Message Bus

Agents can communicate through the built-in message bus:

```python
# Subscribe to events
def on_task_completed(data):
    print(f"Task {data['task_id']} completed by {data['agent_id']}")

coordinator.message_bus.subscribe("task_completed", on_task_completed)

# Available events:
# - agent_spawned
# - agent_terminated
# - task_submitted
# - task_assigned
# - task_completed
```

## Team Patterns

### Pre-built Teams

```python
# Development team (architect, developers, reviewer, tester, documenter)
dev_team = await create_development_team(coordinator)

# Code review pipeline (reviewer, tester, security reviewer)
review_team = await create_review_pipeline(coordinator)

# Research team (lead, analysts)
research_team = await create_research_team(coordinator)
```

### Dynamic Team Creation

```python
# Spawn agents for specific capabilities
agents = await coordinator.spawn_agents_for_capabilities(
    required_capabilities={AgentCapability.CODE_GENERATION, AgentCapability.TESTING},
    count=3
)
```

## Advanced Features

### Adaptive Scaling

The system can automatically spawn new agents when workload increases:

```python
# If no suitable agent exists for a task, one will be created
task_id = await coordinator.create_and_submit_task(
    description="Analyze ML model performance",
    required_capabilities={AgentCapability.DATA_ANALYSIS, AgentCapability.RESEARCH}
)
# System automatically spawns an agent with these capabilities
```

### Agent Lifecycle Management

```python
# Monitor agent performance
status = await coordinator.get_system_status()
for agent_id, info in status["agents"].items():
    print(f"{info['name']}: {info['capacity']} capacity")

# Terminate underutilized agents
await coordinator.terminate_agent(agent_id)
# Tasks are automatically requeued
```

### Custom Agent Types

Extend the base agent class:

```python
from claude_code_sdk.dynamic_multi_agent import DynamicAgent

class SpecializedAgent(DynamicAgent):
    async def process_task(self, task):
        # Custom processing logic
        result = await self.specialized_processing(task)
        return result
    
    async def specialized_processing(self, task):
        # Your implementation
        pass

# Register custom agent type
coordinator.registry.register_agent_type("specialized", SpecializedAgent)
```

## Examples

### Example 1: Basic Task Distribution

```python
# Spawn agents
dev = await coordinator.spawn_agent(profile="developer")
tester = await coordinator.spawn_agent(profile="tester")

# Submit tasks - automatically routed to appropriate agents
await coordinator.create_and_submit_task(
    "Implement login feature",
    {AgentCapability.CODE_GENERATION}
)
await coordinator.create_and_submit_task(
    "Write login tests",
    {AgentCapability.TESTING}
)
```

### Example 2: Complex Project Workflow

```python
# Define project phases
phases = [
    ("Design API architecture", {AgentCapability.ARCHITECTURE}, 10),
    ("Implement services", {AgentCapability.CODE_GENERATION}, 8),
    ("Write tests", {AgentCapability.TESTING}, 7),
    ("Security review", {AgentCapability.SECURITY, AgentCapability.CODE_REVIEW}, 9),
    ("Create documentation", {AgentCapability.DOCUMENTATION}, 6)
]

# Submit all tasks
for description, capabilities, priority in phases:
    await coordinator.create_and_submit_task(
        description, capabilities, priority
    )

# System handles agent allocation and task execution
```

### Example 3: Event-Driven Coordination

```python
# Track progress
completed_count = 0

async def on_task_completed(data):
    global completed_count
    completed_count += 1
    
    # Trigger follow-up tasks
    if data["task_id"].startswith("test-"):
        await coordinator.create_and_submit_task(
            f"Deploy {data['task_id']}",
            {AgentCapability.DEVOPS}
        )

coordinator.message_bus.subscribe("task_completed", on_task_completed)
```

## Best Practices

1. **Capability Design**: Keep capabilities focused and composable
2. **Task Granularity**: Break large tasks into smaller, focused units
3. **Priority Management**: Use priorities to ensure critical tasks complete first
4. **Resource Monitoring**: Monitor agent utilization and scale accordingly
5. **Error Handling**: Implement retry logic for failed tasks

## Performance Considerations

- Agents process tasks concurrently up to their `max_concurrent_tasks` limit
- Task assignment considers both capability match and current agent load
- The message bus operates asynchronously to prevent blocking
- Agent spawning is lightweight - create agents as needed

## Integration with Claude Code SDK

The dynamic multi-agent system fully integrates with Claude Code SDK:

```python
# Agents use Claude Code options
profile = AgentProfile(
    name="Custom Agent",
    capabilities={AgentCapability.CODE_GENERATION},
    tools=["Read", "Write", "Edit", "Bash"],
    model="claude-3.5-sonnet",  # Specific model
    system_prompt="Custom instructions..."
)
```

## Troubleshooting

### Common Issues

1. **No suitable agent found**
   - Check task capabilities match available agents
   - Consider using `spawn_agents_for_capabilities()`

2. **Tasks stuck in pending**
   - Verify agents have capacity
   - Check for circular dependencies

3. **Agent not accepting tasks**
   - Agent may be at max capacity
   - Check agent is still active

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Detailed logs will show:
# - Agent spawning
# - Task assignment decisions
# - Message bus activity
# - Agent lifecycle events
```

## Future Enhancements

Planned features:
- Persistent task queue with recovery
- Agent performance metrics and learning
- Distributed agent execution
- Task result caching
- Advanced scheduling algorithms

## Contributing

When extending the multi-agent system:

1. Follow the existing patterns for agents and tasks
2. Ensure new capabilities are well-defined
3. Add appropriate tests
4. Document new agent profiles

## License

Part of the Claude Code SDK - see main project license.