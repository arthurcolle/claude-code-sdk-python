# Agent System Integration Guide

The Claude Code SDK can be integrated with the Claude Code Agent System to create sophisticated multi-agent applications. This guide explains how to use the SDK within the agent framework and build complex AI workflows.

## Overview

The Agent System is a separate package that builds on top of the Claude Code SDK, providing:

- **Multi-Agent Architecture**: Specialized agents for different tasks
- **Task Orchestration**: Intelligent routing and coordination
- **Tool Management**: Dynamic tool creation and execution
- **State Persistence**: Maintain context across agent interactions
- **Scalable Execution**: Distributed agent processing

## Installation

The agent system will be available as a separate package:

```bash
# Install the agent system (includes SDK as dependency)
pip install claude-code-agent-system

# Or install with full features
pip install claude-code-agent-system[full]
```

For development, install from the repository:

```bash
cd agent_system
pip install -e .
```

## Integration Patterns

### Pattern 1: SDK-Enhanced Agents

Replace direct CLI calls with SDK integration:

```python
from claude_code_sdk import query, ClaudeCodeOptions
from claude_code_sdk.types import AssistantMessage, TextBlock
from claude_code_agent_system import BaseAgent

class SDKEnhancedAgent(BaseAgent):
    """Base class for agents that use Claude Code SDK."""
    
    async def query_claude(
        self, 
        prompt: str, 
        tools: list[str] = None,
        **kwargs
    ) -> str:
        """Query Claude using the SDK."""
        options = ClaudeCodeOptions(
            allowed_tools=tools or [],
            max_thinking_tokens=kwargs.get('max_thinking_tokens', 8000),
            permission_mode=kwargs.get('permission_mode', 'default')
        )
        
        response = ""
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response += block.text
        
        return response
```

### Pattern 2: Tool-Aware Agents

Create agents that can dynamically use Claude's tools:

```python
class ToolAwareCoderAgent(SDKEnhancedAgent):
    """Coder agent that intelligently uses tools."""
    
    async def generate_code_with_context(
        self,
        task_description: str,
        project_path: str
    ) -> dict:
        """Generate code with full project context."""
        
        # First, explore the project structure
        exploration_prompt = f"""
        Explore the project at {project_path} and understand its structure.
        Look for existing patterns, dependencies, and coding standards.
        """
        
        context = await self.query_claude(
            exploration_prompt,
            tools=["Read", "Glob", "Grep"]
        )
        
        # Then generate code following discovered patterns
        generation_prompt = f"""
        Based on the project analysis:
        {context}
        
        Now implement: {task_description}
        
        Follow the existing code patterns and style.
        """
        
        code = await self.query_claude(
            generation_prompt,
            tools=["Write", "Edit", "MultiEdit"]
        )
        
        return {
            "code": code,
            "context": context,
            "files_created": self.extract_created_files(code)
        }
```

### Pattern 3: Multi-Agent Workflows

Coordinate multiple SDK-enhanced agents:

```python
from claude_code_agent_system import Workflow, OrchestratorAgent

class SDKOrchestratorAgent(OrchestratorAgent):
    """Orchestrator that coordinates SDK-enhanced agents."""
    
    async def plan_and_execute(self, task: dict) -> dict:
        """Plan and execute a complex task."""
        
        # Use Claude to create an execution plan
        planning_prompt = f"""
        Create a detailed execution plan for: {task['description']}
        
        Break it down into steps that can be handled by:
        - Researcher: For gathering information
        - Coder: For implementation
        - Validator: For testing
        - Executor: For running code
        
        Return a structured plan.
        """
        
        plan = await self.query_claude(
            planning_prompt,
            tools=["Task"],  # Use Task tool for planning
            max_thinking_tokens=15000
        )
        
        # Execute the plan
        workflow = Workflow("SDKWorkflow")
        
        for step in self.parse_plan(plan):
            workflow.add_step(
                name=step['name'],
                agent=step['agent'],
                inputs=step['inputs'],
                depends_on=step.get('depends_on', [])
            )
        
        return await workflow.execute()
```

## Real-World Examples

### Example 1: Full-Stack Application Generator

```python
from claude_code_agent_system import AgentSystem
from claude_code_sdk import ClaudeCodeOptions

async def generate_full_stack_app(requirements: dict):
    """Generate a complete full-stack application."""
    
    # Initialize the agent system
    system = AgentSystem()
    
    # Create specialized agents with SDK integration
    researcher = SDKResearcherAgent()
    architect = SDKArchitectAgent()
    frontend_dev = SDKFrontendAgent()
    backend_dev = SDKBackendAgent()
    devops = SDKDevOpsAgent()
    
    # Register agents
    system.register_agent(researcher)
    system.register_agent(architect)
    system.register_agent(frontend_dev)
    system.register_agent(backend_dev)
    system.register_agent(devops)
    
    # Submit the task
    task_result = await system.submit_task({
        "type": "full_stack_app",
        "requirements": requirements,
        "workflow": [
            {"agent": "researcher", "action": "analyze_requirements"},
            {"agent": "architect", "action": "design_architecture"},
            {"agent": "backend_dev", "action": "implement_api"},
            {"agent": "frontend_dev", "action": "build_ui"},
            {"agent": "devops", "action": "setup_deployment"}
        ]
    })
    
    return task_result

# Usage
app_requirements = {
    "name": "TaskManager",
    "features": ["user auth", "task CRUD", "real-time updates"],
    "tech_stack": {
        "frontend": "React with TypeScript",
        "backend": "FastAPI",
        "database": "PostgreSQL"
    }
}

result = await generate_full_stack_app(app_requirements)
```

### Example 2: Code Review and Refactoring System

```python
class CodeReviewAgent(SDKEnhancedAgent):
    """Agent specialized in code review and refactoring."""
    
    async def review_and_refactor(self, file_path: str) -> dict:
        """Review code and suggest/implement improvements."""
        
        # Step 1: Analyze the code
        analysis = await self.query_claude(
            f"Analyze the code in {file_path} for potential improvements",
            tools=["Read", "Grep"]
        )
        
        # Step 2: Generate refactoring plan
        plan = await self.query_claude(
            f"Based on this analysis: {analysis}\n"
            f"Create a refactoring plan with specific improvements",
            max_thinking_tokens=10000
        )
        
        # Step 3: Implement refactoring
        result = await self.query_claude(
            f"Implement the refactoring plan: {plan}",
            tools=["Edit", "MultiEdit"],
            permission_mode="acceptEdits"
        )
        
        # Step 4: Validate changes
        validation = await self.query_claude(
            "Validate that the refactored code maintains functionality",
            tools=["Read", "Bash"]
        )
        
        return {
            "analysis": analysis,
            "plan": plan,
            "changes": result,
            "validation": validation
        }
```

### Example 3: Intelligent Documentation Generator

```python
class DocumentationAgent(SDKEnhancedAgent):
    """Generate comprehensive documentation using SDK."""
    
    async def document_codebase(self, project_path: str) -> dict:
        """Generate full documentation for a codebase."""
        
        # Discover all code files
        files = await self.query_claude(
            f"Find all Python files in {project_path}",
            tools=["Glob", "LS"]
        )
        
        docs = {}
        
        for file in self.parse_file_list(files):
            # Analyze each file
            analysis = await self.query_claude(
                f"Analyze {file} and extract functions, classes, and their purposes",
                tools=["Read", "Grep"]
            )
            
            # Generate documentation
            doc = await self.query_claude(
                f"Generate comprehensive documentation for: {analysis}",
                max_thinking_tokens=5000
            )
            
            # Write documentation
            await self.query_claude(
                f"Write the documentation to appropriate files",
                tools=["Write", "Edit"],
                permission_mode="acceptEdits"
            )
            
            docs[file] = doc
        
        # Generate overall documentation
        overview = await self.query_claude(
            "Generate project overview documentation based on all files",
            tools=["Write"]
        )
        
        return {
            "file_docs": docs,
            "overview": overview,
            "files_documented": len(docs)
        }
```

## Advanced Integration Techniques

### Custom Tool Integration

Create custom tools that agents can use:

```python
from claude_code_agent_system import ToolRegistry

class DatabaseAgent(SDKEnhancedAgent):
    """Agent with custom database tools."""
    
    def __init__(self):
        super().__init__()
        self.register_custom_tools()
    
    def register_custom_tools(self):
        """Register database-specific tools."""
        registry = ToolRegistry()
        
        registry.create_tool({
            "name": "DatabaseQuery",
            "description": "Execute SQL queries safely",
            "parameters": {
                "query": {"type": "string", "required": True},
                "database": {"type": "string", "required": True}
            },
            "implementation": self.execute_sql_query
        })
        
        registry.create_tool({
            "name": "SchemaAnalyzer",
            "description": "Analyze database schema",
            "parameters": {
                "database": {"type": "string", "required": True}
            },
            "implementation": self.analyze_schema
        })
    
    async def execute_sql_query(self, query: str, database: str) -> str:
        """Safely execute SQL queries."""
        # Implementation here
        pass
    
    async def optimize_database(self, connection_string: str) -> dict:
        """Use Claude with custom tools to optimize database."""
        
        # First analyze the schema
        schema_analysis = await self.query_claude(
            "Analyze the database schema and identify optimization opportunities",
            tools=["SchemaAnalyzer", "DatabaseQuery"]
        )
        
        # Generate optimization plan
        optimization_plan = await self.query_claude(
            f"Based on schema analysis: {schema_analysis}\n"
            "Create an optimization plan",
            max_thinking_tokens=10000
        )
        
        return {
            "analysis": schema_analysis,
            "plan": optimization_plan
        }
```

### State Management

Maintain state across agent interactions:

```python
from claude_code_agent_system import AgentState

class StatefulProjectAgent(SDKEnhancedAgent):
    """Agent that maintains project state."""
    
    def __init__(self):
        super().__init__()
        self.state = AgentState()
    
    async def initialize_project(self, project_path: str):
        """Initialize and analyze project state."""
        
        # Analyze project structure
        structure = await self.query_claude(
            f"Analyze the complete structure of {project_path}",
            tools=["Read", "Glob", "Grep"]
        )
        
        # Store in state
        self.state.set("project_structure", structure)
        self.state.set("project_path", project_path)
        
        # Identify key components
        components = await self.query_claude(
            "Identify key components, dependencies, and patterns",
            tools=["Read", "Grep"]
        )
        
        self.state.set("components", components)
        
        return self.state.to_dict()
    
    async def make_consistent_changes(self, changes: list[dict]):
        """Make changes while maintaining consistency."""
        
        # Retrieve state
        structure = self.state.get("project_structure")
        components = self.state.get("components")
        
        for change in changes:
            # Use state information to ensure consistency
            result = await self.query_claude(
                f"Implement change: {change}\n"
                f"Ensure consistency with: {components}\n"
                f"Project structure: {structure}",
                tools=["Edit", "MultiEdit"],
                permission_mode="acceptEdits"
            )
            
            # Update state with changes
            self.state.append("changes_made", result)
```

### Error Handling and Recovery

Implement robust error handling:

```python
class ResilientAgent(SDKEnhancedAgent):
    """Agent with advanced error handling."""
    
    async def execute_with_recovery(
        self,
        task: dict,
        max_retries: int = 3
    ) -> dict:
        """Execute task with automatic recovery."""
        
        for attempt in range(max_retries):
            try:
                # Try to execute the task
                result = await self.query_claude(
                    task["prompt"],
                    tools=task.get("tools", []),
                    **task.get("options", {})
                )
                
                # Validate result
                validation = await self.query_claude(
                    f"Validate this result: {result}",
                    tools=["Read", "Bash"]
                )
                
                if "error" not in validation.lower():
                    return {
                        "status": "success",
                        "result": result,
                        "attempts": attempt + 1
                    }
                
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    # Ask Claude to diagnose and fix
                    recovery_plan = await self.query_claude(
                        f"The previous attempt failed with: {e}\n"
                        "Diagnose the issue and suggest a fix",
                        max_thinking_tokens=5000
                    )
                    
                    # Modify task based on recovery plan
                    task = self.modify_task_for_recovery(task, recovery_plan)
                else:
                    return {
                        "status": "failed",
                        "error": str(e),
                        "attempts": attempt + 1
                    }
```

## Performance Optimization

### Parallel Agent Execution

```python
import asyncio

class ParallelOrchestrator(SDKOrchestratorAgent):
    """Orchestrator that runs agents in parallel."""
    
    async def execute_parallel_tasks(self, tasks: list[dict]) -> dict:
        """Execute multiple tasks in parallel."""
        
        # Group tasks by agent
        agent_tasks = {}
        for task in tasks:
            agent = task["agent"]
            if agent not in agent_tasks:
                agent_tasks[agent] = []
            agent_tasks[agent].append(task)
        
        # Execute in parallel
        results = await asyncio.gather(*[
            self.execute_agent_tasks(agent, tasks)
            for agent, tasks in agent_tasks.items()
        ])
        
        return {
            "results": results,
            "execution_time": self.calculate_execution_time()
        }
```

### Caching and Memoization

```python
from functools import lru_cache
import hashlib

class CachedAgent(SDKEnhancedAgent):
    """Agent with response caching."""
    
    def __init__(self):
        super().__init__()
        self.cache = {}
    
    def _get_cache_key(self, prompt: str, options: dict) -> str:
        """Generate cache key for prompt and options."""
        key_data = f"{prompt}:{str(sorted(options.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def query_claude_cached(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """Query Claude with caching."""
        
        cache_key = self._get_cache_key(prompt, kwargs)
        
        if cache_key in self.cache:
            self.logger.info("Using cached response")
            return self.cache[cache_key]
        
        response = await self.query_claude(prompt, **kwargs)
        self.cache[cache_key] = response
        
        return response
```

## Monitoring and Debugging

### Agent Performance Monitoring

```python
from claude_code_agent_system import AgentMonitor

class MonitoredAgent(SDKEnhancedAgent):
    """Agent with built-in monitoring."""
    
    def __init__(self):
        super().__init__()
        self.monitor = AgentMonitor()
    
    async def query_claude(self, prompt: str, **kwargs) -> str:
        """Query Claude with monitoring."""
        
        # Start monitoring
        self.monitor.start_operation("claude_query")
        
        try:
            response = await super().query_claude(prompt, **kwargs)
            
            # Record metrics
            self.monitor.record_success("claude_query")
            self.monitor.record_tokens_used(
                self.estimate_tokens(prompt + response)
            )
            
            return response
            
        except Exception as e:
            self.monitor.record_failure("claude_query", str(e))
            raise
        
        finally:
            self.monitor.end_operation("claude_query")
```

## Best Practices

1. **Use Type Hints**: Ensure all agent methods have proper type hints
2. **Handle Errors Gracefully**: Always implement error recovery
3. **Log Important Operations**: Use structured logging for debugging
4. **Validate Inputs and Outputs**: Ensure data integrity
5. **Implement Timeouts**: Prevent hanging operations
6. **Cache When Possible**: Reduce redundant API calls
7. **Monitor Resource Usage**: Track token usage and costs
8. **Test Agent Interactions**: Write comprehensive tests

## Migration Guide

### Migrating from CLI-based Agents

```python
# Before: CLI-based agent
class OldCoderAgent(BaseAgent):
    async def generate_code(self, prompt: str) -> str:
        result = subprocess.run(
            ["claude", "query", prompt],
            capture_output=True
        )
        return result.stdout.decode()

# After: SDK-based agent
class NewCoderAgent(SDKEnhancedAgent):
    async def generate_code(self, prompt: str) -> str:
        return await self.query_claude(
            prompt,
            tools=["Write", "Edit"],
            max_thinking_tokens=10000
        )
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure both SDK and agent system are installed
2. **Tool Not Found**: Verify tool names match Claude's available tools
3. **Rate Limiting**: Implement proper backoff and retry logic
4. **Memory Issues**: Use streaming for large responses
5. **Timeout Errors**: Adjust timeout settings for long operations

### Debug Mode

Enable detailed debugging:

```python
import logging

# Enable debug logging for agents
logging.getLogger("claude_code_agent_system").setLevel(logging.DEBUG)
logging.getLogger("claude_code_sdk").setLevel(logging.DEBUG)

# Or for specific agent
agent = SDKEnhancedAgent()
agent.logger.setLevel(logging.DEBUG)
```

## Further Resources

- [Agent System README](../agent_system/README.md)
- [SDK Documentation](../README.md)
- [Example Applications](../examples/)
- [API Reference](./api-reference.md)