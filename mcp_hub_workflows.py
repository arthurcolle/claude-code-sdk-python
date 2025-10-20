#!/usr/bin/env python3
"""
MCP Hub Workflow Engine

Supports:
- DAG (Directed Acyclic Graph) execution
- Conditional branching
- Parallel execution paths
- Error handling and retries
- Workflow templates
- Event-driven execution
- Workflow versioning
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Callable, Union, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import defaultdict
import uuid


class StepStatus(Enum):
    """Status of a workflow step"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class WorkflowStatus(Enum):
    """Status of entire workflow"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """A single step in a workflow"""
    step_id: str
    tool_name: str
    parameters: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)  # Step IDs this depends on
    condition: Optional[str] = None  # Python expression to evaluate
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: Optional[int] = None
    on_error: str = "fail"  # "fail", "skip", "retry"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepExecution:
    """Record of a step execution"""
    step_id: str
    execution_id: str
    status: StepStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['status'] = self.status.value
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data


@dataclass
class Workflow:
    """A complete workflow definition"""
    workflow_id: str
    name: str
    steps: List[WorkflowStep]
    description: str = ""
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate workflow definition"""
        # Check for cycles
        if self._has_cycle():
            return False, "Workflow contains circular dependencies"

        # Check that dependencies exist
        step_ids = {step.step_id for step in self.steps}
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    return False, f"Step {step.step_id} depends on non-existent step {dep}"

        return True, None

    def _has_cycle(self) -> bool:
        """Check if workflow has circular dependencies (cycle detection)"""
        graph = defaultdict(list)
        for step in self.steps:
            for dep in step.depends_on:
                graph[dep].append(step.step_id)

        visited = set()
        rec_stack = set()

        def has_cycle_util(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    if has_cycle_util(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for step in self.steps:
            if step.step_id not in visited:
                if has_cycle_util(step.step_id):
                    return True

        return False

    def get_execution_order(self) -> List[List[str]]:
        """Get topological order for execution (layers that can run in parallel)"""
        # Build dependency graph
        graph = defaultdict(set)
        in_degree = defaultdict(int)

        all_steps = {step.step_id for step in self.steps}

        for step in self.steps:
            if step.step_id not in in_degree:
                in_degree[step.step_id] = 0
            for dep in step.depends_on:
                graph[dep].add(step.step_id)
                in_degree[step.step_id] += 1

        # Kahn's algorithm for topological sort with levels
        layers = []
        queue = [sid for sid in all_steps if in_degree[sid] == 0]

        while queue:
            # All steps in current layer can run in parallel
            layers.append(queue.copy())

            next_layer = []
            for node in queue:
                for neighbor in graph[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_layer.append(neighbor)

            queue = next_layer

        return layers

    def to_dict(self) -> Dict[str, Any]:
        return {
            'workflow_id': self.workflow_id,
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'steps': [step.to_dict() for step in self.steps],
            'tags': self.tags,
            'metadata': self.metadata
        }


@dataclass
class WorkflowExecution:
    """Track execution of a workflow"""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    step_executions: Dict[str, StepExecution] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)  # Shared context
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'execution_id': self.execution_id,
            'workflow_id': self.workflow_id,
            'status': self.status.value,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'step_executions': {k: v.to_dict() for k, v in self.step_executions.items()},
            'context': self.context,
            'error': self.error
        }


class WorkflowEngine:
    """Execute workflows with DAG support"""

    def __init__(self, tool_executor: Callable):
        """
        Args:
            tool_executor: Async function that executes a tool
                          Should have signature: async def(tool_name, params) -> result
        """
        self.tool_executor = tool_executor
        self.workflows: Dict[str, Workflow] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.templates: Dict[str, Workflow] = {}

    def register_workflow(self, workflow: Workflow) -> Tuple[bool, Optional[str]]:
        """Register a workflow"""
        is_valid, error = workflow.validate()
        if not is_valid:
            return False, error

        self.workflows[workflow.workflow_id] = workflow
        return True, None

    def register_template(self, name: str, workflow: Workflow):
        """Register a reusable workflow template"""
        self.templates[name] = workflow

    def create_from_template(self, template_name: str, **params) -> Optional[Workflow]:
        """Create a workflow instance from template"""
        template = self.templates.get(template_name)
        if not template:
            return None

        # Clone template and customize with params
        workflow = Workflow(
            workflow_id=str(uuid.uuid4()),
            name=f"{template.name} - {datetime.now().isoformat()}",
            steps=[WorkflowStep(**asdict(step)) for step in template.steps],
            description=template.description,
            version=template.version,
            metadata={**template.metadata, 'template': template_name, **params}
        )

        return workflow

    async def execute_workflow(self, workflow_id: str,
                               initial_context: Optional[Dict[str, Any]] = None) -> WorkflowExecution:
        """Execute a workflow"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")

        execution_id = str(uuid.uuid4())
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            started_at=datetime.now(),
            context=initial_context or {}
        )
        self.executions[execution_id] = execution

        try:
            # Get execution order (parallel layers)
            layers = workflow.get_execution_order()

            # Execute each layer
            for layer in layers:
                # Get steps for this layer
                layer_steps = [s for s in workflow.steps if s.step_id in layer]

                # Execute all steps in layer in parallel
                tasks = [
                    self._execute_step(step, execution)
                    for step in layer_steps
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

                # Check if any critical failures
                failed_steps = [
                    step_id for step_id in layer
                    if execution.step_executions[step_id].status == StepStatus.FAILED
                ]

                if failed_steps:
                    # Check if we should continue
                    should_fail = any(
                        s.on_error == "fail"
                        for s in layer_steps
                        if s.step_id in failed_steps
                    )
                    if should_fail:
                        execution.status = WorkflowStatus.FAILED
                        execution.error = f"Steps failed: {failed_steps}"
                        break

            # Mark as completed if not failed
            if execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.COMPLETED

        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)

        execution.completed_at = datetime.now()
        return execution

    async def _execute_step(self, step: WorkflowStep, execution: WorkflowExecution):
        """Execute a single step"""
        step_exec = StepExecution(
            step_id=step.step_id,
            execution_id=execution.execution_id,
            status=StepStatus.RUNNING,
            started_at=datetime.now()
        )
        execution.step_executions[step.step_id] = step_exec

        try:
            # Check condition
            if step.condition:
                if not self._evaluate_condition(step.condition, execution.context):
                    step_exec.status = StepStatus.SKIPPED
                    step_exec.completed_at = datetime.now()
                    return

            # Wait for dependencies
            await self._wait_for_dependencies(step, execution)

            # Resolve parameters (may reference context)
            resolved_params = self._resolve_parameters(step.parameters, execution.context)

            # Execute with timeout
            if step.timeout_seconds:
                result = await asyncio.wait_for(
                    self.tool_executor(step.tool_name, resolved_params),
                    timeout=step.timeout_seconds
                )
            else:
                result = await self.tool_executor(step.tool_name, resolved_params)

            step_exec.result = result
            step_exec.status = StepStatus.COMPLETED

            # Update context with result
            execution.context[f"step_{step.step_id}_result"] = result

        except asyncio.TimeoutError:
            step_exec.error = "Timeout"
            await self._handle_step_error(step, step_exec, execution)

        except Exception as e:
            step_exec.error = str(e)
            await self._handle_step_error(step, step_exec, execution)

        step_exec.completed_at = datetime.now()

    async def _handle_step_error(self, step: WorkflowStep,
                                 step_exec: StepExecution,
                                 execution: WorkflowExecution):
        """Handle step execution error"""
        if step.on_error == "retry" and step_exec.retry_count < step.max_retries:
            step_exec.status = StepStatus.RETRYING
            step_exec.retry_count += 1
            await asyncio.sleep(2 ** step_exec.retry_count)  # Exponential backoff
            await self._execute_step(step, execution)
        elif step.on_error == "skip":
            step_exec.status = StepStatus.SKIPPED
        else:
            step_exec.status = StepStatus.FAILED

    async def _wait_for_dependencies(self, step: WorkflowStep, execution: WorkflowExecution):
        """Wait for step dependencies to complete"""
        while True:
            all_complete = True
            for dep_id in step.depends_on:
                dep_status = execution.step_executions.get(dep_id)
                if not dep_status or dep_status.status not in [StepStatus.COMPLETED, StepStatus.SKIPPED]:
                    all_complete = False
                    break

            if all_complete:
                break

            await asyncio.sleep(0.1)

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition expression"""
        try:
            # Safe eval with limited context
            return eval(condition, {"__builtins__": {}}, context)
        except:
            return False

    def _resolve_parameters(self, params: Dict[str, Any],
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve parameter values from context"""
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$"):
                # Reference to context variable
                context_key = value[1:]
                resolved[key] = context.get(context_key, value)
            else:
                resolved[key] = value
        return resolved

    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get execution details"""
        return self.executions.get(execution_id)

    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all registered workflows"""
        return [wf.to_dict() for wf in self.workflows.values()]

    def get_workflow_stats(self, workflow_id: str) -> Dict[str, Any]:
        """Get statistics for a workflow"""
        executions = [
            e for e in self.executions.values()
            if e.workflow_id == workflow_id
        ]

        total = len(executions)
        if total == 0:
            return {'total_executions': 0}

        completed = sum(1 for e in executions if e.status == WorkflowStatus.COMPLETED)
        failed = sum(1 for e in executions if e.status == WorkflowStatus.FAILED)

        avg_duration = 0
        if completed > 0:
            durations = [
                (e.completed_at - e.started_at).total_seconds()
                for e in executions
                if e.status == WorkflowStatus.COMPLETED and e.completed_at
            ]
            avg_duration = sum(durations) / len(durations)

        return {
            'total_executions': total,
            'completed': completed,
            'failed': failed,
            'success_rate': completed / total if total > 0 else 0,
            'average_duration_seconds': avg_duration
        }


# ============================================================================
# Workflow Templates
# ============================================================================

def create_data_pipeline_template() -> Workflow:
    """Template for data pipeline: fetch -> transform -> load"""
    return Workflow(
        workflow_id="template-data-pipeline",
        name="Data Pipeline Template",
        description="Fetch data from source, transform, and load to destination",
        steps=[
            WorkflowStep(
                step_id="fetch",
                tool_name="http_get",
                parameters={"url": "$source_url", "headers": {}},
            ),
            WorkflowStep(
                step_id="transform",
                tool_name="execute_python",
                parameters={
                    "code": "$transform_code",
                    "timeout": 60
                },
                depends_on=["fetch"]
            ),
            WorkflowStep(
                step_id="load",
                tool_name="sql_query",
                parameters={
                    "query": "$insert_query",
                    "params": "$step_transform_result"
                },
                depends_on=["transform"]
            ),
        ],
        tags=["data", "etl", "pipeline"]
    )


def create_deployment_template() -> Workflow:
    """Template for application deployment"""
    return Workflow(
        workflow_id="template-deployment",
        name="Deployment Template",
        description="Build, test, and deploy application",
        steps=[
            WorkflowStep(
                step_id="git_pull",
                tool_name="git_pull",
                parameters={"repo_path": "$repo_path", "remote": "origin"},
            ),
            WorkflowStep(
                step_id="build",
                tool_name="docker_build",
                parameters={
                    "path": "$repo_path",
                    "tag": "$image_tag"
                },
                depends_on=["git_pull"]
            ),
            WorkflowStep(
                step_id="test",
                tool_name="execute_python",
                parameters={
                    "code": "pytest tests/",
                    "timeout": 300
                },
                depends_on=["build"],
                on_error="fail"
            ),
            WorkflowStep(
                step_id="deploy",
                tool_name="docker_run",
                parameters={
                    "image": "$image_tag",
                    "detach": True,
                    "ports": {"80": "8000"}
                },
                depends_on=["test"],
                condition="step_test_result['success']"
            ),
            WorkflowStep(
                step_id="notify",
                tool_name="send_slack",
                parameters={
                    "channel": "#deployments",
                    "message": "Deployment completed successfully"
                },
                depends_on=["deploy"]
            ),
        ],
        tags=["deployment", "ci-cd"]
    )


def create_monitoring_template() -> Workflow:
    """Template for monitoring and alerting"""
    return Workflow(
        workflow_id="template-monitoring",
        name="Monitoring Template",
        description="Collect metrics, analyze, and alert",
        steps=[
            WorkflowStep(
                step_id="collect_metrics",
                tool_name="get_metrics",
                parameters={
                    "metric_name": "$metric_name",
                    "time_range": "$time_range"
                },
            ),
            WorkflowStep(
                step_id="analyze",
                tool_name="execute_python",
                parameters={
                    "code": "$analysis_code",
                    "timeout": 60
                },
                depends_on=["collect_metrics"]
            ),
            WorkflowStep(
                step_id="alert_if_needed",
                tool_name="send_email",
                parameters={
                    "to": "$alert_email",
                    "subject": "Alert: $metric_name",
                    "body": "$step_analyze_result"
                },
                depends_on=["analyze"],
                condition="step_analyze_result['alert_needed']"
            ),
        ],
        tags=["monitoring", "alerting"]
    )


# ============================================================================
# Workflow Builder - Fluent API
# ============================================================================

class WorkflowBuilder:
    """Fluent API for building workflows"""

    def __init__(self, name: str, description: str = ""):
        self.workflow_id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.steps: List[WorkflowStep] = []
        self.tags: List[str] = []
        self.metadata: Dict[str, Any] = {}

    def add_step(self, step_id: str, tool_name: str,
                 parameters: Dict[str, Any],
                 depends_on: Optional[List[str]] = None,
                 **kwargs) -> 'WorkflowBuilder':
        """Add a step to the workflow"""
        step = WorkflowStep(
            step_id=step_id,
            tool_name=tool_name,
            parameters=parameters,
            depends_on=depends_on or [],
            **kwargs
        )
        self.steps.append(step)
        return self

    def add_parallel_steps(self, steps: List[Tuple[str, str, Dict[str, Any]]]) -> 'WorkflowBuilder':
        """Add multiple steps that run in parallel"""
        for step_id, tool_name, parameters in steps:
            self.add_step(step_id, tool_name, parameters)
        return self

    def add_sequential_steps(self, steps: List[Tuple[str, str, Dict[str, Any]]]) -> 'WorkflowBuilder':
        """Add steps that run sequentially"""
        prev_step_id = None
        for step_id, tool_name, parameters in steps:
            depends_on = [prev_step_id] if prev_step_id else []
            self.add_step(step_id, tool_name, parameters, depends_on=depends_on)
            prev_step_id = step_id
        return self

    def with_tags(self, *tags: str) -> 'WorkflowBuilder':
        """Add tags to workflow"""
        self.tags.extend(tags)
        return self

    def with_metadata(self, **metadata) -> 'WorkflowBuilder':
        """Add metadata to workflow"""
        self.metadata.update(metadata)
        return self

    def build(self) -> Workflow:
        """Build the workflow"""
        return Workflow(
            workflow_id=self.workflow_id,
            name=self.name,
            description=self.description,
            steps=self.steps,
            tags=self.tags,
            metadata=self.metadata
        )


# ============================================================================
# Example Usage
# ============================================================================

async def demo_workflow_engine():
    """Demonstrate workflow engine"""
    from distributed_mcp_hub import SimpleMCPClient

    print("=" * 80)
    print("Workflow Engine Demo")
    print("=" * 80)
    print()

    # Mock tool executor
    async def mock_executor(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.1)  # Simulate work
        return {
            'success': True,
            'tool': tool_name,
            'params': params,
            'result': f"Result from {tool_name}"
        }

    engine = WorkflowEngine(tool_executor=mock_executor)

    # Build a workflow using fluent API
    workflow = (
        WorkflowBuilder("Data Processing Pipeline", "Fetch, process, and store data")
        .add_step("fetch_api1", "http_get", {"url": "https://api1.com"})
        .add_step("fetch_api2", "http_get", {"url": "https://api2.com"})
        .add_step("merge", "execute_python",
                 {"code": "merge_data()", "timeout": 60},
                 depends_on=["fetch_api1", "fetch_api2"])
        .add_step("store", "sql_query",
                 {"query": "INSERT INTO data VALUES (?)", "params": {}},
                 depends_on=["merge"])
        .with_tags("data", "pipeline")
        .build()
    )

    # Register workflow
    success, error = engine.register_workflow(workflow)
    print(f"✓ Registered workflow: {workflow.name}")
    print(f"  Steps: {len(workflow.steps)}")
    print(f"  Execution order: {workflow.get_execution_order()}")
    print()

    # Execute workflow
    print("Executing workflow...")
    execution = await engine.execute_workflow(workflow.workflow_id)

    print(f"✓ Workflow completed: {execution.status.value}")
    print(f"  Duration: {(execution.completed_at - execution.started_at).total_seconds():.2f}s")
    print(f"  Steps executed: {len(execution.step_executions)}")
    print()

    # Show step details
    print("Step Execution Details:")
    for step_id, step_exec in execution.step_executions.items():
        duration = 0
        if step_exec.completed_at and step_exec.started_at:
            duration = (step_exec.completed_at - step_exec.started_at).total_seconds()
        print(f"  {step_id}: {step_exec.status.value} ({duration:.2f}s)")
    print()

    # Stats
    stats = engine.get_workflow_stats(workflow.workflow_id)
    print(f"Workflow Statistics:")
    print(f"  Total executions: {stats['total_executions']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print()


if __name__ == "__main__":
    asyncio.run(demo_workflow_engine())
