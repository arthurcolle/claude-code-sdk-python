"""Training data collector for capturing agent interactions and task executions."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable, Union
from pathlib import Path
import json
import asyncio
from contextlib import asynccontextmanager

from .types import (
    Message,
    AssistantMessage,
    UserMessage,
    ToolUseBlock,
    ToolResultBlock,
    TextBlock,
    ResultMessage,
)
from .training_data import (
    TaskDefinition,
    TaskExecution,
    AgentAction,
    TrainingExample,
    TaskStatus,
    AgentRole,
    TrainingDataStorage,
    SQLiteTrainingStorage,
    DuckDBTrainingStorage,
)
from .conversation_manager import ConversationManager
from .conversation_chains import ConversationChain, ChainStep


class TrainingDataCollector:
    """Collects training data from Claude SDK interactions."""

    def __init__(
        self,
        storage: TrainingDataStorage,
        auto_save: bool = True,
        quality_scorer: Optional[Callable[[TaskExecution], float]] = None,
    ):
        self.storage = storage
        self.auto_save = auto_save
        self.quality_scorer = quality_scorer or self._default_quality_scorer
        self.current_executions: Dict[str, TaskExecution] = {}
        self.current_tasks: Dict[str, TaskDefinition] = {}

    def _default_quality_scorer(self, execution: TaskExecution) -> float:
        """Default quality scoring based on execution metrics."""
        score = 0.0

        # Success contributes 40%
        if execution.success:
            score += 0.4

        # Efficiency contributes 30%
        if execution.completed_at and execution.started_at:
            duration = (execution.completed_at - execution.started_at).total_seconds()
            # Score based on speed (faster is better, with diminishing returns)
            if duration < 60:  # Under 1 minute
                score += 0.3
            elif duration < 300:  # Under 5 minutes
                score += 0.2
            elif duration < 600:  # Under 10 minutes
                score += 0.1

        # Cost efficiency contributes 20%
        if execution.cost_usd < 0.01:
            score += 0.2
        elif execution.cost_usd < 0.05:
            score += 0.15
        elif execution.cost_usd < 0.10:
            score += 0.1
        elif execution.cost_usd < 0.50:
            score += 0.05

        # Tool usage efficiency contributes 10%
        if execution.actions:
            tool_uses = [a for a in execution.actions if a.action_type == "tool_use"]
            if tool_uses:
                # Penalize excessive tool use
                if len(tool_uses) < 5:
                    score += 0.1
                elif len(tool_uses) < 10:
                    score += 0.05

        return min(score, 1.0)

    def start_task(
        self, task: TaskDefinition, session_id: str, agent_id: str = "claude"
    ) -> str:
        """Start tracking a new task execution."""
        execution = TaskExecution(
            task_id=task.id,
            session_id=session_id,
            agent_id=agent_id,
            status=TaskStatus.IN_PROGRESS,
            started_at=datetime.now(),
        )

        self.current_tasks[task.id] = task
        self.current_executions[session_id] = execution

        return execution.id

    def add_message(self, session_id: str, message: Message):
        """Add a message to the current execution."""
        if session_id not in self.current_executions:
            return

        execution = self.current_executions[session_id]

        # Serialize message for storage
        msg_dict = self._serialize_message(message)
        execution.messages.append(msg_dict)

        # Extract actions from assistant messages
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    action = AgentAction(
                        agent_id=execution.agent_id,
                        agent_role=AgentRole.TASK_EXECUTOR,
                        action_type="tool_use",
                        timestamp=datetime.now(),
                        tool_name=block.name,
                        tool_input=block.input,
                        content={"tool_use_id": block.id},
                    )
                    execution.actions.append(action)

                elif isinstance(block, TextBlock) and block.text:
                    # Extract reasoning from text
                    if any(
                        keyword in block.text.lower()
                        for keyword in ["because", "therefore", "since", "reasoning:"]
                    ):
                        if execution.actions:
                            execution.actions[-1].reasoning = block.text

        # Handle result messages
        elif isinstance(message, ResultMessage):
            execution.cost_usd = message.cost_usd or 0.0
            if message.usage:
                execution.tokens_used = dict(message.usage)
            execution.metadata["session_result"] = message.data

    def add_tool_result(
        self,
        session_id: str,
        tool_use_id: str,
        result: Dict[str, Any],
        is_error: bool = False,
    ):
        """Add a tool result to the current execution."""
        if session_id not in self.current_executions:
            return

        execution = self.current_executions[session_id]

        # Find the corresponding action
        for action in reversed(execution.actions):
            if action.content.get("tool_use_id") == tool_use_id:
                action.tool_result = result
                if is_error:
                    action.metadata["error"] = True
                break

    def complete_task(
        self,
        session_id: str,
        success: bool,
        outcome: Optional[str] = None,
        error: Optional[str] = None,
        performance_metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[TrainingExample]:
        """Complete a task execution and optionally save as training example."""
        if session_id not in self.current_executions:
            return None

        execution = self.current_executions[session_id]
        execution.completed_at = datetime.now()
        execution.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        execution.success = success
        execution.outcome = outcome
        execution.error = error
        execution.performance_metrics = performance_metrics or {}

        # Calculate quality score
        quality_score = self.quality_scorer(execution)

        # Create training example
        task = self.current_tasks.get(execution.task_id)
        if not task:
            return None

        example = TrainingExample(
            task_definition=task,
            execution=execution,
            quality_score=quality_score,
            is_golden=quality_score >= 0.8,  # High quality examples are golden
            tags=self._generate_tags(task, execution),
            created_at=datetime.now(),
        )

        # Save if auto-save is enabled
        if self.auto_save:
            self.storage.save_training_example(example)

        # Clean up
        del self.current_executions[session_id]

        return example

    def _serialize_message(self, message: Message) -> Dict[str, Any]:
        """Serialize a message for storage."""
        if isinstance(message, UserMessage):
            return {"type": "user", "content": message.content}
        elif isinstance(message, AssistantMessage):
            content_blocks = []
            for block in message.content:
                if isinstance(block, TextBlock):
                    content_blocks.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolUseBlock):
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )
                elif isinstance(block, ToolResultBlock):
                    content_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.tool_use_id,
                            "content": block.content,
                            "is_error": block.is_error,
                        }
                    )

            return {"type": "assistant", "content": content_blocks}
        else:
            return {
                "type": message.__class__.__name__,
                "data": message.data if hasattr(message, "data") else str(message),
            }

    def _generate_tags(
        self, task: TaskDefinition, execution: TaskExecution
    ) -> List[str]:
        """Generate tags for a training example."""
        tags = []

        # Task-based tags
        tags.append(f"category:{task.category}")
        tags.append(f"difficulty:{task.difficulty}")

        # Execution-based tags
        if execution.success:
            tags.append("successful")
        else:
            tags.append("failed")

        # Tool usage tags
        tools_used = set()
        for action in execution.actions:
            if action.tool_name:
                tools_used.add(action.tool_name)

        for tool in tools_used:
            tags.append(f"uses:{tool}")

        # Performance tags
        if execution.cost_usd < 0.01:
            tags.append("low_cost")

        duration = None
        if execution.completed_at and execution.started_at:
            duration = (execution.completed_at - execution.started_at).total_seconds()
            if duration < 60:
                tags.append("fast_execution")

        return tags

    @asynccontextmanager
    async def collect_from_query(
        self,
        task: TaskDefinition,
        prompt: str,
        query_func: Callable,
        options: Optional[Any] = None,
    ):
        """Context manager for collecting training data from a query."""
        session_id = None
        execution_id = None

        try:
            # Start task execution
            execution_id = self.start_task(task, "pending", "claude")

            # Track messages
            messages_collected = []
            tool_results = {}

            async for message in query_func(prompt=prompt, options=options):
                # Extract session ID
                if isinstance(message, ResultMessage) and message.session_id:
                    session_id = message.session_id
                    if execution_id and session_id != "pending":
                        self.current_executions[session_id] = (
                            self.current_executions.pop("pending")
                        )
                        self.current_executions[session_id].session_id = session_id

                # Collect messages
                messages_collected.append(message)

                # Add to execution
                if session_id or execution_id:
                    self.add_message(session_id or "pending", message)

                # Track tool results
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, ToolResultBlock):
                            tool_results[block.tool_use_id] = {
                                "content": block.content,
                                "is_error": block.is_error,
                            }
                            self.add_tool_result(
                                session_id or "pending",
                                block.tool_use_id,
                                {"content": block.content},
                                block.is_error,
                            )

                yield message

            # Task completed successfully
            if session_id:
                example = self.complete_task(
                    session_id, success=True, outcome="Task completed successfully"
                )
                if example:
                    print(
                        f"Training example saved with quality score: {example.quality_score:.2f}"
                    )

        except Exception as e:
            # Task failed
            if session_id:
                self.complete_task(session_id, success=False, error=str(e))
            raise


class ConversationChainCollector:
    """Collects training data from conversation chains."""

    def __init__(self, storage: TrainingDataStorage):
        self.storage = storage
        self.collector = TrainingDataCollector(storage)

    async def collect_from_chain(
        self,
        chain: ConversationChain,
        task: TaskDefinition,
        initial_prompt: str,
        options: Optional[Any] = None,
    ) -> TrainingExample:
        """Collect training data from a conversation chain execution."""
        # Create a wrapped chain that collects data
        original_steps = chain.steps.copy()
        session_id = None
        execution_id = None

        # Start task
        execution_id = self.collector.start_task(task, "chain_pending", "chain_agent")

        # Wrap each step to collect data
        for i, step in enumerate(chain.steps):
            original_processor = step.result_processor

            async def wrapped_processor(result, context, step_name=step.name):
                # Collect messages from the step
                if "messages" in result:
                    for msg in result["messages"]:
                        self.collector.add_message(session_id or "chain_pending", msg)

                # Collect session ID
                nonlocal session_id
                if "session_id" in result and not session_id:
                    session_id = result["session_id"]
                    # Update execution with real session ID
                    if "chain_pending" in self.collector.current_executions:
                        self.collector.current_executions[session_id] = (
                            self.collector.current_executions.pop("chain_pending")
                        )
                        self.collector.current_executions[
                            session_id
                        ].session_id = session_id

                # Call original processor
                if original_processor:
                    return await original_processor(result, context)
                return result

            step.result_processor = wrapped_processor

        try:
            # Execute chain
            result = await chain.execute(initial_prompt, options)

            # Complete task
            success = result.get("status") == "completed"
            outcome = result.get("final_output", "Chain execution completed")

            example = self.collector.complete_task(
                session_id or "chain_pending",
                success=success,
                outcome=outcome,
                performance_metrics={
                    "completed_steps": len(result.get("completed_steps", [])),
                    "total_steps": len(chain.steps),
                },
            )

            return example

        finally:
            # Restore original steps
            chain.steps = original_steps


class MultiAgentCollector:
    """Collects training data from multi-agent interactions."""

    def __init__(self, storage: TrainingDataStorage):
        self.storage = storage
        self.agent_executions: Dict[str, List[AgentAction]] = {}

    def track_agent_action(
        self,
        agent_id: str,
        agent_role: AgentRole,
        action_type: str,
        content: Dict[str, Any],
        tool_name: Optional[str] = None,
        reasoning: Optional[str] = None,
        confidence: float = 1.0,
    ):
        """Track an action taken by an agent."""
        action = AgentAction(
            agent_id=agent_id,
            agent_role=agent_role,
            action_type=action_type,
            timestamp=datetime.now(),
            content=content,
            tool_name=tool_name,
            reasoning=reasoning,
            confidence=confidence,
        )

        if agent_id not in self.agent_executions:
            self.agent_executions[agent_id] = []

        self.agent_executions[agent_id].append(action)

    def create_multi_agent_example(
        self,
        task: TaskDefinition,
        coordinator_session_id: str,
        outcome: str,
        success: bool,
        cost_usd: float = 0.0,
    ) -> TrainingExample:
        """Create a training example from multi-agent execution."""
        # Combine all agent actions
        all_actions = []
        for agent_id, actions in self.agent_executions.items():
            all_actions.extend(actions)

        # Sort by timestamp
        all_actions.sort(key=lambda a: a.timestamp)

        # Create execution
        execution = TaskExecution(
            task_id=task.id,
            session_id=coordinator_session_id,
            agent_id="multi_agent_system",
            status=TaskStatus.COMPLETED if success else TaskStatus.FAILED,
            started_at=all_actions[0].timestamp if all_actions else datetime.now(),
            completed_at=all_actions[-1].timestamp if all_actions else datetime.now(),
            actions=all_actions,
            outcome=outcome,
            success=success,
            cost_usd=cost_usd,
            metadata={
                "agent_count": len(self.agent_executions),
                "agent_ids": list(self.agent_executions.keys()),
            },
        )

        # Calculate quality score with multi-agent considerations
        quality_score = self._score_multi_agent_execution(execution)

        # Create example
        example = TrainingExample(
            task_definition=task,
            execution=execution,
            quality_score=quality_score,
            is_golden=quality_score >= 0.85,
            tags=["multi_agent"] + self._generate_multi_agent_tags(execution),
            annotations={
                "coordination_quality": self._assess_coordination(all_actions),
                "agent_efficiency": self._assess_agent_efficiency(
                    self.agent_executions
                ),
            },
        )

        # Save
        self.storage.save_training_example(example)

        # Clear tracked data
        self.agent_executions.clear()

        return example

    def _score_multi_agent_execution(self, execution: TaskExecution) -> float:
        """Score quality of multi-agent execution."""
        base_score = 0.5 if execution.success else 0.0

        # Coordination quality (30%)
        coordination_score = self._assess_coordination(execution.actions)
        base_score += coordination_score * 0.3

        # Efficiency (20%)
        if len(execution.actions) < 20:
            base_score += 0.2
        elif len(execution.actions) < 50:
            base_score += 0.1

        return min(base_score, 1.0)

    def _assess_coordination(self, actions: List[AgentAction]) -> float:
        """Assess quality of agent coordination."""
        if not actions:
            return 0.0

        # Check for good coordination patterns
        score = 0.5

        # Sequential handoffs
        last_agent = None
        handoff_count = 0
        for action in actions:
            if action.agent_id != last_agent:
                handoff_count += 1
                last_agent = action.agent_id

        # Good coordination has some handoffs but not too many
        if 2 <= handoff_count <= 10:
            score += 0.3
        elif handoff_count < 2:
            score += 0.1

        # Check for parallel work (actions close in time from different agents)
        parallel_work = False
        for i in range(len(actions) - 1):
            time_diff = (
                actions[i + 1].timestamp - actions[i].timestamp
            ).total_seconds()
            if time_diff < 1 and actions[i].agent_id != actions[i + 1].agent_id:
                parallel_work = True
                break

        if parallel_work:
            score += 0.2

        return min(score, 1.0)

    def _assess_agent_efficiency(
        self, agent_executions: Dict[str, List[AgentAction]]
    ) -> Dict[str, float]:
        """Assess efficiency of individual agents."""
        efficiency_scores = {}

        for agent_id, actions in agent_executions.items():
            if not actions:
                efficiency_scores[agent_id] = 0.0
                continue

            # Base efficiency
            score = 0.5

            # Penalize too many actions
            if len(actions) < 5:
                score += 0.3
            elif len(actions) < 10:
                score += 0.2
            elif len(actions) < 20:
                score += 0.1

            # Bonus for high confidence actions
            avg_confidence = sum(a.confidence for a in actions) / len(actions)
            score += avg_confidence * 0.2

            efficiency_scores[agent_id] = min(score, 1.0)

        return efficiency_scores

    def _generate_multi_agent_tags(self, execution: TaskExecution) -> List[str]:
        """Generate tags specific to multi-agent execution."""
        tags = []

        agent_count = execution.metadata.get("agent_count", 0)
        tags.append(f"agents:{agent_count}")

        # Tag agent roles used
        roles_used = set()
        for action in execution.actions:
            roles_used.add(action.agent_role.value)

        for role in roles_used:
            tags.append(f"role:{role}")

        return tags
