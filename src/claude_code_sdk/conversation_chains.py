"""Context-aware conversation chains for complex workflows."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, AsyncIterator, Union
from enum import Enum

from .types import Message, ClaudeCodeOptions, ResultMessage
from .conversation_manager import ConversationManager
from . import query


class ChainStatus(Enum):
    """Status of a chain execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class ChainStep:
    """Single step in a conversation chain."""
    name: str
    prompt_template: str
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    options_modifier: Optional[Callable[[ClaudeCodeOptions], ClaudeCodeOptions]] = None
    result_processor: Optional[Callable[[List[Message]], Dict[str, Any]]] = None
    retry_on_failure: bool = False
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)  # Names of steps that must complete first
    
    def should_execute(self, context: Dict[str, Any]) -> bool:
        """Check if this step should execute based on condition."""
        if self.condition is None:
            return True
        return self.condition(context)
    
    def format_prompt(self, context: Dict[str, Any]) -> str:
        """Format the prompt with context."""
        return self.prompt_template.format(**context)


@dataclass
class ChainResult:
    """Result of a chain execution."""
    status: ChainStatus
    completed_steps: List[str]
    failed_steps: List[str]
    context: Dict[str, Any]
    total_cost: float
    session_ids: Dict[str, str]  # step_name -> session_id mapping
    error: Optional[str] = None


class ConversationChain:
    """Manages a chain of conversation steps with context passing."""
    
    def __init__(
        self,
        name: str,
        steps: List[ChainStep],
        initial_context: Optional[Dict[str, Any]] = None,
        manager: Optional[ConversationManager] = None,
    ):
        """Initialize a conversation chain.
        
        Args:
            name: Name of the chain
            steps: List of steps to execute
            initial_context: Initial context values
            manager: Optional ConversationManager for persistence
        """
        self.name = name
        self.steps = steps
        self.initial_context = initial_context or {}
        self.manager = manager or ConversationManager()
        
        # Validate dependencies
        step_names = {step.name for step in steps}
        for step in steps:
            for dep in step.dependencies:
                if dep not in step_names:
                    raise ValueError(f"Step '{step.name}' depends on unknown step '{dep}'")
    
    async def execute(
        self,
        context_overrides: Optional[Dict[str, Any]] = None,
        parallel: bool = False,
    ) -> ChainResult:
        """Execute the conversation chain.
        
        Args:
            context_overrides: Override initial context values
            parallel: Execute independent steps in parallel
            
        Returns:
            ChainResult with execution details
        """
        # Initialize execution state
        context = self.initial_context.copy()
        if context_overrides:
            context.update(context_overrides)
        
        completed_steps = set()
        failed_steps = []
        session_ids = {}
        total_cost = 0.0
        
        # Build dependency graph
        step_map = {step.name: step for step in self.steps}
        
        async def execute_step(step: ChainStep) -> bool:
            """Execute a single step."""
            # Check dependencies
            for dep in step.dependencies:
                if dep not in completed_steps:
                    return False
            
            # Check condition
            if not step.should_execute(context):
                completed_steps.add(step.name)
                return True
            
            # Execute with retries
            for attempt in range(step.max_retries if step.retry_on_failure else 1):
                try:
                    # Format prompt
                    prompt = step.format_prompt(context)
                    
                    # Prepare options
                    options = ClaudeCodeOptions()
                    if step.options_modifier:
                        options = step.options_modifier(options)
                    
                    # Execute query
                    messages = []
                    step_cost = 0.0
                    session_id = None
                    
                    async for message in query(prompt=prompt, options=options):
                        messages.append(message)
                        if isinstance(message, ResultMessage):
                            session_id = message.session_id
                            step_cost = message.cost_usd
                    
                    # Process results
                    if step.result_processor:
                        step_results = step.result_processor(messages)
                        context.update(step_results)
                    
                    # Update state
                    completed_steps.add(step.name)
                    session_ids[step.name] = session_id
                    nonlocal total_cost
                    total_cost += step_cost
                    
                    return True
                    
                except Exception as e:
                    if attempt == (step.max_retries - 1 if step.retry_on_failure else 0):
                        failed_steps.append(step.name)
                        context[f"{step.name}_error"] = str(e)
                        return False
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            return False
        
        # Execute steps
        if parallel:
            # Group steps by dependency level
            levels = self._compute_dependency_levels()
            
            for level in levels:
                # Execute steps at this level in parallel
                tasks = [execute_step(step_map[name]) for name in level]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check for failures
                if not all(r is True or r is False for r in results):
                    return ChainResult(
                        status=ChainStatus.FAILED,
                        completed_steps=list(completed_steps),
                        failed_steps=failed_steps,
                        context=context,
                        total_cost=total_cost,
                        session_ids=session_ids,
                        error="Parallel execution error",
                    )
        else:
            # Sequential execution
            for step in self.steps:
                success = await execute_step(step)
                if not success and step.name not in completed_steps:
                    return ChainResult(
                        status=ChainStatus.FAILED,
                        completed_steps=list(completed_steps),
                        failed_steps=failed_steps,
                        context=context,
                        total_cost=total_cost,
                        session_ids=session_ids,
                        error=f"Step '{step.name}' failed",
                    )
        
        # Determine final status
        if failed_steps:
            status = ChainStatus.FAILED
        else:
            status = ChainStatus.COMPLETED
        
        return ChainResult(
            status=status,
            completed_steps=list(completed_steps),
            failed_steps=failed_steps,
            context=context,
            total_cost=total_cost,
            session_ids=session_ids,
        )
    
    def _compute_dependency_levels(self) -> List[List[str]]:
        """Compute dependency levels for parallel execution."""
        levels = []
        remaining = {step.name: set(step.dependencies) for step in self.steps}
        completed = set()
        
        while remaining:
            # Find steps with no remaining dependencies
            current_level = [
                name for name, deps in remaining.items()
                if not deps - completed
            ]
            
            if not current_level:
                raise ValueError("Circular dependency detected")
            
            levels.append(current_level)
            completed.update(current_level)
            
            # Remove completed steps
            for name in current_level:
                del remaining[name]
        
        return levels


# Pre-defined chain examples

def create_full_development_chain() -> ConversationChain:
    """Create a chain for full development workflow."""
    steps = [
        ChainStep(
            name="analyze_requirements",
            prompt_template="Analyze these requirements and create a development plan: {requirements}",
            result_processor=lambda msgs: {"development_plan": _extract_text(msgs)},
        ),
        ChainStep(
            name="create_structure",
            prompt_template="Based on this plan: {development_plan}, create the project structure",
            dependencies=["analyze_requirements"],
            options_modifier=lambda opts: ClaudeCodeOptions(
                allowed_tools=["Write", "Bash"],
                permission_mode="acceptEdits",
            ),
        ),
        ChainStep(
            name="implement_core",
            prompt_template="Implement the core functionality as outlined in the plan",
            dependencies=["create_structure"],
            options_modifier=lambda opts: ClaudeCodeOptions(
                allowed_tools=["Write", "Edit", "Read"],
                permission_mode="acceptEdits",
            ),
        ),
        ChainStep(
            name="write_tests",
            prompt_template="Write comprehensive tests for the implemented functionality",
            dependencies=["implement_core"],
            options_modifier=lambda opts: ClaudeCodeOptions(
                allowed_tools=["Write", "Read", "Bash"],
                permission_mode="acceptEdits",
            ),
        ),
        ChainStep(
            name="add_documentation",
            prompt_template="Create documentation for the project",
            dependencies=["implement_core"],
            options_modifier=lambda opts: ClaudeCodeOptions(
                allowed_tools=["Write", "Read"],
                permission_mode="acceptEdits",
            ),
        ),
        ChainStep(
            name="review_and_refine",
            prompt_template="Review the entire implementation and suggest improvements",
            dependencies=["write_tests", "add_documentation"],
            options_modifier=lambda opts: ClaudeCodeOptions(
                allowed_tools=["Read", "Edit"],
            ),
        ),
    ]
    
    return ConversationChain(
        name="full_development",
        steps=steps,
        initial_context={"requirements": ""},
    )


def create_debugging_chain() -> ConversationChain:
    """Create a chain for systematic debugging."""
    steps = [
        ChainStep(
            name="reproduce_issue",
            prompt_template="Try to reproduce this issue: {issue_description}",
            options_modifier=lambda opts: ClaudeCodeOptions(
                allowed_tools=["Read", "Bash"],
            ),
            result_processor=lambda msgs: {"reproduction_steps": _extract_text(msgs)},
        ),
        ChainStep(
            name="analyze_code",
            prompt_template="Analyze the code related to: {reproduction_steps}",
            dependencies=["reproduce_issue"],
            options_modifier=lambda opts: ClaudeCodeOptions(
                allowed_tools=["Read", "Grep"],
            ),
            result_processor=lambda msgs: {"analysis": _extract_text(msgs)},
        ),
        ChainStep(
            name="identify_root_cause",
            prompt_template="Based on the analysis: {analysis}, identify the root cause",
            dependencies=["analyze_code"],
            result_processor=lambda msgs: {"root_cause": _extract_text(msgs)},
        ),
        ChainStep(
            name="implement_fix",
            prompt_template="Fix the root cause: {root_cause}",
            dependencies=["identify_root_cause"],
            options_modifier=lambda opts: ClaudeCodeOptions(
                allowed_tools=["Edit", "Write"],
                permission_mode="acceptEdits",
            ),
        ),
        ChainStep(
            name="verify_fix",
            prompt_template="Verify the fix resolves the original issue",
            dependencies=["implement_fix"],
            options_modifier=lambda opts: ClaudeCodeOptions(
                allowed_tools=["Bash", "Read"],
            ),
        ),
        ChainStep(
            name="add_test",
            prompt_template="Add a test to prevent regression of this issue",
            dependencies=["verify_fix"],
            condition=lambda ctx: "test" not in ctx.get("skip_steps", []),
            options_modifier=lambda opts: ClaudeCodeOptions(
                allowed_tools=["Write", "Edit"],
                permission_mode="acceptEdits",
            ),
        ),
    ]
    
    return ConversationChain(
        name="debugging",
        steps=steps,
        initial_context={"issue_description": ""},
    )


def create_refactoring_chain() -> ConversationChain:
    """Create a chain for safe refactoring."""
    steps = [
        ChainStep(
            name="analyze_current_code",
            prompt_template="Analyze the code in {target_file} for refactoring opportunities",
            options_modifier=lambda opts: ClaudeCodeOptions(
                allowed_tools=["Read", "Grep"],
            ),
            result_processor=lambda msgs: {"current_analysis": _extract_text(msgs)},
        ),
        ChainStep(
            name="create_refactoring_plan",
            prompt_template="Create a detailed refactoring plan based on: {current_analysis}",
            dependencies=["analyze_current_code"],
            result_processor=lambda msgs: {"refactoring_plan": _extract_text(msgs)},
        ),
        ChainStep(
            name="backup_original",
            prompt_template="Create a backup of the original code",
            dependencies=["create_refactoring_plan"],
            options_modifier=lambda opts: ClaudeCodeOptions(
                allowed_tools=["Bash", "Write"],
                permission_mode="acceptEdits",
            ),
        ),
        ChainStep(
            name="implement_refactoring",
            prompt_template="Implement the refactoring according to: {refactoring_plan}",
            dependencies=["backup_original"],
            options_modifier=lambda opts: ClaudeCodeOptions(
                allowed_tools=["Edit", "Write"],
                permission_mode="acceptEdits",
            ),
        ),
        ChainStep(
            name="run_tests",
            prompt_template="Run existing tests to ensure functionality is preserved",
            dependencies=["implement_refactoring"],
            options_modifier=lambda opts: ClaudeCodeOptions(
                allowed_tools=["Bash"],
            ),
            retry_on_failure=True,
        ),
        ChainStep(
            name="update_documentation",
            prompt_template="Update any documentation affected by the refactoring",
            dependencies=["run_tests"],
            condition=lambda ctx: ctx.get("has_documentation", True),
            options_modifier=lambda opts: ClaudeCodeOptions(
                allowed_tools=["Edit", "Write"],
                permission_mode="acceptEdits",
            ),
        ),
    ]
    
    return ConversationChain(
        name="refactoring",
        steps=steps,
        initial_context={"target_file": ""},
    )


# Helper function
def _extract_text(messages: List[Message]) -> str:
    """Extract text content from messages."""
    from .types import AssistantMessage, TextBlock
    
    text_parts = []
    for msg in messages:
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
    return "\n".join(text_parts)