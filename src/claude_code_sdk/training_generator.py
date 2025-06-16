"""Training data generator with task templates for various agent scenarios."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime
import random
import asyncio
from pathlib import Path

from .training_data import TaskDefinition, TaskExecution, TrainingExample, AgentRole
from .training_collector import (
    TrainingDataCollector,
    ConversationChainCollector,
    MultiAgentCollector,
)
from .conversation_chains import ConversationChain, ChainStep
from .conversation_templates import ConversationTemplate
from . import query, ClaudeCodeOptions


class TaskTemplateLibrary:
    """Library of task templates for training data generation."""

    @staticmethod
    def get_code_review_tasks() -> List[TaskDefinition]:
        """Generate code review task templates."""
        return [
            TaskDefinition(
                name="Python Code Review - Security",
                description="Review Python code for security vulnerabilities and suggest improvements",
                category="code_review",
                difficulty="medium",
                required_tools=["Read", "Grep", "Edit"],
                expected_outcomes=[
                    "Identify security vulnerabilities",
                    "Suggest secure coding practices",
                    "Fix critical issues",
                ],
                constraints=[
                    "Must check for SQL injection",
                    "Must check for XSS vulnerabilities",
                ],
            ),
            TaskDefinition(
                name="JavaScript Performance Review",
                description="Analyze JavaScript code for performance bottlenecks and optimize",
                category="code_review",
                difficulty="hard",
                required_tools=["Read", "Grep", "Edit", "Bash"],
                expected_outcomes=[
                    "Identify performance bottlenecks",
                    "Suggest optimization strategies",
                    "Implement key optimizations",
                ],
                constraints=["Focus on async operations", "Consider bundle size"],
            ),
            TaskDefinition(
                name="API Design Review",
                description="Review REST API design for best practices and consistency",
                category="code_review",
                difficulty="medium",
                required_tools=["Read", "Grep"],
                expected_outcomes=[
                    "Evaluate RESTful design principles",
                    "Check API consistency",
                    "Suggest improvements",
                ],
                constraints=[
                    "Follow REST best practices",
                    "Consider versioning strategy",
                ],
            ),
        ]

    @staticmethod
    def get_debugging_tasks() -> List[TaskDefinition]:
        """Generate debugging task templates."""
        return [
            TaskDefinition(
                name="Fix Failing Unit Test",
                description="Debug and fix a failing unit test in the test suite",
                category="debugging",
                difficulty="easy",
                required_tools=["Read", "Bash", "Edit"],
                expected_outcomes=[
                    "Identify root cause of test failure",
                    "Fix the issue",
                    "Verify test passes",
                ],
                constraints=[
                    "Don't modify test logic unless it's wrong",
                    "Maintain code quality",
                ],
            ),
            TaskDefinition(
                name="Memory Leak Investigation",
                description="Investigate and fix a memory leak in a Node.js application",
                category="debugging",
                difficulty="expert",
                required_tools=["Read", "Grep", "Bash", "Edit"],
                expected_outcomes=[
                    "Identify memory leak source",
                    "Implement fix",
                    "Add monitoring",
                ],
                constraints=["Use profiling tools", "Document findings"],
            ),
            TaskDefinition(
                name="Race Condition Debug",
                description="Debug and fix a race condition in concurrent code",
                category="debugging",
                difficulty="hard",
                required_tools=["Read", "Grep", "Edit"],
                expected_outcomes=[
                    "Identify race condition",
                    "Implement proper synchronization",
                    "Add tests",
                ],
                constraints=["Maintain performance", "Avoid deadlocks"],
            ),
        ]

    @staticmethod
    def get_refactoring_tasks() -> List[TaskDefinition]:
        """Generate refactoring task templates."""
        return [
            TaskDefinition(
                name="Extract Reusable Components",
                description="Refactor code to extract reusable components and reduce duplication",
                category="refactoring",
                difficulty="medium",
                required_tools=["Read", "Grep", "Edit", "Write"],
                expected_outcomes=[
                    "Identify duplicate code",
                    "Extract reusable components",
                    "Update imports",
                ],
                constraints=[
                    "Maintain backward compatibility",
                    "Follow project conventions",
                ],
            ),
            TaskDefinition(
                name="Async/Await Migration",
                description="Migrate callback-based code to use async/await patterns",
                category="refactoring",
                difficulty="medium",
                required_tools=["Read", "Grep", "Edit"],
                expected_outcomes=[
                    "Convert callbacks to async/await",
                    "Handle errors properly",
                    "Maintain functionality",
                ],
                constraints=["Preserve error handling", "Don't break existing APIs"],
            ),
            TaskDefinition(
                name="Design Pattern Implementation",
                description="Refactor code to implement appropriate design patterns",
                category="refactoring",
                difficulty="hard",
                required_tools=["Read", "Edit", "Write"],
                expected_outcomes=[
                    "Identify applicable patterns",
                    "Implement pattern",
                    "Document changes",
                ],
                constraints=["Don't over-engineer", "Keep it simple"],
            ),
        ]

    @staticmethod
    def get_implementation_tasks() -> List[TaskDefinition]:
        """Generate implementation task templates."""
        return [
            TaskDefinition(
                name="Add Caching Layer",
                description="Implement a caching layer for frequently accessed data",
                category="implementation",
                difficulty="medium",
                required_tools=["Read", "Write", "Edit"],
                expected_outcomes=[
                    "Design cache strategy",
                    "Implement caching",
                    "Add cache invalidation",
                ],
                constraints=[
                    "Use existing cache libraries",
                    "Handle cache misses gracefully",
                ],
            ),
            TaskDefinition(
                name="API Rate Limiting",
                description="Implement rate limiting for API endpoints",
                category="implementation",
                difficulty="medium",
                required_tools=["Read", "Write", "Edit", "Bash"],
                expected_outcomes=[
                    "Implement rate limiter",
                    "Add configuration",
                    "Test implementation",
                ],
                constraints=["Use standard algorithms", "Make it configurable"],
            ),
            TaskDefinition(
                name="Data Export Feature",
                description="Implement data export functionality in multiple formats",
                category="implementation",
                difficulty="easy",
                required_tools=["Read", "Write", "Edit"],
                expected_outcomes=[
                    "Support JSON export",
                    "Support CSV export",
                    "Add download endpoint",
                ],
                constraints=["Handle large datasets", "Include metadata"],
            ),
        ]

    @staticmethod
    def get_testing_tasks() -> List[TaskDefinition]:
        """Generate testing task templates."""
        return [
            TaskDefinition(
                name="Write Integration Tests",
                description="Write comprehensive integration tests for API endpoints",
                category="testing",
                difficulty="medium",
                required_tools=["Read", "Write", "Bash"],
                expected_outcomes=[
                    "Test happy paths",
                    "Test error cases",
                    "Test edge cases",
                ],
                constraints=["Use existing test framework", "Mock external services"],
            ),
            TaskDefinition(
                name="Add Property-Based Tests",
                description="Implement property-based tests for core algorithms",
                category="testing",
                difficulty="hard",
                required_tools=["Read", "Write", "Edit"],
                expected_outcomes=[
                    "Identify properties to test",
                    "Implement generators",
                    "Write property tests",
                ],
                constraints=["Use hypothesis or similar", "Focus on invariants"],
            ),
            TaskDefinition(
                name="Performance Test Suite",
                description="Create performance tests for critical paths",
                category="testing",
                difficulty="medium",
                required_tools=["Read", "Write", "Bash"],
                expected_outcomes=[
                    "Identify critical paths",
                    "Write performance tests",
                    "Set up benchmarks",
                ],
                constraints=["Measure consistently", "Set realistic thresholds"],
            ),
        ]

    @staticmethod
    def get_documentation_tasks() -> List[TaskDefinition]:
        """Generate documentation task templates."""
        return [
            TaskDefinition(
                name="API Documentation",
                description="Generate comprehensive API documentation",
                category="documentation",
                difficulty="easy",
                required_tools=["Read", "Write"],
                expected_outcomes=[
                    "Document all endpoints",
                    "Include examples",
                    "Add authentication info",
                ],
                constraints=["Use OpenAPI/Swagger format", "Keep examples realistic"],
            ),
            TaskDefinition(
                name="Architecture Documentation",
                description="Document system architecture and design decisions",
                category="documentation",
                difficulty="medium",
                required_tools=["Read", "Write", "Grep"],
                expected_outcomes=[
                    "Create architecture diagram",
                    "Document key decisions",
                    "Explain trade-offs",
                ],
                constraints=["Use standard notation", "Keep it up to date"],
            ),
        ]

    @staticmethod
    def get_multi_agent_tasks() -> List[TaskDefinition]:
        """Generate multi-agent coordination task templates."""
        return [
            TaskDefinition(
                name="Full Stack Feature Implementation",
                description="Implement a complete feature requiring frontend, backend, and database changes",
                category="multi_agent",
                difficulty="expert",
                required_tools=["Read", "Write", "Edit", "Bash", "Grep"],
                expected_outcomes=[
                    "Design API endpoints",
                    "Implement backend logic",
                    "Create frontend components",
                    "Update database schema",
                    "Write tests",
                ],
                constraints=["Maintain consistency", "Follow existing patterns"],
                metadata={
                    "requires_agents": ["backend", "frontend", "database", "tester"]
                },
            ),
            TaskDefinition(
                name="Security Audit and Fixes",
                description="Conduct security audit and fix vulnerabilities across the codebase",
                category="multi_agent",
                difficulty="expert",
                required_tools=["Read", "Grep", "Edit", "Bash"],
                expected_outcomes=[
                    "Scan for vulnerabilities",
                    "Prioritize issues",
                    "Fix critical vulnerabilities",
                    "Update dependencies",
                    "Document findings",
                ],
                constraints=[
                    "Don't break functionality",
                    "Follow security best practices",
                ],
                metadata={
                    "requires_agents": ["security_auditor", "developer", "reviewer"]
                },
            ),
            TaskDefinition(
                name="Performance Optimization Sprint",
                description="Optimize application performance across multiple components",
                category="multi_agent",
                difficulty="hard",
                required_tools=["Read", "Edit", "Bash", "Grep"],
                expected_outcomes=[
                    "Profile application",
                    "Identify bottlenecks",
                    "Optimize database queries",
                    "Optimize frontend rendering",
                    "Implement caching",
                ],
                constraints=["Maintain functionality", "Document changes"],
                metadata={
                    "requires_agents": [
                        "profiler",
                        "backend_optimizer",
                        "frontend_optimizer",
                        "tester",
                    ]
                },
            ),
        ]


class TrainingDataGenerator:
    """Generates training data by executing tasks with various strategies."""

    def __init__(self, storage, task_library: Optional[TaskTemplateLibrary] = None):
        self.storage = storage
        self.task_library = task_library or TaskTemplateLibrary()
        self.collector = TrainingDataCollector(storage)

    async def generate_single_task_examples(
        self,
        task_category: str,
        count: int = 5,
        options: Optional[ClaudeCodeOptions] = None,
    ) -> List[TrainingExample]:
        """Generate training examples for single-task scenarios."""
        examples = []

        # Get tasks for category
        tasks = self._get_tasks_by_category(task_category)
        if not tasks:
            raise ValueError(f"No tasks found for category: {task_category}")

        # Generate examples
        for i in range(count):
            task = random.choice(tasks)
            prompt = self._generate_task_prompt(task)

            if not options:
                options = ClaudeCodeOptions(
                    allowed_tools=task.required_tools, max_thinking_tokens=10000
                )

            example = None
            async with self.collector.collect_from_query(
                task=task, prompt=prompt, query_func=query, options=options
            ) as message_stream:
                # Process messages (they're automatically collected)
                async for message in message_stream:
                    pass

            # Example is saved automatically by collector
            examples.append(example)

            # Add variation
            await asyncio.sleep(1)  # Prevent rate limiting

        return examples

    async def generate_chain_examples(
        self, chain_type: str, count: int = 3
    ) -> List[TrainingExample]:
        """Generate training examples using conversation chains."""
        examples = []
        chain_collector = ConversationChainCollector(self.storage)

        for i in range(count):
            # Create chain based on type
            chain = self._create_chain(chain_type)
            task = self._create_task_for_chain(chain_type)

            # Generate initial prompt
            prompt = self._generate_task_prompt(task)

            # Collect from chain execution
            example = await chain_collector.collect_from_chain(
                chain=chain, task=task, initial_prompt=prompt
            )

            if example:
                examples.append(example)

            await asyncio.sleep(1)

        return examples

    async def generate_multi_agent_examples(
        self, scenario: str, count: int = 2
    ) -> List[TrainingExample]:
        """Generate training examples from multi-agent scenarios."""
        examples = []

        multi_agent_tasks = TaskTemplateLibrary.get_multi_agent_tasks()
        scenario_tasks = [t for t in multi_agent_tasks if scenario in t.name.lower()]

        if not scenario_tasks:
            scenario_tasks = multi_agent_tasks

        for i in range(count):
            task = random.choice(scenario_tasks)

            # Simulate multi-agent execution
            example = await self._simulate_multi_agent_execution(task)
            if example:
                examples.append(example)

            await asyncio.sleep(2)

        return examples

    def _get_tasks_by_category(self, category: str) -> List[TaskDefinition]:
        """Get tasks for a specific category."""
        method_map = {
            "code_review": TaskTemplateLibrary.get_code_review_tasks,
            "debugging": TaskTemplateLibrary.get_debugging_tasks,
            "refactoring": TaskTemplateLibrary.get_refactoring_tasks,
            "implementation": TaskTemplateLibrary.get_implementation_tasks,
            "testing": TaskTemplateLibrary.get_testing_tasks,
            "documentation": TaskTemplateLibrary.get_documentation_tasks,
            "multi_agent": TaskTemplateLibrary.get_multi_agent_tasks,
        }

        if category in method_map:
            return method_map[category]()

        # Return all tasks if category not found
        all_tasks = []
        for method in method_map.values():
            all_tasks.extend(method())
        return all_tasks

    def _generate_task_prompt(self, task: TaskDefinition) -> str:
        """Generate a prompt for a task."""
        prompt_parts = [task.description, "", "Requirements:"]

        for outcome in task.expected_outcomes:
            prompt_parts.append(f"- {outcome}")

        if task.constraints:
            prompt_parts.append("")
            prompt_parts.append("Constraints:")
            for constraint in task.constraints:
                prompt_parts.append(f"- {constraint}")

        return "\n".join(prompt_parts)

    def _create_chain(self, chain_type: str) -> ConversationChain:
        """Create a conversation chain for testing."""
        if chain_type == "debugging":
            return ConversationChain(
                name="debugging_chain",
                steps=[
                    ChainStep(
                        name="identify_issue",
                        prompt="First, identify the root cause of the issue",
                        allowed_tools=["Read", "Grep", "Bash"],
                    ),
                    ChainStep(
                        name="reproduce",
                        prompt="Reproduce the issue to confirm understanding",
                        allowed_tools=["Bash"],
                        depends_on=["identify_issue"],
                    ),
                    ChainStep(
                        name="fix",
                        prompt="Implement a fix for the issue",
                        allowed_tools=["Edit", "Write"],
                        depends_on=["reproduce"],
                    ),
                    ChainStep(
                        name="verify",
                        prompt="Verify the fix works correctly",
                        allowed_tools=["Bash"],
                        depends_on=["fix"],
                    ),
                ],
            )
        elif chain_type == "refactoring":
            return ConversationChain(
                name="refactoring_chain",
                steps=[
                    ChainStep(
                        name="analyze",
                        prompt="Analyze the code structure and identify refactoring opportunities",
                        allowed_tools=["Read", "Grep"],
                    ),
                    ChainStep(
                        name="plan",
                        prompt="Plan the refactoring approach",
                        depends_on=["analyze"],
                    ),
                    ChainStep(
                        name="refactor",
                        prompt="Implement the refactoring",
                        allowed_tools=["Edit", "Write"],
                        depends_on=["plan"],
                    ),
                    ChainStep(
                        name="test",
                        prompt="Run tests to ensure nothing broke",
                        allowed_tools=["Bash"],
                        depends_on=["refactor"],
                    ),
                ],
            )
        else:
            # Default chain
            return ConversationChain(
                name="default_chain",
                steps=[
                    ChainStep(
                        name="understand",
                        prompt="Understand the requirements",
                        allowed_tools=["Read", "Grep"],
                    ),
                    ChainStep(
                        name="implement",
                        prompt="Implement the solution",
                        allowed_tools=["Write", "Edit"],
                        depends_on=["understand"],
                    ),
                    ChainStep(
                        name="verify",
                        prompt="Verify the implementation",
                        allowed_tools=["Bash", "Read"],
                        depends_on=["implement"],
                    ),
                ],
            )

    def _create_task_for_chain(self, chain_type: str) -> TaskDefinition:
        """Create a task definition for a chain type."""
        if chain_type == "debugging":
            return TaskDefinition(
                name=f"Chain Debugging Task",
                description="Debug an issue using a systematic chain approach",
                category="debugging",
                difficulty="medium",
                required_tools=["Read", "Grep", "Edit", "Bash"],
                expected_outcomes=["Issue identified", "Fix implemented", "Tests pass"],
            )
        elif chain_type == "refactoring":
            return TaskDefinition(
                name=f"Chain Refactoring Task",
                description="Refactor code using a systematic chain approach",
                category="refactoring",
                difficulty="medium",
                required_tools=["Read", "Grep", "Edit", "Write", "Bash"],
                expected_outcomes=[
                    "Code analyzed",
                    "Refactoring completed",
                    "Tests pass",
                ],
            )
        else:
            return TaskDefinition(
                name=f"Chain Task",
                description="Complete a task using a conversation chain",
                category="general",
                difficulty="medium",
                required_tools=["Read", "Write", "Edit", "Bash", "Grep"],
                expected_outcomes=["Task completed successfully"],
            )

    async def _simulate_multi_agent_execution(
        self, task: TaskDefinition
    ) -> Optional[TrainingExample]:
        """Simulate a multi-agent execution for training data."""
        collector = MultiAgentCollector(self.storage)

        # Simulate agent actions based on task requirements
        required_agents = task.metadata.get("requires_agents", ["agent1", "agent2"])

        # Coordinator starts
        collector.track_agent_action(
            agent_id="coordinator",
            agent_role=AgentRole.COORDINATOR,
            action_type="planning",
            content={"plan": f"Executing task: {task.name}"},
            reasoning="Breaking down task into subtasks for agents",
        )

        # Simulate agent interactions
        for i, agent_id in enumerate(required_agents):
            # Agent receives task
            collector.track_agent_action(
                agent_id=agent_id,
                agent_role=AgentRole.SPECIALIST,
                action_type="task_received",
                content={"subtask": f"Part {i + 1} of {task.name}"},
                confidence=0.9,
            )

            # Agent performs work
            tools_to_use = random.sample(
                task.required_tools, min(3, len(task.required_tools))
            )

            for tool in tools_to_use:
                collector.track_agent_action(
                    agent_id=agent_id,
                    agent_role=AgentRole.SPECIALIST,
                    action_type="tool_use",
                    content={"tool": tool, "purpose": "Executing subtask"},
                    tool_name=tool,
                    confidence=random.uniform(0.7, 1.0),
                )

            # Agent completes subtask
            collector.track_agent_action(
                agent_id=agent_id,
                agent_role=AgentRole.SPECIALIST,
                action_type="task_completed",
                content={"result": f"Completed part {i + 1}"},
                reasoning="Subtask completed successfully",
            )

        # Coordinator reviews
        collector.track_agent_action(
            agent_id="coordinator",
            agent_role=AgentRole.COORDINATOR,
            action_type="review",
            content={"status": "All subtasks completed"},
            reasoning="Reviewing agent outputs and preparing final result",
        )

        # Create example
        success = random.random() > 0.2  # 80% success rate
        example = collector.create_multi_agent_example(
            task=task,
            coordinator_session_id=f"multi_agent_sim_{datetime.now().timestamp()}",
            outcome="Task completed by multi-agent system"
            if success
            else "Task failed",
            success=success,
            cost_usd=random.uniform(0.01, 0.10),
        )

        return example


class BatchTrainingGenerator:
    """Generate training data in batches with various scenarios."""

    def __init__(self, storage, output_dir: Optional[Path] = None):
        self.storage = storage
        self.generator = TrainingDataGenerator(storage)
        self.output_dir = output_dir or Path("training_data")
        self.output_dir.mkdir(exist_ok=True)

    async def generate_diverse_dataset(
        self,
        examples_per_category: int = 5,
        include_chains: bool = True,
        include_multi_agent: bool = True,
    ) -> Dict[str, Any]:
        """Generate a diverse dataset covering multiple categories."""
        stats = {
            "total_examples": 0,
            "by_category": {},
            "by_difficulty": {},
            "golden_examples": 0,
            "failed_examples": 0,
        }

        # Generate single-task examples
        categories = [
            "code_review",
            "debugging",
            "refactoring",
            "implementation",
            "testing",
        ]

        for category in categories:
            print(f"Generating {category} examples...")
            try:
                examples = await self.generator.generate_single_task_examples(
                    task_category=category, count=examples_per_category
                )
                stats["by_category"][category] = len(examples)
                stats["total_examples"] += len(examples)

                # Count golden examples
                stats["golden_examples"] += sum(
                    1 for ex in examples if ex and ex.is_golden
                )

            except Exception as e:
                print(f"Error generating {category} examples: {e}")
                stats["by_category"][category] = 0

        # Generate chain examples
        if include_chains:
            print("Generating chain examples...")
            try:
                chain_examples = []
                for chain_type in ["debugging", "refactoring"]:
                    examples = await self.generator.generate_chain_examples(
                        chain_type=chain_type, count=3
                    )
                    chain_examples.extend(examples)

                stats["by_category"]["chains"] = len(chain_examples)
                stats["total_examples"] += len(chain_examples)

            except Exception as e:
                print(f"Error generating chain examples: {e}")
                stats["by_category"]["chains"] = 0

        # Generate multi-agent examples
        if include_multi_agent:
            print("Generating multi-agent examples...")
            try:
                multi_examples = await self.generator.generate_multi_agent_examples(
                    scenario="feature", count=3
                )
                stats["by_category"]["multi_agent"] = len(multi_examples)
                stats["total_examples"] += len(multi_examples)

            except Exception as e:
                print(f"Error generating multi-agent examples: {e}")
                stats["by_category"]["multi_agent"] = 0

        # Export dataset
        print("Exporting dataset...")
        export_path = self.storage.export_for_training(
            format="jsonl",
            output_path=self.output_dir
            / f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        )

        stats["export_path"] = str(export_path)

        # Get analytics if using DuckDB
        if hasattr(self.storage, "get_analytics"):
            stats["analytics"] = self.storage.get_analytics()

        return stats

    async def generate_golden_examples(self, count: int = 10) -> List[TrainingExample]:
        """Generate high-quality golden examples with manual curation."""
        golden_examples = []

        # Focus on well-defined tasks with clear outcomes
        golden_tasks = [
            TaskDefinition(
                name="Fix Simple Type Error",
                description="Fix a TypeScript type error in a function",
                category="debugging",
                difficulty="easy",
                required_tools=["Read", "Edit"],
                expected_outcomes=["Type error resolved", "Function works correctly"],
                constraints=["Don't change function behavior"],
            ),
            TaskDefinition(
                name="Add Input Validation",
                description="Add input validation to an API endpoint",
                category="implementation",
                difficulty="easy",
                required_tools=["Read", "Edit"],
                expected_outcomes=[
                    "Validate required fields",
                    "Return appropriate errors",
                ],
                constraints=["Use existing validation library"],
            ),
            TaskDefinition(
                name="Extract Duplicate Code",
                description="Extract duplicate code into a reusable function",
                category="refactoring",
                difficulty="medium",
                required_tools=["Read", "Grep", "Edit", "Write"],
                expected_outcomes=[
                    "Identify duplication",
                    "Create reusable function",
                    "Update call sites",
                ],
                constraints=["Maintain existing behavior"],
            ),
        ]

        # Generate examples with specific options for quality
        for i in range(count):
            task = golden_tasks[i % len(golden_tasks)]

            options = ClaudeCodeOptions(
                allowed_tools=task.required_tools,
                max_thinking_tokens=15000,  # More thinking for quality
                permission_mode="acceptEdits",
            )

            # Generate with collection
            async with self.generator.collector.collect_from_query(
                task=task,
                prompt=self.generator._generate_task_prompt(task),
                query_func=query,
                options=options,
            ) as message_stream:
                async for message in message_stream:
                    pass

            # Manually mark as golden if quality score is high
            examples = self.storage.get_training_examples(
                filters={"min_quality": 0.7}, limit=1
            )

            if examples:
                example = examples[0]
                example.is_golden = True
                example.validated = True
                example.validator_notes = "Auto-generated golden example"
                self.storage.save_training_example(example)
                golden_examples.append(example)

        return golden_examples
