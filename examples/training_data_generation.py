#!/usr/bin/env python3
"""
Training Data Generation Example

This example demonstrates how to generate, collect, and export training data
for advanced task-executing agents using the Claude SDK.

Features demonstrated:
- Creating task definitions for various categories
- Collecting training data from SDK interactions
- Using both SQLite and DuckDB storage backends
- Generating diverse datasets with quality scoring
- Exporting data in multiple formats for ML frameworks
- Analytics and visualization of training data
"""

import asyncio
from pathlib import Path
from datetime import datetime
import json

# Training data components
from claude_code_sdk.training_data import (
    TaskDefinition, SQLiteTrainingStorage, DuckDBTrainingStorage,
    TrainingExample, AgentRole
)
from claude_code_sdk.training_collector import (
    TrainingDataCollector, ConversationChainCollector, MultiAgentCollector
)
from claude_code_sdk.training_generator import (
    TrainingDataGenerator, BatchTrainingGenerator, TaskTemplateLibrary
)
from claude_code_sdk.training_export import TrainingDataExporter

# SDK components
from claude_code_sdk import query, ClaudeCodeOptions
from claude_code_sdk.conversation_chains import ConversationChain, ChainStep


async def demonstrate_basic_collection():
    """Demonstrate basic training data collection."""
    print("=== Basic Training Data Collection ===\n")
    
    # Initialize storage
    storage = SQLiteTrainingStorage("training_data/basic_examples.db")
    collector = TrainingDataCollector(storage)
    
    # Define a simple task
    task = TaskDefinition(
        name="Fix Python Import Error",
        description="Fix the import error in main.py by installing the missing package",
        category="debugging",
        difficulty="easy",
        required_tools=["Read", "Bash"],
        expected_outcomes=["Import error resolved", "Package installed"],
        constraints=["Use pip to install packages"]
    )
    
    # Collect training data from a query
    print(f"Executing task: {task.name}")
    
    async with collector.collect_from_query(
        task=task,
        prompt="There's an import error in main.py. The module 'requests' is not found. Please fix it.",
        query_func=query,
        options=ClaudeCodeOptions(
            allowed_tools=task.required_tools,
            permission_mode="acceptEdits"
        )
    ) as message_stream:
        async for message in message_stream:
            # Messages are automatically collected
            pass
    
    print("✓ Training example collected and saved\n")


async def demonstrate_task_templates():
    """Demonstrate using task templates for various categories."""
    print("=== Task Template Library ===\n")
    
    # Show available task categories
    categories = {
        "Code Review": TaskTemplateLibrary.get_code_review_tasks(),
        "Debugging": TaskTemplateLibrary.get_debugging_tasks(),
        "Refactoring": TaskTemplateLibrary.get_refactoring_tasks(),
        "Implementation": TaskTemplateLibrary.get_implementation_tasks(),
        "Testing": TaskTemplateLibrary.get_testing_tasks(),
        "Multi-Agent": TaskTemplateLibrary.get_multi_agent_tasks()
    }
    
    for category, tasks in categories.items():
        print(f"{category} Tasks ({len(tasks)}):")
        for task in tasks[:2]:  # Show first 2 tasks per category
            print(f"  - {task.name} ({task.difficulty})")
        if len(tasks) > 2:
            print(f"  ... and {len(tasks) - 2} more")
        print()


async def demonstrate_batch_generation():
    """Demonstrate batch training data generation."""
    print("=== Batch Training Data Generation ===\n")
    
    # Use DuckDB for better analytics
    storage = DuckDBTrainingStorage("training_data/batch_examples.duckdb")
    batch_generator = BatchTrainingGenerator(storage)
    
    print("Generating diverse dataset...")
    print("(This will generate synthetic examples - in production, use real interactions)")
    
    # For demonstration, we'll simulate some examples
    generator = TrainingDataGenerator(storage)
    
    # Generate a few examples for each category
    categories = ["debugging", "refactoring", "implementation"]
    
    for category in categories:
        print(f"\nGenerating {category} examples...")
        
        # Get tasks for this category
        tasks = TaskTemplateLibrary.get_debugging_tasks() if category == "debugging" else \
                TaskTemplateLibrary.get_refactoring_tasks() if category == "refactoring" else \
                TaskTemplateLibrary.get_implementation_tasks()
        
        # Generate 2 examples per category (reduced for demo)
        for i, task in enumerate(tasks[:2]):
            print(f"  Task {i+1}: {task.name}")
            
            # Create a simulated execution
            example = TrainingExample(
                task_definition=task,
                execution=_create_simulated_execution(task),
                quality_score=0.75 + (i * 0.1),  # Vary quality scores
                is_golden=i == 0,  # First example is golden
                tags=[f"category:{category}", "simulated"]
            )
            
            storage.save_training_example(example)
    
    print("\n✓ Batch generation complete")
    
    # Show analytics
    if hasattr(storage, 'get_analytics'):
        print("\nDataset Analytics:")
        analytics = storage.get_analytics()
        print(f"  Total examples: {analytics['total_examples']}")
        print(f"  Golden examples: {analytics['golden_examples']}")
        print(f"  Average quality: {analytics['avg_quality_score']:.2f}")


async def demonstrate_export_formats():
    """Demonstrate exporting training data in various formats."""
    print("\n=== Export Formats Demo ===\n")
    
    # Use existing storage
    storage = SQLiteTrainingStorage("training_data/basic_examples.db")
    exporter = TrainingDataExporter(storage)
    
    export_dir = Path("training_data/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Export for fine-tuning
    print("1. Exporting for LLM fine-tuning...")
    
    # JSONL format
    jsonl_path = exporter.export_for_fine_tuning(
        output_path=export_dir / "fine_tuning.jsonl",
        format="jsonl"
    )
    print(f"   ✓ JSONL export: {jsonl_path}")
    
    # Conversations format
    conv_path = exporter.export_for_fine_tuning(
        output_path=export_dir / "conversations.json",
        format="conversations"
    )
    print(f"   ✓ Conversations export: {conv_path}")
    
    # Instruction format
    instruct_path = exporter.export_for_fine_tuning(
        output_path=export_dir / "instructions.jsonl",
        format="instruct"
    )
    print(f"   ✓ Instruction format: {instruct_path}")
    
    # Export for evaluation
    print("\n2. Exporting for evaluation...")
    eval_path = exporter.export_for_evaluation(
        output_path=export_dir / "evaluation_dataset.json",
        format="json",
        include_golden_only=True
    )
    print(f"   ✓ Evaluation dataset: {eval_path}")
    
    # Export analytics
    print("\n3. Exporting analytics...")
    analytics_path = exporter.export_analytics(
        output_dir=export_dir / "analytics",
        format="html"
    )
    print(f"   ✓ Analytics dashboard: {analytics_path}")
    
    # Export for ML frameworks
    print("\n4. Exporting for ML frameworks...")
    hf_paths = exporter.export_for_ml_frameworks(
        output_dir=export_dir / "huggingface_dataset",
        framework="huggingface",
        split_ratio=(0.8, 0.1, 0.1)
    )
    print("   ✓ HuggingFace dataset splits:")
    for split, path in hf_paths.items():
        print(f"     - {split}: {path}")


async def demonstrate_multi_agent_collection():
    """Demonstrate multi-agent training data collection."""
    print("\n=== Multi-Agent Collection Demo ===\n")
    
    storage = SQLiteTrainingStorage("training_data/multi_agent.db")
    collector = MultiAgentCollector(storage)
    
    # Define a multi-agent task
    task = TaskDefinition(
        name="Implement User Authentication",
        description="Implement complete user authentication with frontend and backend",
        category="multi_agent",
        difficulty="hard",
        required_tools=["Read", "Write", "Edit", "Bash"],
        expected_outcomes=[
            "Backend authentication API",
            "Frontend login form",
            "Database schema for users",
            "Unit tests"
        ],
        metadata={"requires_agents": ["backend_dev", "frontend_dev", "db_admin", "tester"]}
    )
    
    print(f"Simulating multi-agent execution for: {task.name}")
    
    # Simulate agent interactions
    # Backend developer starts
    collector.track_agent_action(
        agent_id="backend_dev",
        agent_role=AgentRole.SPECIALIST,
        action_type="planning",
        content={"plan": "Design authentication API endpoints"},
        reasoning="Need to define API structure before implementation"
    )
    
    collector.track_agent_action(
        agent_id="backend_dev",
        agent_role=AgentRole.SPECIALIST,
        action_type="tool_use",
        tool_name="Write",
        content={"file": "auth_api.py", "action": "create"},
        reasoning="Creating authentication endpoints"
    )
    
    # Frontend developer works in parallel
    collector.track_agent_action(
        agent_id="frontend_dev",
        agent_role=AgentRole.SPECIALIST,
        action_type="tool_use",
        tool_name="Write",
        content={"file": "LoginForm.jsx", "action": "create"},
        reasoning="Creating login UI component"
    )
    
    # Database admin sets up schema
    collector.track_agent_action(
        agent_id="db_admin",
        agent_role=AgentRole.SPECIALIST,
        action_type="tool_use",
        tool_name="Write",
        content={"file": "schema.sql", "action": "create"},
        reasoning="Setting up user tables and indexes"
    )
    
    # Tester validates
    collector.track_agent_action(
        agent_id="tester",
        agent_role=AgentRole.SPECIALIST,
        action_type="tool_use",
        tool_name="Write",
        content={"file": "test_auth.py", "action": "create"},
        reasoning="Writing integration tests for authentication"
    )
    
    # Create training example
    example = collector.create_multi_agent_example(
        task=task,
        coordinator_session_id="multi_agent_demo_001",
        outcome="Successfully implemented user authentication system",
        success=True,
        cost_usd=0.05
    )
    
    print(f"✓ Multi-agent example saved with quality score: {example.quality_score:.2f}")
    print(f"  Coordination quality: {example.annotations['coordination_quality']:.2f}")
    print("  Agent efficiency scores:")
    for agent, score in example.annotations['agent_efficiency'].items():
        print(f"    - {agent}: {score:.2f}")


async def demonstrate_custom_quality_scoring():
    """Demonstrate custom quality scoring for training examples."""
    print("\n=== Custom Quality Scoring ===\n")
    
    # Define custom quality scorer
    def custom_quality_scorer(execution):
        """Custom scoring based on specific criteria."""
        score = 0.0
        
        # Base score for completion
        if execution.success:
            score += 0.3
        
        # Efficiency scoring
        action_count = len(execution.actions)
        if action_count <= 5:
            score += 0.3
        elif action_count <= 10:
            score += 0.2
        elif action_count <= 15:
            score += 0.1
        
        # Tool usage efficiency
        tools_used = set(a.tool_name for a in execution.actions if a.tool_name)
        if len(tools_used) <= 3:
            score += 0.2
        elif len(tools_used) <= 5:
            score += 0.1
        
        # Reasoning quality
        reasoning_count = sum(1 for a in execution.actions if a.reasoning)
        if reasoning_count >= len(execution.actions) * 0.8:
            score += 0.2
        elif reasoning_count >= len(execution.actions) * 0.5:
            score += 0.1
        
        return min(score, 1.0)
    
    # Create collector with custom scorer
    storage = SQLiteTrainingStorage("training_data/custom_scoring.db")
    collector = TrainingDataCollector(
        storage=storage,
        quality_scorer=custom_quality_scorer
    )
    
    print("Using custom quality scorer that evaluates:")
    print("  - Task completion (30%)")
    print("  - Action efficiency (30%)")
    print("  - Tool usage efficiency (20%)")
    print("  - Reasoning quality (20%)")
    
    # Example scoring
    task = TaskDefinition(
        name="Optimize Database Query",
        description="Optimize a slow database query",
        category="optimization",
        difficulty="medium",
        required_tools=["Read", "Edit"]
    )
    
    # Simulate execution for scoring demo
    from claude_code_sdk.training_data import TaskExecution, TaskStatus, AgentAction
    
    execution = TaskExecution(
        task_id=task.id,
        session_id="demo_session",
        agent_id="optimizer",
        status=TaskStatus.COMPLETED,
        success=True,
        started_at=datetime.now(),
        completed_at=datetime.now(),
        actions=[
            AgentAction(
                agent_id="optimizer",
                agent_role=AgentRole.TASK_EXECUTOR,
                action_type="tool_use",
                tool_name="Read",
                reasoning="First, I need to examine the slow query"
            ),
            AgentAction(
                agent_id="optimizer",
                agent_role=AgentRole.TASK_EXECUTOR,
                action_type="analysis",
                content={"finding": "Missing index on user_id column"},
                reasoning="The query is slow because it's doing a full table scan"
            ),
            AgentAction(
                agent_id="optimizer",
                agent_role=AgentRole.TASK_EXECUTOR,
                action_type="tool_use",
                tool_name="Edit",
                reasoning="Adding index to improve query performance"
            )
        ]
    )
    
    quality_score = custom_quality_scorer(execution)
    print(f"\nExample execution scored: {quality_score:.2f}")
    print("  ✓ Task completed successfully: +0.30")
    print("  ✓ Only 3 actions (efficient): +0.30")
    print("  ✓ Used 2 tools (efficient): +0.20")
    print("  ✓ All actions have reasoning: +0.20")


def _create_simulated_execution(task):
    """Create a simulated execution for demonstration."""
    from claude_code_sdk.training_data import TaskExecution, TaskStatus, AgentAction
    import uuid
    import random
    
    execution = TaskExecution(
        id=str(uuid.uuid4()),
        task_id=task.id,
        session_id=f"sim_{uuid.uuid4()}",
        agent_id="simulated_agent",
        status=TaskStatus.COMPLETED,
        success=True,
        started_at=datetime.now(),
        completed_at=datetime.now(),
        actions=[],
        messages=[
            {"type": "user", "content": task.description},
            {"type": "assistant", "content": [
                {"type": "text", "text": f"I'll help you with {task.name}"}
            ]}
        ],
        outcome=f"Successfully completed {task.name}",
        cost_usd=random.uniform(0.01, 0.10),
        tokens_used={"input": random.randint(100, 1000), "output": random.randint(200, 2000)}
    )
    
    # Add some simulated actions
    for i, tool in enumerate(task.required_tools[:3]):
        execution.actions.append(AgentAction(
            agent_id="simulated_agent",
            agent_role=AgentRole.TASK_EXECUTOR,
            action_type="tool_use",
            tool_name=tool,
            content={"step": i+1},
            reasoning=f"Using {tool} to complete step {i+1}",
            confidence=random.uniform(0.7, 1.0)
        ))
    
    return execution


async def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("Claude SDK Training Data Generation Demo")
    print("=" * 60)
    print()
    
    # Create output directory
    Path("training_data").mkdir(exist_ok=True)
    
    try:
        # Run demonstrations
        await demonstrate_basic_collection()
        await demonstrate_task_templates()
        await demonstrate_batch_generation()
        await demonstrate_export_formats()
        await demonstrate_multi_agent_collection()
        await demonstrate_custom_quality_scoring()
        
        print("\n" + "=" * 60)
        print("✅ All demonstrations completed successfully!")
        print("\nGenerated files can be found in the 'training_data' directory:")
        print("  - basic_examples.db: SQLite database with basic examples")
        print("  - batch_examples.duckdb: DuckDB database with batch examples")
        print("  - multi_agent.db: Multi-agent training examples")
        print("  - exports/: Various export formats")
        print("    - fine_tuning.jsonl: Ready for LLM fine-tuning")
        print("    - evaluation_dataset.json: For model evaluation")
        print("    - analytics/: HTML analytics dashboard")
        print("    - huggingface_dataset/: HuggingFace-compatible dataset")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())