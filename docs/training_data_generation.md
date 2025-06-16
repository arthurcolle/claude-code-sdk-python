# Training Data Generation for Task-Executing Agents

The Claude SDK includes a comprehensive training data generation system for creating high-quality datasets for training advanced task-executing agents. This system supports both SQLite and DuckDB storage backends.

## Overview

The training data system consists of four main components:

1. **Data Models** (`training_data.py`) - Core data structures for tasks, executions, and training examples
2. **Collectors** (`training_collector.py`) - Capture interactions and agent behaviors
3. **Generators** (`training_generator.py`) - Create diverse training datasets
4. **Exporters** (`training_export.py`) - Export data in various ML-ready formats

## Quick Start

```python
import asyncio
from claude_code_sdk.training_data import (
    TaskDefinition, SQLiteTrainingStorage
)
from claude_code_sdk.training_collector import TrainingDataCollector
from claude_code_sdk import query, ClaudeCodeOptions

async def collect_training_data():
    # Initialize storage
    storage = SQLiteTrainingStorage("training_data.db")
    collector = TrainingDataCollector(storage)
    
    # Define a task
    task = TaskDefinition(
        name="Fix Type Error",
        description="Fix the TypeScript type error in utils.ts",
        category="debugging",
        difficulty="easy",
        required_tools=["Read", "Edit"],
        expected_outcomes=["Type error resolved"]
    )
    
    # Collect training data
    async with collector.collect_from_query(
        task=task,
        prompt="Fix the type error in utils.ts line 42",
        query_func=query,
        options=ClaudeCodeOptions(allowed_tools=task.required_tools)
    ) as messages:
        async for msg in messages:
            pass  # Messages are automatically collected
    
    print("Training example saved!")

asyncio.run(collect_training_data())
```

## Storage Backends

### SQLite Storage

Best for smaller datasets and simple queries:

```python
from claude_code_sdk.training_data import SQLiteTrainingStorage

storage = SQLiteTrainingStorage("training_data.db")
```

### DuckDB Storage

Better for large datasets and analytics:

```python
from claude_code_sdk.training_data import DuckDBTrainingStorage

storage = DuckDBTrainingStorage("training_data.duckdb")

# DuckDB includes built-in analytics
analytics = storage.get_analytics()
print(f"Total examples: {analytics['total_examples']}")
print(f"Success rate: {analytics['success_rate']:.1%}")
```

## Task Templates

Pre-defined task templates for common scenarios:

```python
from claude_code_sdk.training_generator import TaskTemplateLibrary

# Get tasks by category
review_tasks = TaskTemplateLibrary.get_code_review_tasks()
debug_tasks = TaskTemplateLibrary.get_debugging_tasks()
refactor_tasks = TaskTemplateLibrary.get_refactoring_tasks()

# Categories available:
# - code_review: Security reviews, performance analysis, API design
# - debugging: Fix tests, memory leaks, race conditions
# - refactoring: Extract components, async migration, design patterns
# - implementation: Caching, rate limiting, data export
# - testing: Integration tests, property tests, performance tests
# - multi_agent: Full stack features, security audits, optimization
```

## Batch Generation

Generate complete datasets efficiently:

```python
from claude_code_sdk.training_generator import BatchTrainingGenerator

async def generate_dataset():
    storage = DuckDBTrainingStorage("dataset.duckdb")
    generator = BatchTrainingGenerator(storage)
    
    # Generate diverse dataset
    stats = await generator.generate_diverse_dataset(
        examples_per_category=10,
        include_chains=True,
        include_multi_agent=True
    )
    
    print(f"Generated {stats['total_examples']} examples")
    print(f"Golden examples: {stats['golden_examples']}")
```

## Quality Scoring

Examples are automatically scored for quality (0-1 scale):

- **Success** (40%): Did the task complete successfully?
- **Efficiency** (30%): How quickly was it completed?
- **Cost** (20%): How much did it cost in API usage?
- **Tool Usage** (10%): Were tools used efficiently?

Custom quality scorers can be provided:

```python
def custom_scorer(execution):
    score = 0.0
    if execution.success:
        score += 0.5
    if len(execution.actions) < 10:
        score += 0.3
    if execution.cost_usd < 0.05:
        score += 0.2
    return min(score, 1.0)

collector = TrainingDataCollector(storage, quality_scorer=custom_scorer)
```

## Export Formats

### For Fine-Tuning LLMs

```python
from claude_code_sdk.training_export import TrainingDataExporter

exporter = TrainingDataExporter(storage)

# JSONL format (OpenAI style)
exporter.export_for_fine_tuning(
    output_path="fine_tuning.jsonl",
    format="jsonl"
)

# Conversation format
exporter.export_for_fine_tuning(
    output_path="conversations.json",
    format="conversations"
)

# Instruction-following format
exporter.export_for_fine_tuning(
    output_path="instructions.jsonl",
    format="instruct"
)
```

### For Evaluation

```python
# Export golden examples for evaluation
exporter.export_for_evaluation(
    output_path="eval_dataset.json",
    format="json",
    include_golden_only=True
)
```

### For ML Frameworks

```python
# HuggingFace datasets format
paths = exporter.export_for_ml_frameworks(
    output_dir="hf_dataset",
    framework="huggingface",
    split_ratio=(0.8, 0.1, 0.1)  # train/val/test
)
```

### Analytics Export

```python
# HTML dashboard with visualizations
exporter.export_analytics(
    output_dir="analytics",
    format="html"
)

# CSV files for analysis
exporter.export_analytics(
    output_dir="analytics",
    format="csv"
)
```

## Multi-Agent Training Data

Collect training data from multi-agent interactions:

```python
from claude_code_sdk.training_collector import MultiAgentCollector
from claude_code_sdk.training_data import AgentRole

collector = MultiAgentCollector(storage)

# Track agent actions
collector.track_agent_action(
    agent_id="backend_dev",
    agent_role=AgentRole.SPECIALIST,
    action_type="tool_use",
    tool_name="Write",
    content={"file": "api.py", "action": "create"},
    reasoning="Creating API endpoints"
)

# Create training example
example = collector.create_multi_agent_example(
    task=task,
    coordinator_session_id="session_123",
    outcome="Feature implemented successfully",
    success=True,
    cost_usd=0.10
)
```

## Conversation Chains

Collect data from complex workflows:

```python
from claude_code_sdk.training_collector import ConversationChainCollector
from claude_code_sdk.conversation_chains import ConversationChain, ChainStep

# Define a debugging chain
chain = ConversationChain(
    name="debug_chain",
    steps=[
        ChainStep(
            name="identify",
            prompt="Identify the root cause",
            allowed_tools=["Read", "Grep"]
        ),
        ChainStep(
            name="fix",
            prompt="Fix the issue",
            allowed_tools=["Edit"],
            depends_on=["identify"]
        ),
        ChainStep(
            name="verify",
            prompt="Verify the fix",
            allowed_tools=["Bash"],
            depends_on=["fix"]
        )
    ]
)

# Collect from chain execution
collector = ConversationChainCollector(storage)
example = await collector.collect_from_chain(
    chain=chain,
    task=task,
    initial_prompt="Debug the failing test"
)
```

## Filtering and Querying

Retrieve specific training examples:

```python
# Get high-quality examples
examples = storage.get_training_examples(
    filters={
        "min_quality": 0.8,
        "category": "debugging",
        "is_golden": True
    },
    limit=100
)

# Get examples for specific difficulty
examples = storage.get_training_examples(
    filters={"difficulty": "hard"},
    limit=50
)
```

## Best Practices

1. **Task Definition**: Be specific about expected outcomes and constraints
2. **Quality Thresholds**: Set `is_golden=True` for examples with quality_score >= 0.8
3. **Diverse Data**: Generate examples across all categories and difficulties
4. **Validation**: Review and validate golden examples before using for training
5. **Storage Choice**: Use SQLite for <100k examples, DuckDB for larger datasets
6. **Export Format**: Choose format based on your training framework requirements

## Example: Complete Training Pipeline

See `examples/training_data_generation.py` for a complete demonstration of:

- Basic collection from SDK interactions
- Using task templates
- Batch generation
- Multi-agent scenarios
- Custom quality scoring
- Exporting in various formats
- Analytics generation

Run the example:

```bash
python examples/training_data_generation.py
```

This will generate sample training data in the `training_data/` directory with various storage backends and export formats.