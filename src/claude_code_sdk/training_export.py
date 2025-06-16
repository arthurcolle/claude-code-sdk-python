"""Export utilities for training data in various formats."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
import json
import csv
from datetime import datetime
from dataclasses import asdict
from pathlib import Path

# Optional imports with fallback
try:
    import yaml
except ImportError:
    yaml = None

try:
    import pandas as pd
except ImportError:
    pd = None

import sqlite3
import duckdb

from .training_data import TrainingExample, TaskExecution, AgentAction


class TrainingDataExporter:
    """Export training data in various formats for different ML frameworks."""

    def __init__(self, storage):
        self.storage = storage

    def export_for_fine_tuning(
        self,
        output_path: Path,
        format: str = "jsonl",
        filters: Optional[Dict[str, Any]] = None,
        transform_fn: Optional[Callable[[TrainingExample], Dict[str, Any]]] = None,
    ) -> Path:
        """Export data formatted for fine-tuning LLMs."""
        examples = self.storage.get_training_examples(filters=filters)

        if format == "jsonl":
            return self._export_jsonl_fine_tuning(examples, output_path, transform_fn)
        elif format == "conversations":
            return self._export_conversations_format(examples, output_path)
        elif format == "instruct":
            return self._export_instruct_format(examples, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_jsonl_fine_tuning(
        self,
        examples: List[TrainingExample],
        output_path: Path,
        transform_fn: Optional[Callable] = None,
    ) -> Path:
        """Export in JSONL format for fine-tuning."""
        with open(output_path, "w") as f:
            for example in examples:
                if transform_fn:
                    record = transform_fn(example)
                else:
                    record = self._default_fine_tuning_transform(example)

                f.write(json.dumps(record) + "\n")

        return output_path

    def _default_fine_tuning_transform(
        self, example: TrainingExample
    ) -> Dict[str, Any]:
        """Default transformation for fine-tuning format."""
        # Convert to conversation format
        messages = []

        # Add system message with task context
        system_content = f"""You are an expert task-executing agent. 
Task: {example.task_definition.name}
Category: {example.task_definition.category}
Required tools: {", ".join(example.task_definition.required_tools)}
Expected outcomes: {"; ".join(example.task_definition.expected_outcomes)}"""

        if example.task_definition.constraints:
            system_content += (
                f"\nConstraints: {'; '.join(example.task_definition.constraints)}"
            )

        messages.append({"role": "system", "content": system_content})

        # Add conversation messages
        for msg in example.execution.messages:
            if msg["type"] == "user":
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["type"] == "assistant":
                # Reconstruct assistant message
                content_parts = []
                for block in msg.get("content", []):
                    if block["type"] == "text":
                        content_parts.append(block["text"])
                    elif block["type"] == "tool_use":
                        content_parts.append(
                            f"Using tool {block['name']}: {json.dumps(block['input'])}"
                        )

                messages.append(
                    {"role": "assistant", "content": "\n".join(content_parts)}
                )

        return {
            "messages": messages,
            "metadata": {
                "task_id": example.task_definition.id,
                "quality_score": example.quality_score,
                "is_golden": example.is_golden,
                "success": example.execution.success,
                "cost_usd": example.execution.cost_usd,
            },
        }

    def _export_conversations_format(
        self, examples: List[TrainingExample], output_path: Path
    ) -> Path:
        """Export in conversations format (for conversational fine-tuning)."""
        conversations = []

        for example in examples:
            conversation = {
                "conversation_id": example.id,
                "task": example.task_definition.name,
                "turns": [],
            }

            # Group messages into turns
            current_turn = None
            for msg in example.execution.messages:
                if msg["type"] == "user":
                    if current_turn:
                        conversation["turns"].append(current_turn)
                    current_turn = {
                        "user": msg["content"],
                        "assistant": "",
                        "tools_used": [],
                    }
                elif msg["type"] == "assistant" and current_turn:
                    assistant_text = []
                    for block in msg.get("content", []):
                        if block["type"] == "text":
                            assistant_text.append(block["text"])
                        elif block["type"] == "tool_use":
                            current_turn["tools_used"].append(
                                {"name": block["name"], "input": block["input"]}
                            )
                    current_turn["assistant"] = "\n".join(assistant_text)

            if current_turn:
                conversation["turns"].append(current_turn)

            conversations.append(conversation)

        with open(output_path, "w") as f:
            json.dump(conversations, f, indent=2)

        return output_path

    def _export_instruct_format(
        self, examples: List[TrainingExample], output_path: Path
    ) -> Path:
        """Export in instruction-following format."""
        records = []

        for example in examples:
            # Create instruction from task
            instruction = f"{example.task_definition.description}\n\n"
            instruction += "Requirements:\n"
            for outcome in example.task_definition.expected_outcomes:
                instruction += f"- {outcome}\n"

            # Extract response from execution
            response_parts = []
            for action in example.execution.actions:
                if action.reasoning:
                    response_parts.append(f"Reasoning: {action.reasoning}")
                if action.tool_name:
                    response_parts.append(f"Action: Use {action.tool_name}")
                    if action.tool_result:
                        response_parts.append(
                            f"Result: {json.dumps(action.tool_result)}"
                        )

            if example.execution.outcome:
                response_parts.append(f"Outcome: {example.execution.outcome}")

            record = {
                "instruction": instruction,
                "response": "\n\n".join(response_parts),
                "metadata": {
                    "category": example.task_definition.category,
                    "difficulty": example.task_definition.difficulty,
                    "success": example.execution.success,
                    "quality_score": example.quality_score,
                },
            }

            records.append(record)

        with open(output_path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        return output_path

    def export_for_evaluation(
        self, output_path: Path, format: str = "json", include_golden_only: bool = False
    ) -> Path:
        """Export data formatted for evaluation benchmarks."""
        filters = {"is_golden": True} if include_golden_only else None
        examples = self.storage.get_training_examples(filters=filters)

        if format == "json":
            return self._export_evaluation_json(examples, output_path)
        elif format == "yaml":
            return self._export_evaluation_yaml(examples, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_evaluation_json(
        self, examples: List[TrainingExample], output_path: Path
    ) -> Path:
        """Export evaluation dataset in JSON format."""
        eval_data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "total_examples": len(examples),
            "tasks": [],
        }

        for example in examples:
            task_data = {
                "id": example.id,
                "task": {
                    "name": example.task_definition.name,
                    "description": example.task_definition.description,
                    "category": example.task_definition.category,
                    "difficulty": example.task_definition.difficulty,
                    "required_tools": example.task_definition.required_tools,
                    "expected_outcomes": example.task_definition.expected_outcomes,
                    "constraints": example.task_definition.constraints,
                },
                "reference_execution": {
                    "success": example.execution.success,
                    "outcome": example.execution.outcome,
                    "actions": [
                        {
                            "type": action.action_type,
                            "tool": action.tool_name,
                            "reasoning": action.reasoning,
                        }
                        for action in example.execution.actions
                    ],
                    "cost_usd": example.execution.cost_usd,
                    "duration_seconds": (
                        example.execution.completed_at - example.execution.started_at
                    ).total_seconds()
                    if example.execution.completed_at
                    else None,
                },
                "quality_metrics": {
                    "quality_score": example.quality_score,
                    "is_golden": example.is_golden,
                    "validated": example.validated,
                },
            }

            eval_data["tasks"].append(task_data)

        with open(output_path, "w") as f:
            json.dump(eval_data, f, indent=2, default=str)

        return output_path

    def _export_evaluation_yaml(
        self, examples: List[TrainingExample], output_path: Path
    ) -> Path:
        """Export evaluation dataset in YAML format."""
        if yaml is None:
            raise ImportError(
                "PyYAML is required for YAML export. Install with: pip install pyyaml"
            )

        eval_data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "tasks": [],
        }

        for example in examples:
            task_data = {
                "id": example.id,
                "task_name": example.task_definition.name,
                "category": example.task_definition.category,
                "expected_tools": example.task_definition.required_tools,
                "success_criteria": example.task_definition.expected_outcomes,
                "reference": {
                    "success": example.execution.success,
                    "quality_score": example.quality_score,
                },
            }

            eval_data["tasks"].append(task_data)

        with open(output_path, "w") as f:
            yaml.dump(eval_data, f, default_flow_style=False)

        return output_path

    def export_analytics(self, output_dir: Path, format: str = "html") -> Path:
        """Export analytics and visualizations of the training data."""
        examples = self.storage.get_training_examples()

        if format == "html":
            return self._export_html_analytics(examples, output_dir)
        elif format == "csv":
            return self._export_csv_analytics(examples, output_dir)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_html_analytics(
        self, examples: List[TrainingExample], output_dir: Path
    ) -> Path:
        """Export HTML analytics dashboard."""
        if pd is None:
            # Fallback to basic HTML without pandas
            return self._export_basic_html_analytics(examples, output_dir)

        output_dir.mkdir(exist_ok=True)

        # Prepare data for visualization
        data = []
        for example in examples:
            duration = None
            if example.execution.completed_at and example.execution.started_at:
                duration = (
                    example.execution.completed_at - example.execution.started_at
                ).total_seconds()

            data.append(
                {
                    "task_name": example.task_definition.name,
                    "category": example.task_definition.category,
                    "difficulty": example.task_definition.difficulty,
                    "success": example.execution.success,
                    "quality_score": example.quality_score,
                    "is_golden": example.is_golden,
                    "cost_usd": example.execution.cost_usd,
                    "duration_seconds": duration,
                    "tool_count": len(
                        set(
                            a.tool_name
                            for a in example.execution.actions
                            if a.tool_name
                        )
                    ),
                    "action_count": len(example.execution.actions),
                }
            )

        df = pd.DataFrame(data)

        # Generate HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Training Data Analytics</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f0f0f0; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .success {{ color: green; }}
        .failure {{ color: red; }}
    </style>
</head>
<body>
    <h1>Training Data Analytics</h1>
    
    <div class="metrics">
        <div class="metric">
            <h3>Total Examples</h3>
            <p>{len(examples)}</p>
        </div>
        <div class="metric">
            <h3>Success Rate</h3>
            <p>{df["success"].mean():.1%}</p>
        </div>
        <div class="metric">
            <h3>Golden Examples</h3>
            <p>{df["is_golden"].sum()}</p>
        </div>
        <div class="metric">
            <h3>Avg Quality Score</h3>
            <p>{df["quality_score"].mean():.2f}</p>
        </div>
        <div class="metric">
            <h3>Total Cost</h3>
            <p>${df["cost_usd"].sum():.2f}</p>
        </div>
    </div>
    
    <h2>Category Breakdown</h2>
    <table>
        <tr>
            <th>Category</th>
            <th>Count</th>
            <th>Success Rate</th>
            <th>Avg Quality</th>
            <th>Avg Cost</th>
        </tr>
"""

        for category in df["category"].unique():
            cat_df = df[df["category"] == category]
            html_content += f"""
        <tr>
            <td>{category}</td>
            <td>{len(cat_df)}</td>
            <td>{cat_df["success"].mean():.1%}</td>
            <td>{cat_df["quality_score"].mean():.2f}</td>
            <td>${cat_df["cost_usd"].mean():.3f}</td>
        </tr>
"""

        html_content += """
    </table>
    
    <h2>Difficulty Analysis</h2>
    <table>
        <tr>
            <th>Difficulty</th>
            <th>Count</th>
            <th>Success Rate</th>
            <th>Avg Duration (s)</th>
        </tr>
"""

        for difficulty in ["easy", "medium", "hard", "expert"]:
            diff_df = df[df["difficulty"] == difficulty]
            if len(diff_df) > 0:
                avg_duration = diff_df["duration_seconds"].mean()
                html_content += f"""
        <tr>
            <td>{difficulty}</td>
            <td>{len(diff_df)}</td>
            <td>{diff_df["success"].mean():.1%}</td>
            <td>{avg_duration:.1f if not pd.isna(avg_duration) else 'N/A'}</td>
        </tr>
"""

        html_content += """
    </table>
</body>
</html>
"""

        output_path = output_dir / "analytics.html"
        with open(output_path, "w") as f:
            f.write(html_content)

        # Also save raw data as CSV
        df.to_csv(output_dir / "training_data_summary.csv", index=False)

        return output_path

    def _export_basic_html_analytics(
        self, examples: List[TrainingExample], output_dir: Path
    ) -> Path:
        """Export basic HTML analytics without pandas."""
        output_dir.mkdir(exist_ok=True)

        # Calculate basic statistics
        total = len(examples)
        successful = sum(1 for e in examples if e.execution.success)
        golden = sum(1 for e in examples if e.is_golden)
        avg_quality = sum(e.quality_score for e in examples) / total if total > 0 else 0
        total_cost = sum(e.execution.cost_usd for e in examples)

        # Category counts
        category_counts = {}
        for example in examples:
            cat = example.task_definition.category
            if cat not in category_counts:
                category_counts[cat] = 0
            category_counts[cat] += 1

        # Generate simple HTML
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Training Data Analytics</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f0f0f0; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <h1>Training Data Analytics</h1>
    
    <div class="metrics">
        <div class="metric">
            <h3>Total Examples</h3>
            <p>{total}</p>
        </div>
        <div class="metric">
            <h3>Success Rate</h3>
            <p>{(successful / total * 100):.1f}%</p>
        </div>
        <div class="metric">
            <h3>Golden Examples</h3>
            <p>{golden}</p>
        </div>
        <div class="metric">
            <h3>Avg Quality Score</h3>
            <p>{avg_quality:.2f}</p>
        </div>
        <div class="metric">
            <h3>Total Cost</h3>
            <p>${total_cost:.2f}</p>
        </div>
    </div>
    
    <h2>Category Breakdown</h2>
    <table>
        <tr>
            <th>Category</th>
            <th>Count</th>
            <th>Percentage</th>
        </tr>
"""

        for category, count in sorted(category_counts.items()):
            percentage = (count / total * 100) if total > 0 else 0
            html_content += f"""        <tr>
            <td>{category}</td>
            <td>{count}</td>
            <td>{percentage:.1f}%</td>
        </tr>
"""

        html_content += """    </table>
</body>
</html>"""

        output_path = output_dir / "analytics.html"
        with open(output_path, "w") as f:
            f.write(html_content)

        return output_path

    def _export_csv_analytics(
        self, examples: List[TrainingExample], output_dir: Path
    ) -> Path:
        """Export analytics data as CSV files."""
        output_dir.mkdir(exist_ok=True)

        if pd is None:
            # Fallback to basic CSV export without pandas
            return self._export_basic_csv_analytics(examples, output_dir)

        # Task summary
        task_data = []
        for example in examples:
            task_data.append(
                {
                    "task_id": example.task_definition.id,
                    "task_name": example.task_definition.name,
                    "category": example.task_definition.category,
                    "difficulty": example.task_definition.difficulty,
                    "tool_count": len(example.task_definition.required_tools),
                    "constraint_count": len(example.task_definition.constraints),
                }
            )

        task_df = pd.DataFrame(task_data)
        task_df.to_csv(output_dir / "tasks.csv", index=False)

        # Execution summary
        exec_data = []
        for example in examples:
            duration = None
            if example.execution.completed_at and example.execution.started_at:
                duration = (
                    example.execution.completed_at - example.execution.started_at
                ).total_seconds()

            exec_data.append(
                {
                    "execution_id": example.execution.id,
                    "task_id": example.task_definition.id,
                    "success": example.execution.success,
                    "duration_seconds": duration,
                    "action_count": len(example.execution.actions),
                    "cost_usd": example.execution.cost_usd,
                    "quality_score": example.quality_score,
                    "is_golden": example.is_golden,
                }
            )

        exec_df = pd.DataFrame(exec_data)
        exec_df.to_csv(output_dir / "executions.csv", index=False)

        # Tool usage
        tool_data = []
        for example in examples:
            for action in example.execution.actions:
                if action.tool_name:
                    tool_data.append(
                        {
                            "execution_id": example.execution.id,
                            "tool_name": action.tool_name,
                            "action_type": action.action_type,
                            "confidence": action.confidence,
                            "has_reasoning": action.reasoning is not None,
                        }
                    )

        if tool_data:
            tool_df = pd.DataFrame(tool_data)
            tool_df.to_csv(output_dir / "tool_usage.csv", index=False)

        return output_dir

    def _export_basic_csv_analytics(
        self, examples: List[TrainingExample], output_dir: Path
    ) -> Path:
        """Export basic CSV analytics without pandas."""
        # Export task summary
        with open(output_dir / "tasks.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "task_id",
                    "task_name",
                    "category",
                    "difficulty",
                    "tool_count",
                    "constraint_count",
                ],
            )
            writer.writeheader()

            for example in examples:
                writer.writerow(
                    {
                        "task_id": example.task_definition.id,
                        "task_name": example.task_definition.name,
                        "category": example.task_definition.category,
                        "difficulty": example.task_definition.difficulty,
                        "tool_count": len(example.task_definition.required_tools),
                        "constraint_count": len(example.task_definition.constraints),
                    }
                )

        # Export execution summary
        with open(output_dir / "executions.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "execution_id",
                    "task_id",
                    "success",
                    "duration_seconds",
                    "action_count",
                    "cost_usd",
                    "quality_score",
                    "is_golden",
                ],
            )
            writer.writeheader()

            for example in examples:
                duration = None
                if example.execution.completed_at and example.execution.started_at:
                    duration = (
                        example.execution.completed_at - example.execution.started_at
                    ).total_seconds()

                writer.writerow(
                    {
                        "execution_id": example.execution.id,
                        "task_id": example.task_definition.id,
                        "success": example.execution.success,
                        "duration_seconds": duration,
                        "action_count": len(example.execution.actions),
                        "cost_usd": example.execution.cost_usd,
                        "quality_score": example.quality_score,
                        "is_golden": example.is_golden,
                    }
                )

        return output_dir

    def export_for_ml_frameworks(
        self,
        output_dir: Path,
        framework: str = "huggingface",
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> Dict[str, Path]:
        """Export data formatted for specific ML frameworks."""
        output_dir.mkdir(exist_ok=True)

        if framework == "huggingface":
            return self._export_huggingface_dataset(output_dir, split_ratio)
        elif framework == "tensorflow":
            return self._export_tensorflow_dataset(output_dir, split_ratio)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    def _export_huggingface_dataset(
        self, output_dir: Path, split_ratio: Tuple[float, float, float]
    ) -> Dict[str, Path]:
        """Export dataset in HuggingFace datasets format."""
        examples = self.storage.get_training_examples()

        # Shuffle and split
        import random

        random.shuffle(examples)

        total = len(examples)
        train_size = int(total * split_ratio[0])
        val_size = int(total * split_ratio[1])

        train_examples = examples[:train_size]
        val_examples = examples[train_size : train_size + val_size]
        test_examples = examples[train_size + val_size :]

        # Export splits
        paths = {}
        for split_name, split_examples in [
            ("train", train_examples),
            ("validation", val_examples),
            ("test", test_examples),
        ]:
            split_path = output_dir / f"{split_name}.jsonl"
            with open(split_path, "w") as f:
                for example in split_examples:
                    record = self._default_fine_tuning_transform(example)
                    f.write(json.dumps(record) + "\n")
            paths[split_name] = split_path

        # Create dataset card
        card_content = f"""---
dataset_info:
  dataset_name: task_execution_training
  dataset_size: {total}
  splits:
    - name: train
      num_examples: {len(train_examples)}
    - name: validation
      num_examples: {len(val_examples)}
    - name: test
      num_examples: {len(test_examples)}
---

# Task Execution Training Dataset

This dataset contains examples of task execution by AI agents.

## Dataset Statistics
- Total examples: {total}
- Categories: {len(set(e.task_definition.category for e in examples))}
- Average quality score: {sum(e.quality_score for e in examples) / total:.2f}
"""

        with open(output_dir / "README.md", "w") as f:
            f.write(card_content)

        return paths

    def _export_tensorflow_dataset(
        self, output_dir: Path, split_ratio: Tuple[float, float, float]
    ) -> Dict[str, Path]:
        """Export dataset in TensorFlow format."""
        # Similar to HuggingFace but with TFRecord format
        # Implementation would use tf.data.Dataset
        pass
