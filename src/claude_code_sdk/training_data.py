"""Training data generation and storage for advanced task-executing agents."""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import json
import sqlite3
import duckdb
from enum import Enum
import uuid

from .types import Message, AssistantMessage, UserMessage, ToolUseBlock, ToolResultBlock


class TaskStatus(Enum):
    """Status of a task execution."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentRole(Enum):
    """Roles for different types of agents."""

    TASK_EXECUTOR = "task_executor"
    COORDINATOR = "coordinator"
    PLANNER = "planner"
    REVIEWER = "reviewer"
    SPECIALIST = "specialist"


@dataclass
class TaskDefinition:
    """Definition of a task for training data."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: str = ""
    difficulty: str = "medium"  # easy, medium, hard, expert
    required_tools: List[str] = field(default_factory=list)
    expected_outcomes: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AgentAction:
    """Represents a single action taken by an agent."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    agent_role: AgentRole = AgentRole.TASK_EXECUTOR
    action_type: str = ""  # tool_use, message, decision, planning
    timestamp: datetime = field(default_factory=datetime.now)
    content: Dict[str, Any] = field(default_factory=dict)
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskExecution:
    """Complete execution trace of a task."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    session_id: str = ""
    agent_id: str = ""
    status: TaskStatus = TaskStatus.PENDING
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    actions: List[AgentAction] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)  # Serialized messages
    outcome: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    tokens_used: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingExample:
    """A complete training example for task-executing agents."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_definition: TaskDefinition = field(default_factory=TaskDefinition)
    execution: TaskExecution = field(default_factory=TaskExecution)
    quality_score: float = 0.0  # 0-1 score for quality of execution
    is_golden: bool = False  # Whether this is a high-quality reference example
    tags: List[str] = field(default_factory=list)
    annotations: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    validated: bool = False
    validator_notes: Optional[str] = None


class TrainingDataStorage:
    """Base class for training data storage backends."""

    def save_task_definition(self, task: TaskDefinition) -> str:
        """Save a task definition."""
        raise NotImplementedError

    def save_execution(self, execution: TaskExecution) -> str:
        """Save a task execution."""
        raise NotImplementedError

    def save_training_example(self, example: TrainingExample) -> str:
        """Save a complete training example."""
        raise NotImplementedError

    def get_training_examples(
        self, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None
    ) -> List[TrainingExample]:
        """Retrieve training examples with optional filters."""
        raise NotImplementedError

    def export_for_training(
        self,
        format: str = "jsonl",
        output_path: Optional[Path] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Export training data in specified format."""
        raise NotImplementedError


class SQLiteTrainingStorage(TrainingDataStorage):
    """SQLite backend for training data storage."""

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS task_definitions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    category TEXT,
                    difficulty TEXT,
                    required_tools TEXT,  -- JSON array
                    expected_outcomes TEXT,  -- JSON array
                    constraints TEXT,  -- JSON array
                    metadata TEXT,  -- JSON object
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS task_executions (
                    id TEXT PRIMARY KEY,
                    task_id TEXT,
                    session_id TEXT,
                    agent_id TEXT,
                    status TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    actions TEXT,  -- JSON array
                    messages TEXT,  -- JSON array
                    outcome TEXT,
                    success BOOLEAN,
                    error TEXT,
                    performance_metrics TEXT,  -- JSON object
                    cost_usd REAL,
                    tokens_used TEXT,  -- JSON object
                    metadata TEXT,  -- JSON object
                    FOREIGN KEY (task_id) REFERENCES task_definitions(id)
                );
                
                CREATE TABLE IF NOT EXISTS training_examples (
                    id TEXT PRIMARY KEY,
                    task_definition_id TEXT,
                    execution_id TEXT,
                    quality_score REAL,
                    is_golden BOOLEAN,
                    tags TEXT,  -- JSON array
                    annotations TEXT,  -- JSON object
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    validated BOOLEAN DEFAULT FALSE,
                    validator_notes TEXT,
                    FOREIGN KEY (task_definition_id) REFERENCES task_definitions(id),
                    FOREIGN KEY (execution_id) REFERENCES task_executions(id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_examples_quality ON training_examples(quality_score);
                CREATE INDEX IF NOT EXISTS idx_examples_golden ON training_examples(is_golden);
                CREATE INDEX IF NOT EXISTS idx_executions_status ON task_executions(status);
                CREATE INDEX IF NOT EXISTS idx_tasks_category ON task_definitions(category);
            """)

    def save_task_definition(self, task: TaskDefinition) -> str:
        """Save a task definition."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO task_definitions 
                (id, name, description, category, difficulty, required_tools, 
                 expected_outcomes, constraints, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    task.id,
                    task.name,
                    task.description,
                    task.category,
                    task.difficulty,
                    json.dumps(task.required_tools),
                    json.dumps(task.expected_outcomes),
                    json.dumps(task.constraints),
                    json.dumps(task.metadata),
                    task.created_at.isoformat(),
                ),
            )
        return task.id

    def save_execution(self, execution: TaskExecution) -> str:
        """Save a task execution."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO task_executions
                (id, task_id, session_id, agent_id, status, started_at, completed_at,
                 actions, messages, outcome, success, error, performance_metrics,
                 cost_usd, tokens_used, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    execution.id,
                    execution.task_id,
                    execution.session_id,
                    execution.agent_id,
                    execution.status.value,
                    execution.started_at.isoformat(),
                    execution.completed_at.isoformat()
                    if execution.completed_at
                    else None,
                    json.dumps([asdict(a) for a in execution.actions]),
                    json.dumps(execution.messages),
                    execution.outcome,
                    execution.success,
                    execution.error,
                    json.dumps(execution.performance_metrics),
                    execution.cost_usd,
                    json.dumps(execution.tokens_used),
                    json.dumps(execution.metadata),
                ),
            )
        return execution.id

    def save_training_example(self, example: TrainingExample) -> str:
        """Save a complete training example."""
        # Save task definition
        self.save_task_definition(example.task_definition)
        # Save execution
        self.save_execution(example.execution)

        # Save training example
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO training_examples
                (id, task_definition_id, execution_id, quality_score, is_golden,
                 tags, annotations, created_at, validated, validator_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    example.id,
                    example.task_definition.id,
                    example.execution.id,
                    example.quality_score,
                    example.is_golden,
                    json.dumps(example.tags),
                    json.dumps(example.annotations),
                    example.created_at.isoformat(),
                    example.validated,
                    example.validator_notes,
                ),
            )
        return example.id

    def get_training_examples(
        self, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None
    ) -> List[TrainingExample]:
        """Retrieve training examples with optional filters."""
        query = """
            SELECT 
                te.*,
                td.name, td.description, td.category, td.difficulty, 
                td.required_tools, td.expected_outcomes, td.constraints,
                td.metadata as task_metadata,
                tx.session_id, tx.agent_id, tx.status, tx.started_at, 
                tx.completed_at, tx.actions, tx.messages, tx.outcome,
                tx.success, tx.error, tx.performance_metrics, tx.cost_usd,
                tx.tokens_used, tx.metadata as exec_metadata
            FROM training_examples te
            JOIN task_definitions td ON te.task_definition_id = td.id
            JOIN task_executions tx ON te.execution_id = tx.id
            WHERE 1=1
        """

        params = []
        if filters:
            if "is_golden" in filters:
                query += " AND te.is_golden = ?"
                params.append(filters["is_golden"])
            if "min_quality" in filters:
                query += " AND te.quality_score >= ?"
                params.append(filters["min_quality"])
            if "category" in filters:
                query += " AND td.category = ?"
                params.append(filters["category"])
            if "validated" in filters:
                query += " AND te.validated = ?"
                params.append(filters["validated"])

        query += " ORDER BY te.quality_score DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        examples = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            for row in conn.execute(query, params):
                # Reconstruct objects from database
                task_def = TaskDefinition(
                    id=row["task_definition_id"],
                    name=row["name"],
                    description=row["description"],
                    category=row["category"],
                    difficulty=row["difficulty"],
                    required_tools=json.loads(row["required_tools"]),
                    expected_outcomes=json.loads(row["expected_outcomes"]),
                    constraints=json.loads(row["constraints"]),
                    metadata=json.loads(row["task_metadata"]),
                )

                actions = []
                for action_data in json.loads(row["actions"]):
                    actions.append(AgentAction(**action_data))

                execution = TaskExecution(
                    id=row["execution_id"],
                    task_id=row["task_definition_id"],
                    session_id=row["session_id"],
                    agent_id=row["agent_id"],
                    status=TaskStatus(row["status"]),
                    started_at=datetime.fromisoformat(row["started_at"]),
                    completed_at=datetime.fromisoformat(row["completed_at"])
                    if row["completed_at"]
                    else None,
                    actions=actions,
                    messages=json.loads(row["messages"]),
                    outcome=row["outcome"],
                    success=bool(row["success"]),
                    error=row["error"],
                    performance_metrics=json.loads(row["performance_metrics"]),
                    cost_usd=row["cost_usd"],
                    tokens_used=json.loads(row["tokens_used"]),
                    metadata=json.loads(row["exec_metadata"]),
                )

                example = TrainingExample(
                    id=row["id"],
                    task_definition=task_def,
                    execution=execution,
                    quality_score=row["quality_score"],
                    is_golden=bool(row["is_golden"]),
                    tags=json.loads(row["tags"]),
                    annotations=json.loads(row["annotations"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                    validated=bool(row["validated"]),
                    validator_notes=row["validator_notes"],
                )
                examples.append(example)

        return examples

    def export_for_training(
        self,
        format: str = "jsonl",
        output_path: Optional[Path] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Export training data in specified format."""
        examples = self.get_training_examples(filters=filters)

        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"training_data_{timestamp}.{format}")

        if format == "jsonl":
            with open(output_path, "w") as f:
                for example in examples:
                    # Create training-ready format
                    training_record = {
                        "task": {
                            "name": example.task_definition.name,
                            "description": example.task_definition.description,
                            "category": example.task_definition.category,
                            "required_tools": example.task_definition.required_tools,
                            "constraints": example.task_definition.constraints,
                        },
                        "execution": {
                            "messages": example.execution.messages,
                            "actions": [asdict(a) for a in example.execution.actions],
                            "outcome": example.execution.outcome,
                            "success": example.execution.success,
                        },
                        "metadata": {
                            "quality_score": example.quality_score,
                            "is_golden": example.is_golden,
                            "tags": example.tags,
                            "cost_usd": example.execution.cost_usd,
                            "tokens_used": example.execution.tokens_used,
                        },
                    }
                    f.write(json.dumps(training_record) + "\n")

        elif format == "json":
            all_records = []
            for example in examples:
                training_record = {
                    "task": asdict(example.task_definition),
                    "execution": asdict(example.execution),
                    "quality_score": example.quality_score,
                    "is_golden": example.is_golden,
                    "tags": example.tags,
                    "annotations": example.annotations,
                }
                all_records.append(training_record)

            with open(output_path, "w") as f:
                json.dump(all_records, f, indent=2, default=str)

        else:
            raise ValueError(f"Unsupported format: {format}")

        return output_path


class DuckDBTrainingStorage(TrainingDataStorage):
    """DuckDB backend for training data storage with advanced analytics."""

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize DuckDB schema with analytics-friendly structure."""
        con = duckdb.connect(str(self.db_path))

        # Create main tables
        con.execute("""
            CREATE TABLE IF NOT EXISTS task_definitions (
                id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                description TEXT,
                category VARCHAR,
                difficulty VARCHAR,
                required_tools JSON,
                expected_outcomes JSON,
                constraints JSON,
                metadata JSON,
                created_at TIMESTAMP
            )
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS task_executions (
                id VARCHAR PRIMARY KEY,
                task_id VARCHAR REFERENCES task_definitions(id),
                session_id VARCHAR,
                agent_id VARCHAR,
                status VARCHAR,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                duration_seconds DOUBLE,
                actions JSON,
                messages JSON,
                outcome TEXT,
                success BOOLEAN,
                error TEXT,
                performance_metrics JSON,
                cost_usd DOUBLE,
                tokens_used JSON,
                total_tokens INTEGER,
                metadata JSON
            )
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS agent_actions (
                id VARCHAR PRIMARY KEY,
                execution_id VARCHAR REFERENCES task_executions(id),
                agent_id VARCHAR,
                agent_role VARCHAR,
                action_type VARCHAR,
                timestamp TIMESTAMP,
                tool_name VARCHAR,
                tool_input JSON,
                tool_result JSON,
                reasoning TEXT,
                confidence DOUBLE,
                metadata JSON
            )
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS training_examples (
                id VARCHAR PRIMARY KEY,
                task_definition_id VARCHAR REFERENCES task_definitions(id),
                execution_id VARCHAR REFERENCES task_executions(id),
                quality_score DOUBLE,
                is_golden BOOLEAN,
                tags JSON,
                annotations JSON,
                created_at TIMESTAMP,
                validated BOOLEAN,
                validator_notes TEXT
            )
        """)

        # Create analytical views
        con.execute("""
            CREATE OR REPLACE VIEW execution_analytics AS
            SELECT 
                te.id,
                td.category,
                td.difficulty,
                te.status,
                te.success,
                te.duration_seconds,
                te.cost_usd,
                te.total_tokens,
                json_array_length(te.actions) as action_count,
                json_array_length(te.messages) as message_count
            FROM task_executions te
            JOIN task_definitions td ON te.task_id = td.id
        """)

        con.execute("""
            CREATE OR REPLACE VIEW tool_usage_stats AS
            SELECT 
                aa.tool_name,
                COUNT(*) as usage_count,
                AVG(aa.confidence) as avg_confidence,
                COUNT(DISTINCT aa.execution_id) as unique_executions
            FROM agent_actions aa
            WHERE aa.tool_name IS NOT NULL
            GROUP BY aa.tool_name
        """)

        con.close()

    def save_task_definition(self, task: TaskDefinition) -> str:
        """Save a task definition."""
        con = duckdb.connect(str(self.db_path))
        con.execute(
            """
            INSERT OR REPLACE INTO task_definitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                task.id,
                task.name,
                task.description,
                task.category,
                task.difficulty,
                task.required_tools,
                task.expected_outcomes,
                task.constraints,
                task.metadata,
                task.created_at,
            ],
        )
        con.close()
        return task.id

    def save_execution(self, execution: TaskExecution) -> str:
        """Save a task execution with calculated metrics."""
        con = duckdb.connect(str(self.db_path))

        # Calculate duration
        duration = None
        if execution.completed_at and execution.started_at:
            duration = (execution.completed_at - execution.started_at).total_seconds()

        # Calculate total tokens
        total_tokens = (
            sum(execution.tokens_used.values()) if execution.tokens_used else 0
        )

        # Save execution
        con.execute(
            """
            INSERT OR REPLACE INTO task_executions VALUES 
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                execution.id,
                execution.task_id,
                execution.session_id,
                execution.agent_id,
                execution.status.value,
                execution.started_at,
                execution.completed_at,
                duration,
                [asdict(a) for a in execution.actions],
                execution.messages,
                execution.outcome,
                execution.success,
                execution.error,
                execution.performance_metrics,
                execution.cost_usd,
                execution.tokens_used,
                total_tokens,
                execution.metadata,
            ],
        )

        # Save individual actions for analytics
        for action in execution.actions:
            con.execute(
                """
                INSERT OR REPLACE INTO agent_actions VALUES 
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    action.id,
                    execution.id,
                    action.agent_id,
                    action.agent_role.value,
                    action.action_type,
                    action.timestamp,
                    action.tool_name,
                    action.tool_input,
                    action.tool_result,
                    action.reasoning,
                    action.confidence,
                    action.metadata,
                ],
            )

        con.close()
        return execution.id

    def save_training_example(self, example: TrainingExample) -> str:
        """Save a complete training example."""
        self.save_task_definition(example.task_definition)
        self.save_execution(example.execution)

        con = duckdb.connect(str(self.db_path))
        con.execute(
            """
            INSERT OR REPLACE INTO training_examples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                example.id,
                example.task_definition.id,
                example.execution.id,
                example.quality_score,
                example.is_golden,
                example.tags,
                example.annotations,
                example.created_at,
                example.validated,
                example.validator_notes,
            ],
        )
        con.close()
        return example.id

    def get_training_examples(
        self, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None
    ) -> List[TrainingExample]:
        """Retrieve training examples with optional filters."""
        con = duckdb.connect(str(self.db_path))

        query = """
            SELECT * FROM training_examples te
            JOIN task_definitions td ON te.task_definition_id = td.id
            JOIN task_executions tx ON te.execution_id = tx.id
            WHERE 1=1
        """

        if filters:
            conditions = []
            if "is_golden" in filters:
                conditions.append(f"te.is_golden = {filters['is_golden']}")
            if "min_quality" in filters:
                conditions.append(f"te.quality_score >= {filters['min_quality']}")
            if "category" in filters:
                conditions.append(f"td.category = '{filters['category']}'")
            if "validated" in filters:
                conditions.append(f"te.validated = {filters['validated']}")

            if conditions:
                query += " AND " + " AND ".join(conditions)

        query += " ORDER BY te.quality_score DESC"

        if limit:
            query += f" LIMIT {limit}"

        result = con.execute(query).fetchall()
        con.close()

        # Convert to TrainingExample objects
        examples = []
        for row in result:
            # Parse the row and reconstruct objects
            # (Implementation similar to SQLite version but using DuckDB result format)
            pass  # Simplified for brevity

        return examples

    def export_for_training(
        self,
        format: str = "jsonl",
        output_path: Optional[Path] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Export training data with DuckDB's efficient export capabilities."""
        con = duckdb.connect(str(self.db_path))

        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"training_data_{timestamp}.{format}")

        # Build query for export
        query = """
            SELECT 
                te.*,
                td.* EXCLUDE (id),
                tx.* EXCLUDE (id, task_id)
            FROM training_examples te
            JOIN task_definitions td ON te.task_definition_id = td.id
            JOIN task_executions tx ON te.execution_id = tx.id
        """

        if filters:
            # Add filter conditions
            pass

        if format == "jsonl":
            # Use DuckDB's JSON export
            con.execute(f"""
                COPY ({query}) TO '{output_path}' (FORMAT JSON, ARRAY false)
            """)
        elif format == "parquet":
            # Export to Parquet for efficient storage
            con.execute(f"""
                COPY ({query}) TO '{output_path}' (FORMAT PARQUET)
            """)

        con.close()
        return output_path

    def get_analytics(self) -> Dict[str, Any]:
        """Get training data analytics using DuckDB's analytical capabilities."""
        con = duckdb.connect(str(self.db_path))

        analytics = {
            "total_examples": con.execute(
                "SELECT COUNT(*) FROM training_examples"
            ).fetchone()[0],
            "golden_examples": con.execute(
                "SELECT COUNT(*) FROM training_examples WHERE is_golden"
            ).fetchone()[0],
            "avg_quality_score": con.execute(
                "SELECT AVG(quality_score) FROM training_examples"
            ).fetchone()[0],
            "task_categories": con.execute(
                "SELECT category, COUNT(*) as count FROM task_definitions GROUP BY category"
            ).fetchall(),
            "success_rate": con.execute(
                "SELECT AVG(CAST(success AS DOUBLE)) FROM task_executions"
            ).fetchone()[0],
            "avg_cost_per_task": con.execute(
                "SELECT AVG(cost_usd) FROM task_executions"
            ).fetchone()[0],
            "tool_usage": con.execute(
                "SELECT * FROM tool_usage_stats ORDER BY usage_count DESC"
            ).fetchall(),
            "execution_stats": con.execute("""
                SELECT 
                    category,
                    difficulty,
                    COUNT(*) as count,
                    AVG(duration_seconds) as avg_duration,
                    AVG(cost_usd) as avg_cost,
                    AVG(CAST(success AS DOUBLE)) as success_rate
                FROM execution_analytics
                GROUP BY category, difficulty
            """).fetchall(),
        }

        con.close()
        return analytics
