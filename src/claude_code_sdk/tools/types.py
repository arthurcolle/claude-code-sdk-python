"""Type definitions for Tool Management API."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from typing_extensions import TypedDict, NotRequired


class ToolActionType(str, Enum):
    """Tool action types."""
    HTTP = "http"
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    SERVICE = "service"


class ToolOutputType(str, Enum):
    """Tool output types."""
    JSON = "json"
    TEXT = "text"
    BINARY = "binary"
    AI = "ai"


class HTTPActionConfig(TypedDict):
    """HTTP action configuration."""
    method: str
    url: str
    headers: NotRequired[dict[str, str]]


class PythonActionConfig(TypedDict):
    """Python action configuration."""
    code: str
    function_name: str


class JavaScriptActionConfig(TypedDict):
    """JavaScript action configuration."""
    code: str
    function_name: str


class ServiceActionConfig(TypedDict):
    """Service action configuration."""
    service_id: str
    endpoint_path: str
    endpoint_method: str
    base_url: str


@dataclass
class ToolAction:
    """Tool action configuration."""
    type: ToolActionType
    http: HTTPActionConfig | None = None
    python: PythonActionConfig | None = None
    javascript: JavaScriptActionConfig | None = None
    service_config: ServiceActionConfig | None = None


@dataclass
class ToolOutput:
    """Tool output configuration."""
    type: ToolOutputType
    content: str = ""


@dataclass
class Tool:
    """Tool definition."""
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    namespace: str | None = None
    code: str | None = None
    action: ToolAction | None = None
    output: ToolOutput | None = None
    id: str | None = None


@dataclass
class ToolResponse:
    """Tool response from API."""
    id: str
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    namespace: str | None = None
    action: ToolAction | None = None
    output: ToolOutput | None = None


@dataclass
class ToolSearchRequest:
    """Tool search request."""
    query: str
    limit: int = 5
    strategy: Literal["vector_similarity", "text"] = "vector_similarity"


@dataclass
class ToolSearchResponse:
    """Tool search response."""
    id: str
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    namespace: str | None = None
    action: ToolAction | None = None
    output: ToolOutput | None = None
    score: float | None = None
    embedding: list[float] | None = None


@dataclass
class EmbeddingResponse:
    """Tool embedding response."""
    id: str
    name: str
    embedding: list[float]


@dataclass
class ToolExecutionRequest:
    """Tool execution request."""
    input_data: dict[str, Any]
    tool_id: str | None = None
    tool_name: str | None = None
    user_settings: dict[str, Any] | None = None
    format_type: str | None = None


@dataclass
class ToolExecutionResponse:
    """Tool execution response."""
    tool_id: str
    output_data: dict[str, Any]


@dataclass
class ToolIdentifier:
    """Tool identifier."""
    id: str | None = None
    name: str | None = None


@dataclass
class SequentialToolExecutionRequest:
    """Sequential tool execution request."""
    tool_ids: list[str | ToolIdentifier]
    initial_input: dict[str, Any]


@dataclass
class ParallelToolExecutionRequest:
    """Parallel tool execution request."""
    tool_ids: list[str | ToolIdentifier]
    input_data: dict[str, dict[str, Any]]


@dataclass
class ToolBatchRequest:
    """Tool batch creation request."""
    tools: list[Tool]