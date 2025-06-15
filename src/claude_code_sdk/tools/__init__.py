"""Tool Management API integration."""

from .client import ToolManagementClient
from .discovery import ToolDiscovery, ToolRegistry
from .enhanced_client import EnhancedClient, query_with_tools
from .hooks import ToolExecutionHooks, ToolRegistryOptions
from .types import (
    EmbeddingResponse,
    HTTPActionConfig,
    JavaScriptActionConfig,
    ParallelToolExecutionRequest,
    PythonActionConfig,
    SequentialToolExecutionRequest,
    ServiceActionConfig,
    Tool,
    ToolAction,
    ToolActionType,
    ToolBatchRequest,
    ToolExecutionRequest,
    ToolExecutionResponse,
    ToolIdentifier,
    ToolOutput,
    ToolOutputType,
    ToolResponse,
    ToolSearchRequest,
    ToolSearchResponse,
)

__all__ = [
    # Client and utilities
    "ToolManagementClient",
    "EnhancedClient",
    "query_with_tools",
    "ToolExecutionHooks",
    "ToolRegistryOptions",
    "ToolDiscovery",
    "ToolRegistry",
    # Types
    "Tool",
    "ToolResponse",
    "ToolSearchRequest",
    "ToolSearchResponse",
    "ToolExecutionRequest",
    "ToolExecutionResponse",
    "ToolIdentifier",
    "SequentialToolExecutionRequest",
    "ParallelToolExecutionRequest",
    "ToolBatchRequest",
    "EmbeddingResponse",
    "ToolAction",
    "ToolActionType",
    "ToolOutput",
    "ToolOutputType",
    "HTTPActionConfig",
    "PythonActionConfig",
    "JavaScriptActionConfig",
    "ServiceActionConfig",
]
