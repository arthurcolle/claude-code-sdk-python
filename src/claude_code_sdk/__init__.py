"""Claude SDK for Python."""

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ._errors import (
    ClaudeSDKError,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ProcessError,
)
from ._internal.client import InternalClient
from .types import (
    AssistantMessage,
    ClaudeCodeOptions,
    ContentBlock,
    McpServerConfig,
    Message,
    PermissionMode,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from .streaming import (
    StreamController,
    StreamState,
    ControlledStream,
    create_controlled_stream,
)
from .retry import (
    RetryConfig,
    RetryExhaustedError,
    RetryableQuery,
    with_retry,
    query_with_retry,
)
from .pool import (
    ConnectionPool,
    PooledConnection,
    get_global_pool,
    pooled_query,
)
from .batch import (
    BatchProcessor,
    BatchItem,
    BatchResult,
    BatchStatus,
    StreamingBatchProcessor,
    batch_query,
    batch_query_streaming,
)
from .progress import (
    ProgressTracker,
    ProgressInfo,
    ProgressEvent,
    ProgressReporter,
    ProgressBar,
    ProgressBarReporter,
    query_with_progress,
)
from .cache import (
    QueryCache,
    CacheConfig,
    CacheEntry,
    CacheBackend,
    MemoryCacheBackend,
    FileCacheBackend,
    get_global_cache,
    cached_query,
)
from .middleware import (
    Middleware,
    MiddlewareChain,
    LoggingMiddleware,
    MetricsMiddleware,
    AuthenticationMiddleware,
    RateLimitMiddleware,
    TransformMiddleware,
    Plugin,
    PluginManager,
    query_with_middleware,
)
from .auth import (
    ClaudeAuth,
    OAuthConfig,
    OAuthFlow,
    AuthToken,
    TokenStorage,
    AuthenticationError,
    login,
    logout,
    get_auth_headers,
)
from .authenticated_query import (
    authenticated_query,
    query_with_oauth,
    query_with_api_key,
)
from .logging import (
    LogConfig,
    LogLevel,
    LogManager,
    PerformanceLogger,
    configure_logging,
    get_logger,
    get_performance_logger,
    log_context,
)
from .metrics import (
    MetricsCollector,
    MetricsManager,
    ConversationMetrics,
    MetricPoint,
    MetricSummary,
    MetricType,
    AggregationType,
    MetricExporter,
    ConsoleExporter,
    PrometheusExporter,
    create_metrics_manager,
)
from .persistence import (
    ConversationState,
    ConversationCheckpoint,
    ConversationPersistence,
    StorageBackend,
    FileStorageBackend,
    MemoryStorageBackend,
    SQLiteStorageBackend,
    create_persistence,
)
from .quota import (
    QuotaManager,
    QuotaLimit,
    QuotaPeriod,
    QuotaExceededError,
    RateLimitError,
    get_global_quota_manager,
    query_with_quota,
)
from .rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    QuotaConfig,
    RateLimitStrategy,
    QuotaType,
    RateLimitExceeded,
    QuotaExceeded,
    UsageStats,
    create_rate_limiter,
)
from .recovery import (
    RecoveryConfig,
    RecoveryStrategy,
    CircuitBreaker,
    RecoveryManager,
    query_with_recovery,
    query_with_fallback,
    query_with_circuit_breaker,
)
from .tools_api import (
    CustomTool,
    ToolParameter,
    ToolRegistry,
    create_tool,
    get_global_tool_registry,
    query_with_tools,
)

# Lazy import for tools module
if TYPE_CHECKING:
    from . import tools

__version__ = "0.0.10"

__all__ = [
    # Main function
    "query",
    "query_with_retry",
    "pooled_query",
    "batch_query",
    "batch_query_streaming",
    "query_with_progress",
    # CLI command functions
    "update",
    "mcp",
    "config",
    "doctor",
    "version",
    # Types
    "PermissionMode",
    "McpServerConfig",
    "UserMessage",
    "AssistantMessage",
    "SystemMessage",
    "ResultMessage",
    "Message",
    "ClaudeCodeOptions",
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ContentBlock",
    # Errors
    "ClaudeSDKError",
    "CLIConnectionError",
    "CLINotFoundError",
    "ProcessError",
    "CLIJSONDecodeError",
    "RetryExhaustedError",
    # Streaming control
    "StreamController",
    "StreamState",
    "ControlledStream",
    "create_controlled_stream",
    # Retry functionality
    "RetryConfig",
    "RetryableQuery",
    "with_retry",
    # Connection pooling
    "ConnectionPool",
    "PooledConnection",
    "get_global_pool",
    # Batch processing
    "BatchProcessor",
    "BatchItem",
    "BatchResult",
    "BatchStatus",
    "StreamingBatchProcessor",
    # Progress tracking
    "ProgressTracker",
    "ProgressInfo",
    "ProgressEvent",
    "ProgressReporter",
    "ProgressBar",
    "ProgressBarReporter",
    # Cache
    "QueryCache",
    "CacheConfig",
    "CacheEntry",
    "CacheBackend",
    "MemoryCacheBackend",
    "FileCacheBackend",
    "get_global_cache",
    "cached_query",
    # Middleware
    "Middleware",
    "MiddlewareChain",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "AuthenticationMiddleware",
    "RateLimitMiddleware",
    "TransformMiddleware",
    "Plugin",
    "PluginManager",
    "query_with_middleware",
    # Authentication
    "ClaudeAuth",
    "OAuthConfig",
    "OAuthFlow",
    "AuthToken",
    "TokenStorage",
    "AuthenticationError",
    "login",
    "logout",
    "get_auth_headers",
    "authenticated_query",
    "query_with_oauth",
    "query_with_api_key",
    # Logging
    "LogConfig",
    "LogLevel",
    "LogManager",
    "PerformanceLogger",
    "configure_logging",
    "get_logger",
    "get_performance_logger",
    "log_context",
    # Metrics
    "MetricsCollector",
    "MetricsManager",
    "ConversationMetrics",
    "MetricPoint",
    "MetricSummary",
    "MetricType",
    "AggregationType",
    "MetricExporter",
    "ConsoleExporter",
    "PrometheusExporter",
    "create_metrics_manager",
    # Persistence
    "ConversationState",
    "ConversationCheckpoint",
    "ConversationPersistence",
    "StorageBackend",
    "FileStorageBackend",
    "MemoryStorageBackend",
    "SQLiteStorageBackend",
    "create_persistence",
    # Quota management
    "QuotaManager",
    "QuotaLimit",
    "QuotaPeriod",
    "QuotaExceededError",
    "RateLimitError",
    "get_global_quota_manager",
    "query_with_quota",
    # Rate limiting
    "RateLimiter",
    "RateLimitConfig",
    "QuotaConfig",
    "RateLimitStrategy",
    "QuotaType",
    "RateLimitExceeded",
    "QuotaExceeded",
    "UsageStats",
    "create_rate_limiter",
    # Recovery
    "RecoveryConfig",
    "RecoveryStrategy",
    "CircuitBreaker",
    "RecoveryManager",
    "query_with_recovery",
    "query_with_fallback",
    "query_with_circuit_breaker",
    # Custom tools
    "CustomTool",
    "ToolParameter",
    "ToolRegistry",
    "create_tool",
    "get_global_tool_registry",
    "query_with_tools",
    # Tools module (lazy import)
    "tools",
]


async def query(
    *, prompt: str, options: ClaudeCodeOptions | None = None
) -> AsyncIterator[Message]:
    """
    Query Claude Code.

    Python SDK for interacting with Claude Code.

    Args:
        prompt: The prompt to send to Claude
        options: Optional configuration (defaults to ClaudeCodeOptions() if None).
                 Set options.permission_mode to control tool execution:
                 - 'default': CLI prompts for dangerous tools
                 - 'acceptEdits': Auto-accept file edits
                 - 'bypassPermissions': Allow all tools (use with caution)
                 Set options.cwd for working directory.

    Yields:
        Messages from the conversation


    Example:
        ```python
        # Simple usage
        async for message in query(prompt="Hello"):
            print(message)

        # With options
        async for message in query(
            prompt="Hello",
            options=ClaudeCodeOptions(
                system_prompt="You are helpful",
                cwd="/home/user"
            )
        ):
            print(message)
        ```
    """
    if options is None:
        options = ClaudeCodeOptions()

    os.environ["CLAUDE_CODE_ENTRYPOINT"] = "sdk-py"

    client = InternalClient()

    async for message in client.process_query(prompt=prompt, options=options):
        yield message


async def _run_cli_command(
    command: list[str], cli_path: str | Path | None = None
) -> str:
    """Run a CLI command and return output."""
    import shutil
    import anyio
    from subprocess import PIPE

    # Find CLI path
    if cli_path is None:
        cli_path = shutil.which("claude")
        if not cli_path:
            locations = [
                Path.home() / ".npm-global/bin/claude",
                Path("/usr/local/bin/claude"),
                Path.home() / ".local/bin/claude",
                Path.home() / "node_modules/.bin/claude",
                Path.home() / ".yarn/bin/claude",
            ]
            for path in locations:
                if path.exists() and path.is_file():
                    cli_path = str(path)
                    break

        if not cli_path:
            raise CLINotFoundError(
                "Claude Code not found. Install with:\n"
                "  npm install -g @anthropic-ai/claude-code"
            )

    # Run command
    cmd = [str(cli_path)] + command
    process = await anyio.open_process(
        cmd,
        stdin=None,
        stdout=PIPE,
        stderr=PIPE,
    )

    stdout_bytes, stderr_bytes = await process.communicate()
    stdout = stdout_bytes.decode() if stdout_bytes else ""
    stderr = stderr_bytes.decode() if stderr_bytes else ""

    if process.returncode != 0:
        raise ProcessError(
            f"Command failed: {' '.join(cmd)}",
            exit_code=process.returncode,
            stderr=stderr or stdout,
        )

    return stdout


async def update(cli_path: str | Path | None = None) -> str:
    """
    Check for and install updates to Claude Code.

    Args:
        cli_path: Optional path to the CLI binary

    Returns:
        Update status message

    Example:
        ```python
        result = await update()
        print(result)
        ```
    """
    return await _run_cli_command(["update"], cli_path)


async def mcp(
    subcommand: list[str] | None = None, cli_path: str | Path | None = None
) -> str:
    """
    Configure and manage Model Context Protocol (MCP) servers.

    Args:
        subcommand: MCP subcommand and arguments (e.g., ["list"], ["add", "server-name"])
        cli_path: Optional path to the CLI binary

    Returns:
        MCP command output

    Example:
        ```python
        # List MCP servers
        result = await mcp(["list"])

        # Add MCP server
        result = await mcp(["add", "my-server"])
        ```
    """
    cmd = ["mcp"]
    if subcommand:
        cmd.extend(subcommand)
    return await _run_cli_command(cmd, cli_path)


async def config(
    subcommand: list[str] | None = None, cli_path: str | Path | None = None
) -> str:
    """
    Manage Claude Code configuration.

    Args:
        subcommand: Config subcommand and arguments (e.g., ["get", "key"], ["set", "key", "value"])
        cli_path: Optional path to the CLI binary

    Returns:
        Config command output

    Example:
        ```python
        # Get config value
        result = await config(["get", "model"])

        # Set config value
        result = await config(["set", "model", "claude-sonnet-4"])
        ```
    """
    cmd = ["config"]
    if subcommand:
        cmd.extend(subcommand)
    return await _run_cli_command(cmd, cli_path)


async def doctor(cli_path: str | Path | None = None) -> str:
    """
    Check health of Claude Code auto-updater.

    Args:
        cli_path: Optional path to the CLI binary

    Returns:
        Health check results

    Example:
        ```python
        result = await doctor()
        print(result)
        ```
    """
    return await _run_cli_command(["doctor"], cli_path)


async def version(cli_path: str | Path | None = None) -> str:
    """
    Get Claude Code version.

    Args:
        cli_path: Optional path to the CLI binary

    Returns:
        Version string

    Example:
        ```python
        ver = await version()
        print(f"Claude Code version: {ver}")
        ```
    """
    return await _run_cli_command(["--version"], cli_path)


def __getattr__(name: str) -> Any:
    """Lazy import for tools module."""
    if name == "tools":
        from . import tools

        return tools
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
