"""Tests for batch processing functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncIterator

from claude_code_sdk import ClaudeCodeOptions
from claude_code_sdk.types import (
    AssistantMessage, UserMessage, ResultMessage, SystemMessage,
    TextBlock, ToolUseBlock, ToolResultBlock
)
from claude_code_sdk.batch import (
    BatchProcessor, BatchResult, BatchOptions, BatchStrategy,
    batch_query, batch_with_callbacks, batch_with_concurrency,
    batch_with_strategy, batch_with_retry
)


@pytest.fixture
def mock_query():
    """Create a mock query function."""
    async def _mock_query(prompt: str, options: ClaudeCodeOptions | None = None) -> AsyncIterator:
        # Simulate different responses based on prompt
        if "error" in prompt:
            yield SystemMessage(subtype="error", data={"error": "Test error"})
            yield ResultMessage(
                subtype="error",
                cost_usd=0.0,
                duration_ms=100,
                session_id="error-session",
                total_cost_usd=0.0
            )
        else:
            yield AssistantMessage(content=[
                TextBlock(text=f"Response to: {prompt}")
            ])
            yield ResultMessage(
                subtype="success",
                cost_usd=0.01,
                duration_ms=500,
                session_id=f"session-{prompt[:5]}",
                total_cost_usd=0.01,
                usage={
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30
                }
            )
    
    return _mock_query


@pytest.mark.asyncio
async def test_batch_processor_basic(mock_query):
    """Test basic batch processing."""
    processor = BatchProcessor()
    prompts = ["Hello", "World", "Test"]
    
    with patch("claude_code_sdk.batch.query", mock_query):
        results = await processor.process_batch(prompts)
    
    assert len(results) == 3
    for i, result in enumerate(results):
        assert result.prompt == prompts[i]
        assert result.success
        assert result.cost_usd == 0.01
        assert len(result.messages) == 2


@pytest.mark.asyncio
async def test_batch_processor_with_errors(mock_query):
    """Test batch processing with errors."""
    processor = BatchProcessor()
    prompts = ["Hello", "error prompt", "Test"]
    
    with patch("claude_code_sdk.batch.query", mock_query):
        results = await processor.process_batch(prompts)
    
    assert len(results) == 3
    assert results[0].success
    assert not results[1].success
    assert results[1].error == "Test error"
    assert results[2].success


@pytest.mark.asyncio
async def test_batch_processor_concurrency(mock_query):
    """Test concurrent batch processing."""
    processor = BatchProcessor(BatchOptions(
        max_concurrent=2,
        strategy=BatchStrategy.CONCURRENT
    ))
    prompts = ["A", "B", "C", "D"]
    
    with patch("claude_code_sdk.batch.query", mock_query):
        results = await processor.process_batch(prompts)
    
    assert len(results) == 4
    assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_batch_processor_sequential(mock_query):
    """Test sequential batch processing."""
    processor = BatchProcessor(BatchOptions(
        strategy=BatchStrategy.SEQUENTIAL
    ))
    prompts = ["First", "Second"]
    
    with patch("claude_code_sdk.batch.query", mock_query):
        results = await processor.process_batch(prompts)
    
    assert len(results) == 2
    assert results[0].prompt == "First"
    assert results[1].prompt == "Second"


@pytest.mark.asyncio
async def test_batch_processor_parallel(mock_query):
    """Test parallel batch processing."""
    processor = BatchProcessor(BatchOptions(
        strategy=BatchStrategy.PARALLEL
    ))
    prompts = ["One", "Two", "Three"]
    
    with patch("claude_code_sdk.batch.query", mock_query):
        results = await processor.process_batch(prompts)
    
    assert len(results) == 3
    assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_batch_processor_stop_on_error(mock_query):
    """Test stop on error behavior."""
    processor = BatchProcessor(BatchOptions(
        strategy=BatchStrategy.SEQUENTIAL,
        stop_on_error=True
    ))
    prompts = ["Good", "error prompt", "Should not run"]
    
    with patch("claude_code_sdk.batch.query", mock_query):
        results = await processor.process_batch(prompts)
    
    assert len(results) == 2  # Should stop after error
    assert results[0].success
    assert not results[1].success


@pytest.mark.asyncio
async def test_batch_processor_callbacks():
    """Test batch processing with callbacks."""
    started = []
    completed = []
    failed = []
    
    processor = BatchProcessor(BatchOptions(
        on_start=lambda p: started.append(p),
        on_complete=lambda r: completed.append(r.prompt),
        on_error=lambda e, p: failed.append(p)
    ))
    
    prompts = ["Test1", "error test", "Test2"]
    
    async def custom_query(prompt, options=None):
        if "error" in prompt:
            raise Exception("Test error")
        async for msg in mock_query()(prompt, options):
            yield msg
    
    mock_query_instance = mock_query()
    with patch("claude_code_sdk.batch.query", custom_query):
        results = await processor.process_batch(prompts)
    
    assert started == prompts
    assert len(completed) == 2
    assert len(failed) == 1
    assert failed[0] == "error test"


@pytest.mark.asyncio
async def test_batch_processor_retry(mock_query):
    """Test batch processing with retry."""
    call_count = 0
    
    async def flaky_query(prompt, options=None):
        nonlocal call_count
        call_count += 1
        if prompt == "flaky" and call_count < 3:
            raise Exception("Temporary error")
        async for msg in mock_query()(prompt, options):
            yield msg
    
    processor = BatchProcessor(BatchOptions(
        retry_failed=True,
        max_retries=3
    ))
    
    with patch("claude_code_sdk.batch.query", flaky_query):
        results = await processor.process_batch(["flaky"])
    
    assert len(results) == 1
    assert results[0].success
    assert call_count == 3


@pytest.mark.asyncio
async def test_batch_query_function(mock_query):
    """Test the convenience batch_query function."""
    with patch("claude_code_sdk.batch.query", mock_query):
        results = await batch_query(["Hello", "World"])
    
    assert len(results) == 2
    assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_batch_with_callbacks(mock_query):
    """Test batch_with_callbacks convenience function."""
    completed = []
    
    with patch("claude_code_sdk.batch.query", mock_query):
        results = await batch_with_callbacks(
            prompts=["Test"],
            on_complete=lambda r: completed.append(r)
        )
    
    assert len(results) == 1
    assert len(completed) == 1


@pytest.mark.asyncio
async def test_batch_with_concurrency(mock_query):
    """Test batch_with_concurrency convenience function."""
    with patch("claude_code_sdk.batch.query", mock_query):
        results = await batch_with_concurrency(
            prompts=["A", "B", "C"],
            max_concurrent=2
        )
    
    assert len(results) == 3


@pytest.mark.asyncio
async def test_batch_with_strategy(mock_query):
    """Test batch_with_strategy convenience function."""
    with patch("claude_code_sdk.batch.query", mock_query):
        results = await batch_with_strategy(
            prompts=["One", "Two"],
            strategy=BatchStrategy.SEQUENTIAL
        )
    
    assert len(results) == 2


@pytest.mark.asyncio
async def test_batch_with_retry(mock_query):
    """Test batch_with_retry convenience function."""
    with patch("claude_code_sdk.batch.query", mock_query):
        results = await batch_with_retry(
            prompts=["Test"],
            max_retries=3
        )
    
    assert len(results) == 1


@pytest.mark.asyncio
async def test_batch_result_aggregation():
    """Test BatchResult aggregation methods."""
    results = [
        BatchResult(
            prompt="A",
            success=True,
            messages=[AssistantMessage(content=[TextBlock(text="Response A")])],
            cost_usd=0.01,
            duration_ms=100,
            session_id="session-a",
            usage={"total_tokens": 10}
        ),
        BatchResult(
            prompt="B",
            success=True,
            messages=[AssistantMessage(content=[TextBlock(text="Response B")])],
            cost_usd=0.02,
            duration_ms=200,
            session_id="session-b",
            usage={"total_tokens": 20}
        ),
        BatchResult(
            prompt="C",
            success=False,
            messages=[],
            error="Error C"
        )
    ]
    
    # Test successful results
    successful = BatchResult.successful_results(results)
    assert len(successful) == 2
    assert all(r.success for r in successful)
    
    # Test failed results
    failed = BatchResult.failed_results(results)
    assert len(failed) == 1
    assert not failed[0].success
    
    # Test total cost
    total_cost = BatchResult.total_cost(results)
    assert total_cost == 0.03
    
    # Test average duration
    avg_duration = BatchResult.average_duration(results)
    assert avg_duration == 150.0


@pytest.mark.asyncio
async def test_batch_processor_with_options(mock_query):
    """Test batch processing with Claude options."""
    options = ClaudeCodeOptions(
        allowed_tools=["Read", "Write"],
        max_thinking_tokens=1000
    )
    
    processor = BatchProcessor(BatchOptions(
        default_options=options
    ))
    
    with patch("claude_code_sdk.batch.query", mock_query):
        results = await processor.process_batch(["Test"])
    
    assert len(results) == 1
    assert results[0].success


@pytest.mark.asyncio
async def test_batch_processor_empty_batch():
    """Test batch processing with empty input."""
    processor = BatchProcessor()
    results = await processor.process_batch([])
    
    assert len(results) == 0


@pytest.mark.asyncio
async def test_batch_processor_exception_handling(mock_query):
    """Test exception handling in batch processing."""
    async def error_query(prompt, options=None):
        if prompt == "exception":
            raise ValueError("Test exception")
        async for msg in mock_query()(prompt, options):
            yield msg
    
    processor = BatchProcessor()
    
    with patch("claude_code_sdk.batch.query", error_query):
        results = await processor.process_batch(["exception", "normal"])
    
    assert len(results) == 2
    assert not results[0].success
    assert "Test exception" in results[0].error
    assert results[1].success


@pytest.mark.asyncio
async def test_batch_processor_progress_tracking():
    """Test progress tracking during batch processing."""
    progress_updates = []
    
    processor = BatchProcessor(BatchOptions(
        on_progress=lambda completed, total: progress_updates.append((completed, total))
    ))
    
    prompts = ["A", "B", "C"]
    
    # Use the fixture's mock_query
    async def tracking_query(prompt, options=None):
        async for msg in mock_query()(prompt, options):
            yield msg
    
    mock_query_instance = mock_query()
    with patch("claude_code_sdk.batch.query", tracking_query):
        await processor.process_batch(prompts)
    
    # Should have progress updates for each completed prompt
    assert len(progress_updates) >= len(prompts)
    assert progress_updates[-1] == (3, 3)


@pytest.mark.asyncio
async def test_batch_processor_timeout():
    """Test timeout handling in batch processing."""
    import asyncio
    
    async def slow_query(prompt, options=None):
        await asyncio.sleep(2)  # Simulate slow response
        yield AssistantMessage(content=[TextBlock(text="Slow response")])
    
    processor = BatchProcessor(BatchOptions(
        timeout_seconds=0.1  # Very short timeout
    ))
    
    with patch("claude_code_sdk.batch.query", slow_query):
        results = await processor.process_batch(["Test"])
    
    assert len(results) == 1
    assert not results[0].success
    assert "timeout" in results[0].error.lower()