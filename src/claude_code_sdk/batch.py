"""Batch processing for multiple prompts."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union
from collections.abc import AsyncIterator
from datetime import datetime
from enum import Enum

from .types import Message, ClaudeCodeOptions, ResultMessage
from .pool import ConnectionPool, get_global_pool


class BatchStatus(Enum):
    """Batch processing status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchItem:
    """A single item in a batch."""
    
    id: str
    prompt: str
    options: ClaudeCodeOptions | None = None
    status: BatchStatus = BatchStatus.PENDING
    messages: list[Message] = field(default_factory=list)
    error: Exception | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    
    @property
    def duration_ms(self) -> int | None:
        """Get duration in milliseconds."""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None
    
    @property
    def cost_usd(self) -> float:
        """Get total cost for this item."""
        for msg in self.messages:
            if isinstance(msg, ResultMessage):
                return msg.cost_usd
        return 0.0


@dataclass
class BatchResult:
    """Result of batch processing."""
    
    items: list[BatchItem]
    total_duration_ms: int
    successful_count: int
    failed_count: int
    total_cost_usd: float
    
    @classmethod
    def from_items(cls, items: list[BatchItem], duration_ms: int) -> "BatchResult":
        """Create result from processed items."""
        successful = sum(1 for item in items if item.status == BatchStatus.COMPLETED)
        failed = sum(1 for item in items if item.status == BatchStatus.FAILED)
        total_cost = sum(item.cost_usd for item in items)
        
        return cls(
            items=items,
            total_duration_ms=duration_ms,
            successful_count=successful,
            failed_count=failed,
            total_cost_usd=total_cost,
        )


@dataclass
class BatchProcessor:
    """Processes multiple queries in batch."""
    
    max_concurrent: int = 5
    pool: Optional[ConnectionPool] = None
    on_item_start: Optional[Callable[[BatchItem], None]] = None
    on_item_complete: Optional[Callable[[BatchItem], None]] = None
    on_item_error: Optional[Callable[[BatchItem, Exception], None]] = None
    retry_failed: bool = True
    continue_on_error: bool = True
    
    _semaphore: asyncio.Semaphore = field(init=False)
    
    def __post_init__(self):
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        if self.pool is None:
            self.pool = get_global_pool()
    
    async def _process_item(self, item: BatchItem) -> None:
        """Process a single batch item."""
        async with self._semaphore:
            try:
                item.status = BatchStatus.RUNNING
                item.started_at = datetime.now()
                
                if self.on_item_start:
                    self.on_item_start(item)
                
                # Execute query
                async for message in self.pool.query(
                    prompt=item.prompt,
                    options=item.options
                ):
                    item.messages.append(message)
                
                item.status = BatchStatus.COMPLETED
                item.completed_at = datetime.now()
                
                if self.on_item_complete:
                    self.on_item_complete(item)
                    
            except Exception as e:
                item.status = BatchStatus.FAILED
                item.error = e
                item.completed_at = datetime.now()
                
                if self.on_item_error:
                    self.on_item_error(item, e)
                
                if not self.continue_on_error:
                    raise
    
    async def process(
        self,
        items: list[tuple[str, str, ClaudeCodeOptions | None]]
    ) -> BatchResult:
        """
        Process multiple queries in batch.
        
        Args:
            items: List of (id, prompt, options) tuples
            
        Returns:
            BatchResult with all processed items
            
        Example:
            ```python
            processor = BatchProcessor(max_concurrent=10)
            
            items = [
                ("q1", "Hello", None),
                ("q2", "How are you?", options),
                ("q3", "Goodbye", None),
            ]
            
            result = await processor.process(items)
            print(f"Processed {result.successful_count} queries")
            print(f"Total cost: ${result.total_cost_usd:.4f}")
            ```
        """
        start_time = datetime.now()
        
        # Create batch items
        batch_items = [
            BatchItem(id=item_id, prompt=prompt, options=options)
            for item_id, prompt, options in items
        ]
        
        # Process all items
        tasks = [
            self._process_item(item)
            for item in batch_items
        ]
        
        # Handle retry if enabled
        if self.retry_failed:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Retry failed items once
            retry_tasks = []
            for i, (item, result) in enumerate(zip(batch_items, results)):
                if item.status == BatchStatus.FAILED:
                    item.status = BatchStatus.PENDING  # Reset for retry
                    item.error = None
                    retry_tasks.append((i, self._process_item(item)))
            
            if retry_tasks:
                retry_results = await asyncio.gather(
                    *[task for _, task in retry_tasks],
                    return_exceptions=True
                )
        else:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate duration
        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return BatchResult.from_items(batch_items, duration_ms)


@dataclass
class StreamingBatchProcessor(BatchProcessor):
    """Batch processor with streaming results."""
    
    async def process_streaming(
        self,
        items: list[tuple[str, str, ClaudeCodeOptions | None]]
    ) -> AsyncIterator[tuple[str, Message]]:
        """
        Process batch with streaming results.
        
        Yields:
            Tuples of (item_id, message) as they arrive
            
        Example:
            ```python
            processor = StreamingBatchProcessor()
            
            items = [("q1", "Hello", None), ("q2", "Hi", None)]
            
            async for item_id, message in processor.process_streaming(items):
                print(f"{item_id}: {message}")
            ```
        """
        # Create batch items
        batch_items = {
            item_id: BatchItem(id=item_id, prompt=prompt, options=options)
            for item_id, prompt, options in items
        }
        
        # Create message queues for each item
        queues: dict[str, asyncio.Queue[Message | None]] = {
            item_id: asyncio.Queue()
            for item_id in batch_items
        }
        
        async def _process_with_queue(item: BatchItem) -> None:
            """Process item and put messages in queue."""
            queue = queues[item.id]
            try:
                async with self._semaphore:
                    item.status = BatchStatus.RUNNING
                    item.started_at = datetime.now()
                    
                    async for message in self.pool.query(
                        prompt=item.prompt,
                        options=item.options
                    ):
                        item.messages.append(message)
                        await queue.put(message)
                    
                    item.status = BatchStatus.COMPLETED
            except Exception as e:
                item.status = BatchStatus.FAILED
                item.error = e
            finally:
                item.completed_at = datetime.now()
                await queue.put(None)  # Sentinel
        
        # Start all processors
        tasks = [
            asyncio.create_task(_process_with_queue(item))
            for item in batch_items.values()
        ]
        
        # Stream results as they arrive
        active_queues = set(queues.keys())
        
        while active_queues:
            # Get next available message
            for item_id in list(active_queues):
                queue = queues[item_id]
                try:
                    message = queue.get_nowait()
                    if message is None:
                        active_queues.remove(item_id)
                    else:
                        yield (item_id, message)
                except asyncio.QueueEmpty:
                    pass
            
            if active_queues:
                await asyncio.sleep(0.01)  # Small delay
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)


# Convenience functions
async def batch_query(
    prompts: list[str],
    options: ClaudeCodeOptions | None = None,
    max_concurrent: int = 5,
    pool: Optional[ConnectionPool] = None
) -> BatchResult:
    """
    Execute multiple queries in batch with same options.
    
    Args:
        prompts: List of prompts to process
        options: Options to use for all queries
        max_concurrent: Maximum concurrent queries
        pool: Connection pool (uses global if None)
        
    Returns:
        BatchResult with processed items
        
    Example:
        ```python
        prompts = ["Hello", "How are you?", "What's 2+2?"]
        result = await batch_query(prompts, max_concurrent=10)
        
        for item in result.items:
            print(f"{item.prompt}: {item.status}")
        ```
    """
    processor = BatchProcessor(
        max_concurrent=max_concurrent,
        pool=pool
    )
    
    items = [
        (f"q{i}", prompt, options)
        for i, prompt in enumerate(prompts)
    ]
    
    return await processor.process(items)


async def batch_query_streaming(
    prompts: list[str],
    options: ClaudeCodeOptions | None = None,
    max_concurrent: int = 5,
    pool: Optional[ConnectionPool] = None
) -> AsyncIterator[tuple[int, Message]]:
    """
    Execute batch queries with streaming results.
    
    Yields:
        Tuples of (prompt_index, message)
        
    Example:
        ```python
        prompts = ["Hello", "Hi", "Hey"]
        
        async for idx, message in batch_query_streaming(prompts):
            print(f"Prompt {idx}: {message}")
        ```
    """
    processor = StreamingBatchProcessor(
        max_concurrent=max_concurrent,
        pool=pool
    )
    
    items = [
        (str(i), prompt, options)
        for i, prompt in enumerate(prompts)
    ]
    
    async for item_id, message in processor.process_streaming(items):
        yield (int(item_id), message)