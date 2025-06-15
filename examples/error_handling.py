"""Examples of error handling and recovery patterns with Claude SDK."""

import asyncio
import logging
import time
from typing import AsyncIterator, Optional

import anyio

from claude_code_sdk import (
    AssistantMessage,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ClaudeCodeOptions,
    Message,
    ProcessError,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    query,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Example 1: Basic error handling
async def basic_error_handling():
    """Demonstrate basic error handling patterns."""
    print("\n=== Example 1: Basic Error Handling ===")
    
    try:
        async for message in query(prompt="Hello Claude"):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text}")
                        
    except CLINotFoundError:
        print("Error: Claude Code CLI not found!")
        print("Please install it with: npm install -g @anthropic-ai/claude-code")
        
    except ProcessError as e:
        print(f"Process error (exit code {e.exit_code}): {e}")
        if e.stderr:
            print(f"Details: {e.stderr}")
            
    except CLIConnectionError as e:
        print(f"Connection error: {e}")
        print("Check if Claude Code is running and accessible")
        
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")


# Example 2: Retry logic with exponential backoff
async def query_with_retry(
    prompt: str,
    options: Optional[ClaudeCodeOptions] = None,
    max_retries: int = 3,
    initial_delay: float = 1.0
) -> AsyncIterator[Message]:
    """Query with automatic retry on failure."""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}")
            
            async for message in query(prompt=prompt, options=options):
                yield message
            return  # Success
            
        except CLINotFoundError:
            # Can't recover from this - re-raise immediately
            logger.error("Claude Code not installed - cannot retry")
            raise
            
        except (CLIConnectionError, ProcessError) as e:
            last_error = e
            
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                logger.warning(
                    f"Error on attempt {attempt + 1}: {e}. "
                    f"Retrying in {delay} seconds..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {max_retries} attempts failed")
                
    if last_error:
        raise last_error


async def retry_example():
    """Demonstrate retry logic."""
    print("\n=== Example 2: Retry with Exponential Backoff ===")
    
    try:
        async for message in query_with_retry(
            prompt="Tell me about error handling",
            max_retries=3
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text[:100]}...")
                        
    except Exception as e:
        print(f"Failed after retries: {e}")


# Example 3: Handling tool errors
async def handle_tool_errors():
    """Demonstrate handling of tool execution errors."""
    print("\n=== Example 3: Handling Tool Errors ===")
    
    options = ClaudeCodeOptions(
        allowed_tools=["Read"],
        permission_mode="bypassPermissions"
    )
    
    # Ask Claude to read a non-existent file
    prompt = "Please read the file /this/file/does/not/exist.txt"
    
    tool_errors = []
    
    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text}")
                    elif isinstance(block, ToolResultBlock) and block.is_error:
                        tool_errors.append(block)
                        print(f"Tool error detected: {block.content}")
                        
    except Exception as e:
        print(f"Query error: {e}")
        
    if tool_errors:
        print(f"\nFound {len(tool_errors)} tool errors during execution")


# Example 4: Timeout handling
async def handle_timeouts():
    """Demonstrate timeout handling for long operations."""
    print("\n=== Example 4: Timeout Handling ===")
    
    options = ClaudeCodeOptions(
        allowed_tools=["Bash"],
        permission_mode="bypassPermissions"
    )
    
    # Simulate a long-running operation
    prompt = "Run a command that takes 5 seconds: sleep 5 && echo 'Done'"
    
    try:
        # Set a 3-second timeout (will fail)
        async with asyncio.timeout(3):
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    print("Claude is working...")
                    
    except asyncio.TimeoutError:
        print("Operation timed out after 3 seconds")
        print("Tip: Increase timeout for long operations")


# Example 5: Graceful degradation
class RobustClaudeClient:
    """A robust client with fallback strategies."""
    
    def __init__(self):
        self.fallback_mode = False
        self.error_count = 0
        self.max_errors = 3
        
    async def query_with_fallback(
        self,
        prompt: str,
        options: Optional[ClaudeCodeOptions] = None
    ) -> AsyncIterator[Message]:
        """Query with fallback to limited functionality."""
        
        if self.fallback_mode:
            # In fallback mode, use minimal options
            logger.warning("Running in fallback mode with limited tools")
            safe_options = ClaudeCodeOptions(
                allowed_tools=["Read"],  # Only safe read operations
                max_turns=1,
                permission_mode="default"
            )
        else:
            safe_options = options or ClaudeCodeOptions()
            
        try:
            async for message in query(prompt=prompt, options=safe_options):
                yield message
                
            # Success - reset error count
            self.error_count = 0
            
        except (CLIConnectionError, ProcessError) as e:
            self.error_count += 1
            logger.error(f"Error #{self.error_count}: {e}")
            
            if self.error_count >= self.max_errors:
                logger.warning("Entering fallback mode due to repeated errors")
                self.fallback_mode = True
                
            raise


async def graceful_degradation_example():
    """Demonstrate graceful degradation."""
    print("\n=== Example 5: Graceful Degradation ===")
    
    client = RobustClaudeClient()
    
    prompts = [
        "What is 2 + 2?",
        "Read the README.md file",
        "Run ls command",  # Might fail in fallback mode
    ]
    
    for prompt in prompts:
        try:
            print(f"\nQuery: {prompt}")
            async for message in client.query_with_fallback(prompt):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(f"Response: {block.text[:100]}...")
                            
        except Exception as e:
            print(f"Failed: {e}")


# Example 6: Circuit breaker pattern
class CircuitBreaker:
    """Simple circuit breaker implementation."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half_open
        
    def call_succeeded(self):
        """Record successful call."""
        self.failure_count = 0
        self.state = "closed"
        
    def call_failed(self):
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
    def can_proceed(self) -> bool:
        """Check if calls can proceed."""
        if self.state == "closed":
            return True
            
        if self.state == "open":
            if (self.last_failure_time and 
                time.time() - self.last_failure_time > self.recovery_timeout):
                self.state = "half_open"
                logger.info("Circuit breaker entering half-open state")
                return True
            return False
            
        # half_open - allow one test call
        return True


async def circuit_breaker_example():
    """Demonstrate circuit breaker pattern."""
    print("\n=== Example 6: Circuit Breaker Pattern ===")
    
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=5.0)
    
    async def protected_query(prompt: str) -> bool:
        """Query with circuit breaker protection."""
        if not breaker.can_proceed():
            print("Circuit breaker is OPEN - request blocked")
            return False
            
        try:
            async for message in query(prompt=prompt):
                if isinstance(message, AssistantMessage):
                    print("Success!")
                    breaker.call_succeeded()
                    return True
                    
        except Exception as e:
            print(f"Failed: {e}")
            breaker.call_failed()
            return False
            
        return False
    
    # Simulate multiple failures
    prompts = ["Test 1", "Test 2", "Test 3", "Test 4"]
    
    for i, prompt in enumerate(prompts):
        print(f"\nAttempt {i+1}: {prompt}")
        await protected_query(prompt)
        
        # Add delay between attempts
        if i < len(prompts) - 1:
            await asyncio.sleep(1)


# Example 7: Comprehensive error monitoring
class ErrorMonitor:
    """Monitor and report on errors."""
    
    def __init__(self):
        self.errors: list[tuple[float, Exception]] = []
        self.start_time = time.time()
        
    def record_error(self, error: Exception):
        """Record an error with timestamp."""
        self.errors.append((time.time(), error))
        
    def get_error_summary(self) -> dict:
        """Get summary of errors."""
        if not self.errors:
            return {"total_errors": 0}
            
        error_types = {}
        for _, error in self.errors:
            error_type = type(error).__name__
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
        return {
            "total_errors": len(self.errors),
            "uptime_seconds": time.time() - self.start_time,
            "error_types": error_types,
            "error_rate": len(self.errors) / (time.time() - self.start_time),
            "last_error": str(self.errors[-1][1]) if self.errors else None,
        }


async def error_monitoring_example():
    """Demonstrate error monitoring."""
    print("\n=== Example 7: Error Monitoring ===")
    
    monitor = ErrorMonitor()
    
    test_prompts = [
        ("Valid prompt", "What is 2 + 2?", None),
        ("Tool error", "Read /invalid/path.txt", 
         ClaudeCodeOptions(allowed_tools=["Read"], permission_mode="bypassPermissions")),
        ("Another query", "Hello Claude", None),
    ]
    
    for name, prompt, options in test_prompts:
        print(f"\nTesting: {name}")
        try:
            message_count = 0
            async for message in query(prompt=prompt, options=options):
                message_count += 1
                if isinstance(message, ResultMessage):
                    print(f"Success! Cost: ${message.cost_usd:.4f}")
                    
        except Exception as e:
            monitor.record_error(e)
            print(f"Error recorded: {type(e).__name__}")
            
    # Print error summary
    print("\n=== Error Summary ===")
    summary = monitor.get_error_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")


# Main execution
async def main():
    """Run all examples."""
    examples = [
        ("Basic Error Handling", basic_error_handling),
        ("Retry with Backoff", retry_example),
        ("Tool Error Handling", handle_tool_errors),
        ("Timeout Handling", handle_timeouts),
        ("Graceful Degradation", graceful_degradation_example),
        ("Circuit Breaker", circuit_breaker_example),
        ("Error Monitoring", error_monitoring_example),
    ]
    
    print("Claude SDK Error Handling Examples")
    print("==================================")
    
    for name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"\nExample '{name}' failed: {e}")
            
        # Small delay between examples
        await asyncio.sleep(1)
        
    print("\n\nAll examples completed!")


if __name__ == "__main__":
    anyio.run(main)