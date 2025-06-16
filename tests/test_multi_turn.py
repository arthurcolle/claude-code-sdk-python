"""Tests for multi-turn conversation functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from claude_code_sdk import (
    query,
    ClaudeCodeOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
)
from claude_code_sdk._internal.client import InternalClient


@pytest.mark.asyncio
async def test_multi_turn_with_session_id(monkeypatch):
    """Test multi-turn conversation using session ID."""
    mock_client = AsyncMock(spec=InternalClient)
    session_id = "test-session-123"
    
    # Mock responses for two turns
    turn1_messages = [
        AssistantMessage(content=[TextBlock(text="I'll remember that.")]),
        ResultMessage(
            subtype="success",
            cost_usd=0.01,
            duration_ms=1000,
            duration_api_ms=800,
            is_error=False,
            num_turns=1,
            session_id=session_id,
            total_cost_usd=0.01,
        ),
    ]
    
    turn2_messages = [
        AssistantMessage(content=[TextBlock(text="Your name is Alice.")]),
        ResultMessage(
            subtype="success",
            cost_usd=0.01,
            duration_ms=900,
            duration_api_ms=700,
            is_error=False,
            num_turns=2,
            session_id=session_id,
            total_cost_usd=0.02,
        ),
    ]
    
    # Set up mock to return different messages for each call
    call_count = 0
    
    async def mock_process_query(*args, **kwargs):
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            for msg in turn1_messages:
                yield msg
        else:
            for msg in turn2_messages:
                yield msg
    
    mock_client.process_query = mock_process_query
    
    # Patch the client creation
    monkeypatch.setattr(
        "claude_code_sdk.InternalClient",
        lambda: mock_client
    )
    
    # First turn
    captured_session_id = None
    messages = []
    async for message in query(prompt="Remember my name is Alice"):
        messages.append(message)
        if isinstance(message, ResultMessage):
            captured_session_id = message.session_id
    
    assert captured_session_id == session_id
    assert len(messages) == 2
    assert isinstance(messages[0], AssistantMessage)
    assert messages[0].content[0].text == "I'll remember that."
    
    # Second turn with session ID
    messages = []
    async for message in query(
        prompt="What's my name?",
        options=ClaudeCodeOptions(resume=session_id)
    ):
        messages.append(message)
    
    assert len(messages) == 2
    assert isinstance(messages[0], AssistantMessage)
    assert messages[0].content[0].text == "Your name is Alice."
    assert messages[1].total_cost_usd == 0.02  # Cumulative cost


@pytest.mark.asyncio
async def test_continue_conversation_flag(monkeypatch):
    """Test using continue_conversation flag."""
    mock_client = AsyncMock(spec=InternalClient)
    
    # Mock response
    messages = [
        AssistantMessage(content=[TextBlock(text="Continuing our conversation...")]),
        ResultMessage(
            subtype="success",
            cost_usd=0.01,
            duration_ms=1000,
            duration_api_ms=800,
            is_error=False,
            num_turns=2,
            session_id="continued-session",
            total_cost_usd=0.02,
        ),
    ]
    
    async def mock_process_query(prompt, options):
        # Verify continue_conversation is set
        assert options.continue_conversation is True
        for msg in messages:
            yield msg
    
    mock_client.process_query = mock_process_query
    
    monkeypatch.setattr(
        "claude_code_sdk.InternalClient",
        lambda: mock_client
    )
    
    # Use continue_conversation
    result_messages = []
    async for message in query(
        prompt="What were we discussing?",
        options=ClaudeCodeOptions(continue_conversation=True)
    ):
        result_messages.append(message)
    
    assert len(result_messages) == 2
    assert result_messages[0].content[0].text == "Continuing our conversation..."


@pytest.mark.asyncio
async def test_new_conversation_without_resume(monkeypatch):
    """Test that omitting resume/continue_conversation starts a new conversation."""
    mock_client = AsyncMock(spec=InternalClient)
    
    messages = [
        AssistantMessage(content=[TextBlock(text="Hello! This is a new conversation.")]),
        ResultMessage(
            subtype="success",
            cost_usd=0.01,
            duration_ms=1000,
            duration_api_ms=800,
            is_error=False,
            num_turns=1,
            session_id="new-session-456",
            total_cost_usd=0.01,
        ),
    ]
    
    async def mock_process_query(prompt, options):
        # Verify no resume or continue_conversation is set
        assert options.resume is None
        assert options.continue_conversation is False
        for msg in messages:
            yield msg
    
    mock_client.process_query = mock_process_query
    
    monkeypatch.setattr(
        "claude_code_sdk.InternalClient",
        lambda: mock_client
    )
    
    # Start new conversation
    result_messages = []
    async for message in query(prompt="Hello Claude"):
        result_messages.append(message)
    
    assert len(result_messages) == 2
    assert result_messages[1].num_turns == 1  # New conversation


@pytest.mark.asyncio
async def test_max_turns_limit(monkeypatch):
    """Test that max_turns option is passed correctly."""
    mock_client = AsyncMock(spec=InternalClient)
    
    async def mock_process_query(prompt, options):
        # Verify max_turns is set
        assert options.max_turns == 5
        yield AssistantMessage(content=[TextBlock(text="Limited conversation")])
    
    mock_client.process_query = mock_process_query
    
    monkeypatch.setattr(
        "claude_code_sdk.InternalClient",
        lambda: mock_client
    )
    
    # Use max_turns
    async for message in query(
        prompt="Hello",
        options=ClaudeCodeOptions(max_turns=5)
    ):
        if isinstance(message, AssistantMessage):
            assert message.content[0].text == "Limited conversation"


@pytest.mark.asyncio
async def test_session_tracking_across_multiple_turns(monkeypatch):
    """Test tracking session across multiple conversation turns."""
    mock_client = AsyncMock(spec=InternalClient)
    session_id = "multi-turn-session"
    
    # Simulate a conversation with 3 turns
    turns = [
        {
            "prompt": "My favorite color is blue",
            "response": "I'll remember that your favorite color is blue.",
            "turn_num": 1,
            "total_cost": 0.01
        },
        {
            "prompt": "I also like pizza",
            "response": "Got it! You like blue and pizza.",
            "turn_num": 2,
            "total_cost": 0.02
        },
        {
            "prompt": "What do you know about me?",
            "response": "I know your favorite color is blue and you like pizza.",
            "turn_num": 3,
            "total_cost": 0.03
        }
    ]
    
    call_count = 0
    
    async def mock_process_query(prompt, options):
        nonlocal call_count
        turn = turns[call_count]
        
        # Verify resume is used after first turn
        if call_count > 0:
            assert options.resume == session_id
        
        yield AssistantMessage(content=[TextBlock(text=turn["response"])])
        yield ResultMessage(
            subtype="success",
            cost_usd=0.01,
            duration_ms=1000,
            duration_api_ms=800,
            is_error=False,
            num_turns=turn["turn_num"],
            session_id=session_id,
            total_cost_usd=turn["total_cost"],
        )
        
        call_count += 1
    
    mock_client.process_query = mock_process_query
    
    monkeypatch.setattr(
        "claude_code_sdk.InternalClient",
        lambda: mock_client
    )
    
    # Run the conversation
    captured_session_id = None
    
    for i, turn in enumerate(turns):
        options = ClaudeCodeOptions()
        if captured_session_id:
            options.resume = captured_session_id
        
        messages = []
        async for message in query(prompt=turn["prompt"], options=options):
            messages.append(message)
            if isinstance(message, ResultMessage):
                captured_session_id = message.session_id
                assert message.num_turns == turn["turn_num"]
                assert message.total_cost_usd == turn["total_cost"]
        
        # Verify response
        assert isinstance(messages[0], AssistantMessage)
        assert messages[0].content[0].text == turn["response"]