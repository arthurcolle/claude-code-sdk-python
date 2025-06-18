"""
Tests for Dynamic Multi-Agent System
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from claude_code_sdk.dynamic_multi_agent import (
    AgentCapability,
    AgentProfile,
    DynamicTask,
    DynamicAgent,
    ClaudeDynamicAgent,
    AgentRegistry,
    DynamicMultiAgentCoordinator,
    MessageBus,
    create_development_team,
    create_review_pipeline,
    create_research_team
)


class TestAgentProfile:
    """Test AgentProfile dataclass"""
    
    def test_agent_profile_creation(self):
        profile = AgentProfile(
            name="Test Agent",
            capabilities={AgentCapability.CODE_GENERATION, AgentCapability.TESTING},
            tools=["Read", "Write"],
            priority=7
        )
        
        assert profile.name == "Test Agent"
        assert AgentCapability.CODE_GENERATION in profile.capabilities
        assert AgentCapability.TESTING in profile.capabilities
        assert profile.tools == ["Read", "Write"]
        assert profile.priority == 7
        assert profile.max_concurrent_tasks == 3  # default


class TestDynamicTask:
    """Test DynamicTask dataclass"""
    
    def test_task_creation(self):
        task = DynamicTask(
            id="task-123",
            description="Test task",
            required_capabilities={AgentCapability.CODE_GENERATION},
            priority=8
        )
        
        assert task.id == "task-123"
        assert task.description == "Test task"
        assert AgentCapability.CODE_GENERATION in task.required_capabilities
        assert task.priority == 8
        assert task.status == "pending"
        assert task.assigned_agent is None


class TestDynamicAgent:
    """Test base DynamicAgent functionality"""
    
    @pytest.mark.asyncio
    async def test_can_handle_task(self):
        profile = AgentProfile(
            name="Test Agent",
            capabilities={AgentCapability.CODE_GENERATION, AgentCapability.TESTING},
            tools=["Read", "Write"]
        )
        
        # Create concrete implementation for testing
        class TestAgent(DynamicAgent):
            async def process_task(self, task):
                return "processed"
        
        agent = TestAgent("agent-1", profile)
        
        # Task with matching capabilities
        task1 = DynamicTask(
            id="task-1",
            description="Generate code",
            required_capabilities={AgentCapability.CODE_GENERATION}
        )
        assert await agent.can_handle_task(task1) is True
        
        # Task with non-matching capabilities
        task2 = DynamicTask(
            id="task-2",
            description="Review security",
            required_capabilities={AgentCapability.SECURITY}
        )
        assert await agent.can_handle_task(task2) is False
    
    @pytest.mark.asyncio
    async def test_accept_task(self):
        profile = AgentProfile(
            name="Test Agent",
            capabilities={AgentCapability.CODE_GENERATION},
            tools=["Read"],
            max_concurrent_tasks=2
        )
        
        class TestAgent(DynamicAgent):
            async def process_task(self, task):
                return "processed"
        
        agent = TestAgent("agent-1", profile)
        
        # Accept first task
        task1 = DynamicTask("task-1", "Task 1", {AgentCapability.CODE_GENERATION})
        assert await agent.accept_task(task1) is True
        assert len(agent.current_tasks) == 1
        assert task1.assigned_agent == "agent-1"
        assert task1.status == "in_progress"
        
        # Accept second task
        task2 = DynamicTask("task-2", "Task 2", {AgentCapability.CODE_GENERATION})
        assert await agent.accept_task(task2) is True
        assert len(agent.current_tasks) == 2
        
        # Reject third task (at capacity)
        task3 = DynamicTask("task-3", "Task 3", {AgentCapability.CODE_GENERATION})
        assert await agent.accept_task(task3) is False
        assert len(agent.current_tasks) == 2
    
    @pytest.mark.asyncio
    async def test_complete_task(self):
        profile = AgentProfile(
            name="Test Agent",
            capabilities={AgentCapability.CODE_GENERATION},
            tools=["Read"]
        )
        
        class TestAgent(DynamicAgent):
            async def process_task(self, task):
                return "processed"
        
        agent = TestAgent("agent-1", profile)
        
        task = DynamicTask("task-1", "Task 1", {AgentCapability.CODE_GENERATION})
        await agent.accept_task(task)
        
        # Complete task
        result = {"output": "task completed"}
        await agent.complete_task(task, result)
        
        assert task.status == "completed"
        assert task.result == result
        assert task.completed_at is not None
        assert len(agent.current_tasks) == 0
        assert len(agent.completed_tasks) == 1


class TestClaudeDynamicAgent:
    """Test ClaudeDynamicAgent implementation"""
    
    def test_system_prompt_generation(self):
        profile = AgentProfile(
            name="Test Developer",
            capabilities={AgentCapability.CODE_GENERATION, AgentCapability.TESTING},
            tools=["Read", "Write"]
        )
        
        agent = ClaudeDynamicAgent("agent-1", profile)
        prompt = agent._generate_system_prompt()
        
        assert "Test Developer" in prompt
        assert "generate high-quality, efficient code" in prompt
        assert "write comprehensive tests" in prompt
    
    @pytest.mark.asyncio
    async def test_process_task(self):
        profile = AgentProfile(
            name="Test Agent",
            capabilities={AgentCapability.CODE_GENERATION},
            tools=["Read", "Write"]
        )
        
        agent = ClaudeDynamicAgent("agent-1", profile)
        
        # Mock the query function
        with patch('claude_code_sdk.query') as mock_query:
            # Mock response
            mock_message = MagicMock()
            mock_message.content = [
                MagicMock(text="Task completed successfully")
            ]
            mock_query.return_value.__aiter__.return_value = [mock_message]
            
            task = DynamicTask(
                id="task-1",
                description="Generate a function",
                required_capabilities={AgentCapability.CODE_GENERATION},
                context={"language": "python"}
            )
            
            result = await agent.process_task(task)
            
            assert result["agent_id"] == "agent-1"
            assert result["task_id"] == "task-1"
            assert "result" in result
            assert "timestamp" in result


class TestAgentRegistry:
    """Test AgentRegistry functionality"""
    
    def test_initialize_templates(self):
        registry = AgentRegistry()
        templates = registry._profile_templates
        
        assert "developer" in templates
        assert "reviewer" in templates
        assert "tester" in templates
        assert "architect" in templates
        assert "documenter" in templates
        assert "devops" in templates
        assert "coordinator" in templates
    
    def test_register_and_get_profile(self):
        registry = AgentRegistry()
        
        custom_profile = AgentProfile(
            name="Custom Agent",
            capabilities={AgentCapability.RESEARCH},
            tools=["WebSearch"]
        )
        
        registry.register_profile("custom", custom_profile)
        retrieved = registry.get_profile("custom")
        
        assert retrieved == custom_profile
    
    def test_create_agent_with_template(self):
        registry = AgentRegistry()
        
        agent = registry.create_agent("claude", "developer")
        
        assert isinstance(agent, ClaudeDynamicAgent)
        assert agent.profile.name == "Developer Agent"
        assert AgentCapability.CODE_GENERATION in agent.profile.capabilities
    
    def test_create_agent_with_custom_profile(self):
        registry = AgentRegistry()
        
        custom_profile = AgentProfile(
            name="Custom Agent",
            capabilities={AgentCapability.RESEARCH},
            tools=["WebSearch"]
        )
        
        agent = registry.create_agent("claude", custom_profile)
        
        assert isinstance(agent, ClaudeDynamicAgent)
        assert agent.profile == custom_profile
    
    def test_list_profiles(self):
        registry = AgentRegistry()
        
        # Add custom profile
        custom_profile = AgentProfile(
            name="Custom Agent",
            capabilities={AgentCapability.RESEARCH},
            tools=["WebSearch"]
        )
        registry.register_profile("custom", custom_profile)
        
        profiles = registry.list_profiles()
        
        # Should include both templates and custom profiles
        assert "developer" in profiles
        assert "custom" in profiles
        assert profiles["custom"] == custom_profile


class TestDynamicMultiAgentCoordinator:
    """Test DynamicMultiAgentCoordinator functionality"""
    
    @pytest.mark.asyncio
    async def test_spawn_agent(self):
        coordinator = DynamicMultiAgentCoordinator()
        await coordinator.start()
        
        agent_id = await coordinator.spawn_agent(profile="developer")
        
        assert agent_id in coordinator.agents
        assert isinstance(coordinator.agents[agent_id], ClaudeDynamicAgent)
        
        await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_spawn_agents_for_capabilities(self):
        coordinator = DynamicMultiAgentCoordinator()
        await coordinator.start()
        
        required_caps = {AgentCapability.CODE_GENERATION, AgentCapability.TESTING}
        agent_ids = await coordinator.spawn_agents_for_capabilities(required_caps, count=2)
        
        assert len(agent_ids) == 2
        for agent_id in agent_ids:
            agent = coordinator.agents[agent_id]
            assert required_caps.issubset(agent.profile.capabilities)
        
        await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_submit_task(self):
        coordinator = DynamicMultiAgentCoordinator()
        await coordinator.start()
        
        task = DynamicTask(
            id="test-task",
            description="Test task",
            required_capabilities={AgentCapability.CODE_GENERATION}
        )
        
        task_id = await coordinator.submit_task(task)
        
        assert task_id == "test-task"
        assert task_id in coordinator.pending_tasks
        assert coordinator.task_queue.qsize() == 1
        
        await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_find_suitable_agent(self):
        coordinator = DynamicMultiAgentCoordinator()
        await coordinator.start()
        
        # Spawn agents with different capabilities
        dev_id = await coordinator.spawn_agent(profile="developer")
        test_id = await coordinator.spawn_agent(profile="tester")
        
        # Task requiring code generation
        task = DynamicTask(
            id="task-1",
            description="Generate code",
            required_capabilities={AgentCapability.CODE_GENERATION}
        )
        
        suitable_agent = await coordinator._find_suitable_agent(task)
        
        assert suitable_agent is not None
        assert suitable_agent.id == dev_id  # Developer has CODE_GENERATION capability
        
        await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_terminate_agent(self):
        coordinator = DynamicMultiAgentCoordinator()
        await coordinator.start()
        
        agent_id = await coordinator.spawn_agent(profile="developer")
        
        # Add a task to the agent
        task = DynamicTask("task-1", "Test", {AgentCapability.CODE_GENERATION})
        agent = coordinator.agents[agent_id]
        await agent.accept_task(task)
        
        # Terminate agent
        await coordinator.terminate_agent(agent_id)
        
        assert agent_id not in coordinator.agents
        assert not agent.is_active
        # Task should be requeued
        assert coordinator.task_queue.qsize() == 1
        
        await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_get_system_status(self):
        coordinator = DynamicMultiAgentCoordinator()
        await coordinator.start()
        
        # Spawn some agents
        await coordinator.spawn_agent(profile="developer")
        await coordinator.spawn_agent(profile="tester")
        
        # Submit a task
        await coordinator.create_and_submit_task(
            description="Test task",
            required_capabilities={AgentCapability.CODE_GENERATION}
        )
        
        status = await coordinator.get_system_status()
        
        assert "agents" in status
        assert len(status["agents"]) == 2
        assert status["pending_tasks"] == 1
        assert status["completed_tasks"] == 0
        
        await coordinator.stop()


class TestMessageBus:
    """Test MessageBus functionality"""
    
    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        bus = MessageBus()
        received_messages = []
        
        def callback(message):
            received_messages.append(message)
        
        bus.subscribe("test_topic", callback)
        
        await bus.publish("test_topic", {"data": "test"})
        
        assert len(received_messages) == 1
        assert received_messages[0] == {"data": "test"}
    
    @pytest.mark.asyncio
    async def test_async_subscriber(self):
        bus = MessageBus()
        received_messages = []
        
        async def async_callback(message):
            received_messages.append(message)
        
        bus.subscribe("test_topic", async_callback)
        
        await bus.publish("test_topic", {"data": "async test"})
        
        assert len(received_messages) == 1
        assert received_messages[0] == {"data": "async test"}
    
    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        bus = MessageBus()
        received_messages = []
        
        def callback(message):
            received_messages.append(message)
        
        bus.subscribe("test_topic", callback)
        bus.unsubscribe("test_topic", callback)
        
        await bus.publish("test_topic", {"data": "test"})
        
        assert len(received_messages) == 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        bus = MessageBus()
        
        def failing_callback(message):
            raise Exception("Test error")
        
        def working_callback(message):
            message["processed"] = True
        
        bus.subscribe("test_topic", failing_callback)
        bus.subscribe("test_topic", working_callback)
        
        message = {"data": "test"}
        await bus.publish("test_topic", message)
        
        # Working callback should still process despite failing callback
        assert message.get("processed") is True


class TestFactoryFunctions:
    """Test factory functions for creating teams"""
    
    @pytest.mark.asyncio
    async def test_create_development_team(self):
        coordinator = DynamicMultiAgentCoordinator()
        await coordinator.start()
        
        team = await create_development_team(coordinator)
        
        assert len(team) == 6  # architect, 2 developers, reviewer, tester, documenter
        
        # Verify agent capabilities
        capabilities_found = set()
        for agent_id in team:
            agent = coordinator.agents[agent_id]
            capabilities_found.update(agent.profile.capabilities)
        
        # Should have various development capabilities
        assert AgentCapability.ARCHITECTURE in capabilities_found
        assert AgentCapability.CODE_GENERATION in capabilities_found
        assert AgentCapability.CODE_REVIEW in capabilities_found
        assert AgentCapability.TESTING in capabilities_found
        assert AgentCapability.DOCUMENTATION in capabilities_found
        
        await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_create_review_pipeline(self):
        coordinator = DynamicMultiAgentCoordinator()
        await coordinator.start()
        
        pipeline = await create_review_pipeline(coordinator)
        
        assert len(pipeline) == 3  # reviewer, tester, security reviewer
        
        # Verify capabilities
        capabilities_found = set()
        for agent_id in pipeline:
            agent = coordinator.agents[agent_id]
            capabilities_found.update(agent.profile.capabilities)
        
        assert AgentCapability.CODE_REVIEW in capabilities_found
        assert AgentCapability.TESTING in capabilities_found
        assert AgentCapability.SECURITY in capabilities_found
        
        await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_create_research_team(self):
        coordinator = DynamicMultiAgentCoordinator()
        await coordinator.start()
        
        team = await create_research_team(coordinator)
        
        assert len(team) == 3  # research lead + 2 analysts
        
        # Verify capabilities
        capabilities_found = set()
        for agent_id in team:
            agent = coordinator.agents[agent_id]
            capabilities_found.update(agent.profile.capabilities)
        
        assert AgentCapability.RESEARCH in capabilities_found
        assert AgentCapability.DATA_ANALYSIS in capabilities_found
        assert AgentCapability.PLANNING in capabilities_found
        assert AgentCapability.COORDINATION in capabilities_found
        
        await coordinator.stop()


@pytest.mark.asyncio
async def test_integration_task_flow():
    """Integration test for complete task flow"""
    coordinator = DynamicMultiAgentCoordinator()
    await coordinator.start()
    
    # Track events
    events = []
    
    def track_event(topic):
        def handler(data):
            events.append((topic, data))
        return handler
    
    # Subscribe to events
    coordinator.message_bus.subscribe("agent_spawned", track_event("agent_spawned"))
    coordinator.message_bus.subscribe("task_submitted", track_event("task_submitted"))
    coordinator.message_bus.subscribe("task_assigned", track_event("task_assigned"))
    
    # Spawn an agent
    agent_id = await coordinator.spawn_agent(profile="developer")
    
    # Submit a task
    task_id = await coordinator.create_and_submit_task(
        description="Test integration task",
        required_capabilities={AgentCapability.CODE_GENERATION},
        priority=8
    )
    
    # Give time for task assignment
    await asyncio.sleep(0.5)
    
    # Verify events occurred
    event_types = [e[0] for e in events]
    assert "agent_spawned" in event_types
    assert "task_submitted" in event_types
    assert "task_assigned" in event_types
    
    # Verify task was assigned to our agent
    task = coordinator.pending_tasks.get(task_id)
    if task:
        assert task.assigned_agent == agent_id
    
    await coordinator.stop()