"""Advanced multi-agent system examples using Claude SDK."""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Type

import anyio

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    Message,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    query,
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Core agent framework
class AgentCapability(Enum):
    """Capabilities that agents can have."""
    CODE_WRITING = "code_writing"
    CODE_REVIEW = "code_review"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DEBUGGING = "debugging"
    DEPLOYMENT = "deployment"
    DATA_ANALYSIS = "data_analysis"


class AgentRole(Enum):
    """Predefined agent roles."""
    ARCHITECT = "architect"
    DEVELOPER = "developer"
    REVIEWER = "reviewer"
    TESTER = "tester"
    DOCUMENTER = "documenter"
    SECURITY_AUDITOR = "security_auditor"
    DEVOPS = "devops"
    DATA_SCIENTIST = "data_scientist"
    PROJECT_MANAGER = "project_manager"


@dataclass
class AgentMessage:
    """Message passed between agents."""
    sender: str
    recipient: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    attachments: List[str] = field(default_factory=list)  # File paths
    priority: int = 0  # Higher number = higher priority
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())


@dataclass
class AgentTask:
    """Task assigned to an agent."""
    id: str
    description: str
    assigned_to: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    completed_at: Optional[float] = None


class Agent(ABC):
    """Base agent class."""
    
    def __init__(
        self,
        name: str,
        role: AgentRole,
        capabilities: List[AgentCapability],
        system_prompt: Optional[str] = None
    ):
        self.name = name
        self.role = role
        self.capabilities = capabilities
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self.session_id: Optional[str] = None
        self.conversation_history: List[Dict[str, Any]] = []
        
    def _default_system_prompt(self) -> str:
        """Generate default system prompt based on role."""
        prompts = {
            AgentRole.ARCHITECT: "You are a software architect. Focus on system design, patterns, and architecture decisions.",
            AgentRole.DEVELOPER: "You are a software developer. Write clean, efficient, and well-documented code.",
            AgentRole.REVIEWER: "You are a code reviewer. Look for bugs, security issues, and suggest improvements.",
            AgentRole.TESTER: "You are a QA engineer. Write comprehensive tests and find edge cases.",
            AgentRole.DOCUMENTER: "You are a technical writer. Create clear, comprehensive documentation.",
            AgentRole.SECURITY_AUDITOR: "You are a security expert. Find vulnerabilities and suggest security improvements.",
            AgentRole.DEVOPS: "You are a DevOps engineer. Focus on deployment, CI/CD, and infrastructure.",
            AgentRole.DATA_SCIENTIST: "You are a data scientist. Analyze data and create insights.",
            AgentRole.PROJECT_MANAGER: "You are a project manager. Coordinate tasks and track progress.",
        }
        return prompts.get(self.role, "You are a helpful AI assistant.")
        
    async def send_message(self, recipient: 'Agent', content: str, **kwargs):
        """Send a message to another agent."""
        message = AgentMessage(
            sender=self.name,
            recipient=recipient.name,
            content=content,
            **kwargs
        )
        await recipient.message_queue.put(message)
        
    async def receive_message(self) -> AgentMessage:
        """Receive a message from the queue."""
        return await self.message_queue.get()
        
    async def query_claude(self, prompt: str, tools: Optional[List[str]] = None) -> List[Message]:
        """Query Claude with agent-specific configuration."""
        options = ClaudeCodeOptions(
            system_prompt=self.system_prompt,
            resume=self.session_id,
            allowed_tools=tools or [],
            permission_mode="acceptEdits"
        )
        
        messages = []
        async for message in query(prompt=prompt, options=options):
            messages.append(message)
            if isinstance(message, ResultMessage):
                self.session_id = message.session_id
                
        return messages
        
    @abstractmethod
    async def process_task(self, task: AgentTask) -> Any:
        """Process a task assigned to this agent."""
        pass
        
    @abstractmethod
    async def collaborate(self, other_agent: 'Agent', context: Dict[str, Any]) -> Any:
        """Collaborate with another agent."""
        pass


# Specialized agent implementations
class ArchitectAgent(Agent):
    """Agent specialized in system architecture."""
    
    def __init__(self, name: str = "Architect"):
        super().__init__(
            name=name,
            role=AgentRole.ARCHITECT,
            capabilities=[
                AgentCapability.ARCHITECTURE,
                AgentCapability.CODE_REVIEW,
                AgentCapability.DOCUMENTATION
            ]
        )
        
    async def process_task(self, task: AgentTask) -> Any:
        """Design system architecture."""
        prompt = f"""
        Task: {task.description}
        
        Please provide:
        1. High-level architecture design
        2. Component breakdown
        3. Technology recommendations
        4. Integration points
        5. Scalability considerations
        """
        
        messages = await self.query_claude(prompt, tools=["Write"])
        
        # Extract architecture design
        design = ""
        for message in messages:
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        design += block.text + "\n"
                        
        return {"design": design, "task_id": task.id}
        
    async def collaborate(self, other_agent: Agent, context: Dict[str, Any]) -> Any:
        """Collaborate with other agents on architecture decisions."""
        if other_agent.role == AgentRole.DEVELOPER:
            # Provide architecture guidance to developer
            return await self._guide_developer(other_agent, context)
        elif other_agent.role == AgentRole.REVIEWER:
            # Review architecture with reviewer
            return await self._architecture_review(other_agent, context)
        else:
            return await self._general_collaboration(other_agent, context)
            
    async def _guide_developer(self, developer: Agent, context: Dict[str, Any]) -> Any:
        """Provide architecture guidance to developer."""
        guidance = f"""
        Based on the architecture design for {context.get('project', 'the project')}:
        
        Key architectural decisions:
        - {context.get('patterns', 'Use appropriate design patterns')}
        - {context.get('structure', 'Follow the defined component structure')}
        
        Please implement according to these guidelines.
        """
        
        await self.send_message(developer, guidance)
        return {"guidance_sent": True}
        
    async def _architecture_review(self, reviewer: Agent, context: Dict[str, Any]) -> Any:
        """Conduct architecture review."""
        review_request = f"""
        Please review the architecture for:
        - Design patterns appropriateness
        - Scalability concerns
        - Security considerations
        - Integration complexity
        
        Context: {json.dumps(context, indent=2)}
        """
        
        await self.send_message(reviewer, review_request)
        review_response = await self.receive_message()
        
        return {"review": review_response.content}
        
    async def _general_collaboration(self, agent: Agent, context: Dict[str, Any]) -> Any:
        """General collaboration with any agent."""
        await self.send_message(agent, f"Collaboration request: {context}")
        response = await self.receive_message()
        return {"response": response.content}


class DeveloperAgent(Agent):
    """Agent specialized in code development."""
    
    def __init__(self, name: str = "Developer", specialization: str = "python"):
        super().__init__(
            name=name,
            role=AgentRole.DEVELOPER,
            capabilities=[
                AgentCapability.CODE_WRITING,
                AgentCapability.DEBUGGING,
                AgentCapability.TESTING
            ]
        )
        self.specialization = specialization
        
    async def process_task(self, task: AgentTask) -> Any:
        """Implement code based on task requirements."""
        prompt = f"""
        Task: {task.description}
        Specialization: {self.specialization}
        
        Please implement the required functionality with:
        1. Clean, readable code
        2. Proper error handling
        3. Type hints (if applicable)
        4. Basic documentation
        """
        
        tools = ["Read", "Write", "Edit", "Bash"]
        messages = await self.query_claude(prompt, tools=tools)
        
        # Track created files
        created_files = []
        for message in messages:
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock) and block.name == "Write":
                        created_files.append(block.input.get("file_path"))
                        
        return {
            "task_id": task.id,
            "created_files": created_files,
            "status": "completed"
        }
        
    async def collaborate(self, other_agent: Agent, context: Dict[str, Any]) -> Any:
        """Collaborate with other agents on development."""
        if other_agent.role == AgentRole.TESTER:
            return await self._collaborate_with_tester(other_agent, context)
        elif other_agent.role == AgentRole.REVIEWER:
            return await self._prepare_for_review(other_agent, context)
        else:
            return await self._general_dev_collaboration(other_agent, context)
            
    async def _collaborate_with_tester(self, tester: Agent, context: Dict[str, Any]) -> Any:
        """Work with tester to ensure code testability."""
        code_info = f"""
        I've implemented {context.get('feature', 'the feature')} in these files:
        {json.dumps(context.get('files', []), indent=2)}
        
        Key functions to test:
        {json.dumps(context.get('functions', []), indent=2)}
        
        Please create comprehensive tests.
        """
        
        await self.send_message(tester, code_info)
        return {"collaboration": "test_request_sent"}
        
    async def _prepare_for_review(self, reviewer: Agent, context: Dict[str, Any]) -> Any:
        """Prepare code for review."""
        review_package = {
            "files": context.get("files", []),
            "changes": context.get("changes", "New implementation"),
            "testing": context.get("testing", "Unit tests included"),
            "documentation": context.get("documentation", "Inline docs added")
        }
        
        await self.send_message(
            reviewer,
            f"Code ready for review: {json.dumps(review_package, indent=2)}"
        )
        
        return {"status": "submitted_for_review"}
        
    async def _general_dev_collaboration(self, agent: Agent, context: Dict[str, Any]) -> Any:
        """General development collaboration."""
        await self.send_message(
            agent,
            f"Development update: {context.get('update', 'Progress made')}"
        )
        return {"collaboration": "completed"}


class ReviewerAgent(Agent):
    """Agent specialized in code review."""
    
    def __init__(self, name: str = "Reviewer"):
        super().__init__(
            name=name,
            role=AgentRole.REVIEWER,
            capabilities=[
                AgentCapability.CODE_REVIEW,
                AgentCapability.SECURITY,
                AgentCapability.PERFORMANCE
            ]
        )
        self.review_checklist = [
            "Code correctness",
            "Error handling",
            "Security vulnerabilities",
            "Performance issues",
            "Code style and readability",
            "Test coverage",
            "Documentation"
        ]
        
    async def process_task(self, task: AgentTask) -> Any:
        """Review code based on task description."""
        files_to_review = task.metadata.get("files", [])
        
        review_results = []
        for file_path in files_to_review:
            prompt = f"""
            Review the code in {file_path} for:
            {chr(10).join(f'- {item}' for item in self.review_checklist)}
            
            Provide specific feedback and suggestions.
            """
            
            messages = await self.query_claude(prompt, tools=["Read", "Grep"])
            
            review = ""
            for message in messages:
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            review += block.text + "\n"
                            
            review_results.append({
                "file": file_path,
                "review": review,
                "status": "reviewed"
            })
            
        return {
            "task_id": task.id,
            "reviews": review_results,
            "overall_status": "completed"
        }
        
    async def collaborate(self, other_agent: Agent, context: Dict[str, Any]) -> Any:
        """Collaborate with other agents on reviews."""
        if other_agent.role == AgentRole.DEVELOPER:
            return await self._provide_feedback(other_agent, context)
        elif other_agent.role == AgentRole.SECURITY_AUDITOR:
            return await self._security_collaboration(other_agent, context)
        else:
            return {"collaboration": "completed"}
            
    async def _provide_feedback(self, developer: Agent, context: Dict[str, Any]) -> Any:
        """Provide review feedback to developer."""
        feedback = {
            "issues_found": context.get("issues", []),
            "suggestions": context.get("suggestions", []),
            "approval_status": context.get("status", "needs_changes")
        }
        
        await self.send_message(
            developer,
            f"Review feedback: {json.dumps(feedback, indent=2)}"
        )
        
        return {"feedback_sent": True}
        
    async def _security_collaboration(self, auditor: Agent, context: Dict[str, Any]) -> Any:
        """Collaborate on security review."""
        security_concerns = context.get("security_concerns", [])
        
        await self.send_message(
            auditor,
            f"Potential security issues found: {json.dumps(security_concerns, indent=2)}"
        )
        
        audit_response = await self.receive_message()
        return {"security_review": audit_response.content}


class TesterAgent(Agent):
    """Agent specialized in testing."""
    
    def __init__(self, name: str = "Tester"):
        super().__init__(
            name=name,
            role=AgentRole.TESTER,
            capabilities=[
                AgentCapability.TESTING,
                AgentCapability.DEBUGGING
            ]
        )
        
    async def process_task(self, task: AgentTask) -> Any:
        """Create and run tests."""
        code_files = task.metadata.get("files", [])
        test_type = task.metadata.get("test_type", "unit")
        
        prompt = f"""
        Create {test_type} tests for the following files:
        {json.dumps(code_files, indent=2)}
        
        Include:
        1. Happy path tests
        2. Edge cases
        3. Error conditions
        4. Performance tests (if applicable)
        """
        
        tools = ["Read", "Write", "Bash"]
        messages = await self.query_claude(prompt, tools=tools)
        
        test_results = {
            "task_id": task.id,
            "test_files_created": [],
            "test_results": []
        }
        
        for message in messages:
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        if block.name == "Write":
                            test_results["test_files_created"].append(
                                block.input.get("file_path")
                            )
                        elif block.name == "Bash" and "test" in block.input.get("command", ""):
                            test_results["test_results"].append({
                                "command": block.input.get("command"),
                                "status": "executed"
                            })
                            
        return test_results
        
    async def collaborate(self, other_agent: Agent, context: Dict[str, Any]) -> Any:
        """Collaborate with other agents on testing."""
        if other_agent.role == AgentRole.DEVELOPER:
            return await self._test_developer_code(other_agent, context)
        else:
            return {"collaboration": "completed"}
            
    async def _test_developer_code(self, developer: Agent, context: Dict[str, Any]) -> Any:
        """Test code provided by developer."""
        test_report = {
            "tested_files": context.get("files", []),
            "tests_passed": context.get("passed", 0),
            "tests_failed": context.get("failed", 0),
            "coverage": context.get("coverage", "unknown")
        }
        
        await self.send_message(
            developer,
            f"Test report: {json.dumps(test_report, indent=2)}"
        )
        
        return {"test_report_sent": True}


# Multi-agent orchestration
class MultiAgentOrchestrator:
    """Orchestrates multiple agents working together."""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.tasks: List[AgentTask] = []
        self.task_queue: asyncio.Queue[AgentTask] = asyncio.Queue()
        self.completed_tasks: List[AgentTask] = []
        
    def register_agent(self, agent: Agent):
        """Register an agent with the orchestrator."""
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name} ({agent.role.value})")
        
    def create_task(
        self,
        description: str,
        assigned_to: Optional[str] = None,
        dependencies: Optional[List[str]] = None
    ) -> AgentTask:
        """Create a new task."""
        task = AgentTask(
            id=f"task_{len(self.tasks)}",
            description=description,
            assigned_to=assigned_to,
            dependencies=dependencies or []
        )
        self.tasks.append(task)
        return task
        
    async def assign_task(self, task: AgentTask, agent_name: str):
        """Assign a task to an agent."""
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not found")
            
        task.assigned_to = agent_name
        task.status = "assigned"
        await self.task_queue.put(task)
        logger.info(f"Assigned task {task.id} to {agent_name}")
        
    async def execute_tasks(self):
        """Execute all tasks in parallel."""
        async def agent_worker(agent_name: str):
            agent = self.agents[agent_name]
            while True:
                try:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                    if task.assigned_to == agent_name:
                        logger.info(f"{agent_name} processing task {task.id}")
                        task.status = "in_progress"
                        
                        try:
                            result = await agent.process_task(task)
                            task.result = result
                            task.status = "completed"
                            task.completed_at = asyncio.get_event_loop().time()
                            self.completed_tasks.append(task)
                            logger.info(f"{agent_name} completed task {task.id}")
                        except Exception as e:
                            task.status = "failed"
                            task.result = {"error": str(e)}
                            logger.error(f"{agent_name} failed task {task.id}: {e}")
                    else:
                        # Not for this agent, put it back
                        await self.task_queue.put(task)
                except asyncio.TimeoutError:
                    # No tasks available, check if we should exit
                    if self.task_queue.empty() and all(
                        t.status in ["completed", "failed"] for t in self.tasks
                    ):
                        break
                        
        # Start workers for all agents
        workers = [
            asyncio.create_task(agent_worker(agent_name))
            for agent_name in self.agents
        ]
        
        await asyncio.gather(*workers)
        
    async def facilitate_collaboration(
        self,
        agent1_name: str,
        agent2_name: str,
        context: Dict[str, Any]
    ) -> Any:
        """Facilitate collaboration between two agents."""
        agent1 = self.agents.get(agent1_name)
        agent2 = self.agents.get(agent2_name)
        
        if not agent1 or not agent2:
            raise ValueError("One or both agents not found")
            
        logger.info(f"Facilitating collaboration: {agent1_name} <-> {agent2_name}")
        
        # Agent1 initiates collaboration
        result1 = await agent1.collaborate(agent2, context)
        
        # Agent2 responds
        result2 = await agent2.collaborate(agent1, {**context, "response_to": result1})
        
        return {
            "collaboration": f"{agent1_name} <-> {agent2_name}",
            "results": [result1, result2]
        }
        
    def get_task_status(self) -> Dict[str, Any]:
        """Get status of all tasks."""
        status = {
            "total": len(self.tasks),
            "pending": sum(1 for t in self.tasks if t.status == "pending"),
            "in_progress": sum(1 for t in self.tasks if t.status == "in_progress"),
            "completed": sum(1 for t in self.tasks if t.status == "completed"),
            "failed": sum(1 for t in self.tasks if t.status == "failed")
        }
        return status


# Example: Software development team
async def software_development_team_example():
    """Example of a complete software development team."""
    print("\n=== Software Development Team Example ===")
    
    # Create orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    # Create specialized agents
    architect = ArchitectAgent("Alice_Architect")
    backend_dev = DeveloperAgent("Bob_Backend", specialization="python")
    frontend_dev = DeveloperAgent("Carol_Frontend", specialization="javascript")
    reviewer = ReviewerAgent("Dave_Reviewer")
    tester = TesterAgent("Eve_Tester")
    
    # Register agents
    for agent in [architect, backend_dev, frontend_dev, reviewer, tester]:
        orchestrator.register_agent(agent)
        
    # Create project tasks
    tasks = [
        orchestrator.create_task(
            "Design architecture for a task management API",
            assigned_to="Alice_Architect"
        ),
        orchestrator.create_task(
            "Implement REST API endpoints for tasks",
            assigned_to="Bob_Backend",
            dependencies=["task_0"]
        ),
        orchestrator.create_task(
            "Create React frontend for task management",
            assigned_to="Carol_Frontend",
            dependencies=["task_0"]
        ),
        orchestrator.create_task(
            "Review backend implementation",
            assigned_to="Dave_Reviewer",
            dependencies=["task_1"]
        ),
        orchestrator.create_task(
            "Write tests for API endpoints",
            assigned_to="Eve_Tester",
            dependencies=["task_1"]
        )
    ]
    
    # Assign all tasks
    for task in tasks:
        if task.assigned_to:
            await orchestrator.assign_task(task, task.assigned_to)
            
    # Execute tasks
    print("\nExecuting tasks...")
    await orchestrator.execute_tasks()
    
    # Show results
    print("\nTask Status:")
    print(json.dumps(orchestrator.get_task_status(), indent=2))
    
    # Facilitate collaboration
    print("\nFacilitating collaborations...")
    
    # Backend dev collaborates with tester
    collab1 = await orchestrator.facilitate_collaboration(
        "Bob_Backend",
        "Eve_Tester",
        {
            "feature": "Task API",
            "files": ["api/tasks.py", "api/models.py"],
            "functions": ["create_task", "update_task", "delete_task"]
        }
    )
    print(f"Collaboration 1: {json.dumps(collab1, indent=2)}")


# Example: Code review pipeline
async def code_review_pipeline_example():
    """Example of an automated code review pipeline."""
    print("\n=== Code Review Pipeline Example ===")
    
    orchestrator = MultiAgentOrchestrator()
    
    # Create review team
    lead_reviewer = ReviewerAgent("Lead_Reviewer")
    security_auditor = Agent(
        name="Security_Auditor",
        role=AgentRole.SECURITY_AUDITOR,
        capabilities=[AgentCapability.SECURITY, AgentCapability.CODE_REVIEW]
    )
    performance_expert = Agent(
        name="Performance_Expert",
        role=AgentRole.DEVELOPER,
        capabilities=[AgentCapability.PERFORMANCE, AgentCapability.CODE_REVIEW]
    )
    
    # Register agents
    for agent in [lead_reviewer, security_auditor, performance_expert]:
        orchestrator.register_agent(agent)
        
    # Create review tasks
    files_to_review = ["src/auth.py", "src/database.py", "src/api.py"]
    
    for file_path in files_to_review:
        # General review
        orchestrator.create_task(
            f"Review {file_path} for code quality",
            assigned_to="Lead_Reviewer"
        )
        
        # Security review
        orchestrator.create_task(
            f"Security audit of {file_path}",
            assigned_to="Security_Auditor"
        )
        
        # Performance review
        orchestrator.create_task(
            f"Performance review of {file_path}",
            assigned_to="Performance_Expert"
        )
        
    # Execute reviews
    print("\nExecuting reviews...")
    await orchestrator.execute_tasks()
    
    print("\nReview pipeline completed!")


# Example: Microservices development team
async def microservices_team_example():
    """Example of a team developing microservices."""
    print("\n=== Microservices Development Team Example ===")
    
    orchestrator = MultiAgentOrchestrator()
    
    # Create specialized microservice developers
    auth_service_dev = DeveloperAgent("Auth_Service_Dev", specialization="python")
    user_service_dev = DeveloperAgent("User_Service_Dev", specialization="python")
    api_gateway_dev = DeveloperAgent("API_Gateway_Dev", specialization="nodejs")
    
    # Create infrastructure agents
    devops = Agent(
        name="DevOps_Engineer",
        role=AgentRole.DEVOPS,
        capabilities=[AgentCapability.DEPLOYMENT, AgentCapability.ARCHITECTURE]
    )
    
    # Register all agents
    for agent in [auth_service_dev, user_service_dev, api_gateway_dev, devops]:
        orchestrator.register_agent(agent)
        
    # Create microservices tasks
    services = [
        ("Authentication Service", "Auth_Service_Dev"),
        ("User Management Service", "User_Service_Dev"),
        ("API Gateway", "API_Gateway_Dev"),
    ]
    
    for service_name, developer in services:
        # Development task
        dev_task = orchestrator.create_task(
            f"Develop {service_name} with REST API",
            assigned_to=developer
        )
        
        # Deployment task
        orchestrator.create_task(
            f"Create Docker container for {service_name}",
            assigned_to="DevOps_Engineer",
            dependencies=[dev_task.id]
        )
        
    # Execute development
    print("\nDeveloping microservices...")
    await orchestrator.execute_tasks()
    
    print("\nMicroservices development completed!")


# Example: AI-powered debugging team
class DebuggerAgent(Agent):
    """Agent specialized in debugging."""
    
    def __init__(self, name: str = "Debugger"):
        super().__init__(
            name=name,
            role=AgentRole.DEVELOPER,
            capabilities=[AgentCapability.DEBUGGING, AgentCapability.TESTING]
        )
        
    async def process_task(self, task: AgentTask) -> Any:
        """Debug issues in code."""
        error_info = task.metadata.get("error", {})
        files = task.metadata.get("files", [])
        
        prompt = f"""
        Debug the following issue:
        Error: {error_info.get('message', 'Unknown error')}
        Stack trace: {error_info.get('stack_trace', 'Not available')}
        
        Files involved: {json.dumps(files, indent=2)}
        
        Please:
        1. Identify the root cause
        2. Suggest a fix
        3. Implement the fix if possible
        """
        
        tools = ["Read", "Edit", "Bash", "Grep"]
        messages = await self.query_claude(prompt, tools=tools)
        
        debug_result = {
            "task_id": task.id,
            "root_cause": "",
            "fix_applied": False,
            "suggestions": []
        }
        
        for message in messages:
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        debug_result["root_cause"] += block.text
                    elif isinstance(block, ToolUseBlock) and block.name == "Edit":
                        debug_result["fix_applied"] = True
                        
        return debug_result
        
    async def collaborate(self, other_agent: Agent, context: Dict[str, Any]) -> Any:
        """Collaborate on debugging."""
        if other_agent.role == AgentRole.TESTER:
            # Work with tester to reproduce issue
            return await self._reproduce_with_tester(other_agent, context)
        else:
            return {"collaboration": "completed"}
            
    async def _reproduce_with_tester(self, tester: Agent, context: Dict[str, Any]) -> Any:
        """Work with tester to reproduce issue."""
        reproduction_request = {
            "error": context.get("error"),
            "steps": context.get("reproduction_steps", []),
            "environment": context.get("environment", {})
        }
        
        await self.send_message(
            tester,
            f"Please help reproduce: {json.dumps(reproduction_request, indent=2)}"
        )
        
        return {"reproduction_requested": True}


async def debugging_team_example():
    """Example of an AI-powered debugging team."""
    print("\n=== AI-Powered Debugging Team Example ===")
    
    orchestrator = MultiAgentOrchestrator()
    
    # Create debugging team
    debugger = DebuggerAgent("Debug_Master")
    tester = TesterAgent("Test_Expert")
    reviewer = ReviewerAgent("Code_Reviewer")
    
    # Register agents
    for agent in [debugger, tester, reviewer]:
        orchestrator.register_agent(agent)
        
    # Create debugging scenario
    bug_report = {
        "error": {
            "message": "AttributeError: 'NoneType' object has no attribute 'process'",
            "stack_trace": "File 'processor.py', line 42, in handle_request"
        },
        "files": ["processor.py", "utils.py"],
        "reproduction_steps": ["Send POST request to /api/process", "With empty body"]
    }
    
    # Create debugging tasks
    debug_task = orchestrator.create_task(
        "Debug and fix AttributeError in processor",
        assigned_to="Debug_Master"
    )
    debug_task.metadata = bug_report
    
    test_task = orchestrator.create_task(
        "Create test to prevent regression",
        assigned_to="Test_Expert",
        dependencies=[debug_task.id]
    )
    
    review_task = orchestrator.create_task(
        "Review the fix and test",
        assigned_to="Code_Reviewer",
        dependencies=[debug_task.id, test_task.id]
    )
    
    # Assign tasks
    for task in [debug_task, test_task, review_task]:
        if task.assigned_to:
            await orchestrator.assign_task(task, task.assigned_to)
            
    # Execute debugging
    print("\nDebugging in progress...")
    await orchestrator.execute_tasks()
    
    # Facilitate collaboration
    await orchestrator.facilitate_collaboration(
        "Debug_Master",
        "Test_Expert",
        bug_report
    )
    
    print("\nDebugging completed!")


# Main execution
async def main():
    """Run all multi-agent examples."""
    examples = [
        ("Software Development Team", software_development_team_example),
        ("Code Review Pipeline", code_review_pipeline_example),
        ("Microservices Team", microservices_team_example),
        ("AI-Powered Debugging Team", debugging_team_example),
    ]
    
    print("Advanced Multi-Agent System Examples")
    print("====================================")
    
    for name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"\nExample '{name}' failed: {e}")
            logger.error(f"Example failed: {e}", exc_info=True)
            
        # Small delay between examples
        await asyncio.sleep(2)
        
    print("\n\nAll examples completed!")


if __name__ == "__main__":
    anyio.run(main)