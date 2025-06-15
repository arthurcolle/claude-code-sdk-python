"""
Main application for the Complex Agent System.
"""

import asyncio
import logging
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from agent_system.config import config, AgentRole
from agent_system.core.message import MessageQueue, Message, MessageType
from agent_system.core.task import Task, TaskType, TaskStatus
from agent_system.core.base_agent import BaseAgent
from agent_system.agents import (
    OrchestratorAgent,
    ResearcherAgent,
    CoderAgent,
    ValidatorAgent,
    ExecutorAgent
)


# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentSystem:
    """Main agent system coordinator."""
    
    def __init__(self):
        self.message_queue = MessageQueue()
        self.agents: Dict[str, BaseAgent] = {}
        self.orchestrator: Optional[OrchestratorAgent] = None
        self._running = False
        
    async def initialize(self):
        """Initialize the agent system."""
        logger.info("Initializing Agent System")
        
        # Create orchestrator
        self.orchestrator = OrchestratorAgent(
            name="MainOrchestrator",
            message_queue=self.message_queue
        )
        self.agents[self.orchestrator.id] = self.orchestrator
        
        # Create specialized agents
        agents_to_create = [
            (ResearcherAgent, "Researcher1", 2),
            (CoderAgent, "Coder1", 2),
            (ValidatorAgent, "Validator1", 1),
            (ExecutorAgent, "Executor1", 3)
        ]
        
        for agent_class, base_name, count in agents_to_create:
            for i in range(count):
                agent = agent_class(
                    name=f"{base_name}_{i}",
                    message_queue=self.message_queue
                )
                self.agents[agent.id] = agent
                
                # Register with orchestrator
                await self.orchestrator.register_agent({
                    "agent_id": agent.id,
                    "name": agent.name,
                    "role": agent.role.value,
                    "capabilities": [cap.value for cap in agent.capabilities],
                    "state": agent.state.value
                })
        
        logger.info(f"Initialized {len(self.agents)} agents")
    
    async def start(self):
        """Start the agent system."""
        self._running = True
        
        # Start all agents
        tasks = []
        for agent in self.agents.values():
            tasks.append(asyncio.create_task(agent.start()))
        
        logger.info("Agent System started")
        
        # Wait for all agents to run
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in agent system: {e}")
    
    async def stop(self):
        """Stop the agent system."""
        self._running = False
        
        # Stop all agents
        for agent in self.agents.values():
            await agent.stop()
        
        logger.info("Agent System stopped")
    
    async def submit_task(self, task: Task) -> str:
        """Submit a task to the system."""
        # Send to orchestrator
        await self.orchestrator.assign_task(task)
        return task.id
    
    def get_system_status(self) -> Dict:
        """Get current system status."""
        return {
            "running": self._running,
            "agent_count": len(self.agents),
            "agents": [
                {
                    "id": agent.id,
                    "name": agent.name,
                    "role": agent.role.value,
                    "state": agent.state.value,
                    "metrics": agent.metrics.model_dump()
                }
                for agent in self.agents.values()
            ]
        }


# Create FastAPI application
app = FastAPI(
    title="Complex Agent System",
    version="1.0.0",
    description="A sophisticated multi-agent system integrated with tool registry",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent system instance
agent_system = AgentSystem()


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await agent_system.initialize()
    asyncio.create_task(agent_system.start())
    logger.info("Agent System API started")
    
    yield
    
    # Shutdown
    await agent_system.stop()
    logger.info("Agent System API stopped")


# API Models
class TaskRequest(BaseModel):
    """Request model for creating tasks."""
    name: str
    description: str
    type: TaskType
    priority: int = 5
    input_data: Dict = {}
    dependencies: List[str] = []
    timeout: Optional[int] = None


class TaskResponse(BaseModel):
    """Response model for task creation."""
    task_id: str
    message: str


class SystemStatus(BaseModel):
    """System status response model."""
    status: str
    agent_count: int
    agents: List[Dict]


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Complex Agent System",
        "version": "1.0.0",
        "status": "running",
        "tool_registry_url": config.TOOL_REGISTRY_URL
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent_system": agent_system._running,
        "agent_count": len(agent_system.agents)
    }


@app.post("/tasks", response_model=TaskResponse)
async def create_task(request: TaskRequest):
    """Create a new task."""
    try:
        task = Task(
            name=request.name,
            description=request.description,
            type=request.type,
            priority=request.priority,
            input_data=request.input_data,
            dependencies=request.dependencies,
            timeout=request.timeout,
            created_by="api"
        )
        
        task_id = await agent_system.submit_task(task)
        
        return TaskResponse(
            task_id=task_id,
            message=f"Task created and submitted to orchestrator"
        )
        
    except Exception as e:
        logger.error(f"Failed to create task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Get system status."""
    status = agent_system.get_system_status()
    return SystemStatus(
        status="running" if status["running"] else "stopped",
        agent_count=status["agent_count"],
        agents=status["agents"]
    )


@app.get("/agents")
async def get_agents():
    """Get list of all agents."""
    return {
        "agents": [
            {
                "id": agent.id,
                "name": agent.name,
                "role": agent.role.value,
                "capabilities": [cap.value for cap in agent.capabilities],
                "state": agent.state.value
            }
            for agent in agent_system.agents.values()
        ]
    }


@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get specific agent details."""
    agent = agent_system.agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return {
        "id": agent.id,
        "name": agent.name,
        "role": agent.role.value,
        "capabilities": [cap.value for cap in agent.capabilities],
        "state": agent.state.value,
        "metrics": agent.metrics.model_dump(),
        "knowledge_base": agent.knowledge_base
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    
    try:
        while True:
            # Send system status periodically
            status = agent_system.get_system_status()
            await websocket.send_json({
                "type": "status_update",
                "data": status
            })
            
            await asyncio.sleep(5)  # Send updates every 5 seconds
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


# Example usage endpoints
@app.post("/examples/research-and-create-tool")
async def example_research_and_create_tool(topic: str, tool_type: str = "python"):
    """Example: Research a topic and create a tool."""
    task = Task(
        name=f"Research and create tool for {topic}",
        description=f"Research {topic} and create a {tool_type} tool based on findings",
        type=TaskType.COMPOSITE,
        priority=7,
        input_data={
            "topic": topic,
            "tool_type": tool_type
        },
        created_by="api"
    )
    
    task_id = await agent_system.submit_task(task)
    
    return {
        "task_id": task_id,
        "message": "Composite task created for research and tool creation"
    }


@app.post("/examples/validate-pending-tools")
async def example_validate_pending_tools():
    """Example: Validate all pending tools."""
    task = Task(
        name="Validate all pending tools",
        description="Find and validate all tools pending approval",
        type=TaskType.TOOL_VALIDATION,
        priority=8,
        input_data={
            "scope": "all_pending",
            "auto_update": True
        },
        created_by="api"
    )
    
    task_id = await agent_system.submit_task(task)
    
    return {
        "task_id": task_id,
        "message": "Validation task created for pending tools"
    }


@app.post("/examples/execute-workflow")
async def example_execute_workflow(workflow_name: str):
    """Example: Execute a predefined workflow."""
    # Example workflow definition
    workflow = {
        "name": workflow_name,
        "steps": [
            {
                "name": "Search for data",
                "type": "tool",
                "tool_name": "web_search",
                "input": {"query": "{{topic}}"}
            },
            {
                "name": "Analyze results",
                "type": "tool",
                "tool_name": "data_analyzer",
                "input": {"data": "{{step_0_output}}"}
            },
            {
                "name": "Generate report",
                "type": "tool",
                "tool_name": "report_generator",
                "input": {
                    "analysis": "{{step_1_output}}",
                    "format": "markdown"
                }
            }
        ]
    }
    
    task = Task(
        name=f"Execute workflow: {workflow_name}",
        description=f"Execute the {workflow_name} workflow",
        type=TaskType.TOOL_EXECUTION,
        priority=6,
        input_data={
            "execution_type": "workflow",
            "workflow": workflow,
            "context": {"topic": workflow_name}
        },
        created_by="api"
    )
    
    task_id = await agent_system.submit_task(task)
    
    return {
        "task_id": task_id,
        "message": f"Workflow execution task created for {workflow_name}"
    }


def main():
    """Run the application."""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.AGENT_SYSTEM_PORT,
        reload=False
    )


if __name__ == "__main__":
    main()