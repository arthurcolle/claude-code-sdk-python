"""
Orchestrator Agent - Manages and coordinates other agents in the system.
"""

import asyncio
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
import json

from ..core.base_agent import BaseAgent, AgentState
from ..core.task import Task, TaskResult, TaskType, TaskStatus, TaskQueue
from ..core.message import Message, MessageType, MessagePriority
from ..config import AgentRole, AgentCapability, config


class OrchestratorAgent(BaseAgent):
    """Agent responsible for orchestrating and coordinating other agents."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.role = AgentRole.ORCHESTRATOR
        self.capabilities = {
            AgentCapability.ORCHESTRATION,
            AgentCapability.MONITORING,
            AgentCapability.TOOL_VALIDATION
        }
        
        # Agent management
        self.registered_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_workload: Dict[str, int] = {}
        self.agent_specialties: Dict[AgentRole, List[str]] = {}
        
        # Task management
        self.task_queue = TaskQueue()
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        
        # System state
        self.system_metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "active_agents": 0,
            "system_health": "healthy"
        }
    
    async def initialize(self):
        """Initialize the orchestrator."""
        self.logger.info("Initializing Orchestrator Agent")
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_agents())
        asyncio.create_task(self._task_scheduler())
        asyncio.create_task(self._health_monitor())
    
    async def process_task(self, task: Task) -> TaskResult:
        """Process orchestration tasks."""
        self.logger.info(f"Processing orchestration task: {task.name}")
        
        try:
            if task.type == TaskType.COMPOSITE:
                return await self._handle_composite_task(task)
            elif task.type == TaskType.MONITORING:
                return await self._handle_monitoring_task(task)
            else:
                # Delegate to appropriate agent
                return await self._delegate_task(task)
                
        except Exception as e:
            self.logger.error(f"Error processing task {task.id}: {e}")
            return TaskResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def register_agent(self, agent_info: Dict[str, Any]):
        """Register a new agent in the system."""
        agent_id = agent_info["agent_id"]
        self.registered_agents[agent_id] = agent_info
        self.agent_workload[agent_id] = 0
        
        # Track agents by role
        role = AgentRole(agent_info["role"])
        if role not in self.agent_specialties:
            self.agent_specialties[role] = []
        self.agent_specialties[role].append(agent_id)
        
        self.logger.info(f"Registered agent: {agent_info['name']} ({role.value})")
        
        # Send welcome message
        await self.send_message(
            Message(
                type=MessageType.STATUS_UPDATE,
                sender_id=self.id,
                recipient_id=agent_id,
                payload={
                    "status": "registered",
                    "message": "Welcome to the agent system"
                }
            )
        )
    
    async def _handle_composite_task(self, task: Task) -> TaskResult:
        """Break down and coordinate composite tasks."""
        subtasks = self._decompose_task(task)
        self.logger.info(f"Decomposed task {task.id} into {len(subtasks)} subtasks")
        
        # Add subtasks to queue
        for subtask in subtasks:
            self.task_queue.add(subtask)
        
        # Wait for all subtasks to complete
        subtask_results = {}
        completed_count = 0
        
        while completed_count < len(subtasks):
            await asyncio.sleep(1)  # Check every second
            
            for subtask in subtasks:
                if subtask.id in subtask_results:
                    continue
                    
                if subtask.status == TaskStatus.COMPLETED:
                    subtask_results[subtask.id] = subtask.output_data
                    completed_count += 1
                elif subtask.status == TaskStatus.FAILED:
                    # Handle failure
                    if subtask.can_retry():
                        self.logger.warning(f"Retrying failed subtask {subtask.id}")
                        subtask.retry()
                        self.task_queue.add(subtask)
                    else:
                        return TaskResult(
                            success=False,
                            output=None,
                            error=f"Subtask {subtask.id} failed: {subtask.output_data.error}"
                        )
            
            # Check for timeout
            if task.is_timeout():
                return TaskResult(
                    success=False,
                    output=None,
                    error="Composite task timed out"
                )
        
        # Aggregate results
        aggregated_output = self._aggregate_results(subtask_results, task)
        
        return TaskResult(
            success=True,
            output=aggregated_output,
            metadata={
                "subtasks_count": len(subtasks),
                "subtask_ids": [st.id for st in subtasks]
            }
        )
    
    async def _handle_monitoring_task(self, task: Task) -> TaskResult:
        """Handle system monitoring tasks."""
        monitoring_type = task.input_data.get("type", "system")
        
        if monitoring_type == "system":
            report = await self._generate_system_report()
        elif monitoring_type == "agents":
            report = await self._generate_agent_report()
        elif monitoring_type == "tasks":
            report = await self._generate_task_report()
        else:
            report = {"error": f"Unknown monitoring type: {monitoring_type}"}
        
        return TaskResult(
            success=True,
            output=report,
            metadata={"monitoring_type": monitoring_type}
        )
    
    async def _delegate_task(self, task: Task) -> TaskResult:
        """Delegate a task to the most appropriate agent."""
        # Find best agent for the task
        best_agent = await self._find_best_agent(task)
        
        if not best_agent:
            return TaskResult(
                success=False,
                output=None,
                error="No suitable agent found for task"
            )
        
        # Assign task
        self.task_assignments[task.id] = best_agent
        self.agent_workload[best_agent] += 1
        
        # Send task to agent
        await self.send_message(
            Message(
                type=MessageType.TASK_ASSIGNMENT,
                sender_id=self.id,
                recipient_id=best_agent,
                priority=MessagePriority.HIGH,
                payload={"task": task.dict()}
            )
        )
        
        # Wait for result
        timeout = task.timeout or config.TASK_TIMEOUT
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout:
            # Check for task completion
            if task.status == TaskStatus.COMPLETED:
                self.agent_workload[best_agent] -= 1
                return task.output_data
            elif task.status == TaskStatus.FAILED:
                self.agent_workload[best_agent] -= 1
                return task.output_data
            
            await asyncio.sleep(0.5)
        
        # Timeout
        return TaskResult(
            success=False,
            output=None,
            error="Task delegation timed out"
        )
    
    async def _find_best_agent(self, task: Task) -> Optional[str]:
        """Find the best agent for a given task."""
        # Determine required role based on task type
        required_role = self._get_required_role(task.type)
        
        # Get agents with the required role
        eligible_agents = self.agent_specialties.get(required_role, [])
        
        if not eligible_agents:
            self.logger.warning(f"No agents found with role: {required_role}")
            return None
        
        # Filter by agent state and workload
        available_agents = []
        for agent_id in eligible_agents:
            agent_info = self.registered_agents.get(agent_id)
            if not agent_info:
                continue
                
            # Check if agent is healthy
            if agent_info.get("state") in [AgentState.IDLE, AgentState.BUSY]:
                workload = self.agent_workload.get(agent_id, 0)
                if workload < config.MAX_CONCURRENT_TASKS:
                    available_agents.append((agent_id, workload))
        
        if not available_agents:
            self.logger.warning("All eligible agents are at capacity")
            return None
        
        # Sort by workload (ascending) and return agent with lowest workload
        available_agents.sort(key=lambda x: x[1])
        return available_agents[0][0]
    
    def _get_required_role(self, task_type: TaskType) -> AgentRole:
        """Map task type to required agent role."""
        role_mapping = {
            TaskType.WEB_SEARCH: AgentRole.RESEARCHER,
            TaskType.DATA_ANALYSIS: AgentRole.RESEARCHER,
            TaskType.CODE_GENERATION: AgentRole.CODER,
            TaskType.TOOL_CREATION: AgentRole.CODER,
            TaskType.TOOL_VALIDATION: AgentRole.VALIDATOR,
            TaskType.TOOL_EXECUTION: AgentRole.EXECUTOR,
            TaskType.MONITORING: AgentRole.MONITOR
        }
        return role_mapping.get(task_type, AgentRole.EXECUTOR)
    
    def _decompose_task(self, task: Task) -> List[Task]:
        """Decompose a composite task into subtasks."""
        subtasks = []
        
        # Example decomposition logic
        if "research and create tool" in task.description.lower():
            # Create research subtask
            research_task = Task(
                type=TaskType.WEB_SEARCH,
                name=f"Research for {task.name}",
                description=f"Research requirements for: {task.description}",
                created_by=self.id,
                parent_task_id=task.id,
                priority=task.priority,
                input_data={"query": task.input_data.get("topic", task.name)}
            )
            subtasks.append(research_task)
            
            # Create tool creation subtask (depends on research)
            tool_task = Task(
                type=TaskType.TOOL_CREATION,
                name=f"Create tool for {task.name}",
                description=f"Create tool based on research",
                created_by=self.id,
                parent_task_id=task.id,
                priority=task.priority,
                dependencies=[research_task.id],
                input_data={"type": "python"}
            )
            subtasks.append(tool_task)
            
        elif "analyze and report" in task.description.lower():
            # Create analysis subtask
            analysis_task = Task(
                type=TaskType.DATA_ANALYSIS,
                name=f"Analyze {task.name}",
                description=f"Analyze data for: {task.description}",
                created_by=self.id,
                parent_task_id=task.id,
                priority=task.priority,
                input_data=task.input_data
            )
            subtasks.append(analysis_task)
            
            # Create monitoring subtask
            monitor_task = Task(
                type=TaskType.MONITORING,
                name=f"Monitor {task.name}",
                description=f"Generate report for analysis",
                created_by=self.id,
                parent_task_id=task.id,
                priority=task.priority,
                dependencies=[analysis_task.id],
                input_data={"type": "tasks"}
            )
            subtasks.append(monitor_task)
        
        else:
            # Default: create a single subtask
            subtask = Task(
                type=TaskType.TOOL_EXECUTION,
                name=task.name,
                description=task.description,
                created_by=self.id,
                parent_task_id=task.id,
                priority=task.priority,
                input_data=task.input_data
            )
            subtasks.append(subtask)
        
        return subtasks
    
    def _aggregate_results(self, subtask_results: Dict[str, TaskResult], parent_task: Task) -> Dict[str, Any]:
        """Aggregate results from subtasks."""
        aggregated = {
            "task_id": parent_task.id,
            "task_name": parent_task.name,
            "subtask_count": len(subtask_results),
            "results": []
        }
        
        for task_id, result in subtask_results.items():
            aggregated["results"].append({
                "subtask_id": task_id,
                "success": result.success,
                "output": result.output,
                "error": result.error
            })
        
        # Summarize
        success_count = sum(1 for r in aggregated["results"] if r["success"])
        aggregated["summary"] = {
            "total_subtasks": len(subtask_results),
            "successful": success_count,
            "failed": len(subtask_results) - success_count
        }
        
        return aggregated
    
    async def _monitor_agents(self):
        """Monitor agent health and status."""
        while self._running:
            try:
                # Request status from all agents
                await self.send_message(
                    Message(
                        type=MessageType.QUERY,
                        sender_id=self.id,
                        payload={"query": "status"}
                    )
                )
                
                # Update system metrics
                self.system_metrics["active_agents"] = len([
                    a for a in self.registered_agents.values()
                    if a.get("state") != AgentState.SHUTDOWN
                ])
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in agent monitoring: {e}")
    
    async def _task_scheduler(self):
        """Schedule tasks from the queue to available agents."""
        while self._running:
            try:
                # Get next ready task
                task = self.task_queue.get_next_ready_task()
                if task:
                    # Process the task
                    asyncio.create_task(self._execute_task(task))
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in task scheduler: {e}")
    
    async def _health_monitor(self):
        """Monitor overall system health."""
        while self._running:
            try:
                # Check system health metrics
                health_issues = []
                
                # Check agent availability
                if self.system_metrics["active_agents"] == 0:
                    health_issues.append("No active agents")
                
                # Check task backlog
                pending_tasks = len(self.task_queue.get_tasks_by_status(TaskStatus.PENDING))
                if pending_tasks > 50:
                    health_issues.append(f"High task backlog: {pending_tasks}")
                
                # Check failure rate
                if self.system_metrics["total_tasks"] > 0:
                    failure_rate = self.system_metrics["failed_tasks"] / self.system_metrics["total_tasks"]
                    if failure_rate > 0.2:
                        health_issues.append(f"High failure rate: {failure_rate:.2%}")
                
                # Update health status
                if not health_issues:
                    self.system_metrics["system_health"] = "healthy"
                elif len(health_issues) == 1:
                    self.system_metrics["system_health"] = "degraded"
                else:
                    self.system_metrics["system_health"] = "unhealthy"
                
                self.logger.info(f"System health: {self.system_metrics['system_health']}")
                if health_issues:
                    self.logger.warning(f"Health issues: {', '.join(health_issues)}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
    
    async def _generate_system_report(self) -> Dict[str, Any]:
        """Generate a comprehensive system report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": self.system_metrics,
            "agent_count": len(self.registered_agents),
            "agents_by_role": {
                role.value: len(agents) 
                for role, agents in self.agent_specialties.items()
            },
            "task_queue_size": len(self.task_queue.get_tasks_by_status(TaskStatus.PENDING)),
            "active_tasks": len(self.task_assignments),
            "system_uptime": (datetime.now() - self._start_time).total_seconds()
        }
    
    async def _generate_agent_report(self) -> Dict[str, Any]:
        """Generate a report on all agents."""
        agent_reports = []
        
        for agent_id, agent_info in self.registered_agents.items():
            agent_reports.append({
                "id": agent_id,
                "name": agent_info.get("name"),
                "role": agent_info.get("role"),
                "state": agent_info.get("state"),
                "workload": self.agent_workload.get(agent_id, 0),
                "metrics": agent_info.get("metrics", {})
            })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_agents": len(agent_reports),
            "agents": agent_reports
        }
    
    async def _generate_task_report(self) -> Dict[str, Any]:
        """Generate a report on tasks."""
        all_tasks = {
            "pending": self.task_queue.get_tasks_by_status(TaskStatus.PENDING),
            "assigned": self.task_queue.get_tasks_by_status(TaskStatus.ASSIGNED),
            "in_progress": self.task_queue.get_tasks_by_status(TaskStatus.IN_PROGRESS),
            "completed": self.task_queue.get_tasks_by_status(TaskStatus.COMPLETED),
            "failed": self.task_queue.get_tasks_by_status(TaskStatus.FAILED)
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "task_counts": {status: len(tasks) for status, tasks in all_tasks.items()},
            "total_tasks": sum(len(tasks) for tasks in all_tasks.values()),
            "assignments": self.task_assignments
        }
    
    async def _handle_message(self, message: Message):
        """Handle incoming messages with orchestrator-specific logic."""
        await super()._handle_message(message)
        
        if message.type == MessageType.HEARTBEAT:
            # Update agent information
            agent_id = message.sender_id
            if agent_id in self.registered_agents:
                self.registered_agents[agent_id].update({
                    "state": message.payload.get("state"),
                    "metrics": message.payload.get("metrics"),
                    "last_heartbeat": datetime.now()
                })
        
        elif message.type == MessageType.TASK_RESULT:
            # Update task tracking
            task_id = message.payload.get("task_id")
            if task_id in self.task_assignments:
                del self.task_assignments[task_id]
                self.system_metrics["completed_tasks"] += 1
        
        elif message.type == MessageType.ERROR:
            # Handle errors
            task_id = message.payload.get("task_id")
            if task_id:
                self.system_metrics["failed_tasks"] += 1