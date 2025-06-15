"""
Executor Agent - Specializes in executing tools and managing execution workflows.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from ..core.base_agent import BaseAgent
from ..core.task import Task, TaskResult, TaskType
from ..config import AgentRole, AgentCapability
from ..integrations import get_tool_registry_client


class ExecutorAgent(BaseAgent):
    """Agent specialized in tool execution and workflow management."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.role = AgentRole.EXECUTOR
        self.capabilities = {
            AgentCapability.TOOL_EXECUTION,
            AgentCapability.MONITORING
        }
        self.tool_registry = get_tool_registry_client()
        
        # Execution tracking
        self.execution_history: List[Dict[str, Any]] = []
        self.tool_cache: Dict[str, Dict[str, Any]] = {}
        self.execution_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0
        }
    
    async def initialize(self):
        """Initialize the executor agent."""
        self.logger.info("Initializing Executor Agent")
        
        # Verify tool registry connection
        try:
            health = await self.tool_registry.health_check()
            self.logger.info(f"Tool registry health: {health['status']}")
            
            # Pre-cache frequently used tools
            await self._cache_common_tools()
        except Exception as e:
            self.logger.error(f"Failed to connect to tool registry: {e}")
    
    async def process_task(self, task: Task) -> TaskResult:
        """Process execution tasks."""
        self.logger.info(f"Processing execution task: {task.name}")
        
        try:
            if task.type == TaskType.TOOL_EXECUTION:
                return await self._execute_tool_task(task)
            else:
                # Try to execute as a general task
                return await self._execute_general_task(task)
                
        except Exception as e:
            self.logger.error(f"Error processing task {task.id}: {e}")
            return TaskResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def _execute_tool_task(self, task: Task) -> TaskResult:
        """Execute a specific tool or workflow."""
        execution_type = task.input_data.get("execution_type", "single")
        
        if execution_type == "single":
            return await self._execute_single_tool(task)
        elif execution_type == "sequential":
            return await self._execute_sequential_tools(task)
        elif execution_type == "parallel":
            return await self._execute_parallel_tools(task)
        elif execution_type == "workflow":
            return await self._execute_workflow(task)
        else:
            return TaskResult(
                success=False,
                output=None,
                error=f"Unknown execution type: {execution_type}"
            )
    
    async def _execute_single_tool(self, task: Task) -> TaskResult:
        """Execute a single tool."""
        tool_name = task.input_data.get("tool_name")
        tool_id = task.input_data.get("tool_id")
        tool_input = task.input_data.get("input", {})
        format_type = task.input_data.get("format_type")
        
        self.logger.info(f"Executing tool: {tool_name or tool_id}")
        
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = tool_id or tool_name
            if cache_key in self.tool_cache:
                self.logger.debug(f"Using cached tool info for {cache_key}")
            
            # Execute tool
            result = await self.tool_registry.execute_tool(
                tool_id=tool_id,
                tool_name=tool_name,
                input_data=tool_input,
                format_type=format_type
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update metrics
            self.execution_metrics["total_executions"] += 1
            self.execution_metrics["successful_executions"] += 1
            self.metrics.tools_executed += 1
            self._update_average_execution_time(execution_time)
            
            # Record execution
            self._record_execution({
                "task_id": task.id,
                "tool_name": tool_name,
                "tool_id": tool_id,
                "input": tool_input,
                "output": result,
                "execution_time": execution_time,
                "success": True,
                "timestamp": datetime.now().isoformat()
            })
            
            return TaskResult(
                success=True,
                output=result,
                metadata={
                    "execution_time": execution_time,
                    "tool_identifier": cache_key
                }
            )
            
        except Exception as e:
            self.logger.error(f"Tool execution failed: {e}")
            
            # Update failure metrics
            self.execution_metrics["total_executions"] += 1
            self.execution_metrics["failed_executions"] += 1
            
            # Record failed execution
            self._record_execution({
                "task_id": task.id,
                "tool_name": tool_name,
                "tool_id": tool_id,
                "input": tool_input,
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds(),
                "success": False,
                "timestamp": datetime.now().isoformat()
            })
            
            return TaskResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def _execute_sequential_tools(self, task: Task) -> TaskResult:
        """Execute tools sequentially, passing output from one to the next."""
        tool_ids = task.input_data.get("tool_ids", [])
        tool_names = task.input_data.get("tool_names", [])
        initial_input = task.input_data.get("initial_input", {})
        format_type = task.input_data.get("format_type")
        
        # Convert names to IDs if needed
        if tool_names and not tool_ids:
            tool_ids = await self._resolve_tool_names_to_ids(tool_names)
        
        if not tool_ids:
            return TaskResult(
                success=False,
                output=None,
                error="No tools specified for sequential execution"
            )
        
        self.logger.info(f"Executing {len(tool_ids)} tools sequentially")
        
        try:
            results = await self.tool_registry.execute_tools_sequential(
                tool_ids=tool_ids,
                initial_input=initial_input,
                format_type=format_type
            )
            
            # Process results
            execution_summary = {
                "tool_count": len(tool_ids),
                "results": results,
                "final_output": results[-1]["output_data"] if results else None
            }
            
            return TaskResult(
                success=True,
                output=execution_summary,
                metadata={
                    "execution_type": "sequential",
                    "tool_count": len(tool_ids)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Sequential execution failed: {e}")
            return TaskResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def _execute_parallel_tools(self, task: Task) -> TaskResult:
        """Execute multiple tools in parallel."""
        tool_configs = task.input_data.get("tools", [])
        format_type = task.input_data.get("format_type")
        
        if not tool_configs:
            return TaskResult(
                success=False,
                output=None,
                error="No tools specified for parallel execution"
            )
        
        # Prepare tool IDs and inputs
        tool_ids = []
        input_data = {}
        
        for config in tool_configs:
            tool_id = config.get("tool_id") or await self._resolve_tool_name_to_id(
                config.get("tool_name")
            )
            if tool_id:
                tool_ids.append(tool_id)
                input_data[tool_id] = config.get("input", {})
        
        if not tool_ids:
            return TaskResult(
                success=False,
                output=None,
                error="Failed to resolve tool identifiers"
            )
        
        self.logger.info(f"Executing {len(tool_ids)} tools in parallel")
        
        try:
            results = await self.tool_registry.execute_tools_parallel(
                tool_ids=tool_ids,
                input_data=input_data,
                format_type=format_type
            )
            
            # Process results
            execution_summary = {
                "tool_count": len(tool_ids),
                "results": {
                    result["tool_id"]: result["output_data"] 
                    for result in results
                }
            }
            
            return TaskResult(
                success=True,
                output=execution_summary,
                metadata={
                    "execution_type": "parallel",
                    "tool_count": len(tool_ids)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Parallel execution failed: {e}")
            return TaskResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def _execute_workflow(self, task: Task) -> TaskResult:
        """Execute a complex workflow with conditional logic."""
        workflow_def = task.input_data.get("workflow", {})
        initial_context = task.input_data.get("context", {})
        
        self.logger.info(f"Executing workflow: {workflow_def.get('name', 'unnamed')}")
        
        # Initialize workflow context
        context = initial_context.copy()
        workflow_results = {
            "steps": [],
            "final_output": None,
            "workflow_name": workflow_def.get("name"),
            "start_time": datetime.now().isoformat()
        }
        
        try:
            # Execute workflow steps
            steps = workflow_def.get("steps", [])
            for i, step in enumerate(steps):
                step_result = await self._execute_workflow_step(
                    step, context, i
                )
                
                workflow_results["steps"].append(step_result)
                
                if not step_result["success"]:
                    # Handle step failure
                    if step.get("continue_on_error", False):
                        self.logger.warning(f"Step {i} failed, continuing workflow")
                    else:
                        return TaskResult(
                            success=False,
                            output=workflow_results,
                            error=f"Workflow failed at step {i}: {step_result['error']}"
                        )
                
                # Update context with step output
                if step_result["output"]:
                    context[f"step_{i}_output"] = step_result["output"]
                    
                    # Check conditions for next steps
                    if "conditions" in step:
                        if not self._evaluate_conditions(
                            step["conditions"], context
                        ):
                            self.logger.info(f"Conditions not met, skipping remaining steps")
                            break
            
            # Set final output
            workflow_results["final_output"] = context
            workflow_results["end_time"] = datetime.now().isoformat()
            
            return TaskResult(
                success=True,
                output=workflow_results,
                metadata={
                    "execution_type": "workflow",
                    "steps_executed": len(workflow_results["steps"])
                }
            )
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            return TaskResult(
                success=False,
                output=workflow_results,
                error=str(e)
            )
    
    async def _execute_workflow_step(
        self, 
        step: Dict[str, Any], 
        context: Dict[str, Any],
        step_index: int
    ) -> Dict[str, Any]:
        """Execute a single workflow step."""
        step_type = step.get("type", "tool")
        step_name = step.get("name", f"Step {step_index}")
        
        self.logger.info(f"Executing workflow step: {step_name}")
        
        try:
            if step_type == "tool":
                # Execute tool
                tool_result = await self.tool_registry.execute_tool(
                    tool_name=step.get("tool_name"),
                    tool_id=step.get("tool_id"),
                    input_data=self._resolve_step_inputs(step.get("input", {}), context)
                )
                
                return {
                    "name": step_name,
                    "type": step_type,
                    "success": True,
                    "output": tool_result
                }
            
            elif step_type == "condition":
                # Evaluate condition
                condition_result = self._evaluate_conditions(
                    step.get("conditions", {}), context
                )
                
                return {
                    "name": step_name,
                    "type": step_type,
                    "success": True,
                    "output": {"condition_met": condition_result}
                }
            
            elif step_type == "transform":
                # Transform data
                transform_result = self._apply_transformation(
                    step.get("transformation", {}), context
                )
                
                return {
                    "name": step_name,
                    "type": step_type,
                    "success": True,
                    "output": transform_result
                }
            
            else:
                return {
                    "name": step_name,
                    "type": step_type,
                    "success": False,
                    "error": f"Unknown step type: {step_type}"
                }
                
        except Exception as e:
            return {
                "name": step_name,
                "type": step_type,
                "success": False,
                "error": str(e)
            }
    
    async def _execute_general_task(self, task: Task) -> TaskResult:
        """Execute a general task by finding and using appropriate tools."""
        # Search for relevant tools
        relevant_tools = await self.tool_registry.search_tools(
            prompt=task.description,
            limit=3
        )
        
        if not relevant_tools:
            return TaskResult(
                success=False,
                output=None,
                error="No relevant tools found for task"
            )
        
        # Use the most relevant tool
        best_tool = relevant_tools[0]
        
        # Prepare input based on task data
        tool_input = self._prepare_tool_input(best_tool, task.input_data)
        
        # Execute the tool
        execution_task = Task(
            type=TaskType.TOOL_EXECUTION,
            name=f"Execute {best_tool['name']} for {task.name}",
            description=f"Execute tool to complete task",
            created_by=self.id,
            input_data={
                "tool_name": best_tool["name"],
                "input": tool_input
            }
        )
        
        return await self._execute_single_tool(execution_task)
    
    async def _cache_common_tools(self):
        """Pre-cache commonly used tools."""
        try:
            # Get list of all tools
            tools = await self.tool_registry.get_tools(limit=20)
            
            for tool in tools[:10]:  # Cache top 10 tools
                self.tool_cache[tool["id"]] = tool
                if tool.get("name"):
                    self.tool_cache[tool["name"]] = tool
            
            self.logger.info(f"Cached {len(self.tool_cache)} tools")
            
        except Exception as e:
            self.logger.error(f"Failed to cache tools: {e}")
    
    async def _resolve_tool_names_to_ids(self, tool_names: List[str]) -> List[str]:
        """Resolve tool names to IDs."""
        tool_ids = []
        
        for name in tool_names:
            tool_id = await self._resolve_tool_name_to_id(name)
            if tool_id:
                tool_ids.append(tool_id)
        
        return tool_ids
    
    async def _resolve_tool_name_to_id(self, tool_name: str) -> Optional[str]:
        """Resolve a single tool name to ID."""
        # Check cache first
        if tool_name in self.tool_cache:
            return self.tool_cache[tool_name].get("id")
        
        # Fetch from registry
        try:
            tool = await self.tool_registry.get_tool_by_name(tool_name)
            if tool:
                self.tool_cache[tool_name] = tool
                return tool.get("id")
        except Exception as e:
            self.logger.error(f"Failed to resolve tool name {tool_name}: {e}")
        
        return None
    
    def _resolve_step_inputs(
        self, 
        input_template: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve input template with context values."""
        resolved = {}
        
        for key, value in input_template.items():
            if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                # Template variable
                var_name = value[2:-2].strip()
                resolved[key] = context.get(var_name, value)
            else:
                resolved[key] = value
        
        return resolved
    
    def _evaluate_conditions(
        self, 
        conditions: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate workflow conditions."""
        condition_type = conditions.get("type", "all")
        checks = conditions.get("checks", [])
        
        results = []
        for check in checks:
            field = check.get("field")
            operator = check.get("operator", "equals")
            value = check.get("value")
            
            field_value = context.get(field)
            
            if operator == "equals":
                results.append(field_value == value)
            elif operator == "not_equals":
                results.append(field_value != value)
            elif operator == "contains":
                results.append(value in str(field_value))
            elif operator == "greater_than":
                results.append(float(field_value) > float(value))
            elif operator == "less_than":
                results.append(float(field_value) < float(value))
            else:
                results.append(False)
        
        if condition_type == "all":
            return all(results)
        elif condition_type == "any":
            return any(results)
        else:
            return False
    
    def _apply_transformation(
        self, 
        transformation: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Any:
        """Apply data transformation."""
        transform_type = transformation.get("type", "extract")
        source = transformation.get("source")
        
        source_value = context.get(source)
        
        if transform_type == "extract":
            # Extract field from source
            field = transformation.get("field")
            if isinstance(source_value, dict):
                return source_value.get(field)
            return None
        
        elif transform_type == "map":
            # Map values
            mapping = transformation.get("mapping", {})
            return mapping.get(str(source_value), source_value)
        
        elif transform_type == "aggregate":
            # Aggregate multiple values
            sources = transformation.get("sources", [])
            values = [context.get(s) for s in sources]
            
            agg_func = transformation.get("function", "concat")
            if agg_func == "concat":
                return " ".join(str(v) for v in values if v)
            elif agg_func == "sum":
                return sum(float(v) for v in values if v is not None)
            elif agg_func == "count":
                return len([v for v in values if v is not None])
        
        return source_value
    
    def _prepare_tool_input(
        self, 
        tool: Dict[str, Any], 
        task_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare input for tool based on its schema."""
        input_schema = tool.get("input_schema", {})
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        
        prepared_input = {}
        
        # Map task input to tool input schema
        for prop_name, prop_schema in properties.items():
            if prop_name in task_input:
                prepared_input[prop_name] = task_input[prop_name]
            elif prop_name in required:
                # Try to infer from other fields
                prepared_input[prop_name] = self._infer_value(
                    prop_name, prop_schema, task_input
                )
        
        return prepared_input
    
    def _infer_value(
        self, 
        field_name: str, 
        field_schema: Dict[str, Any], 
        available_data: Dict[str, Any]
    ) -> Any:
        """Try to infer a value for a required field."""
        field_type = field_schema.get("type", "string")
        
        # Simple inference rules
        if "query" in field_name.lower():
            return available_data.get("query", available_data.get("search", ""))
        elif "text" in field_name.lower():
            return available_data.get("text", available_data.get("content", ""))
        elif "url" in field_name.lower():
            return available_data.get("url", available_data.get("endpoint", ""))
        
        # Return default based on type
        if field_type == "string":
            return ""
        elif field_type == "number":
            return 0
        elif field_type == "boolean":
            return False
        elif field_type == "array":
            return []
        elif field_type == "object":
            return {}
        
        return None
    
    def _record_execution(self, execution_data: Dict[str, Any]):
        """Record execution in history."""
        self.execution_history.append(execution_data)
        
        # Keep only last 100 executions
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
    
    def _update_average_execution_time(self, execution_time: float):
        """Update average execution time metric."""
        total = self.execution_metrics["total_executions"]
        current_avg = self.execution_metrics["average_execution_time"]
        
        # Calculate new average
        new_avg = ((current_avg * (total - 1)) + execution_time) / total
        self.execution_metrics["average_execution_time"] = new_avg
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.tool_registry.close()