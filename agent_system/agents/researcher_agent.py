"""
Researcher Agent - Specializes in searching, analyzing, and gathering information.
"""

import asyncio
from typing import Dict, Any, List
import json

from ..core.base_agent import BaseAgent, AgentState
from ..core.task import Task, TaskResult, TaskType
from ..config import AgentRole, AgentCapability
from ..integrations import get_tool_registry_client


class ResearcherAgent(BaseAgent):
    """Agent specialized in research and information gathering."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.role = AgentRole.RESEARCHER
        self.capabilities = {
            AgentCapability.WEB_SEARCH,
            AgentCapability.DATA_ANALYSIS,
            AgentCapability.TOOL_EXECUTION
        }
        self.tool_registry = get_tool_registry_client()
        
        # Research-specific knowledge
        self.research_cache: Dict[str, Any] = {}
        self.search_history: List[Dict[str, Any]] = []
    
    async def initialize(self):
        """Initialize the researcher agent."""
        self.logger.info("Initializing Researcher Agent")
        
        # Check tool registry connectivity
        try:
            health = await self.tool_registry.health_check()
            self.logger.info(f"Tool registry health: {health['status']}")
        except Exception as e:
            self.logger.error(f"Failed to connect to tool registry: {e}")
    
    async def process_task(self, task: Task) -> TaskResult:
        """Process research tasks."""
        self.logger.info(f"Processing research task: {task.name}")
        
        try:
            if task.type == TaskType.WEB_SEARCH:
                return await self._perform_web_search(task)
            elif task.type == TaskType.DATA_ANALYSIS:
                return await self._perform_data_analysis(task)
            elif task.type == TaskType.TOOL_EXECUTION:
                return await self._execute_research_tool(task)
            else:
                # Try to handle as a general research task
                return await self._perform_general_research(task)
                
        except Exception as e:
            self.logger.error(f"Error processing task {task.id}: {e}")
            return TaskResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def _perform_web_search(self, task: Task) -> TaskResult:
        """Perform web search using available tools."""
        query = task.input_data.get("query", "")
        limit = task.input_data.get("limit", 10)
        
        self.logger.info(f"Searching for: {query}")
        
        try:
            # Search for web search tools in the registry
            search_tools = await self.tool_registry.search_tools(
                prompt="web search internet google",
                limit=5
            )
            
            if not search_tools:
                # Create a basic web search tool if none exists
                tool_data = {
                    "name": f"web_search_{task.id[:8]}",
                    "description": "Search the web for information",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer", "default": 10}
                        },
                        "required": ["query"]
                    },
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "results": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "url": {"type": "string"},
                                        "snippet": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "action": {
                        "type": "http",
                        "http": {
                            "method": "GET",
                            "url": "https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1",
                            "headers": {}
                        }
                    },
                    "output": {
                        "type": "ai",
                        "content": "Search results retrieved"
                    }
                }
                
                created_tool = await self.tool_registry.create_tool(tool_data, self.id)
                tool_name = created_tool["name"]
            else:
                tool_name = search_tools[0]["name"]
            
            # Execute the search tool
            results = await self.tool_registry.execute_tool(
                tool_name=tool_name,
                input_data={"query": query, "limit": limit}
            )
            
            # Cache results
            self.research_cache[query] = results
            self.search_history.append({
                "query": query,
                "timestamp": task.created_at.isoformat(),
                "results_count": len(results.get("results", []))
            })
            
            return TaskResult(
                success=True,
                output=results,
                metadata={"tool_used": tool_name, "cached": True}
            )
            
        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            return TaskResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def _perform_data_analysis(self, task: Task) -> TaskResult:
        """Analyze data using appropriate tools."""
        data = task.input_data.get("data", {})
        analysis_type = task.input_data.get("analysis_type", "summary")
        
        self.logger.info(f"Performing {analysis_type} analysis")
        
        try:
            # Search for data analysis tools
            analysis_tools = await self.tool_registry.search_tools(
                prompt=f"data analysis {analysis_type} statistics",
                limit=3
            )
            
            if analysis_tools:
                # Use existing tool
                tool = analysis_tools[0]
                results = await self.tool_registry.execute_tool(
                    tool_name=tool["name"],
                    input_data={"data": data, "type": analysis_type}
                )
            else:
                # Perform basic analysis
                results = self._basic_data_analysis(data, analysis_type)
            
            return TaskResult(
                success=True,
                output=results,
                metadata={"analysis_type": analysis_type}
            )
            
        except Exception as e:
            self.logger.error(f"Data analysis failed: {e}")
            return TaskResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def _execute_research_tool(self, task: Task) -> TaskResult:
        """Execute a specific research tool."""
        tool_name = task.input_data.get("tool_name")
        tool_id = task.input_data.get("tool_id")
        tool_input = task.input_data.get("tool_input", {})
        
        try:
            result = await self.tool_registry.execute_tool(
                tool_id=tool_id,
                tool_name=tool_name,
                input_data=tool_input
            )
            
            return TaskResult(
                success=True,
                output=result,
                metadata={"tool_id": tool_id, "tool_name": tool_name}
            )
            
        except Exception as e:
            self.logger.error(f"Tool execution failed: {e}")
            return TaskResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def _perform_general_research(self, task: Task) -> TaskResult:
        """Perform general research combining multiple approaches."""
        topic = task.input_data.get("topic", task.description)
        depth = task.input_data.get("depth", "medium")
        
        self.logger.info(f"Researching topic: {topic} (depth: {depth})")
        
        research_results = {
            "topic": topic,
            "findings": [],
            "sources": [],
            "summary": ""
        }
        
        try:
            # 1. Web search
            search_result = await self._perform_web_search(
                Task(
                    type=TaskType.WEB_SEARCH,
                    name=f"Search for {topic}",
                    description=f"Web search for {topic}",
                    created_by=self.id,
                    input_data={"query": topic, "limit": 10 if depth == "deep" else 5}
                )
            )
            
            if search_result.success and search_result.output:
                research_results["sources"].extend(
                    search_result.output.get("results", [])
                )
            
            # 2. Look for specialized tools
            relevant_tools = await self.tool_registry.search_tools(
                prompt=topic,
                limit=3
            )
            
            # 3. Execute relevant tools
            for tool in relevant_tools[:2]:  # Limit to 2 tools
                try:
                    tool_result = await self.tool_registry.execute_tool(
                        tool_name=tool["name"],
                        input_data={"query": topic}
                    )
                    research_results["findings"].append({
                        "source": tool["name"],
                        "data": tool_result
                    })
                except Exception as e:
                    self.logger.warning(f"Tool {tool['name']} failed: {e}")
            
            # 4. Synthesize findings
            research_results["summary"] = self._synthesize_research(research_results)
            
            return TaskResult(
                success=True,
                output=research_results,
                metadata={
                    "depth": depth,
                    "sources_count": len(research_results["sources"]),
                    "tools_used": len(research_results["findings"])
                }
            )
            
        except Exception as e:
            self.logger.error(f"General research failed: {e}")
            return TaskResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    def _basic_data_analysis(self, data: Any, analysis_type: str) -> Dict[str, Any]:
        """Perform basic data analysis without external tools."""
        if isinstance(data, list):
            return {
                "type": analysis_type,
                "count": len(data),
                "sample": data[:5] if len(data) > 5 else data,
                "data_type": "list"
            }
        elif isinstance(data, dict):
            return {
                "type": analysis_type,
                "keys": list(data.keys()),
                "size": len(data),
                "data_type": "dict"
            }
        else:
            return {
                "type": analysis_type,
                "value": str(data)[:200],
                "data_type": type(data).__name__
            }
    
    def _synthesize_research(self, research_data: Dict[str, Any]) -> str:
        """Synthesize research findings into a summary."""
        summary_parts = [
            f"Research on '{research_data['topic']}' yielded {len(research_data['sources'])} sources."
        ]
        
        if research_data["findings"]:
            summary_parts.append(
                f"Analysis was performed using {len(research_data['findings'])} specialized tools."
            )
        
        if research_data["sources"]:
            summary_parts.append(
                "Key sources include: " + 
                ", ".join([s.get("title", "Unknown") for s in research_data["sources"][:3]])
            )
        
        return " ".join(summary_parts)
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.tool_registry.close()