#!/usr/bin/env python3
"""Example showing how to integrate the Claude Code Agent System with the SDK."""

import asyncio
import sys
from pathlib import Path

# Add agent_system to path for this example
agent_system_path = Path(__file__).parent.parent / "agent_system"
sys.path.insert(0, str(agent_system_path))

from claude_code_sdk import query, ClaudeCodeOptions
from claude_code_sdk.types import AssistantMessage, TextBlock, ToolUseBlock, ToolResultBlock

# Import agent system components
try:
    from base_agent import BaseAgent, AgentMessage
    from agents import CoderAgent, ResearcherAgent, ValidatorAgent
except ImportError as e:
    print(f"Error importing agent system: {e}")
    print("Make sure the agent_system directory is in the Python path")
    sys.exit(1)


class SDKIntegratedCoderAgent(CoderAgent):
    """Enhanced Coder Agent that uses Claude Code SDK instead of CLI."""
    
    async def generate_code_with_sdk(self, prompt: str, language: str = "python") -> str:
        """Generate code using Claude Code SDK with enhanced capabilities."""
        
        # Configure Claude options for code generation
        options = ClaudeCodeOptions(
            allowed_tools=["Read", "Write", "Edit", "Grep", "Glob"],
            max_thinking_tokens=10000,
            append_system_prompt=f"You are generating {language} code. Focus on clean, efficient, and well-documented code."
        )
        
        # Collect the full response
        full_response = ""
        code_blocks = []
        tool_uses = []
        
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            full_response += block.text + "\n"
                        elif isinstance(block, ToolUseBlock):
                            tool_uses.append({
                                "tool": block.name,
                                "input": block.input
                            })
                        elif isinstance(block, ToolResultBlock):
                            if block.is_error:
                                self.logger.error(f"Tool error: {block.content}")
                            else:
                                # Extract code from tool results if present
                                if "```" in block.content:
                                    code_blocks.append(block.content)
            
            # Log tool usage for debugging
            if tool_uses:
                self.logger.info(f"Tools used during code generation: {[t['tool'] for t in tool_uses]}")
            
            return full_response
            
        except Exception as e:
            self.logger.error(f"Error generating code with SDK: {e}")
            raise
    
    async def process_task(self, task: dict) -> dict:
        """Override to use SDK-based code generation."""
        self.logger.info(f"Processing coding task with SDK: {task}")
        
        # Extract task details
        description = task.get("description", "")
        requirements = task.get("requirements", [])
        language = task.get("language", "python")
        
        # Build comprehensive prompt
        prompt = f"""
        Task: {description}
        
        Requirements:
        {chr(10).join(f'- {req}' for req in requirements)}
        
        Please generate the {language} code to accomplish this task.
        Include proper error handling, documentation, and tests if applicable.
        """
        
        # Generate code using SDK
        code = await self.generate_code_with_sdk(prompt, language)
        
        return {
            "status": "completed",
            "code": code,
            "language": language,
            "tools_used": True,
            "sdk_version": "0.0.10"
        }


class SDKIntegratedResearcherAgent(ResearcherAgent):
    """Enhanced Researcher Agent that uses Claude Code SDK for analysis."""
    
    async def analyze_with_sdk(self, research_data: dict) -> str:
        """Use Claude SDK to analyze research findings."""
        
        options = ClaudeCodeOptions(
            allowed_tools=["WebSearch", "WebFetch"],
            max_thinking_tokens=8000,
        )
        
        prompt = f"""
        Analyze the following research data and provide insights:
        
        Topic: {research_data.get('topic', 'Unknown')}
        Sources: {len(research_data.get('sources', []))}
        
        Key findings:
        {chr(10).join(research_data.get('findings', []))}
        
        Please provide a comprehensive analysis with actionable insights.
        """
        
        analysis = ""
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        analysis += block.text
        
        return analysis


async def integrated_workflow_example():
    """Example of a complete workflow using SDK-integrated agents."""
    
    print("=== Claude Code SDK + Agent System Integration Example ===\n")
    
    # Initialize agents with SDK integration
    coder = SDKIntegratedCoderAgent()
    researcher = SDKIntegratedResearcherAgent()
    
    # Example 1: Research and Code Generation Workflow
    print("1. Research Task")
    research_task = {
        "topic": "Best practices for Python async web servers",
        "sources": ["official docs", "blog posts", "GitHub examples"],
        "findings": [
            "Use ASGI servers like uvicorn for async support",
            "Implement proper connection pooling",
            "Handle graceful shutdowns",
            "Use structured concurrency with TaskGroups"
        ]
    }
    
    analysis = await researcher.analyze_with_sdk(research_task)
    print(f"Research Analysis:\n{analysis[:500]}...\n")
    
    # Example 2: Code Generation based on Research
    print("2. Code Generation Task")
    coding_task = {
        "description": "Create an async web server with best practices",
        "requirements": [
            "Use FastAPI framework",
            "Implement connection pooling",
            "Add graceful shutdown handling",
            "Include health check endpoint",
            "Use structured logging"
        ],
        "language": "python"
    }
    
    result = await coder.process_task(coding_task)
    print(f"Generated Code:\n{result['code'][:1000]}...\n")
    
    # Example 3: Direct SDK Usage within Agent Context
    print("3. Direct SDK Query")
    options = ClaudeCodeOptions(
        allowed_tools=["Task"],
        max_thinking_tokens=5000
    )
    
    direct_response = ""
    async for message in query(
        prompt="Create a simple task management system architecture",
        options=options
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    direct_response += block.text
    
    print(f"Architecture Design:\n{direct_response[:500]}...\n")
    
    print("\n=== Integration Example Complete ===")


async def multi_agent_collaboration_example():
    """Example showing multiple agents collaborating using SDK."""
    
    print("\n=== Multi-Agent Collaboration Example ===\n")
    
    # Initialize agents
    coder = SDKIntegratedCoderAgent()
    validator = ValidatorAgent()
    
    # Generate code
    code_result = await coder.process_task({
        "description": "Create a fibonacci function with memoization",
        "requirements": ["Handle large numbers", "Include type hints", "Add docstring"],
        "language": "python"
    })
    
    # Validate the generated code
    validation_result = await validator.validate({
        "code": code_result["code"],
        "test_cases": [
            {"input": 0, "expected": 0},
            {"input": 1, "expected": 1},
            {"input": 10, "expected": 55}
        ]
    })
    
    print(f"Code Generation Status: {code_result['status']}")
    print(f"Validation Status: {validation_result.get('status', 'unknown')}")
    print(f"SDK was used: {code_result.get('sdk_version') is not None}")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(integrated_workflow_example())
    asyncio.run(multi_agent_collaboration_example())