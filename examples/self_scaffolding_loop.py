#!/usr/bin/env python3
"""
Self-Scaffolding Loop Example using Claude Code SDK

This example demonstrates how to create a self-improving system where Claude:
1. Analyzes a codebase or project
2. Identifies areas for improvement
3. Implements those improvements
4. Tests the changes
5. Reflects on the results and continues improving
"""

import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from claude_code_sdk import query, ClaudeCodeOptions
from claude_code_sdk.types import AssistantMessage, TextBlock, ToolUseBlock, ToolResultBlock, ResultMessage


class SelfScaffoldingLoop:
    """A self-improving system that uses Claude to iteratively enhance a project."""
    
    def __init__(self, project_path: str, max_iterations: int = 5):
        self.project_path = Path(project_path)
        self.max_iterations = max_iterations
        self.iteration_count = 0
        self.improvements_log = []
        self.session_id = None
        
        # Configure Claude options for file operations
        self.claude_options = ClaudeCodeOptions(
            allowed_tools=["Read", "Write", "Edit", "Bash", "Grep", "Glob", "TodoWrite", "TodoRead"],
            permission_mode="acceptEdits",
            max_thinking_tokens=12000,
            append_system_prompt="You are working on a self-scaffolding project. Be systematic and thorough in your improvements."
        )
    
    async def analyze_project(self) -> str:
        """Ask Claude to analyze the current state of the project."""
        analysis_prompt = f"""
        Please analyze the project at {self.project_path} and:
        1. List the main components and structure
        2. Identify areas that could be improved or extended
        3. Suggest 3-5 specific improvements that could be made
        4. Use the TodoWrite tool to track these improvements
        
        Focus on practical, implementable improvements that add value.
        """
        
        analysis_result = ""
        async for message in query(analysis_prompt, self.claude_options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        analysis_result += block.text + "\n"
            elif isinstance(message, ResultMessage):
                self.session_id = message.session_id
        
        return analysis_result
    
    async def implement_improvement(self, improvement: str) -> Dict[str, Any]:
        """Ask Claude to implement a specific improvement."""
        implementation_prompt = f"""
        Please implement the following improvement: {improvement}
        
        Make sure to:
        1. Update the TodoRead/TodoWrite to mark progress
        2. Write clean, well-documented code
        3. Follow existing patterns in the codebase
        4. Test your changes if possible
        5. Explain what you did and why
        """
        
        # Continue the conversation if we have a session
        options = ClaudeCodeOptions(
            **self.claude_options.__dict__,
            resume=self.session_id
        )
        
        implementation_log = {
            "improvement": improvement,
            "changes": [],
            "result": "",
            "timestamp": datetime.now().isoformat()
        }
        
        async for message in query(implementation_prompt, options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        implementation_log["result"] += block.text + "\n"
                    elif isinstance(block, ToolUseBlock):
                        implementation_log["changes"].append({
                            "tool": block.name,
                            "input": block.input
                        })
            elif isinstance(message, ResultMessage):
                implementation_log["cost_usd"] = message.cost_usd
                implementation_log["duration_ms"] = message.duration_ms
        
        return implementation_log
    
    async def test_changes(self) -> str:
        """Ask Claude to test the recent changes."""
        test_prompt = """
        Please test the recent changes you made:
        1. Run any existing tests
        2. Verify the code works as expected
        3. Check for any regressions
        4. Report the results
        
        Use the Bash tool to run tests if they exist.
        """
        
        options = ClaudeCodeOptions(
            **self.claude_options.__dict__,
            resume=self.session_id
        )
        
        test_result = ""
        async for message in query(test_prompt, options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        test_result += block.text + "\n"
        
        return test_result
    
    async def reflect_on_progress(self) -> str:
        """Ask Claude to reflect on the progress made."""
        reflection_prompt = """
        Please reflect on the improvements made so far:
        1. What worked well?
        2. What could be improved?
        3. Are there any new opportunities that emerged?
        4. Update the todo list with any new insights
        
        Be honest and constructive in your assessment.
        """
        
        options = ClaudeCodeOptions(
            **self.claude_options.__dict__,
            resume=self.session_id
        )
        
        reflection = ""
        async for message in query(reflection_prompt, options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        reflection += block.text + "\n"
        
        return reflection
    
    async def run_loop(self):
        """Run the self-scaffolding loop."""
        print(f"🚀 Starting self-scaffolding loop for {self.project_path}")
        print(f"   Max iterations: {self.max_iterations}\n")
        
        for iteration in range(self.max_iterations):
            self.iteration_count = iteration + 1
            print(f"\n{'='*60}")
            print(f"🔄 Iteration {self.iteration_count}/{self.max_iterations}")
            print(f"{'='*60}\n")
            
            # 1. Analyze the project
            print("📊 Analyzing project...")
            analysis = await self.analyze_project()
            print(f"Analysis complete. Found opportunities for improvement.\n")
            
            # 2. Get the todo list
            print("📝 Checking todo list...")
            todos_prompt = "Please check the current todo list and pick the highest priority item to work on."
            
            options = ClaudeCodeOptions(
                **self.claude_options.__dict__,
                resume=self.session_id
            )
            
            selected_improvement = None
            async for message in query(todos_prompt, options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock) and not selected_improvement:
                            # Extract the first improvement mentioned
                            lines = block.text.split('\n')
                            for line in lines:
                                if line.strip() and not line.startswith('#'):
                                    selected_improvement = line.strip()
                                    break
            
            if not selected_improvement:
                print("✅ No more improvements to make!")
                break
            
            # 3. Implement the improvement
            print(f"\n🔨 Implementing: {selected_improvement}")
            implementation = await self.implement_improvement(selected_improvement)
            self.improvements_log.append(implementation)
            
            # 4. Test the changes
            print("\n🧪 Testing changes...")
            test_result = await self.test_changes()
            
            # 5. Reflect on progress
            print("\n🤔 Reflecting on progress...")
            reflection = await self.reflect_on_progress()
            
            # Save progress log
            self.save_progress_log()
            
            print(f"\n✅ Iteration {self.iteration_count} complete!")
        
        print(f"\n{'='*60}")
        print(f"🎉 Self-scaffolding loop complete!")
        print(f"   Total iterations: {self.iteration_count}")
        print(f"   Improvements made: {len(self.improvements_log)}")
        print(f"   Log saved to: scaffolding_log.json")
        print(f"{'='*60}\n")
    
    def save_progress_log(self):
        """Save the progress log to a JSON file."""
        log_data = {
            "project_path": str(self.project_path),
            "iterations": self.iteration_count,
            "session_id": self.session_id,
            "improvements": self.improvements_log
        }
        
        log_path = self.project_path / "scaffolding_log.json"
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)


async def main():
    """Example usage of the self-scaffolding loop."""
    
    # Example 1: Create a simple project and improve it
    print("Creating example project...")
    
    # Create a simple starter project
    project_dir = Path("./example_project")
    project_dir.mkdir(exist_ok=True)
    
    # Create a basic Python file
    (project_dir / "main.py").write_text('''#!/usr/bin/env python3
"""A simple calculator program that could be improved."""

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

if __name__ == "__main__":
    print("Simple Calculator")
    print(f"2 + 3 = {add(2, 3)}")
    print(f"5 - 2 = {subtract(5, 2)}")
''')
    
    # Create a basic README
    (project_dir / "README.md").write_text('''# Example Project

A simple project to demonstrate self-scaffolding capabilities.

## Features
- Basic arithmetic operations
- Command-line interface

## TODO
- Add more operations
- Add tests
- Improve documentation
''')
    
    # Run the self-scaffolding loop
    scaffolder = SelfScaffoldingLoop(
        project_path=str(project_dir),
        max_iterations=3
    )
    
    await scaffolder.run_loop()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())