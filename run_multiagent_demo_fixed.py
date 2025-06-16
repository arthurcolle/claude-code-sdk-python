#!/usr/bin/env python3
"""
Fixed runner script for the advanced multi-agent system demonstration
Works with the updated SDK and handles API responses correctly
"""

import asyncio
import sys
import argparse
from advanced_multiagent_system import demonstrate_system


async def run_basic_demo():
    """Run the basic demonstration with actual Claude API"""
    print("🤖 Running Multi-Agent System with Claude API")
    print("=" * 80)
    print("Note: This will make actual API calls to Claude")
    print("Make sure you have the Claude Code CLI installed and configured")
    print("=" * 80)
    
    try:
        await demonstrate_system()
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("\nTips:")
        print("- Make sure Claude Code CLI is installed: npm install -g @anthropic-ai/claude-code")
        print("- Ensure you're logged in: claude login")
        print("- Check your API rate limits")
        raise


async def run_simulated_demo():
    """Run the simulated demo without API calls"""
    from simple_multiagent_demo import run_simulated_demo
    await run_simulated_demo()


async def run_custom_scenario():
    """Run a custom scenario with limited API calls"""
    from advanced_multiagent_system import (
        MultiAgentSystem, Task, Priority, TeamType
    )
    from claude_code_sdk import ClaudeCodeOptions
    
    print("🚀 Running Custom Multi-Agent Scenario (Limited API Calls)")
    print("=" * 80)
    
    system = MultiAgentSystem()
    
    # Simple scenario with just 2 tasks to minimize API usage
    print("\n📋 Scenario: Quick Security Audit")
    print("-" * 60)
    
    tasks = [
        Task(
            id="audit_001",
            title="Security vulnerability scan",
            description="Perform a quick security scan of the codebase",
            priority=Priority.HIGH,
            required_skills={"security_testing", "vulnerability_assessment"},
            team_type=TeamType.SECURITY
        ),
        Task(
            id="report_001",  
            title="Generate security report",
            description="Create a brief security report based on the scan",
            priority=Priority.MEDIUM,
            required_skills={"documentation", "reporting"},
            team_type=TeamType.SECURITY,
            dependencies=["audit_001"]
        )
    ]
    
    print(f"Executing {len(tasks)} tasks...")
    results = await system.execute_workflow(tasks)
    
    print(f"\n✅ Workflow Completed!")
    print(f"Tasks Executed: {len(results)}")
    
    for task in tasks:
        status_icon = "✅" if task.status == "completed" else "❌"
        agent_name = "Unassigned"
        if task.assigned_to:
            agent = system.agents.get(task.assigned_to)
            if agent:
                agent_name = agent.name
        print(f"{status_icon} {task.title} - {agent_name}")
    
    # Show results
    for result in results:
        if "outputs" in result and result["outputs"]:
            print(f"\nResults from {result.get('agent', 'Unknown')}:")
            for output in result["outputs"][:2]:
                if isinstance(output, str):
                    print(f"  - {output[:100]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Run the advanced multi-agent system demonstration"
    )
    parser.add_argument(
        '--mode',
        choices=['simulated', 'basic', 'custom'],
        default='simulated',
        help='Demonstration mode: simulated (no API calls), basic (full demo), or custom (limited API calls)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'simulated':
            print("Running simulated demonstration (no API calls)...")
            asyncio.run(run_simulated_demo())
        elif args.mode == 'basic':
            print("Running basic demonstration with Claude API...")
            asyncio.run(run_basic_demo())
        elif args.mode == 'custom':
            print("Running custom scenario with limited API calls...")
            asyncio.run(run_custom_scenario())
    except KeyboardInterrupt:
        print("\n\n👋 Demonstration stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()