"""Quick start example for multi-agent systems."""

import asyncio
import anyio

from claude_code_sdk.multi_agent import (
    AgentCoordinator,
    DeveloperAgent,
    ReviewerAgent,
    TesterAgent,
    create_development_team,
)


async def simple_collaboration_example():
    """Simple example of agents collaborating."""
    print("=== Simple Multi-Agent Collaboration ===\n")
    
    # Create coordinator
    coordinator = AgentCoordinator()
    
    # Create agents
    developer = DeveloperAgent("Alice_Dev")
    reviewer = ReviewerAgent("Bob_Reviewer")
    tester = TesterAgent("Charlie_Tester")
    
    # Register agents
    coordinator.register_agent(developer)
    coordinator.register_agent(reviewer)
    coordinator.register_agent(tester)
    
    # Create tasks
    dev_task = coordinator.create_task(
        name="Create Calculator",
        description="Create a simple Calculator class with add, subtract, multiply, and divide methods",
        assigned_to=developer
    )
    
    test_task = coordinator.create_task(
        name="Test Calculator",
        description="Write unit tests for the Calculator class",
        assigned_to=tester,
        dependencies=[dev_task.id]  # Depends on development task
    )
    
    review_task = coordinator.create_task(
        name="Review Code",
        description="Review the Calculator implementation and tests",
        assigned_to=reviewer,
        dependencies=[dev_task.id, test_task.id]  # Depends on both
    )
    
    # Assign tasks
    await coordinator.assign_task(dev_task, developer)
    await coordinator.assign_task(test_task, tester)
    await coordinator.assign_task(review_task, reviewer)
    
    # Start coordinator (this will start all agents)
    print("Starting multi-agent system...")
    coordinator_task = asyncio.create_task(coordinator.start())
    
    # Wait for tasks to complete
    await asyncio.sleep(30)  # Give agents time to work
    
    # Check results
    print("\n=== Task Results ===")
    for task_id, task in coordinator.tasks.items():
        print(f"\nTask: {task.name}")
        print(f"Status: {task.status.value}")
        if task.result:
            print(f"Result: {task.result}")
            
    # Stop coordinator
    await coordinator.stop()
    coordinator_task.cancel()


async def development_team_example():
    """Example using pre-configured development team."""
    print("\n=== Development Team Example ===\n")
    
    # Create a standard development team
    coordinator, agents = await create_development_team()
    
    # Create a project with multiple tasks
    tasks = []
    
    # Architecture phase
    arch_task = coordinator.create_task(
        name="Design API Architecture",
        description="Design RESTful API architecture for a todo list application",
        assigned_to=agents["architect"]
    )
    tasks.append(arch_task)
    
    # Development phase
    dev_task = coordinator.create_task(
        name="Implement Todo API",
        description="Implement the todo list API with CRUD operations",
        assigned_to=agents["developer"],
        dependencies=[arch_task.id]
    )
    tasks.append(dev_task)
    
    # Testing phase
    test_task = coordinator.create_task(
        name="Write API Tests",
        description="Write comprehensive tests for the todo API",
        assigned_to=agents["tester"],
        dependencies=[dev_task.id]
    )
    tasks.append(test_task)
    
    # Security audit
    security_task = coordinator.create_task(
        name="Security Audit",
        description="Perform security audit on the API implementation",
        assigned_to=agents["security"],
        dependencies=[dev_task.id]
    )
    tasks.append(security_task)
    
    # Code review
    review_task = coordinator.create_task(
        name="Final Review",
        description="Perform final code review of the entire implementation",
        assigned_to=agents["reviewer"],
        dependencies=[dev_task.id, test_task.id, security_task.id]
    )
    tasks.append(review_task)
    
    # Assign all tasks
    for task in tasks:
        await coordinator.assign_task(task, task.assigned_to)
        
    # Start the system
    print("Starting development team...")
    coordinator_task = asyncio.create_task(coordinator.start())
    
    # Monitor progress
    for i in range(60):  # Monitor for 60 seconds
        await asyncio.sleep(5)
        
        completed = coordinator.get_completed_tasks()
        failed = coordinator.get_failed_tasks()
        
        print(f"\nProgress Update {i+1}:")
        print(f"Completed: {len(completed)}/{len(tasks)}")
        print(f"Failed: {len(failed)}")
        
        if len(completed) + len(failed) == len(tasks):
            print("\nAll tasks finished!")
            break
            
    # Show final results
    print("\n=== Final Results ===")
    for task in tasks:
        print(f"\n{task.name}:")
        print(f"  Status: {task.status.value}")
        if task.result:
            print(f"  Files created: {task.result.get('files_created', [])}")
            print(f"  Tools used: {[t['tool'] for t in task.result.get('tools_used', [])]}")
            
    # Stop the system
    await coordinator.stop()
    coordinator_task.cancel()


async def pub_sub_example():
    """Example using publish-subscribe communication."""
    print("\n=== Publish-Subscribe Example ===\n")
    
    coordinator = AgentCoordinator()
    
    # Create specialized agents
    frontend_dev = DeveloperAgent("Frontend_Dev")
    backend_dev = DeveloperAgent("Backend_Dev")
    db_admin = DeveloperAgent("DB_Admin")
    
    # Register agents
    for agent in [frontend_dev, backend_dev, db_admin]:
        coordinator.register_agent(agent)
        
    # Subscribe to topics
    coordinator.subscribe_agent(frontend_dev.id, "api_changes")
    coordinator.subscribe_agent(backend_dev.id, "database_changes")
    coordinator.subscribe_agent(db_admin.id, "schema_updates")
    
    # All agents subscribe to "announcements"
    for agent in [frontend_dev, backend_dev, db_admin]:
        coordinator.subscribe_agent(agent.id, "announcements")
        
    # Start the system
    print("Starting pub-sub system...")
    coordinator_task = asyncio.create_task(coordinator.start())
    
    # Publish messages
    await coordinator.publish(
        "announcements",
        "System",
        "New project starting: E-commerce Platform"
    )
    
    await asyncio.sleep(2)
    
    await coordinator.publish(
        "api_changes",
        backend_dev,
        "New endpoint added: POST /api/products"
    )
    
    await asyncio.sleep(2)
    
    await coordinator.publish(
        "database_changes",
        db_admin,
        "New table created: products (id, name, price, stock)"
    )
    
    await asyncio.sleep(5)
    
    # Stop the system
    await coordinator.stop()
    coordinator_task.cancel()


async def main():
    """Run all examples."""
    print("Multi-Agent System Quick Start Examples")
    print("======================================\n")
    
    examples = [
        ("Simple Collaboration", simple_collaboration_example),
        ("Development Team", development_team_example),
        ("Publish-Subscribe", pub_sub_example),
    ]
    
    for name, example_func in examples:
        try:
            print(f"\nRunning: {name}")
            print("-" * 40)
            await example_func()
        except Exception as e:
            print(f"Example '{name}' failed: {e}")
            
        await asyncio.sleep(2)
        
    print("\n\nAll examples completed!")


if __name__ == "__main__":
    anyio.run(main)