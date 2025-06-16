#!/usr/bin/env python3
"""Examples demonstrating advanced conversation features."""

import asyncio
from pathlib import Path
from datetime import datetime, timedelta

from claude_code_sdk import query, ClaudeCodeOptions
from claude_code_sdk.conversation_manager import ConversationManager
from claude_code_sdk.conversation_templates import (
    TemplateManager,
    CODE_REVIEW_TEMPLATE,
    DEBUGGING_TEMPLATE,
)
from claude_code_sdk.conversation_chains import (
    create_debugging_chain,
    create_refactoring_chain,
    ConversationChain,
    ChainStep,
)


async def example_conversation_manager():
    """Demonstrate conversation management with persistence."""
    print("=== Conversation Manager Example ===\n")
    
    # Create manager with custom storage
    manager = ConversationManager(Path.home() / ".claude_conversations_demo")
    
    # Create a new conversation with metadata
    print("1. Creating new conversation with metadata...")
    session_id, messages = await manager.create_conversation(
        initial_prompt="Let's discuss Python best practices",
        metadata={"project": "demo", "topic": "python"},
        tags=["tutorial", "python", "best-practices"]
    )
    
    print(f"Created conversation: {session_id}")
    
    # Process messages
    async for msg in messages:
        pass
    
    # Continue the conversation
    print("\n2. Continuing conversation...")
    async for msg in manager.continue_conversation(
        session_id,
        "What about error handling best practices?"
    ):
        pass
    
    # Branch the conversation
    print("\n3. Branching conversation...")
    branch_id, branch_messages = await manager.branch_conversation(
        session_id,
        "Let's focus specifically on async error handling"
    )
    
    print(f"Created branch: {branch_id}")
    
    async for msg in branch_messages:
        pass
    
    # List conversations
    print("\n4. Listing conversations...")
    conversations = manager.list_conversations(tags=["python"])
    for conv in conversations:
        print(f"  - {conv.session_id}: {conv.tags} (turns: {conv.turn_count})")
    
    # Export conversation
    print("\n5. Exporting conversation...")
    export_path = Path.home() / "conversation_export.json"
    manager.export_conversation(session_id, export_path)
    print(f"Exported to: {export_path}")
    
    # Show conversation tree
    print("\n6. Conversation tree:")
    tree = manager.get_conversation_tree(session_id)
    print_tree(tree)
    
    print(f"\nTotal cost across all conversations: ${manager.get_total_cost():.4f}")


async def example_conversation_templates():
    """Demonstrate using conversation templates."""
    print("\n=== Conversation Templates Example ===\n")
    
    template_manager = TemplateManager()
    
    # Use code review template
    print("1. Using Code Review Template...")
    template = template_manager.get_template("code_review")
    
    # Format initial prompt with context
    context = {"file_path": "example.py"}
    initial_prompt = template.format_initial_prompt(context)
    options = template.create_options()
    
    print(f"Template: {template.name}")
    print(f"Initial prompt: {initial_prompt}")
    print(f"Allowed tools: {options.allowed_tools}")
    
    # Execute with template
    async for msg in query(prompt=initial_prompt, options=options):
        # Process messages
        pass
    
    # Show follow-up suggestions
    print("\nSuggested follow-ups:")
    for suggestion in template.follow_up_suggestions[:3]:
        print(f"  - {suggestion}")
    
    # Create custom template
    print("\n2. Creating Custom Template...")
    custom_template = template_manager.create_custom_template(
        name="API Design Review",
        description="Review API design for RESTful services",
        system_prompt="You are an API design expert. Focus on REST principles, consistency, and usability.",
        initial_prompts=[
            "Please review the API design in {api_spec_file} for REST compliance and best practices."
        ],
        required_context=["api_spec_file"],
        follow_up_suggestions=[
            "How can we improve the error responses?",
            "Are the endpoints following REST conventions?",
            "What about versioning strategy?",
        ]
    )
    print(f"Created template: {custom_template.name}")
    
    # Suggest template based on task
    print("\n3. Template Suggestion...")
    task = "I need help debugging a performance issue"
    suggested = template_manager.suggest_template(task)
    if suggested:
        print(f"Suggested template for '{task}': {suggested.name}")


async def example_conversation_chains():
    """Demonstrate conversation chains for complex workflows."""
    print("\n=== Conversation Chains Example ===\n")
    
    # Example 1: Debugging chain
    print("1. Running Debugging Chain...")
    debug_chain = create_debugging_chain()
    
    result = await debug_chain.execute(
        context_overrides={
            "issue_description": "The function calculate_total() returns wrong values for negative inputs"
        }
    )
    
    print(f"Chain status: {result.status.value}")
    print(f"Completed steps: {', '.join(result.completed_steps)}")
    print(f"Total cost: ${result.total_cost:.4f}")
    
    # Example 2: Custom chain with conditions
    print("\n2. Custom Chain with Conditional Steps...")
    
    def needs_optimization(context):
        # Condition based on previous analysis
        return context.get("performance_score", 0) < 80
    
    custom_chain = ConversationChain(
        name="performance_optimization",
        steps=[
            ChainStep(
                name="profile_code",
                prompt_template="Profile the performance of {target_function}",
                result_processor=lambda msgs: {"performance_score": 65},  # Mock result
            ),
            ChainStep(
                name="optimize",
                prompt_template="Optimize the function based on profiling results",
                condition=needs_optimization,
                dependencies=["profile_code"],
            ),
            ChainStep(
                name="benchmark",
                prompt_template="Benchmark the optimized code",
                dependencies=["optimize"],
            ),
        ]
    )
    
    result = await custom_chain.execute(
        context_overrides={"target_function": "process_data()"}
    )
    
    print(f"Optimization needed: {needs_optimization(result.context)}")
    print(f"Executed steps: {', '.join(result.completed_steps)}")
    
    # Example 3: Parallel execution
    print("\n3. Parallel Chain Execution...")
    
    parallel_chain = ConversationChain(
        name="parallel_analysis",
        steps=[
            ChainStep(name="analyze_security", prompt_template="Analyze security aspects"),
            ChainStep(name="analyze_performance", prompt_template="Analyze performance"),
            ChainStep(name="analyze_maintainability", prompt_template="Analyze maintainability"),
            ChainStep(
                name="create_report",
                prompt_template="Create a comprehensive report based on all analyses",
                dependencies=["analyze_security", "analyze_performance", "analyze_maintainability"],
            ),
        ]
    )
    
    # Execute with parallel processing
    result = await parallel_chain.execute(parallel=True)
    print(f"Parallel execution completed: {result.status.value}")


async def example_advanced_workflow():
    """Demonstrate a complete advanced workflow."""
    print("\n=== Advanced Workflow Example ===\n")
    
    # Initialize components
    manager = ConversationManager()
    template_manager = TemplateManager()
    
    # Step 1: Start with a template
    print("1. Starting with debugging template...")
    debug_template = template_manager.get_template("debugging")
    
    initial_prompt = debug_template.format_initial_prompt({
        "issue_description": "Memory leak in data processing",
        "file_path": "data_processor.py"
    })
    
    session_id, messages = await manager.create_conversation(
        initial_prompt=initial_prompt,
        options=debug_template.create_options(),
        tags=["debugging", "memory-leak"]
    )
    
    async for msg in messages:
        pass
    
    # Step 2: Branch for different solutions
    print("\n2. Exploring alternative solutions...")
    
    # Branch 1: Quick fix
    branch1_id, _ = await manager.branch_conversation(
        session_id,
        "Let's try a quick fix by clearing caches"
    )
    
    # Branch 2: Refactoring
    branch2_id, _ = await manager.branch_conversation(
        session_id,
        "Let's refactor the entire data processing pipeline"
    )
    
    # Step 3: Create a chain for the refactoring branch
    print("\n3. Running refactoring chain on branch...")
    refactor_chain = create_refactoring_chain()
    
    # Continue in the refactoring branch context
    async for msg in manager.continue_conversation(
        branch2_id,
        "Please proceed with the refactoring plan"
    ):
        pass
    
    # Step 4: Compare branches
    print("\n4. Comparing branches...")
    branch1_context = manager.get_conversation_context(branch1_id)
    branch2_context = manager.get_conversation_context(branch2_id)
    
    print(f"Quick fix branch - Turns: {branch1_context.turn_count}, Cost: ${branch1_context.total_cost:.4f}")
    print(f"Refactor branch - Turns: {branch2_context.turn_count}, Cost: ${branch2_context.total_cost:.4f}")
    
    # Step 5: Export results
    print("\n5. Exporting conversation history...")
    manager.export_conversation(session_id, Path.home() / "debug_session.json")
    manager.export_conversation(branch2_id, Path.home() / "refactor_session.json")
    
    print("\nWorkflow completed!")


def print_tree(tree: dict, level: int = 0):
    """Helper to print conversation tree."""
    indent = "  " * level
    print(f"{indent}- {tree['session_id']} ({tree['turn_count']} turns)")
    for child in tree.get('children', []):
        print_tree(child, level + 1)


async def main():
    """Run all examples."""
    await example_conversation_manager()
    await example_conversation_templates()
    await example_conversation_chains()
    await example_advanced_workflow()


if __name__ == "__main__":
    asyncio.run(main())