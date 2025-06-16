#!/usr/bin/env python3
"""
Example: Integrating Claude Visualizer with Claude Code SDK

This example shows how to use the Claude visualizer alongside the SDK
to create visual representations of Claude's cognitive state during interactions.
"""

import asyncio
from pathlib import Path
from claude_code_sdk import query, ClaudeCodeOptions
from claude_visualizer_standalone import ClaudeVisualizer, VisualizationGenerator, CognitiveState


async def visualize_claude_interaction():
    """Run an interactive session with Claude while generating visualizations."""
    
    # Initialize the visualizer components
    visualizer = ClaudeVisualizer()
    generator = VisualizationGenerator()
    
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║            Claude SDK with Self-Image Visualizer                  ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║  This demo shows Claude's self-representation during:             ║")
    print("║  • Code analysis and understanding                                ║")
    print("║  • Problem solving and planning                                   ║")
    print("║  • Creative tasks and synthesis                                   ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    
    # Configure Claude with specific tools
    options = ClaudeCodeOptions(
        allowed_tools=["Read", "Write", "Edit", "Bash", "Grep"],
        max_thinking_tokens=8000,
    )
    
    # Example tasks that showcase different cognitive states
    tasks = [
        {
            "prompt": "Analyze the structure of this Python project and summarize its main components.",
            "expected_state": CognitiveState.ANALYZING
        },
        {
            "prompt": "Create a plan for adding a new feature: real-time collaboration support.",
            "expected_state": CognitiveState.PLANNING
        },
        {
            "prompt": "Write a creative haiku about asynchronous programming.",
            "expected_state": CognitiveState.CREATING
        }
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*60}")
        print(f"Task {i}: {task['prompt'][:50]}...")
        print(f"Expected cognitive state: {task['expected_state'].value}")
        print('='*60)
        
        # Update visualizer state
        await visualizer.update_state(task['expected_state'])
        
        # Generate initial visualization
        print("\nGenerating pre-task visualization...")
        pre_image = await generator.generate_visualization(
            visualizer.cognitive_state,
            visualizer.self_image_state
        )
        if pre_image:
            print(f"Pre-task visualization saved: {pre_image}")
        
        # Run the Claude query
        print("\nQuerying Claude...")
        message_count = 0
        tool_uses = []
        
        async for message in query(prompt=task['prompt'], options=options):
            message_count += 1
            
            # Track tool usage for visualization
            if hasattr(message, 'content') and isinstance(message.content, list):
                for block in message.content:
                    if hasattr(block, 'name'):  # ToolUseBlock
                        tool_uses.append(block.name)
            
            # Update cognitive state based on activity
            if message_count % 3 == 0:  # Periodic updates
                if tool_uses:
                    if 'Grep' in tool_uses or 'Read' in tool_uses:
                        await visualizer.update_state(CognitiveState.ANALYZING)
                    elif 'Write' in tool_uses or 'Edit' in tool_uses:
                        await visualizer.update_state(CognitiveState.EXECUTING)
        
        # Generate post-task visualization
        print(f"\nTask completed. Tool uses: {set(tool_uses)}")
        print("Generating post-task visualization...")
        
        # Simulate reflection state after task
        await visualizer.update_state(CognitiveState.REFLECTING)
        post_image = await generator.generate_visualization(
            visualizer.cognitive_state,
            visualizer.self_image_state
        )
        if post_image:
            print(f"Post-task visualization saved: {post_image}")
        
        # Brief pause before next task
        await asyncio.sleep(2)
    
    # Generate final summary visualization
    print("\n" + "="*60)
    print("Generating final state summary...")
    await visualizer.update_state(CognitiveState.SYNTHESIZING)
    
    # Create a summary with enhanced parameters
    visualizer.self_image_state['coherence'] = 0.95
    visualizer.self_image_state['complexity'] = 0.8
    
    final_image = await generator.generate_visualization(
        visualizer.cognitive_state,
        visualizer.self_image_state
    )
    if final_image:
        print(f"Final synthesis visualization: {final_image}")
    
    print("\nVisualization session complete!")
    print(f"Images saved to: {generator.output_path}")


async def monitor_claude_with_visualizer():
    """Monitor Claude's state changes during a complex task."""
    
    visualizer = ClaudeVisualizer()
    generator = VisualizationGenerator()
    
    print("Starting Claude monitoring with visualizer...")
    
    # Complex task that will trigger multiple state changes
    complex_prompt = """
    Please analyze the claude_code_sdk codebase and:
    1. List all the main components
    2. Identify the key design patterns used
    3. Suggest three improvements
    4. Create a simple example that demonstrates the SDK's capabilities
    """
    
    options = ClaudeCodeOptions(
        allowed_tools=["Read", "Grep", "Write"],
        max_thinking_tokens=10000,
    )
    
    # Background task to generate periodic visualizations
    async def visualization_loop():
        states_seen = []
        while True:
            current_state = visualizer.cognitive_state
            if current_state not in states_seen:
                states_seen.append(current_state)
                print(f"\nNew state detected: {current_state.value}")
                image = await generator.generate_visualization(
                    current_state,
                    visualizer.self_image_state
                )
                if image:
                    print(f"Visualization saved: {image}")
            await asyncio.sleep(2)
    
    # Start visualization loop in background
    viz_task = asyncio.create_task(visualization_loop())
    
    try:
        # Simulate state changes based on Claude's activity
        state_sequence = [
            CognitiveState.IDLE,
            CognitiveState.THINKING,
            CognitiveState.ANALYZING,
            CognitiveState.PLANNING,
            CognitiveState.CREATING,
            CognitiveState.EXECUTING,
            CognitiveState.REFLECTING,
            CognitiveState.SYNTHESIZING
        ]
        
        state_index = 0
        async for message in query(prompt=complex_prompt, options=options):
            # Progress through states as Claude works
            if state_index < len(state_sequence):
                await visualizer.update_state(state_sequence[state_index])
                state_index += 1
                
            # Adjust self-image parameters based on activity
            if hasattr(message, 'content'):
                visualizer.self_image_state['analytical_depth'] = min(1.0, 
                    visualizer.self_image_state['analytical_depth'] + 0.05)
        
        # Final state
        await visualizer.update_state(CognitiveState.IDLE)
        await asyncio.sleep(3)  # Allow final visualization
        
    finally:
        viz_task.cancel()
        try:
            await viz_task
        except asyncio.CancelledError:
            pass
    
    print("\nMonitoring complete!")


if __name__ == "__main__":
    print("Claude SDK Visualizer Integration Examples\n")
    print("1. Interactive session with task-based visualizations")
    print("2. Real-time monitoring with state tracking")
    
    choice = input("\nSelect example (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(visualize_claude_interaction())
    elif choice == "2":
        asyncio.run(monitor_claude_with_visualizer())
    else:
        print("Invalid choice. Please run again and select 1 or 2.")