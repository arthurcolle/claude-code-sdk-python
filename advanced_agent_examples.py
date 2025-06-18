"""
Advanced Multi-turn Agent Examples
==================================
Demonstrates advanced usage patterns and custom tools.
"""

import asyncio
import json
import os
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any
from multi_turn_agent import tools, MultiTurnAgent, Environment


# ————————————————————————————————————————————————————————————————
# Advanced Tools
# ————————————————————————————————————————————————————————————————

@tools.register(description="Fetch weather data for a location")
async def get_weather(location: str) -> dict:
    """Get current weather for a location (mock implementation)."""
    # In a real implementation, this would call a weather API
    # For demo purposes, return mock data
    import random
    
    weather_conditions = ["sunny", "cloudy", "rainy", "partly cloudy"]
    temp = round(random.uniform(50, 85), 1)
    
    return {
        "location": location,
        "temperature": f"{temp}°F",
        "conditions": random.choice(weather_conditions),
        "humidity": f"{random.randint(40, 80)}%",
        "wind": f"{random.randint(5, 20)} mph",
        "forecast": "Improving throughout the day"
    }


@tools.register(description="Create and manage TODO items")
class TodoManager:
    """Simple TODO list manager."""
    
    todos: List[Dict[str, Any]] = []
    
    @staticmethod
    def add_todo(title: str, priority: str = "medium") -> dict:
        """Add a new TODO item."""
        todo = {
            "id": len(TodoManager.todos) + 1,
            "title": title,
            "priority": priority,
            "completed": False,
            "created_at": datetime.now().isoformat()
        }
        TodoManager.todos.append(todo)
        return {"success": True, "todo": todo}
    
    @staticmethod
    def list_todos(only_incomplete: bool = False) -> list:
        """List all TODO items."""
        todos = TodoManager.todos
        if only_incomplete:
            todos = [t for t in todos if not t["completed"]]
        return todos
    
    @staticmethod
    def complete_todo(todo_id: int) -> dict:
        """Mark a TODO as completed."""
        for todo in TodoManager.todos:
            if todo["id"] == todo_id:
                todo["completed"] = True
                todo["completed_at"] = datetime.now().isoformat()
                return {"success": True, "todo": todo}
        return {"success": False, "error": f"TODO {todo_id} not found"}


# Register TODO methods as individual tools
tools.register(
    TodoManager.add_todo,
    name="add_todo",
    description="Add a new TODO item"
)
tools.register(
    TodoManager.list_todos,
    name="list_todos",
    description="List TODO items"
)
tools.register(
    TodoManager.complete_todo,
    name="complete_todo",
    description="Mark a TODO item as completed"
)


@tools.register(description="Run Python code in a sandboxed environment")
def run_python_code(code: str) -> str:
    """Execute Python code and return the result."""
    # Create a restricted execution environment
    safe_globals = {
        "__builtins__": {
            "print": print,
            "len": len,
            "range": range,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
        }
    }
    
    # Capture output
    import io
    import sys
    
    old_stdout = sys.stdout
    sys.stdout = output_buffer = io.StringIO()
    
    try:
        exec(code, safe_globals)
        result = output_buffer.getvalue()
        return result if result else "Code executed successfully (no output)"
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        sys.stdout = old_stdout


@tools.register(description="Search for information in a knowledge base")
async def search_knowledge(query: str, category: str = "general") -> list:
    """Search a mock knowledge base."""
    # Mock knowledge base
    knowledge_base = {
        "general": [
            {"title": "Python Best Practices", "content": "Use type hints, write tests, follow PEP 8"},
            {"title": "AI Safety", "content": "Ensure AI systems are aligned with human values"},
        ],
        "technical": [
            {"title": "Async Programming", "content": "Use asyncio for concurrent operations"},
            {"title": "Database Design", "content": "Normalize data, use indexes wisely"},
        ],
        "science": [
            {"title": "Climate Change", "content": "Global warming caused by greenhouse gases"},
            {"title": "Quantum Computing", "content": "Uses quantum mechanics for computation"},
        ]
    }
    
    results = []
    search_terms = query.lower().split()
    
    for item in knowledge_base.get(category, []):
        if any(term in item["title"].lower() or term in item["content"].lower() 
               for term in search_terms):
            results.append(item)
    
    return results


@tools.register(description="Generate data visualizations")
def create_chart(data: list, chart_type: str = "bar", title: str = "Chart") -> str:
    """Create a simple ASCII chart."""
    if chart_type == "bar":
        max_value = max(data) if data else 0
        scale = 40 / max_value if max_value > 0 else 1
        
        chart = f"\n{title}\n" + "=" * (len(title) + 10) + "\n"
        for i, value in enumerate(data):
            bar_length = int(value * scale)
            chart += f"{i+1}: {'█' * bar_length} {value}\n"
        
        return chart
    else:
        return f"Chart type '{chart_type}' not supported. Use 'bar'."


# ————————————————————————————————————————————————————————————————
# Advanced Agent Configurations
# ————————————————————————————————————————————————————————————————

class ResearchAgent(MultiTurnAgent):
    """Specialized agent for research tasks."""
    
    def __init__(self):
        super().__init__(
            system_prompt="""You are a research assistant with access to various tools.
            Your goal is to help users find information, analyze data, and manage tasks.
            Always cite your sources when using the search tool.
            Be thorough but concise in your responses.""",
            stream=True
        )


class CodeAssistant(MultiTurnAgent):
    """Specialized agent for coding tasks."""
    
    def __init__(self):
        super().__init__(
            system_prompt="""You are a coding assistant with Python execution capabilities.
            Help users write, debug, and understand code.
            Always test code before providing it to ensure it works.
            Explain your code clearly and suggest improvements.""",
            stream=True
        )


class PersonalAssistant(MultiTurnAgent):
    """Personal assistant for daily tasks."""
    
    def __init__(self):
        super().__init__(
            system_prompt="""You are a helpful personal assistant.
            Help users manage their tasks, check weather, and stay organized.
            Be proactive in suggesting ways to improve productivity.
            Keep responses friendly and conversational.""",
            stream=True
        )


# ————————————————————————————————————————————————————————————————
# Advanced Demos
# ————————————————————————————————————————————————————————————————

async def research_demo():
    """Demo research capabilities."""
    print("Research Assistant Demo")
    print("=" * 50)
    
    agent = ResearchAgent()
    
    queries = [
        "Search for information about Python best practices",
        "What did you find about Python? Can you also search for async programming?",
        "Create a bar chart showing the popularity scores: [85, 92, 78, 88, 95]",
        "Remember that our project focus is on async patterns",
        "What was our project focus again?",
    ]
    
    for query in queries:
        print(f"\nUser: {query}")
        print("Assistant: ", end="")
        await agent.send_user(query)
        print()


async def coding_demo():
    """Demo coding assistant capabilities."""
    print("Coding Assistant Demo")
    print("=" * 50)
    
    agent = CodeAssistant()
    
    tasks = [
        "Write a Python function to calculate fibonacci numbers",
        "Run this code: print([i**2 for i in range(10)])",
        "Can you create a function that checks if a number is prime and test it?",
        "What's 2**10 + 3**5?",
    ]
    
    for task in tasks:
        print(f"\nUser: {task}")
        print("Assistant: ", end="")
        await agent.send_user(task)
        print()


async def personal_assistant_demo():
    """Demo personal assistant capabilities."""
    print("Personal Assistant Demo")
    print("=" * 50)
    
    agent = PersonalAssistant()
    
    interactions = [
        "What's the weather like in San Francisco?",
        "Add a TODO: Review pull requests with high priority",
        "Add another TODO: Prepare for team meeting",
        "Show me my TODO list",
        "I've completed the first task",
        "Show me remaining tasks",
    ]
    
    for interaction in interactions:
        print(f"\nUser: {interaction}")
        print("Assistant: ", end="")
        await agent.send_user(interaction)
        print()


async def multi_agent_collaboration():
    """Demo multiple agents working together."""
    print("Multi-Agent Collaboration Demo")
    print("=" * 50)
    
    # Create different specialized agents
    researcher = ResearchAgent()
    coder = CodeAssistant()
    personal = PersonalAssistant()
    
    # Simulate a complex task requiring multiple agents
    print("\n--- Personal Assistant Planning ---")
    print("User: I need to learn about async programming for my project")
    print("Personal Assistant: ", end="")
    await personal.send_user(
        "I need to learn about async programming for my project. "
        "Can you help me plan this?"
    )
    
    print("\n\n--- Research Assistant Gathering Info ---")
    print("User: Find information about async programming")
    print("Research Assistant: ", end="")
    await researcher.send_user("Search for async programming in technical category")
    
    print("\n\n--- Coding Assistant Implementation ---")
    print("User: Show me a simple async example")
    print("Coding Assistant: ", end="")
    await coder.send_user(
        "Create a simple Python async function example that demonstrates "
        "basic asyncio usage"
    )


async def stateful_conversation_demo():
    """Demo complex stateful conversations."""
    print("Stateful Conversation Demo")
    print("=" * 50)
    
    agent = MultiTurnAgent(
        system_prompt="You are a helpful assistant with perfect memory. "
                     "Track context across the conversation and refer back to "
                     "previous topics when relevant."
    )
    
    # Simulate a complex multi-turn conversation
    conversation = [
        "My name is Alice and I'm working on a machine learning project",
        "I need to process data from 3 sources: CSV files, APIs, and databases",
        "Let's start with CSV files. Remember the filename is 'sales_2024.csv'",
        "Calculate the size in MB if the file has 1 million rows with 20 columns, "
        "assuming 50 bytes per cell on average",
        "What was my name and what file are we working with?",
        "Add a TODO to implement CSV parser for the file we discussed",
        "Now let's move to the API source. The endpoint is https://api.example.com/data",
        "What are all the data sources I mentioned?",
        "Show me all the TODOs",
    ]
    
    for message in conversation:
        print(f"\nUser: {message}")
        print("Assistant: ", end="")
        await agent.send_user(message)
        print()
    
    # Show conversation summary
    print("\n--- Conversation Memory ---")
    history = await agent.get_history()
    print(f"Total messages in history: {len(history)}")
    print(f"System messages: {sum(1 for m in history if m.get('role') == 'system')}")
    print(f"User messages: {sum(1 for m in history if m.get('role') == 'user')}")
    print(f"Assistant messages: {sum(1 for m in history if m.get('role') == 'assistant')}")
    print(f"Tool calls: {sum(1 for m in history if m.get('role') == 'tool')}")


# ————————————————————————————————————————————————————————————————
# Main Demo Runner
# ————————————————————————————————————————————————————————————————

async def run_all_demos():
    """Run all demonstration scenarios."""
    demos = [
        ("Research Assistant", research_demo),
        ("Coding Assistant", coding_demo),
        ("Personal Assistant", personal_assistant_demo),
        ("Multi-Agent Collaboration", multi_agent_collaboration),
        ("Stateful Conversation", stateful_conversation_demo),
    ]
    
    for name, demo_func in demos:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        await demo_func()
        
        # Pause between demos
        print("\nPress Enter to continue to next demo...")
        input()


if __name__ == "__main__":
    # Run individual demos or all demos
    
    # Run all demos
    # asyncio.run(run_all_demos())
    
    # Or run specific demos:
    asyncio.run(personal_assistant_demo())
    # asyncio.run(coding_demo())
    # asyncio.run(research_demo())
    # asyncio.run(stateful_conversation_demo())
    # asyncio.run(multi_agent_collaboration())