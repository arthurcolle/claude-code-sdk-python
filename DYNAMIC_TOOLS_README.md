# Dynamic Tools Framework

A comprehensive Python framework demonstrating dynamic tool creation, meta-recursive analysis, and multi-turn rollout systems with componentized, mutable modules.

## Overview

This framework showcases advanced programming concepts:

1. **Dynamic Tool Creation** - Create tools at runtime from specifications
2. **Dynamic Tool Kit Creation** - Compose tools into reusable kits
3. **Meta-Recursive Analysis** - Code that analyzes and modifies itself
4. **Multi-Turn Rollouts** - Encapsulated execution frames as function calls
5. **Component-Based Architecture** - Everything is a viewable, mutable module

## Core Components

### 1. Dynamic Tool Factory (`dynamic_tools_framework.py`)

```python
# Create tools dynamically at runtime
factory = DynamicToolFactory()
signature = ToolSignature(
    name="Calculator",
    parameters={'a': float, 'b': float},
    returns=float,
    description="Dynamic calculator"
)
factory.create_tool(signature)
```

### 2. Self-Modifying Tools

Tools that can modify their own behavior:

```python
self_mod_tool = SelfModifyingTool("Transformer", "Self-modifying tool")
# Modify behavior at runtime
self_mod_tool.modify_behavior(new_code)
```

### 3. Meta-Recursive Analyzer

Analyzes code including its own:

```python
analyzer = MetaRecursiveAnalyzer()
# Analyzer analyzes itself
self_analysis = analyzer.analyze_self_recursively()
```

### 4. Multi-Turn Rollout Engine

Manages execution as encapsulated frames:

```python
rollout_engine = MultiTurnRolloutEngine()
await rollout_engine.start_rollout("demo", context)
await rollout_engine.add_turn(action="compute", context={...})
summary = rollout_engine.complete_rollout()
```

### 5. Component Registry

All code is componentized and mutable:

```python
registry = ComponentRegistry()
component = CodeComponent(name="sample", source=code)
registry.register(component)
# Evolve components with mutations
evolved = registry.evolve("sample", mutation_fn)
```

## Advanced Features

### Frame Orchestration

```python
orchestrator = FrameOrchestrator()
orchestrator.register_frame_template('computation', {...})
frame = await orchestrator.create_frame('computation', data=...)
```

### Recursive Introspection

```python
introspector = RecursiveIntrospector()
# Introspect any object deeply
analysis = introspector.introspect(obj)
# Introspector introspects itself
meta = introspector.introspect_self()
```

### Code Generation

```python
generator = MetaCodeGenerator()
tool_spec = {
    'class_name': 'DataProcessor',
    'name': 'processor',
    'execute_code': '...'
}
component = generator.generate_tool(tool_spec)
```

## Usage Examples

### Basic Example

```python
# Run the simple demonstration
python simple_demo.py
```

Output:
```
=== Dynamic Tools Framework Demonstration ===

1. Dynamic Tool Creation
----------------------------------------
Calculator result: 10 + 5 = 15.0
Tool created: DynamicCalculator

2. Tool Kit Composition
----------------------------------------
Toolkit execution: 20 * 4 = 80.0

3. Meta-Recursive Analysis
----------------------------------------
Code complexity: 2
Analyzer self-analysis depth: 2

4. Multi-Turn Rollout Frames
----------------------------------------
Rollout completed with 4 frames
```

### Advanced Example

```python
from dynamic_tools_framework import *
from advanced_meta_recursive import *

# Create a self-analyzing, self-modifying pipeline
modifier = SelfAnalyzingModifier()

# First modification
mod1 = modifier.modify_and_analyze('''
async def behavior(x: int) -> int:
    # Add caching
    if not hasattr(self, '_cache'):
        self._cache = {}
    if x in self._cache:
        return self._cache[x]
    result = x * 3
    self._cache[x] = result
    return result
''')

# Analyze modification patterns
patterns = modifier._analyze_modification_patterns()
```

## Key Concepts Demonstrated

1. **Dynamic Tool Creation**: Tools are created from specifications at runtime
2. **Tool Kit Composition**: Tools are composed into kits for organized execution
3. **Meta-Recursive Analysis**: Code analyzes its own structure and behavior
4. **Self-Modification**: Components can modify their own behavior based on performance
5. **Frame-Based Execution**: Multi-turn operations are encapsulated as frames
6. **Component Evolution**: Code components can be mutated and evolved
7. **Deep Introspection**: Recursive analysis of object structures
8. **Execution Trees**: Hierarchical tracking of execution patterns

## Architecture

```
┌─────────────────────────────────────────┐
│         Dynamic Tool Factory            │
│  ┌─────────────┐    ┌────────────────┐ │
│  │ Tool Specs  │───▶│ Runtime Tools  │ │
│  └─────────────┘    └────────────────┘ │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│            Tool Kit                     │
│  ┌─────────────┐    ┌────────────────┐ │
│  │   Tools     │───▶│ Execution Tree │ │
│  └─────────────┘    └────────────────┘ │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│       Multi-Turn Rollout Engine         │
│  ┌─────────────┐    ┌────────────────┐ │
│  │   Frames    │───▶│ Rollout Tree   │ │
│  └─────────────┘    └────────────────┘ │
└─────────────────────────────────────────┘
```

## Files

- `dynamic_tools_framework.py` - Core framework with dynamic tool creation
- `advanced_meta_recursive.py` - Advanced meta-programming capabilities
- `interactive_demo.py` - Full interactive demonstration (complex)
- `simple_demo.py` - Simple working demonstration
- `simple_demo_results.json` - Output from simple demo

## Requirements

- Python 3.10+
- Standard library only (no external dependencies)

## Future Extensions

1. **Distributed Execution**: Frames could be executed across multiple processes
2. **Persistence**: Save and load tool definitions and execution states
3. **Visual Editor**: GUI for creating and modifying tools dynamically
4. **Performance Optimization**: Automatic optimization based on execution patterns
5. **Type System**: Advanced type checking and validation for dynamic tools