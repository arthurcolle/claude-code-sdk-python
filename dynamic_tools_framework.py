#!/usr/bin/env python3
"""
Dynamic Tool Creation Framework

This module demonstrates:
1. Dynamic tool creation at runtime
2. Tool kit composition and management
3. Meta-recursive analysis of code chunks
4. Multi-turn rollouts as encapsulated frames
"""

from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import inspect
import ast
import textwrap
from functools import wraps
import asyncio
from collections import defaultdict
import json


T = TypeVar('T')


@dataclass
class ToolSignature:
    """Represents a tool's signature for dynamic creation"""
    name: str
    parameters: Dict[str, Type]
    returns: Type
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionFrame:
    """Represents an execution frame for multi-turn rollouts"""
    frame_id: str
    parent_frame: Optional['ExecutionFrame'] = None
    children: List['ExecutionFrame'] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    results: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """Base class for all dynamically created tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._execution_history: List[Dict[str, Any]] = []
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters"""
        pass
    
    def analyze_self(self) -> Dict[str, Any]:
        """Meta-recursive analysis of the tool's own code"""
        source = inspect.getsource(self.__class__)
        tree = ast.parse(source)
        
        analysis = {
            'name': self.name,
            'description': self.description,
            'methods': [],
            'attributes': [],
            'execution_count': len(self._execution_history),
            'ast_nodes': self._analyze_ast(tree)
        }
        
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if not name.startswith('_'):
                analysis['methods'].append({
                    'name': name,
                    'signature': str(inspect.signature(method)),
                    'doc': inspect.getdoc(method)
                })
        
        return analysis
    
    def _analyze_ast(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze AST structure recursively"""
        node_info = {
            'type': tree.__class__.__name__,
            'children': []
        }
        
        for child in ast.iter_child_nodes(tree):
            node_info['children'].append(self._analyze_ast(child))
        
        return node_info


class DynamicToolFactory:
    """Factory for creating tools dynamically at runtime"""
    
    def __init__(self):
        self._tool_registry: Dict[str, Type[BaseTool]] = {}
        self._tool_instances: Dict[str, BaseTool] = {}
    
    def create_tool(self, signature: ToolSignature) -> Type[BaseTool]:
        """Dynamically create a new tool class"""
        
        # Create execute method
        async def execute(self, **kwargs):
            # Validate parameters
            for param, param_type in signature.parameters.items():
                if param in kwargs:
                    if not isinstance(kwargs[param], param_type):
                        raise TypeError(f"Parameter {param} must be of type {param_type}")
            
            # Record execution
            self._execution_history.append({
                'timestamp': asyncio.get_event_loop().time(),
                'parameters': kwargs,
                'signature': signature.name
            })
            
            # Execute custom logic if provided
            if 'execute_fn' in signature.metadata:
                return await signature.metadata['execute_fn'](**kwargs)
            
            return f"Executed {signature.name} with {kwargs}"
        
        # Create tool class dynamically
        tool_class = type(
            f"{signature.name}Tool",
            (BaseTool,),
            {
                'execute': execute,
                '__module__': __name__,
                '__doc__': signature.description
            }
        )
        
        self._tool_registry[signature.name] = tool_class
        return tool_class
    
    def instantiate_tool(self, name: str) -> BaseTool:
        """Instantiate a registered tool"""
        if name not in self._tool_registry:
            raise ValueError(f"Tool {name} not found in registry")
        
        if name not in self._tool_instances:
            tool_class = self._tool_registry[name]
            self._tool_instances[name] = tool_class(
                name=name,
                description=tool_class.__doc__ or ""
            )
        
        return self._tool_instances[name]


class ToolKit:
    """Composable toolkit that can contain multiple tools"""
    
    def __init__(self, name: str):
        self.name = name
        self.tools: Dict[str, BaseTool] = {}
        self._execution_frames: List[ExecutionFrame] = []
        self._current_frame: Optional[ExecutionFrame] = None
    
    def add_tool(self, tool: BaseTool):
        """Add a tool to the toolkit"""
        self.tools[tool.name] = tool
    
    def remove_tool(self, tool_name: str):
        """Remove a tool from the toolkit"""
        if tool_name in self.tools:
            del self.tools[tool_name]
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool within the toolkit context"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found in toolkit")
        
        tool = self.tools[tool_name]
        
        # Create execution frame
        frame = ExecutionFrame(
            frame_id=f"{self.name}_{tool_name}_{len(self._execution_frames)}",
            parent_frame=self._current_frame,
            context={'tool_name': tool_name, 'parameters': kwargs}
        )
        
        if self._current_frame:
            self._current_frame.children.append(frame)
        
        self._execution_frames.append(frame)
        
        # Execute in frame context
        previous_frame = self._current_frame
        self._current_frame = frame
        
        try:
            result = await tool.execute(**kwargs)
            frame.results.append(result)
            return result
        finally:
            self._current_frame = previous_frame
    
    def analyze_execution_tree(self) -> Dict[str, Any]:
        """Analyze the execution tree of frames"""
        def frame_to_dict(frame: ExecutionFrame) -> Dict[str, Any]:
            return {
                'frame_id': frame.frame_id,
                'context': frame.context,
                'results': frame.results,
                'children': [frame_to_dict(child) for child in frame.children]
            }
        
        root_frames = [f for f in self._execution_frames if f.parent_frame is None]
        return {
            'toolkit': self.name,
            'total_executions': len(self._execution_frames),
            'execution_tree': [frame_to_dict(frame) for frame in root_frames]
        }


class MetaRecursiveAnalyzer:
    """Analyzer that can recursively analyze its own code and modifications"""
    
    def __init__(self):
        self._analysis_history: List[Dict[str, Any]] = []
        self._code_chunks: Dict[str, str] = {}
    
    def analyze_chunk(self, chunk_name: str, code: str) -> Dict[str, Any]:
        """Analyze a code chunk and store it"""
        self._code_chunks[chunk_name] = code
        
        try:
            tree = ast.parse(code)
            analysis = {
                'chunk_name': chunk_name,
                'ast_analysis': self._deep_ast_analysis(tree),
                'complexity': self._calculate_complexity(tree),
                'dependencies': self._extract_dependencies(tree),
                'mutations_possible': self._identify_mutations(tree)
            }
            
            self._analysis_history.append(analysis)
            return analysis
            
        except SyntaxError as e:
            return {'error': str(e), 'chunk_name': chunk_name}
    
    def _deep_ast_analysis(self, node: ast.AST, depth: int = 0) -> Dict[str, Any]:
        """Perform deep AST analysis with pattern recognition"""
        analysis = {
            'node_type': node.__class__.__name__,
            'depth': depth,
            'attributes': {},
            'children': []
        }
        
        # Extract node attributes
        for attr in node._fields:
            if hasattr(node, attr):
                value = getattr(node, attr)
                if isinstance(value, (str, int, float, bool, type(None))):
                    analysis['attributes'][attr] = value
        
        # Analyze children
        for child in ast.iter_child_nodes(node):
            analysis['children'].append(self._deep_ast_analysis(child, depth + 1))
        
        return analysis
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity
    
    def _extract_dependencies(self, tree: ast.AST) -> List[str]:
        """Extract dependencies from imports"""
        dependencies = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                dependencies.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                dependencies.append(module)
        return list(set(dependencies))
    
    def _identify_mutations(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Identify possible code mutations"""
        mutations = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                mutations.append({
                    'type': 'binary_operation',
                    'operator': node.op.__class__.__name__,
                    'mutations': ['Add', 'Sub', 'Mult', 'Div']
                })
            elif isinstance(node, ast.Compare):
                mutations.append({
                    'type': 'comparison',
                    'operators': [op.__class__.__name__ for op in node.ops],
                    'mutations': ['Lt', 'Gt', 'Eq', 'NotEq']
                })
        
        return mutations
    
    def analyze_self_recursively(self) -> Dict[str, Any]:
        """Recursively analyze the analyzer's own code"""
        own_source = inspect.getsource(self.__class__)
        self_analysis = self.analyze_chunk('MetaRecursiveAnalyzer', own_source)
        
        # Analyze the analysis method itself
        analyze_method_source = inspect.getsource(self.analyze_self_recursively)
        method_analysis = self.analyze_chunk('analyze_self_recursively', analyze_method_source)
        
        return {
            'class_analysis': self_analysis,
            'method_analysis': method_analysis,
            'recursion_depth': len(self._analysis_history),
            'total_chunks_analyzed': len(self._code_chunks)
        }


class MultiTurnRolloutEngine:
    """Engine for managing multi-turn rollouts as encapsulated frames"""
    
    def __init__(self):
        self._rollouts: Dict[str, List[ExecutionFrame]] = defaultdict(list)
        self._active_rollout: Optional[str] = None
        self._frame_factory = self._create_frame_factory()
    
    def _create_frame_factory(self) -> Callable:
        """Create a factory for generating execution frames"""
        frame_counter = 0
        
        def create_frame(rollout_id: str, action: str, context: Dict[str, Any]) -> ExecutionFrame:
            nonlocal frame_counter
            frame_counter += 1
            
            return ExecutionFrame(
                frame_id=f"{rollout_id}_frame_{frame_counter}",
                context={
                    'action': action,
                    'rollout_id': rollout_id,
                    **context
                }
            )
        
        return create_frame
    
    async def start_rollout(self, rollout_id: str, initial_context: Dict[str, Any]):
        """Start a new rollout"""
        self._active_rollout = rollout_id
        
        initial_frame = self._frame_factory(
            rollout_id=rollout_id,
            action='initialize',
            context=initial_context
        )
        
        self._rollouts[rollout_id].append(initial_frame)
        return initial_frame
    
    async def add_turn(self, action: str, context: Dict[str, Any], 
                       tool_kit: Optional[ToolKit] = None) -> ExecutionFrame:
        """Add a turn to the active rollout"""
        if not self._active_rollout:
            raise ValueError("No active rollout")
        
        rollout_id = self._active_rollout
        parent_frame = self._rollouts[rollout_id][-1] if self._rollouts[rollout_id] else None
        
        frame = self._frame_factory(
            rollout_id=rollout_id,
            action=action,
            context=context
        )
        
        if parent_frame:
            frame.parent_frame = parent_frame
            parent_frame.children.append(frame)
        
        # Execute tools if provided
        if tool_kit and 'tool_executions' in context:
            for tool_exec in context['tool_executions']:
                tool_name = tool_exec['tool']
                tool_params = tool_exec['params']
                
                result = await tool_kit.execute_tool(tool_name, **tool_params)
                frame.results.append({
                    'tool': tool_name,
                    'params': tool_params,
                    'result': result
                })
        
        self._rollouts[rollout_id].append(frame)
        return frame
    
    def complete_rollout(self) -> Dict[str, Any]:
        """Complete the active rollout and return summary"""
        if not self._active_rollout:
            raise ValueError("No active rollout")
        
        rollout_id = self._active_rollout
        frames = self._rollouts[rollout_id]
        
        summary = {
            'rollout_id': rollout_id,
            'total_frames': len(frames),
            'frame_tree': self._build_frame_tree(frames[0]) if frames else {},
            'all_results': [
                result 
                for frame in frames 
                for result in frame.results
            ]
        }
        
        self._active_rollout = None
        return summary
    
    def _build_frame_tree(self, root_frame: ExecutionFrame) -> Dict[str, Any]:
        """Build a tree representation of frames"""
        return {
            'frame_id': root_frame.frame_id,
            'action': root_frame.context.get('action', 'unknown'),
            'results': root_frame.results,
            'children': [
                self._build_frame_tree(child) 
                for child in root_frame.children
            ]
        }


# Example: Creating a self-modifying tool
class SelfModifyingTool(BaseTool):
    """A tool that can modify its own behavior"""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self._behavior_code = """
async def behavior(self, x, y):
    return x + y
"""
        self._compile_behavior()
    
    def _compile_behavior(self):
        """Compile and inject the behavior"""
        local_vars = {}
        exec(self._behavior_code, globals(), local_vars)
        self.behavior = local_vars['behavior'].__get__(self, self.__class__)
    
    async def execute(self, **kwargs) -> Any:
        """Execute using the current behavior"""
        return await self.behavior(**kwargs)
    
    def modify_behavior(self, new_code: str):
        """Modify the tool's behavior dynamically"""
        self._behavior_code = new_code
        self._compile_behavior()
        
        # Analyze the modification
        analyzer = MetaRecursiveAnalyzer()
        return analyzer.analyze_chunk(f"{self.name}_behavior", new_code)


# Demonstration function
async def demonstrate_dynamic_tools():
    """Demonstrate all the dynamic capabilities"""
    
    # 1. Create dynamic tools
    factory = DynamicToolFactory()
    
    calculator_sig = ToolSignature(
        name="Calculator",
        parameters={'a': float, 'b': float, 'operation': str},
        returns=float,
        description="Dynamic calculator tool"
    )
    
    async def calc_execute(a: float, b: float, operation: str) -> float:
        operations = {
            'add': lambda x, y: x + y,
            'subtract': lambda x, y: x - y,
            'multiply': lambda x, y: x * y,
            'divide': lambda x, y: x / y if y != 0 else float('inf')
        }
        return operations.get(operation, lambda x, y: 0)(a, b)
    
    calculator_sig.metadata['execute_fn'] = calc_execute
    
    # Create and instantiate tool
    factory.create_tool(calculator_sig)
    calc_tool = factory.instantiate_tool("Calculator")
    
    # 2. Create a toolkit
    toolkit = ToolKit("MathToolkit")
    toolkit.add_tool(calc_tool)
    
    # 3. Create self-modifying tool
    self_mod_tool = SelfModifyingTool("Transformer", "Self-modifying transformation tool")
    toolkit.add_tool(self_mod_tool)
    
    # 4. Set up rollout engine
    rollout_engine = MultiTurnRolloutEngine()
    
    # 5. Start a multi-turn rollout
    await rollout_engine.start_rollout(
        "demo_rollout",
        {'purpose': 'demonstrate dynamic tools'}
    )
    
    # Turn 1: Basic calculation
    await rollout_engine.add_turn(
        action="calculate",
        context={
            'tool_executions': [
                {'tool': 'Calculator', 'params': {'a': 10, 'b': 5, 'operation': 'add'}}
            ]
        },
        tool_kit=toolkit
    )
    
    # Turn 2: Modify the self-modifying tool
    new_behavior = """
async def behavior(self, x, y):
    # Modified to multiply instead of add
    return x * y * 2
"""
    modification_analysis = self_mod_tool.modify_behavior(new_behavior)
    
    await rollout_engine.add_turn(
        action="modify_tool",
        context={
            'modification': 'Changed behavior to multiply and double',
            'analysis': modification_analysis
        }
    )
    
    # Turn 3: Use modified tool
    await rollout_engine.add_turn(
        action="use_modified",
        context={
            'tool_executions': [
                {'tool': 'Transformer', 'params': {'x': 3, 'y': 4}}
            ]
        },
        tool_kit=toolkit
    )
    
    # Complete rollout
    rollout_summary = rollout_engine.complete_rollout()
    
    # 6. Perform meta-recursive analysis
    analyzer = MetaRecursiveAnalyzer()
    self_analysis = analyzer.analyze_self_recursively()
    
    # 7. Analyze toolkit execution tree
    toolkit_analysis = toolkit.analyze_execution_tree()
    
    return {
        'rollout_summary': rollout_summary,
        'self_analysis': self_analysis,
        'toolkit_analysis': toolkit_analysis,
        'tool_introspection': calc_tool.analyze_self()
    }


if __name__ == "__main__":
    # Run demonstration
    results = asyncio.run(demonstrate_dynamic_tools())
    print(json.dumps(results, indent=2, default=str))