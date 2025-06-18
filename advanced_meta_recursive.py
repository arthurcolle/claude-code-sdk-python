#!/usr/bin/env python3
"""
Advanced Meta-Recursive Analysis and Code Generation

This module extends the framework with:
1. Deep introspection capabilities
2. Code generation from specifications
3. Runtime mutation and evolution
4. Component-based architecture
"""

import ast
import inspect
import types
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from dataclasses import dataclass, field
import textwrap
import asyncio
from abc import ABC, abstractmethod
import json
import hashlib
import copy

# Import ExecutionFrame from the base framework
from dynamic_tools_framework import ExecutionFrame, MetaRecursiveAnalyzer


@dataclass
class CodeComponent:
    """Represents a mutable, viewable code component"""
    name: str
    source: str
    ast_tree: Optional[ast.AST] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.source and not self.ast_tree:
            self.ast_tree = ast.parse(self.source)
    
    def get_hash(self) -> str:
        """Get hash of the component for tracking changes"""
        return hashlib.md5(self.source.encode()).hexdigest()
    
    def mutate(self, mutation_fn: Callable[[ast.AST], ast.AST]) -> 'CodeComponent':
        """Apply a mutation to the component"""
        if not self.ast_tree:
            self.ast_tree = ast.parse(self.source)
        
        mutated_tree = mutation_fn(copy.deepcopy(self.ast_tree))
        mutated_source = ast.unparse(mutated_tree)
        
        return CodeComponent(
            name=f"{self.name}_mutated",
            source=mutated_source,
            ast_tree=mutated_tree,
            dependencies=self.dependencies.copy(),
            metadata={**self.metadata, 'parent': self.name}
        )


class ComponentRegistry:
    """Registry for managing code components"""
    
    def __init__(self):
        self._components: Dict[str, CodeComponent] = {}
        self._evolution_history: List[Tuple[str, str, str]] = []  # (from, to, mutation_type)
    
    def register(self, component: CodeComponent):
        """Register a component"""
        self._components[component.name] = component
    
    def get(self, name: str) -> Optional[CodeComponent]:
        """Get a component by name"""
        return self._components.get(name)
    
    def evolve(self, component_name: str, mutation_fn: Callable, 
               new_name: Optional[str] = None) -> CodeComponent:
        """Evolve a component with a mutation"""
        component = self.get(component_name)
        if not component:
            raise ValueError(f"Component {component_name} not found")
        
        mutated = component.mutate(mutation_fn)
        if new_name:
            mutated.name = new_name
        
        self.register(mutated)
        self._evolution_history.append((component_name, mutated.name, "custom"))
        
        return mutated
    
    def get_lineage(self, component_name: str) -> List[str]:
        """Get the evolutionary lineage of a component"""
        lineage = [component_name]
        
        # Trace back through evolution history
        current = component_name
        for from_comp, to_comp, _ in reversed(self._evolution_history):
            if to_comp == current:
                lineage.insert(0, from_comp)
                current = from_comp
        
        return lineage


class MetaCodeGenerator:
    """Generates code from specifications using meta-programming"""
    
    def __init__(self):
        self._templates: Dict[str, str] = self._initialize_templates()
        self._generated_components: List[CodeComponent] = []
    
    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize code generation templates"""
        return {
            'tool': '''class {class_name}(BaseTool):
    """
    {description}
    """
    
    def __init__(self):
        super().__init__("{name}", "{description}")
        {init_code}
    
    async def execute(self, **kwargs) -> Any:
        {execute_code}
        
    def validate_inputs(self, **kwargs) -> bool:
        {validation_code}
        return True
''',
            'analyzer': '''class {class_name}Analyzer:
    """
    Analyzer for {target}
    """
    
    def analyze(self, data: Any) -> Dict[str, Any]:
        results = {{}}
        {analysis_code}
        return results
''',
            'frame_processor': '''async def process_{frame_type}_frame(frame: ExecutionFrame) -> Any:
    """
    Process {frame_type} frames
    """
    {processing_code}
    return frame
'''
        }
    
    def generate_tool(self, spec: Dict[str, Any]) -> CodeComponent:
        """Generate a tool from specification"""
        code = self._templates['tool'].format(
            class_name=spec['class_name'],
            name=spec['name'],
            description=spec['description'],
            init_code=spec.get('init_code', 'pass'),
            execute_code=spec.get('execute_code', 'return {}'),
            validation_code=spec.get('validation_code', 'pass')
        )
        
        component = CodeComponent(
            name=spec['name'],
            source=code,
            metadata={'type': 'generated_tool', 'spec': spec}
        )
        
        self._generated_components.append(component)
        return component
    
    def generate_analyzer(self, target: str, analysis_logic: str) -> CodeComponent:
        """Generate an analyzer for a specific target"""
        code = self._templates['analyzer'].format(
            class_name=target.capitalize(),
            target=target,
            analysis_code=analysis_logic
        )
        
        component = CodeComponent(
            name=f"{target}_analyzer",
            source=code,
            metadata={'type': 'generated_analyzer', 'target': target}
        )
        
        self._generated_components.append(component)
        return component
    
    def introspect_generation_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in generated code"""
        patterns = {
            'total_generated': len(self._generated_components),
            'by_type': {},
            'common_structures': [],
            'generation_graph': []
        }
        
        for component in self._generated_components:
            comp_type = component.metadata.get('type', 'unknown')
            patterns['by_type'][comp_type] = patterns['by_type'].get(comp_type, 0) + 1
            
            # Analyze AST patterns
            if component.ast_tree:
                for node in ast.walk(component.ast_tree):
                    if isinstance(node, ast.ClassDef):
                        patterns['common_structures'].append({
                            'type': 'class',
                            'name': node.name,
                            'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                        })
        
        return patterns


class RecursiveIntrospector:
    """Deep introspection with recursive self-analysis"""
    
    def __init__(self):
        self._introspection_depth = 0
        self._max_depth = 5
        self._introspection_cache: Dict[str, Any] = {}
    
    def introspect(self, obj: Any, depth: int = 0) -> Dict[str, Any]:
        """Recursively introspect an object"""
        if depth > self._max_depth:
            return {'max_depth_reached': True}
        
        obj_id = id(obj)
        if obj_id in self._introspection_cache:
            return {'cached_reference': obj_id}
        
        introspection = {
            'type': type(obj).__name__,
            'module': getattr(obj, '__module__', None),
            'attributes': {},
            'methods': {},
            'introspection_depth': depth
        }
        
        # Cache to prevent infinite recursion
        self._introspection_cache[obj_id] = introspection
        
        # Introspect attributes
        for attr_name in dir(obj):
            if not attr_name.startswith('_'):
                try:
                    attr_value = getattr(obj, attr_name)
                    
                    if callable(attr_value):
                        introspection['methods'][attr_name] = {
                            'signature': str(inspect.signature(attr_value)) if hasattr(attr_value, '__call__') else 'N/A',
                            'doc': inspect.getdoc(attr_value),
                            'is_async': inspect.iscoroutinefunction(attr_value)
                        }
                    else:
                        # Recursively introspect complex attributes
                        if hasattr(attr_value, '__dict__'):
                            introspection['attributes'][attr_name] = self.introspect(attr_value, depth + 1)
                        else:
                            introspection['attributes'][attr_name] = repr(attr_value)
                            
                except Exception as e:
                    introspection['attributes'][attr_name] = f"Error: {str(e)}"
        
        # Introspect source if available
        try:
            source = inspect.getsource(obj.__class__ if hasattr(obj, '__class__') else obj)
            introspection['source_analysis'] = self._analyze_source(source)
        except:
            pass
        
        return introspection
    
    def _analyze_source(self, source: str) -> Dict[str, Any]:
        """Analyze source code structure"""
        tree = ast.parse(source)
        
        analysis = {
            'lines': len(source.splitlines()),
            'classes': 0,
            'functions': 0,
            'async_functions': 0,
            'decorators': [],
            'imports': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                analysis['classes'] += 1
            elif isinstance(node, ast.FunctionDef):
                analysis['functions'] += 1
            elif isinstance(node, ast.AsyncFunctionDef):
                analysis['async_functions'] += 1
            elif isinstance(node, ast.Import):
                analysis['imports'].extend(alias.name for alias in node.names)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                # Track decorator usage
                parent = getattr(node, 'parent', None)
                if isinstance(parent, ast.Decorator):
                    analysis['decorators'].append(node.id)
        
        return analysis
    
    def introspect_self(self) -> Dict[str, Any]:
        """The introspector introspects itself"""
        self._introspection_depth += 1
        
        self_analysis = {
            'class_introspection': self.introspect(self),
            'method_introspection': {},
            'recursive_depth': self._introspection_depth
        }
        
        # Introspect our own introspection method
        introspect_method = self.introspect
        self_analysis['method_introspection']['introspect'] = {
            'source': inspect.getsource(introspect_method),
            'complexity': self._calculate_method_complexity(introspect_method)
        }
        
        # Introspect the introspection of introspection (meta-meta)
        if self._introspection_depth < 3:
            self_analysis['meta_introspection'] = self.introspect_self()
        
        self._introspection_depth -= 1
        return self_analysis
    
    def _calculate_method_complexity(self, method: Callable) -> int:
        """Calculate complexity of a method"""
        try:
            source = inspect.getsource(method)
            tree = ast.parse(source)
            
            complexity = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
                    complexity += 1
                    
            return complexity
        except:
            return -1


class FrameOrchestrator:
    """Orchestrates multi-turn execution frames with rich context"""
    
    def __init__(self):
        self._frame_templates: Dict[str, Dict[str, Any]] = {}
        self._frame_processors: Dict[str, Callable] = {}
        self._frame_stack: List[ExecutionFrame] = []
    
    def register_frame_template(self, frame_type: str, template: Dict[str, Any]):
        """Register a frame template"""
        self._frame_templates[frame_type] = template
    
    def register_frame_processor(self, frame_type: str, processor: Callable):
        """Register a frame processor"""
        self._frame_processors[frame_type] = processor
    
    async def create_frame(self, frame_type: str, **kwargs) -> ExecutionFrame:
        """Create a frame from template"""
        template = self._frame_templates.get(frame_type, {})
        
        frame = ExecutionFrame(
            frame_id=f"{frame_type}_{len(self._frame_stack)}",
            context={
                'type': frame_type,
                'template': template,
                **kwargs
            }
        )
        
        # Link to parent if stack exists
        if self._frame_stack:
            parent = self._frame_stack[-1]
            frame.parent_frame = parent
            parent.children.append(frame)
        
        self._frame_stack.append(frame)
        
        # Process frame if processor exists
        if frame_type in self._frame_processors:
            processor = self._frame_processors[frame_type]
            result = await processor(frame)
            frame.results.append(result)
        
        return frame
    
    def pop_frame(self) -> Optional[ExecutionFrame]:
        """Pop frame from stack"""
        if self._frame_stack:
            return self._frame_stack.pop()
        return None
    
    def visualize_frame_stack(self) -> str:
        """Create a visual representation of the frame stack"""
        lines = ["Frame Stack Visualization:"]
        lines.append("=" * 50)
        
        for i, frame in enumerate(self._frame_stack):
            indent = "  " * i
            frame_type = frame.context.get('type', 'unknown')
            lines.append(f"{indent}[{i}] {frame_type} - {frame.frame_id}")
            
            # Show key context
            for key, value in frame.context.items():
                if key not in ['type', 'template']:
                    lines.append(f"{indent}    {key}: {repr(value)[:50]}...")
            
            # Show results summary
            if frame.results:
                lines.append(f"{indent}    Results: {len(frame.results)} items")
        
        return "\n".join(lines)


# Advanced example: Self-modifying analyzer that analyzes its modifications
class SelfAnalyzingModifier:
    """A component that modifies itself and analyzes the modifications"""
    
    def __init__(self):
        self._modification_history: List[Dict[str, Any]] = []
        self._behavior_versions: Dict[str, str] = {}
        self._current_version = "v1"
        
        # Initial behavior
        self._behavior = self._create_initial_behavior()
    
    def _create_initial_behavior(self) -> Callable:
        """Create initial behavior"""
        async def behavior(x: int) -> int:
            return x * 2
        return behavior
    
    async def execute(self, x: int) -> int:
        """Execute current behavior"""
        return await self._behavior(x)
    
    def modify_and_analyze(self, new_behavior_code: str) -> Dict[str, Any]:
        """Modify behavior and analyze the change"""
        # Store old behavior
        old_source = inspect.getsource(self._behavior)
        self._behavior_versions[self._current_version] = old_source
        
        # Create new behavior
        local_vars = {}
        exec(new_behavior_code, globals(), local_vars)
        new_behavior = local_vars['behavior']
        
        # Analyze differences
        analyzer = MetaRecursiveAnalyzer()
        old_analysis = analyzer.analyze_chunk(f"behavior_{self._current_version}", old_source)
        new_analysis = analyzer.analyze_chunk(f"behavior_new", new_behavior_code)
        
        # Create modification record
        modification = {
            'from_version': self._current_version,
            'to_version': f"v{len(self._behavior_versions) + 2}",
            'old_analysis': old_analysis,
            'new_analysis': new_analysis,
            'complexity_change': new_analysis['complexity'] - old_analysis['complexity'],
            'timestamp': asyncio.get_event_loop().time()
        }
        
        # Apply modification
        self._behavior = new_behavior
        self._current_version = modification['to_version']
        self._modification_history.append(modification)
        
        # Analyze modification patterns
        return self._analyze_modification_patterns()
    
    def _analyze_modification_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in modifications"""
        if not self._modification_history:
            return {'patterns': 'No modifications yet'}
        
        patterns = {
            'total_modifications': len(self._modification_history),
            'complexity_trend': [],
            'version_graph': {},
            'modification_frequency': []
        }
        
        # Analyze complexity trend
        for mod in self._modification_history:
            patterns['complexity_trend'].append({
                'version': mod['to_version'],
                'complexity_change': mod['complexity_change']
            })
        
        # Build version graph
        for mod in self._modification_history:
            patterns['version_graph'][mod['from_version']] = mod['to_version']
        
        return patterns


# Create integrated demonstration
async def demonstrate_advanced_features():
    """Demonstrate advanced meta-recursive capabilities"""
    
    # 1. Initialize components
    registry = ComponentRegistry()
    generator = MetaCodeGenerator()
    introspector = RecursiveIntrospector()
    orchestrator = FrameOrchestrator()
    
    # 2. Generate dynamic components
    tool_spec = {
        'class_name': 'DataProcessor',
        'name': 'data_processor',
        'description': 'Processes data with dynamic logic',
        'init_code': 'self.processing_count = 0',
        'execute_code': '''
        self.processing_count += 1
        data = kwargs.get('data', [])
        return {
            'processed': len(data),
            'count': self.processing_count,
            'result': [item * 2 for item in data]
        }
        '''
    }
    
    generated_tool = generator.generate_tool(tool_spec)
    registry.register(generated_tool)
    
    # 3. Create analyzer component
    analyzer_component = generator.generate_analyzer(
        'performance',
        '''
        results['execution_time'] = data.get('duration', 0)
        results['memory_usage'] = data.get('memory', 0)
        results['efficiency'] = 100 if data.get('errors', 0) == 0 else 50
        '''
    )
    registry.register(analyzer_component)
    
    # 4. Set up frame orchestration
    orchestrator.register_frame_template('analysis', {
        'required_tools': ['analyzer'],
        'output_format': 'json'
    })
    
    async def analyze_frame_processor(frame: ExecutionFrame) -> Dict[str, Any]:
        # Simulate analysis
        return {
            'frame_analysis': f"Analyzed {frame.frame_id}",
            'metrics': {'quality': 0.95, 'completeness': 0.88}
        }
    
    orchestrator.register_frame_processor('analysis', analyze_frame_processor)
    
    # 5. Create self-analyzing modifier
    modifier = SelfAnalyzingModifier()
    
    # 6. Run demonstration sequence
    results = {
        'generated_components': generator.introspect_generation_patterns(),
        'introspection': {},
        'modifications': [],
        'frame_execution': []
    }
    
    # Introspect the introspector
    results['introspection']['self'] = introspector.introspect_self()
    results['introspection']['registry'] = introspector.introspect(registry)
    
    # Perform modifications
    new_behaviors = [
        '''
async def behavior(x: int) -> int:
    return x ** 2  # Square instead of double
''',
        '''
async def behavior(x: int) -> int:
    # Fibonacci-like behavior
    if x <= 1:
        return x
    return x + (x - 1)
'''
    ]
    
    for behavior_code in new_behaviors:
        mod_analysis = modifier.modify_and_analyze(behavior_code)
        results['modifications'].append(mod_analysis)
    
    # Execute frames
    frame1 = await orchestrator.create_frame('analysis', data={'test': 'value'})
    frame2 = await orchestrator.create_frame('analysis', data={'nested': True})
    
    results['frame_execution'] = {
        'stack_visualization': orchestrator.visualize_frame_stack(),
        'frame_count': len(orchestrator._frame_stack)
    }
    
    # Component evolution
    def mutation_add_logging(tree: ast.AST) -> ast.AST:
        # Add logging to all functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                log_stmt = ast.parse('print(f"Executing {self.name}")').body[0]
                node.body.insert(0, log_stmt)
        return tree
    
    evolved = registry.evolve('data_processor', mutation_add_logging, 'data_processor_v2')
    results['evolution'] = {
        'evolved_component': evolved.name,
        'lineage': registry.get_lineage('data_processor_v2')
    }
    
    return results


if __name__ == "__main__":
    # Run advanced demonstration
    results = asyncio.run(demonstrate_advanced_features())
    print(json.dumps(results, indent=2, default=str))