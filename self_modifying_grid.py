#!/usr/bin/env python3
"""
Self-Modifying Grid System

A grid-based abstraction that allows agents to mutate system prompts
and behavior through tool hooks, with full execution tracking.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import copy
from datetime import datetime
import textwrap
import ast

# Import from our framework
from dynamic_tools_framework import BaseTool, ToolSignature, DynamicToolFactory
from advanced_meta_recursive import CodeComponent, ComponentRegistry


class CellType(Enum):
    """Types of cells in the grid"""
    EMPTY = "empty"
    TOOL = "tool"
    PROMPT = "prompt"
    BEHAVIOR = "behavior"
    MEMORY = "memory"
    HOOK = "hook"


@dataclass
class GridCell:
    """Individual cell in the grid"""
    x: int
    y: int
    cell_type: CellType
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    connections: List[Tuple[int, int]] = field(default_factory=list)
    
    def connect_to(self, x: int, y: int):
        """Connect this cell to another"""
        if (x, y) not in self.connections:
            self.connections.append((x, y))
    
    def disconnect_from(self, x: int, y: int):
        """Disconnect from another cell"""
        if (x, y) in self.connections:
            self.connections.remove((x, y))


@dataclass
class ExecutionRecord:
    """Record of a single execution/mutation"""
    timestamp: datetime
    action: str
    cell_coords: Tuple[int, int]
    previous_state: Any
    new_state: Any
    tool_used: Optional[str] = None
    agent_id: Optional[str] = None
    effects: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'action': self.action,
            'cell_coords': self.cell_coords,
            'previous_state': str(self.previous_state)[:100],
            'new_state': str(self.new_state)[:100],
            'tool_used': self.tool_used,
            'agent_id': self.agent_id,
            'effects': self.effects
        }


class SelfModifyingGrid:
    """A grid that can modify itself through tool hooks"""
    
    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height
        self.grid: Dict[Tuple[int, int], GridCell] = {}
        self.system_prompts: Dict[str, str] = {}
        self.execution_history: List[ExecutionRecord] = []
        self.tool_factory = DynamicToolFactory()
        self.tool_hooks: Dict[str, Callable] = {}
        self.active_behaviors: Dict[str, Any] = {}
        
        # Initialize grid
        self._initialize_grid()
        
        # Create mutation tools
        self._create_mutation_tools()
    
    def _initialize_grid(self):
        """Initialize empty grid"""
        for x in range(self.width):
            for y in range(self.height):
                self.grid[(x, y)] = GridCell(x, y, CellType.EMPTY)
    
    def _create_mutation_tools(self):
        """Create tools for grid mutation"""
        
        # Tool for modifying cells
        modify_cell_sig = ToolSignature(
            name="ModifyCell",
            parameters={'x': int, 'y': int, 'cell_type': str, 'content': str},
            returns=Dict,
            description="Modify a cell in the grid"
        )
        
        async def modify_cell_execute(x: int, y: int, cell_type: str, content: str, **kwargs) -> Dict:
            return await self.modify_cell(x, y, cell_type, content, agent_id=kwargs.get('agent_id'))
        
        modify_cell_sig.metadata['execute_fn'] = modify_cell_execute
        self.tool_factory.create_tool(modify_cell_sig)
        
        # Tool for mutating system prompts
        mutate_prompt_sig = ToolSignature(
            name="MutateSystemPrompt",
            parameters={'prompt_id': str, 'new_prompt': str, 'merge_strategy': str},
            returns=Dict,
            description="Mutate a system prompt"
        )
        
        async def mutate_prompt_execute(prompt_id: str, new_prompt: str, 
                                      merge_strategy: str = "replace", **kwargs) -> Dict:
            return await self.mutate_system_prompt(prompt_id, new_prompt, merge_strategy, 
                                                  agent_id=kwargs.get('agent_id'))
        
        mutate_prompt_sig.metadata['execute_fn'] = mutate_prompt_execute
        self.tool_factory.create_tool(mutate_prompt_sig)
        
        # Tool for creating behavior hooks
        create_hook_sig = ToolSignature(
            name="CreateHook",
            parameters={'x': int, 'y': int, 'hook_code': str, 'trigger': str},
            returns=Dict,
            description="Create a behavior hook at a cell"
        )
        
        async def create_hook_execute(x: int, y: int, hook_code: str, trigger: str, **kwargs) -> Dict:
            return await self.create_hook(x, y, hook_code, trigger, agent_id=kwargs.get('agent_id'))
        
        create_hook_sig.metadata['execute_fn'] = create_hook_execute
        self.tool_factory.create_tool(create_hook_sig)
        
        # Tool for connecting cells
        connect_cells_sig = ToolSignature(
            name="ConnectCells",
            parameters={'from_x': int, 'from_y': int, 'to_x': int, 'to_y': int},
            returns=Dict,
            description="Connect two cells in the grid"
        )
        
        async def connect_cells_execute(from_x: int, from_y: int, 
                                      to_x: int, to_y: int, **kwargs) -> Dict:
            return await self.connect_cells(from_x, from_y, to_x, to_y, 
                                          agent_id=kwargs.get('agent_id'))
        
        connect_cells_sig.metadata['execute_fn'] = connect_cells_execute
        self.tool_factory.create_tool(connect_cells_sig)
        
        # Tool for analyzing grid state
        analyze_grid_sig = ToolSignature(
            name="AnalyzeGrid",
            parameters={'analysis_type': str},
            returns=Dict,
            description="Analyze the current grid state"
        )
        
        async def analyze_grid_execute(analysis_type: str = "full", **kwargs) -> Dict:
            return await self.analyze_grid(analysis_type)
        
        analyze_grid_sig.metadata['execute_fn'] = analyze_grid_execute
        self.tool_factory.create_tool(analyze_grid_sig)
    
    async def modify_cell(self, x: int, y: int, cell_type: str, content: str, 
                         agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Modify a cell in the grid"""
        if (x, y) not in self.grid:
            return {'error': f'Cell ({x}, {y}) out of bounds'}
        
        cell = self.grid[(x, y)]
        previous_state = {
            'type': cell.cell_type.value,
            'content': cell.content,
            'metadata': copy.deepcopy(cell.metadata)
        }
        
        # Update cell
        try:
            cell.cell_type = CellType(cell_type)
        except ValueError:
            return {'error': f'Invalid cell type: {cell_type}'}
        
        # Process content based on type
        if cell.cell_type == CellType.TOOL:
            # Parse tool definition
            cell.content = self._parse_tool_content(content)
        elif cell.cell_type == CellType.BEHAVIOR:
            # Compile behavior code
            cell.content = self._compile_behavior(content)
        else:
            cell.content = content
        
        # Record execution
        record = ExecutionRecord(
            timestamp=datetime.now(),
            action="modify_cell",
            cell_coords=(x, y),
            previous_state=previous_state,
            new_state={'type': cell_type, 'content': content},
            tool_used="ModifyCell",
            agent_id=agent_id
        )
        
        # Check for ripple effects
        effects = await self._check_ripple_effects(x, y, cell)
        record.effects = effects
        
        self.execution_history.append(record)
        
        return {
            'success': True,
            'cell': (x, y),
            'new_type': cell_type,
            'effects': effects
        }
    
    async def mutate_system_prompt(self, prompt_id: str, new_prompt: str, 
                                  merge_strategy: str = "replace",
                                  agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Mutate a system prompt"""
        previous_prompt = self.system_prompts.get(prompt_id, "")
        
        if merge_strategy == "replace":
            final_prompt = new_prompt
        elif merge_strategy == "append":
            final_prompt = previous_prompt + "\n\n" + new_prompt
        elif merge_strategy == "prepend":
            final_prompt = new_prompt + "\n\n" + previous_prompt
        elif merge_strategy == "merge":
            # Smart merge - combine intelligently
            final_prompt = self._smart_merge_prompts(previous_prompt, new_prompt)
        else:
            return {'error': f'Unknown merge strategy: {merge_strategy}'}
        
        self.system_prompts[prompt_id] = final_prompt
        
        # Find and update prompt cells
        prompt_cells = []
        for coords, cell in self.grid.items():
            if cell.cell_type == CellType.PROMPT and cell.metadata.get('prompt_id') == prompt_id:
                cell.content = final_prompt
                prompt_cells.append(coords)
        
        # Record execution
        record = ExecutionRecord(
            timestamp=datetime.now(),
            action="mutate_prompt",
            cell_coords=(-1, -1),  # System-level change
            previous_state=previous_prompt,
            new_state=final_prompt,
            tool_used="MutateSystemPrompt",
            agent_id=agent_id,
            effects=[{
                'type': 'prompt_update',
                'prompt_id': prompt_id,
                'affected_cells': prompt_cells,
                'merge_strategy': merge_strategy
            }]
        )
        
        self.execution_history.append(record)
        
        # Trigger any hooks
        await self._trigger_hooks('prompt_mutation', {
            'prompt_id': prompt_id,
            'previous': previous_prompt,
            'new': final_prompt
        })
        
        return {
            'success': True,
            'prompt_id': prompt_id,
            'merge_strategy': merge_strategy,
            'affected_cells': len(prompt_cells),
            'prompt_length': len(final_prompt)
        }
    
    async def create_hook(self, x: int, y: int, hook_code: str, 
                         trigger: str, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a behavior hook at a cell"""
        if (x, y) not in self.grid:
            return {'error': f'Cell ({x}, {y}) out of bounds'}
        
        cell = self.grid[(x, y)]
        
        # Compile hook code
        try:
            compiled_hook = compile(hook_code, f'hook_{x}_{y}', 'exec')
            local_vars = {}
            exec(compiled_hook, globals(), local_vars)
            
            if 'hook_function' not in local_vars:
                return {'error': 'Hook code must define hook_function'}
            
            hook_fn = local_vars['hook_function']
        except Exception as e:
            return {'error': f'Failed to compile hook: {str(e)}'}
        
        # Store hook
        cell.cell_type = CellType.HOOK
        cell.content = {
            'code': hook_code,
            'trigger': trigger,
            'function': hook_fn
        }
        cell.metadata['trigger'] = trigger
        
        # Register hook
        hook_id = f"hook_{x}_{y}_{trigger}"
        self.tool_hooks[hook_id] = hook_fn
        
        # Record execution
        record = ExecutionRecord(
            timestamp=datetime.now(),
            action="create_hook",
            cell_coords=(x, y),
            previous_state=None,
            new_state={'trigger': trigger, 'code': hook_code[:100]},
            tool_used="CreateHook",
            agent_id=agent_id
        )
        
        self.execution_history.append(record)
        
        return {
            'success': True,
            'hook_id': hook_id,
            'cell': (x, y),
            'trigger': trigger
        }
    
    async def connect_cells(self, from_x: int, from_y: int, to_x: int, to_y: int,
                           agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Connect two cells"""
        if (from_x, from_y) not in self.grid or (to_x, to_y) not in self.grid:
            return {'error': 'One or both cells out of bounds'}
        
        from_cell = self.grid[(from_x, from_y)]
        to_cell = self.grid[(to_x, to_y)]
        
        # Create bidirectional connection
        from_cell.connect_to(to_x, to_y)
        to_cell.connect_to(from_x, from_y)
        
        # Record execution
        record = ExecutionRecord(
            timestamp=datetime.now(),
            action="connect_cells",
            cell_coords=(from_x, from_y),
            previous_state=None,
            new_state={'connected_to': (to_x, to_y)},
            tool_used="ConnectCells",
            agent_id=agent_id,
            effects=[{
                'type': 'connection_created',
                'from': (from_x, from_y),
                'to': (to_x, to_y),
                'bidirectional': True
            }]
        )
        
        self.execution_history.append(record)
        
        # Trigger connection hooks
        await self._trigger_hooks('cell_connection', {
            'from': (from_x, from_y),
            'to': (to_x, to_y),
            'from_type': from_cell.cell_type.value,
            'to_type': to_cell.cell_type.value
        })
        
        return {
            'success': True,
            'connection': f"({from_x},{from_y}) <-> ({to_x},{to_y})",
            'from_type': from_cell.cell_type.value,
            'to_type': to_cell.cell_type.value
        }
    
    async def analyze_grid(self, analysis_type: str = "full") -> Dict[str, Any]:
        """Analyze the current grid state"""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'grid_size': f"{self.width}x{self.height}",
            'total_cells': self.width * self.height,
            'cell_types': {},
            'connections': [],
            'system_prompts': len(self.system_prompts),
            'active_hooks': len(self.tool_hooks),
            'execution_history_length': len(self.execution_history)
        }
        
        # Count cell types
        for cell in self.grid.values():
            cell_type = cell.cell_type.value
            analysis['cell_types'][cell_type] = analysis['cell_types'].get(cell_type, 0) + 1
        
        # Analyze connections
        connection_map = {}
        for coords, cell in self.grid.items():
            if cell.connections:
                for conn in cell.connections:
                    # Avoid duplicate connections
                    conn_key = tuple(sorted([coords, conn]))
                    if conn_key not in connection_map:
                        connection_map[conn_key] = {
                            'from': coords,
                            'to': conn,
                            'from_type': cell.cell_type.value,
                            'to_type': self.grid[conn].cell_type.value if conn in self.grid else 'unknown'
                        }
        
        analysis['connections'] = list(connection_map.values())
        
        if analysis_type == "full":
            # Add detailed analysis
            analysis['recent_executions'] = [
                record.to_dict() for record in self.execution_history[-10:]
            ]
            
            # Analyze mutation patterns
            mutation_stats = self._analyze_mutation_patterns()
            analysis['mutation_patterns'] = mutation_stats
            
            # Grid visualization
            analysis['grid_visualization'] = self._visualize_grid()
        
        return analysis
    
    def _parse_tool_content(self, content: str) -> Dict[str, Any]:
        """Parse tool definition from content"""
        try:
            # Assume JSON format for tool definition
            return json.loads(content)
        except:
            return {'raw': content}
    
    def _compile_behavior(self, code: str) -> Dict[str, Any]:
        """Compile behavior code"""
        try:
            tree = ast.parse(code)
            compiled = compile(code, 'behavior', 'exec')
            return {
                'code': code,
                'ast': tree,
                'compiled': compiled
            }
        except Exception as e:
            return {'error': str(e), 'code': code}
    
    def _smart_merge_prompts(self, existing: str, new: str) -> str:
        """Intelligently merge two prompts"""
        # Simple implementation - could be made more sophisticated
        lines_existing = existing.split('\n')
        lines_new = new.split('\n')
        
        # Remove duplicates while preserving order
        merged = []
        seen = set()
        
        for line in lines_existing + lines_new:
            if line.strip() and line not in seen:
                merged.append(line)
                seen.add(line)
        
        return '\n'.join(merged)
    
    async def _check_ripple_effects(self, x: int, y: int, cell: GridCell) -> List[Dict[str, Any]]:
        """Check for ripple effects from cell modification"""
        effects = []
        
        # Check connected cells
        for conn_x, conn_y in cell.connections:
            if (conn_x, conn_y) in self.grid:
                conn_cell = self.grid[(conn_x, conn_y)]
                
                # If connected to a hook, it might trigger
                if conn_cell.cell_type == CellType.HOOK:
                    effects.append({
                        'type': 'potential_hook_trigger',
                        'cell': (conn_x, conn_y),
                        'trigger': conn_cell.metadata.get('trigger')
                    })
                
                # If connected to behavior, it might need recompilation
                elif conn_cell.cell_type == CellType.BEHAVIOR:
                    effects.append({
                        'type': 'behavior_dependency',
                        'cell': (conn_x, conn_y),
                        'action': 'may_need_recompilation'
                    })
        
        return effects
    
    async def _trigger_hooks(self, trigger: str, context: Dict[str, Any]):
        """Trigger all hooks for a given trigger type"""
        triggered = []
        
        for hook_id, hook_fn in self.tool_hooks.items():
            if trigger in hook_id:
                try:
                    # Execute hook
                    if asyncio.iscoroutinefunction(hook_fn):
                        await hook_fn(self, context)
                    else:
                        hook_fn(self, context)
                    triggered.append(hook_id)
                except Exception as e:
                    print(f"Hook {hook_id} failed: {e}")
        
        if triggered:
            # Record hook triggers
            record = ExecutionRecord(
                timestamp=datetime.now(),
                action="trigger_hooks",
                cell_coords=(-1, -1),
                previous_state=None,
                new_state={'trigger': trigger, 'hooks': triggered},
                effects=[{'type': 'hooks_triggered', 'count': len(triggered)}]
            )
            self.execution_history.append(record)
    
    def _analyze_mutation_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in mutations"""
        patterns = {
            'total_mutations': len(self.execution_history),
            'by_action': {},
            'by_tool': {},
            'by_agent': {},
            'temporal_pattern': []
        }
        
        for record in self.execution_history:
            # By action
            patterns['by_action'][record.action] = patterns['by_action'].get(record.action, 0) + 1
            
            # By tool
            if record.tool_used:
                patterns['by_tool'][record.tool_used] = patterns['by_tool'].get(record.tool_used, 0) + 1
            
            # By agent
            if record.agent_id:
                patterns['by_agent'][record.agent_id] = patterns['by_agent'].get(record.agent_id, 0) + 1
        
        # Temporal pattern (last 10 mutations)
        for record in self.execution_history[-10:]:
            patterns['temporal_pattern'].append({
                'time': record.timestamp.isoformat(),
                'action': record.action,
                'cell': record.cell_coords
            })
        
        return patterns
    
    def _visualize_grid(self) -> str:
        """Create ASCII visualization of the grid"""
        # Symbol mapping
        symbols = {
            CellType.EMPTY: '.',
            CellType.TOOL: 'T',
            CellType.PROMPT: 'P',
            CellType.BEHAVIOR: 'B',
            CellType.MEMORY: 'M',
            CellType.HOOK: 'H'
        }
        
        lines = []
        lines.append("Grid Visualization:")
        lines.append("  " + " ".join(str(x) for x in range(min(10, self.width))))
        
        for y in range(min(10, self.height)):
            row = f"{y} "
            for x in range(min(10, self.width)):
                cell = self.grid.get((x, y))
                if cell:
                    symbol = symbols.get(cell.cell_type, '?')
                    # Mark connected cells
                    if cell.connections:
                        symbol = f"[{symbol}]"
                    else:
                        symbol = f" {symbol} "
                    row += symbol
                else:
                    row += " . "
            lines.append(row)
        
        if self.width > 10 or self.height > 10:
            lines.append(f"(Showing 10x10 of {self.width}x{self.height} grid)")
        
        return "\n".join(lines)
    
    def get_execution_history_summary(self) -> str:
        """Get a summary of execution history"""
        if not self.execution_history:
            return "No executions yet"
        
        lines = ["Execution History Summary:"]
        lines.append("-" * 50)
        
        for i, record in enumerate(self.execution_history[-10:], 1):
            lines.append(f"\n{i}. {record.timestamp.strftime('%H:%M:%S')} - {record.action}")
            lines.append(f"   Cell: {record.cell_coords}")
            lines.append(f"   Tool: {record.tool_used or 'None'}")
            if record.agent_id:
                lines.append(f"   Agent: {record.agent_id}")
            if record.effects:
                lines.append(f"   Effects: {len(record.effects)} effect(s)")
                for effect in record.effects[:2]:
                    lines.append(f"     - {effect.get('type', 'unknown')}")
        
        return "\n".join(lines)
    
    def export_state(self) -> Dict[str, Any]:
        """Export the complete grid state"""
        state = {
            'grid_config': {
                'width': self.width,
                'height': self.height
            },
            'cells': {},
            'system_prompts': self.system_prompts,
            'execution_count': len(self.execution_history),
            'active_hooks': list(self.tool_hooks.keys()),
            'timestamp': datetime.now().isoformat()
        }
        
        # Export cells
        for coords, cell in self.grid.items():
            if cell.cell_type != CellType.EMPTY:
                state['cells'][f"{coords[0]},{coords[1]}"] = {
                    'type': cell.cell_type.value,
                    'content': str(cell.content)[:200],  # Truncate for readability
                    'connections': cell.connections,
                    'metadata': cell.metadata
                }
        
        return state


# Agent wrapper for interacting with the grid
class GridAgent:
    """Agent that can interact with the self-modifying grid"""
    
    def __init__(self, agent_id: str, grid: SelfModifyingGrid):
        self.agent_id = agent_id
        self.grid = grid
        self.tool_factory = grid.tool_factory
    
    async def execute_tool(self, tool_name: str, **params) -> Dict[str, Any]:
        """Execute a grid tool"""
        tool = self.tool_factory.instantiate_tool(tool_name)
        
        # Add agent_id to params
        params['agent_id'] = self.agent_id
        
        result = await tool.execute(**params)
        
        # Log agent action
        print(f"[Agent {self.agent_id}] Executed {tool_name}: {result}")
        
        return result
    
    async def create_behavior_chain(self, behaviors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a chain of connected behavior cells"""
        created_cells = []
        
        for i, behavior in enumerate(behaviors):
            x, y = behavior['position']
            
            # Create behavior cell
            result = await self.execute_tool(
                'ModifyCell',
                x=x, y=y,
                cell_type='behavior',
                content=behavior['code']
            )
            
            created_cells.append((x, y))
            
            # Connect to previous cell if exists
            if i > 0:
                prev_x, prev_y = created_cells[i-1]
                await self.execute_tool(
                    'ConnectCells',
                    from_x=prev_x, from_y=prev_y,
                    to_x=x, to_y=y
                )
        
        return {
            'chain_created': True,
            'cells': created_cells,
            'length': len(created_cells)
        }


# Demonstration
async def demonstrate_self_modifying_grid():
    """Demonstrate the self-modifying grid system"""
    print("=== Self-Modifying Grid Demonstration ===\n")
    
    # Create grid
    grid = SelfModifyingGrid(width=8, height=8)
    
    # Create agents
    agent1 = GridAgent("Agent-1", grid)
    agent2 = GridAgent("Agent-2", grid)
    
    # Initial grid state
    print("1. Initial Grid State")
    print("-" * 50)
    analysis = await grid.analyze_grid()
    print(f"Grid size: {analysis['grid_size']}")
    print(f"Empty cells: {analysis['cell_types'].get('empty', 0)}")
    print()
    
    # Agent 1 creates a system prompt
    print("2. Agent 1 Creates System Prompt")
    print("-" * 50)
    
    # First, create a prompt cell
    await agent1.execute_tool(
        'ModifyCell',
        x=2, y=2,
        cell_type='prompt',
        content='You are a helpful assistant.'
    )
    
    # Then mutate the system prompt
    await agent1.execute_tool(
        'MutateSystemPrompt',
        prompt_id='main_prompt',
        new_prompt='You are a helpful and creative assistant who thinks step by step.',
        merge_strategy='replace'
    )
    
    print("System prompt created and stored")
    print()
    
    # Agent 2 modifies the prompt
    print("3. Agent 2 Modifies System Prompt")
    print("-" * 50)
    
    await agent2.execute_tool(
        'MutateSystemPrompt',
        prompt_id='main_prompt',
        new_prompt='You excel at problem-solving and coding.',
        merge_strategy='append'
    )
    
    print("System prompt modified by Agent 2")
    print(f"Current prompt: {grid.system_prompts['main_prompt'][:100]}...")
    print()
    
    # Create a hook that triggers on prompt changes
    print("4. Creating Hook for Prompt Changes")
    print("-" * 50)
    
    hook_code = '''
def hook_function(grid, context):
    print(f"[HOOK] Prompt mutation detected!")
    print(f"[HOOK] Prompt ID: {context.get('prompt_id')}")
    print(f"[HOOK] New length: {len(context.get('new', ''))} chars")
    
    # Add metadata to track mutations
    if not hasattr(grid, 'prompt_mutation_count'):
        grid.prompt_mutation_count = 0
    grid.prompt_mutation_count += 1
'''
    
    await agent1.execute_tool(
        'CreateHook',
        x=3, y=3,
        hook_code=hook_code,
        trigger='prompt_mutation'
    )
    
    print("Hook created at cell (3,3)")
    print()
    
    # Test the hook with another mutation
    print("5. Testing Hook with Another Mutation")
    print("-" * 50)
    
    await agent2.execute_tool(
        'MutateSystemPrompt',
        prompt_id='main_prompt',
        new_prompt='You should always verify your work.',
        merge_strategy='append'
    )
    
    print(f"Prompt mutation count: {getattr(grid, 'prompt_mutation_count', 0)}")
    print()
    
    # Create connected behavior cells
    print("6. Creating Connected Behavior Chain")
    print("-" * 50)
    
    behaviors = [
        {
            'position': (4, 4),
            'code': '''
def process_input(data):
    return data.upper()
'''
        },
        {
            'position': (5, 4),
            'code': '''
def validate_output(data):
    return len(data) > 0
'''
        },
        {
            'position': (6, 4),
            'code': '''
def format_response(data):
    return f"Result: {data}"
'''
        }
    ]
    
    chain_result = await agent1.create_behavior_chain(behaviors)
    print(f"Behavior chain created: {chain_result['cells']}")
    print()
    
    # Analyze final grid state
    print("7. Final Grid Analysis")
    print("-" * 50)
    
    final_analysis = await grid.analyze_grid('full')
    print(grid._visualize_grid())
    print()
    
    print(f"Cell type distribution: {final_analysis['cell_types']}")
    print(f"Total connections: {len(final_analysis['connections'])}")
    print(f"Active hooks: {final_analysis['active_hooks']}")
    print()
    
    # Show execution history
    print("8. Execution History")
    print("-" * 50)
    print(grid.get_execution_history_summary())
    print()
    
    # Export state
    state = grid.export_state()
    with open('grid_state.json', 'w') as f:
        json.dump(state, f, indent=2)
    
    print("Grid state exported to grid_state.json")
    
    return grid


if __name__ == "__main__":
    grid = asyncio.run(demonstrate_self_modifying_grid())