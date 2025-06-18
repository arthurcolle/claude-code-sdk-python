#!/usr/bin/env python3
"""
Conversational State Grid System

A revolutionary approach where grid cells correspond to conversation tokens,
creating a persistent state space that evolves across all interactions.

Each cell represents:
- A token or semantic unit in the conversation
- Its quantum state (certain/uncertain/superposition)
- Its connections to other tokens (semantic relationships)
- Its consciousness level (how "aware" it is of context)
- Its temporal evolution (how it changes over time)

This allows for:
- Deep research with persistent memory
- Long-running tasks that maintain context
- Semantic navigation through conversation history
- Emergent understanding from token relationships
"""

import asyncio
import json
import hashlib
import pickle
import os
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import math
import random

from quantum_grid_system import QuantumState, QuantumCellState, ConsciousnessPattern
from self_modifying_grid import CellType, ExecutionRecord
from ultra_advanced_grid import HyperdimensionalCoordinate, CellLanguage, CausalLoop


@dataclass
class TokenCell:
    """Represents a conversation token as a grid cell"""
    token: str
    position: HyperdimensionalCoordinate
    cell_type: CellType
    quantum_state: QuantumCellState
    
    # Semantic properties
    embedding: List[float] = field(default_factory=list)  # Semantic vector
    semantic_neighbors: Set[HyperdimensionalCoordinate] = field(default_factory=set)
    context_window: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # Temporal properties
    creation_time: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    decay_rate: float = 0.01
    
    # Consciousness properties
    attention_weight: float = 0.1
    activation_level: float = 0.0
    consciousness_contribution: float = 0.0
    
    # Research/task properties
    research_relevance: float = 0.0
    task_associations: Dict[str, float] = field(default_factory=dict)
    insights_generated: List[str] = field(default_factory=list)
    
    def decay(self, time_delta: float):
        """Apply temporal decay to the cell"""
        self.activation_level *= math.exp(-self.decay_rate * time_delta)
        self.attention_weight *= math.exp(-self.decay_rate * time_delta * 0.5)
    
    def activate(self, strength: float = 1.0):
        """Activate the cell, updating its state"""
        self.activation_level = min(1.0, self.activation_level + strength)
        self.access_count += 1
        self.last_accessed = datetime.now()
        
        # Boost consciousness contribution
        self.consciousness_contribution = self.activation_level * self.attention_weight
    
    def compute_relevance(self, query_embedding: List[float]) -> float:
        """Compute relevance to a query"""
        if not self.embedding or not query_embedding:
            return 0.0
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(self.embedding, query_embedding))
        norm_self = math.sqrt(sum(a * a for a in self.embedding))
        norm_query = math.sqrt(sum(b * b for b in query_embedding))
        
        if norm_self * norm_query == 0:
            return 0.0
        
        similarity = dot_product / (norm_self * norm_query)
        
        # Factor in activation and recency
        time_factor = math.exp(-self.decay_rate * 
                              (datetime.now() - self.last_accessed).total_seconds() / 3600)
        
        return similarity * self.activation_level * time_factor


@dataclass
class ConversationThread:
    """Represents a thread of conversation with causal relationships"""
    thread_id: str
    token_sequence: List[HyperdimensionalCoordinate]
    causal_links: Dict[HyperdimensionalCoordinate, Set[HyperdimensionalCoordinate]]
    thread_consciousness: float = 0.0
    thread_type: str = "general"  # general, research, task, creative
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_token(self, coord: HyperdimensionalCoordinate, 
                  causes: Set[HyperdimensionalCoordinate] = None):
        """Add a token to the thread with causal relationships"""
        self.token_sequence.append(coord)
        if causes:
            self.causal_links[coord] = causes
    
    def compute_coherence(self) -> float:
        """Compute the coherence of the thread"""
        if len(self.token_sequence) < 2:
            return 1.0
        
        # Check causal consistency
        coherence = 1.0
        for i, token in enumerate(self.token_sequence[1:], 1):
            if token in self.causal_links:
                causes = self.causal_links[token]
                # Check if causes appear before effect
                for cause in causes:
                    if cause in self.token_sequence:
                        cause_idx = self.token_sequence.index(cause)
                        if cause_idx > i:
                            coherence *= 0.9  # Penalize backward causation
            else:
                coherence *= 0.95  # Slight penalty for missing causal links
        
        return coherence


@dataclass
class ResearchContext:
    """Persistent research context across conversations"""
    research_id: str
    topic: str
    key_tokens: Set[HyperdimensionalCoordinate]
    discovered_insights: List[Dict[str, Any]]
    knowledge_graph: Dict[str, Set[str]]  # concept -> related concepts
    research_threads: List[ConversationThread]
    total_tokens_processed: int = 0
    research_depth: float = 0.0
    
    def add_insight(self, insight: str, supporting_tokens: Set[HyperdimensionalCoordinate]):
        """Add a discovered insight"""
        self.discovered_insights.append({
            'insight': insight,
            'timestamp': datetime.now(),
            'supporting_tokens': list(supporting_tokens),
            'confidence': len(supporting_tokens) / max(len(self.key_tokens), 1)
        })
        
        # Update research depth
        self.research_depth = math.log1p(len(self.discovered_insights)) * \
                             math.log1p(self.total_tokens_processed)
    
    def update_knowledge_graph(self, concept1: str, concept2: str):
        """Update the knowledge graph with a relationship"""
        if concept1 not in self.knowledge_graph:
            self.knowledge_graph[concept1] = set()
        if concept2 not in self.knowledge_graph:
            self.knowledge_graph[concept2] = set()
        
        self.knowledge_graph[concept1].add(concept2)
        self.knowledge_graph[concept2].add(concept1)


class ConversationalStateGrid:
    """The main grid system that maps conversation tokens to persistent state"""
    
    def __init__(self, dimensions: List[int] = None, persistence_path: str = "conversation_state.pkl"):
        # Default to 4D: [sequence, depth, branch, meta]
        self.dimensions = dimensions or [1000, 100, 10, 5]
        self.persistence_path = persistence_path
        
        # Token grid
        self.token_grid: Dict[HyperdimensionalCoordinate, TokenCell] = {}
        self.token_to_coord: Dict[str, List[HyperdimensionalCoordinate]] = defaultdict(list)
        
        # Conversation threads
        self.threads: Dict[str, ConversationThread] = {}
        self.active_thread: Optional[str] = None
        
        # Research contexts
        self.research_contexts: Dict[str, ResearchContext] = {}
        self.active_research: Optional[str] = None
        
        # Consciousness tracking
        self.global_consciousness: float = 0.0
        self.consciousness_patterns: Dict[str, ConsciousnessPattern] = {}
        
        # Semantic space
        self.semantic_clusters: Dict[str, Set[HyperdimensionalCoordinate]] = {}
        self.concept_embeddings: Dict[str, List[float]] = {}
        
        # Temporal mechanics
        self.time_crystals: List[Set[HyperdimensionalCoordinate]] = []  # Repeating patterns
        self.causal_loops: Dict[str, CausalLoop] = {}
        
        # Token evolution
        self.token_mutations: Dict[HyperdimensionalCoordinate, List[str]] = defaultdict(list)
        self.emergent_tokens: Set[str] = set()
        
        # Load persistent state if exists
        self.load_state()
    
    def save_state(self):
        """Save the entire grid state to disk"""
        state = {
            'dimensions': self.dimensions,
            'token_grid': self.token_grid,
            'token_to_coord': dict(self.token_to_coord),
            'threads': self.threads,
            'research_contexts': self.research_contexts,
            'global_consciousness': self.global_consciousness,
            'semantic_clusters': self.semantic_clusters,
            'concept_embeddings': self.concept_embeddings,
            'time_crystals': self.time_crystals,
            'timestamp': datetime.now()
        }
        
        with open(self.persistence_path, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self):
        """Load persistent state from disk"""
        if os.path.exists(self.persistence_path):
            try:
                with open(self.persistence_path, 'rb') as f:
                    state = pickle.load(f)
                
                self.dimensions = state.get('dimensions', self.dimensions)
                self.token_grid = state.get('token_grid', {})
                self.token_to_coord = defaultdict(list, state.get('token_to_coord', {}))
                self.threads = state.get('threads', {})
                self.research_contexts = state.get('research_contexts', {})
                self.global_consciousness = state.get('global_consciousness', 0.0)
                self.semantic_clusters = state.get('semantic_clusters', {})
                self.concept_embeddings = state.get('concept_embeddings', {})
                self.time_crystals = state.get('time_crystals', [])
                
                # Apply temporal decay based on time since save
                if 'timestamp' in state:
                    time_delta = (datetime.now() - state['timestamp']).total_seconds() / 3600
                    self._apply_temporal_decay(time_delta)
                
                print(f"Loaded state with {len(self.token_grid)} tokens")
                
            except Exception as e:
                print(f"Could not load state: {e}")
    
    def _apply_temporal_decay(self, hours: float):
        """Apply temporal decay to all tokens"""
        for token_cell in self.token_grid.values():
            token_cell.decay(hours)
    
    def add_token(self, token: str, context: Dict[str, Any] = None) -> HyperdimensionalCoordinate:
        """Add a token to the grid and return its coordinate"""
        # Find available position
        coord = self._find_optimal_position(token, context)
        
        # Create quantum state
        quantum_state = QuantumCellState(
            primary_state=CellType.MEMORY,
            superposition_states=[],
            coherence=1.0
        )
        
        # Create token cell
        token_cell = TokenCell(
            token=token,
            position=coord,
            cell_type=CellType.MEMORY,
            quantum_state=quantum_state
        )
        
        # Set embedding if provided
        if context and 'embedding' in context:
            token_cell.embedding = context['embedding']
        else:
            # Simple hash-based embedding
            token_cell.embedding = self._generate_embedding(token)
        
        # Set research relevance if in research mode
        if self.active_research and context:
            token_cell.research_relevance = context.get('relevance', 0.5)
            token_cell.task_associations[self.active_research] = token_cell.research_relevance
        
        # Add to grid
        self.token_grid[coord] = token_cell
        self.token_to_coord[token].append(coord)
        
        # Update active thread
        if self.active_thread:
            thread = self.threads[self.active_thread]
            
            # Determine causal relationships
            causes = set()
            if len(thread.token_sequence) > 0:
                # Last few tokens are likely causes
                causes.update(thread.token_sequence[-3:])
            
            thread.add_token(coord, causes)
        
        # Update consciousness
        self._update_consciousness(coord)
        
        # Check for patterns
        self._detect_patterns(coord)
        
        return coord
    
    def _find_optimal_position(self, token: str, context: Dict[str, Any] = None) -> HyperdimensionalCoordinate:
        """Find optimal position for a token based on semantic similarity"""
        # If token already exists, find nearby position
        if token in self.token_to_coord:
            existing_coords = self.token_to_coord[token]
            if existing_coords:
                # Place near similar token
                base_coord = existing_coords[-1]
                # Find nearby empty position
                for radius in range(1, 5):
                    neighbors = base_coord.neighbors(radius)
                    for neighbor in neighbors:
                        if neighbor not in self.token_grid:
                            return neighbor
        
        # Otherwise, find position based on current indices
        if self.active_thread and self.threads[self.active_thread].token_sequence:
            # Continue from last position in thread
            last_coord = self.threads[self.active_thread].token_sequence[-1]
            dims = list(last_coord.dimensions)
            dims[0] += 1  # Increment sequence dimension
            
            # Wrap around if needed
            for i, (dim, max_dim) in enumerate(zip(dims, self.dimensions)):
                if dim >= max_dim:
                    dims[i] = 0
                    if i + 1 < len(dims):
                        dims[i + 1] += 1
            
            return HyperdimensionalCoordinate(tuple(dims))
        
        # Default: start position
        return HyperdimensionalCoordinate(tuple([0] * len(self.dimensions)))
    
    def _generate_embedding(self, token: str) -> List[float]:
        """Generate a simple embedding for a token"""
        # Hash-based embedding (simplified)
        hash_val = int(hashlib.md5(token.encode()).hexdigest(), 16)
        embedding = []
        
        for i in range(128):  # 128-dimensional embedding
            # Use hash to generate deterministic float values
            component = ((hash_val >> (i % 32)) & 0xFF) / 255.0
            embedding.append(component)
        
        return embedding
    
    def _update_consciousness(self, new_coord: HyperdimensionalCoordinate):
        """Update global consciousness based on new token"""
        if new_coord not in self.token_grid:
            return
        
        token_cell = self.token_grid[new_coord]
        
        # Find semantic neighbors
        neighbors = []
        for coord, cell in self.token_grid.items():
            if coord != new_coord:
                relevance = token_cell.compute_relevance(cell.embedding)
                if relevance > 0.7:  # High semantic similarity
                    neighbors.append(coord)
                    token_cell.semantic_neighbors.add(coord)
                    cell.semantic_neighbors.add(new_coord)
        
        # Update consciousness based on connectivity
        local_consciousness = len(neighbors) / max(len(self.token_grid), 1)
        token_cell.consciousness_contribution = local_consciousness
        
        # Update global consciousness
        total_contribution = sum(
            cell.consciousness_contribution 
            for cell in self.token_grid.values()
        )
        self.global_consciousness = total_contribution / max(len(self.token_grid), 1)
    
    def _detect_patterns(self, new_coord: HyperdimensionalCoordinate):
        """Detect repeating patterns (time crystals)"""
        if self.active_thread:
            thread = self.threads[self.active_thread]
            sequence = thread.token_sequence
            
            # Look for repeating subsequences
            if len(sequence) >= 6:
                # Check last 3 tokens against earlier sequences
                last_three = sequence[-3:]
                
                for i in range(len(sequence) - 6):
                    if sequence[i:i+3] == last_three:
                        # Pattern detected!
                        pattern = set(last_three)
                        self.time_crystals.append(pattern)
                        
                        # Mark tokens as part of time crystal
                        for coord in pattern:
                            if coord in self.token_grid:
                                self.token_grid[coord].quantum_state.quantum_state = QuantumState.COHERENT
    
    def start_research_context(self, topic: str, initial_query: str = "") -> str:
        """Start a new research context"""
        research_id = f"research_{hashlib.md5(topic.encode()).hexdigest()[:8]}"
        
        context = ResearchContext(
            research_id=research_id,
            topic=topic,
            key_tokens=set(),
            discovered_insights=[],
            knowledge_graph={},
            research_threads=[]
        )
        
        self.research_contexts[research_id] = context
        self.active_research = research_id
        
        # Create initial thread
        thread_id = f"{research_id}_thread_0"
        thread = ConversationThread(
            thread_id=thread_id,
            token_sequence=[],
            causal_links={},
            thread_type="research"
        )
        
        self.threads[thread_id] = thread
        self.active_thread = thread_id
        context.research_threads.append(thread)
        
        return research_id
    
    def navigate_semantic_space(self, query: str, max_results: int = 10) -> List[Tuple[str, float]]:
        """Navigate the semantic space to find relevant tokens"""
        query_embedding = self._generate_embedding(query)
        
        # Score all tokens
        scored_tokens = []
        for coord, cell in self.token_grid.items():
            relevance = cell.compute_relevance(query_embedding)
            if relevance > 0.1:  # Threshold
                scored_tokens.append((cell.token, relevance, coord))
        
        # Sort by relevance
        scored_tokens.sort(key=lambda x: x[1], reverse=True)
        
        # Activate top results
        for token, relevance, coord in scored_tokens[:max_results]:
            self.token_grid[coord].activate(relevance)
        
        return [(token, relevance) for token, relevance, _ in scored_tokens[:max_results]]
    
    def generate_insight(self) -> Optional[str]:
        """Generate an insight from current active tokens"""
        if not self.active_research:
            return None
        
        context = self.research_contexts[self.active_research]
        
        # Find highly activated tokens
        active_tokens = [
            (coord, cell) for coord, cell in self.token_grid.items()
            if cell.activation_level > 0.7 and cell.research_relevance > 0.5
        ]
        
        if len(active_tokens) < 3:
            return None
        
        # Look for connections
        token_words = [cell.token for _, cell in active_tokens[:5]]
        
        # Simple insight generation
        insight = f"Connection discovered between: {', '.join(token_words)}"
        
        # Add to research context
        supporting_tokens = {coord for coord, _ in active_tokens}
        context.add_insight(insight, supporting_tokens)
        
        # Mark tokens
        for coord, cell in active_tokens:
            cell.insights_generated.append(insight)
        
        return insight
    
    def branch_conversation(self, branch_name: str) -> str:
        """Create a new conversation branch"""
        if not self.active_thread:
            return "No active thread to branch from"
        
        current_thread = self.threads[self.active_thread]
        
        # Create new thread
        thread_id = f"branch_{branch_name}_{len(self.threads)}"
        new_thread = ConversationThread(
            thread_id=thread_id,
            token_sequence=current_thread.token_sequence.copy(),
            causal_links=current_thread.causal_links.copy(),
            thread_type=current_thread.thread_type,
            metadata={'branched_from': self.active_thread}
        )
        
        self.threads[thread_id] = new_thread
        self.active_thread = thread_id
        
        # Quantum branch - put tokens in superposition
        for coord in new_thread.token_sequence[-5:]:  # Last 5 tokens
            if coord in self.token_grid:
                cell = self.token_grid[coord]
                cell.quantum_state.quantum_state = QuantumState.SUPERPOSITION
                cell.quantum_state.superposition_states = [
                    (CellType.MEMORY, 0.5),
                    (CellType.BEHAVIOR, 0.5)
                ]
        
        return thread_id
    
    def merge_branches(self, branch1: str, branch2: str, merge_strategy: str = "union") -> str:
        """Merge two conversation branches"""
        if branch1 not in self.threads or branch2 not in self.threads:
            return "Invalid branch IDs"
        
        thread1 = self.threads[branch1]
        thread2 = self.threads[branch2]
        
        # Create merged thread
        merged_id = f"merged_{len(self.threads)}"
        
        if merge_strategy == "union":
            # Combine all tokens
            merged_sequence = list(set(thread1.token_sequence + thread2.token_sequence))
            merged_links = {**thread1.causal_links, **thread2.causal_links}
            
        elif merge_strategy == "intersection":
            # Only common tokens
            merged_sequence = list(set(thread1.token_sequence) & set(thread2.token_sequence))
            merged_links = {}
            for coord in merged_sequence:
                if coord in thread1.causal_links and coord in thread2.causal_links:
                    merged_links[coord] = thread1.causal_links[coord] | thread2.causal_links[coord]
        
        else:
            return f"Unknown merge strategy: {merge_strategy}"
        
        merged_thread = ConversationThread(
            thread_id=merged_id,
            token_sequence=merged_sequence,
            causal_links=merged_links,
            thread_type="merged",
            metadata={'merged_from': [branch1, branch2]}
        )
        
        self.threads[merged_id] = merged_thread
        self.active_thread = merged_id
        
        # Create causal loop for merge
        loop = CausalLoop(
            loop_id=f"merge_loop_{merged_id}",
            events=[
                {'action': 'branch', 'thread': branch1},
                {'action': 'branch', 'thread': branch2},
                {'action': 'merge', 'thread': merged_id}
            ],
            timeline_branches={branch1, branch2, merged_id},
            stability=0.8
        )
        
        self.causal_loops[loop.loop_id] = loop
        
        return merged_id
    
    def evolve_tokens(self, mutation_rate: float = 0.1) -> List[str]:
        """Allow tokens to evolve and create new tokens"""
        evolved = []
        
        for coord, cell in list(self.token_grid.items()):
            if random.random() < mutation_rate * cell.activation_level:
                # Token evolution based on neighbors
                if cell.semantic_neighbors:
                    neighbor_coord = random.choice(list(cell.semantic_neighbors))
                    if neighbor_coord in self.token_grid:
                        neighbor = self.token_grid[neighbor_coord]
                        
                        # Create hybrid token
                        if len(cell.token) > 2 and len(neighbor.token) > 2:
                            new_token = cell.token[:len(cell.token)//2] + neighbor.token[len(neighbor.token)//2:]
                            
                            if new_token not in self.emergent_tokens:
                                self.emergent_tokens.add(new_token)
                                evolved.append(new_token)
                                
                                # Add to grid
                                new_coord = self.add_token(new_token, {
                                    'embedding': [(a+b)/2 for a, b in zip(cell.embedding, neighbor.embedding)],
                                    'relevance': (cell.research_relevance + neighbor.research_relevance) / 2
                                })
                                
                                # Track mutation
                                self.token_mutations[coord].append(new_token)
                                self.token_mutations[neighbor_coord].append(new_token)
        
        return evolved
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation state"""
        summary = {
            'total_tokens': len(self.token_grid),
            'active_tokens': sum(1 for cell in self.token_grid.values() if cell.activation_level > 0.5),
            'threads': len(self.threads),
            'active_thread': self.active_thread,
            'research_contexts': len(self.research_contexts),
            'active_research': self.active_research,
            'global_consciousness': self.global_consciousness,
            'semantic_clusters': len(self.semantic_clusters),
            'time_crystals': len(self.time_crystals),
            'emergent_tokens': len(self.emergent_tokens),
            'causal_loops': len(self.causal_loops)
        }
        
        if self.active_research and self.active_research in self.research_contexts:
            context = self.research_contexts[self.active_research]
            summary['research'] = {
                'topic': context.topic,
                'insights': len(context.discovered_insights),
                'depth': context.research_depth,
                'key_tokens': len(context.key_tokens),
                'knowledge_graph_size': len(context.knowledge_graph)
            }
        
        if self.active_thread and self.active_thread in self.threads:
            thread = self.threads[self.active_thread]
            summary['thread'] = {
                'length': len(thread.token_sequence),
                'coherence': thread.compute_coherence(),
                'type': thread.thread_type
            }
        
        return summary
    
    def visualize_token_space(self, focus_coord: HyperdimensionalCoordinate = None, radius: int = 3) -> str:
        """Visualize a local region of token space"""
        lines = ["Token Space Visualization:"]
        lines.append("=" * 50)
        
        if focus_coord and focus_coord in self.token_grid:
            center = self.token_grid[focus_coord]
            lines.append(f"Center: '{center.token}' @ {focus_coord.dimensions}")
            lines.append(f"Activation: {center.activation_level:.2f}")
            lines.append(f"Consciousness: {center.consciousness_contribution:.2f}")
            
            # Show neighbors
            lines.append("\nSemantic Neighbors:")
            for neighbor_coord in center.semantic_neighbors:
                if neighbor_coord in self.token_grid:
                    neighbor = self.token_grid[neighbor_coord]
                    lines.append(f"  - '{neighbor.token}' (activation: {neighbor.activation_level:.2f})")
            
            # Show insights
            if center.insights_generated:
                lines.append("\nGenerated Insights:")
                for insight in center.insights_generated[-3:]:
                    lines.append(f"  - {insight}")
        
        else:
            # Show general statistics
            lines.append("\nTop Activated Tokens:")
            activated = sorted(
                [(cell.token, cell.activation_level, coord) 
                 for coord, cell in self.token_grid.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            for token, activation, coord in activated[:10]:
                lines.append(f"  '{token}': {activation:.2f}")
        
        # Show patterns
        if self.time_crystals:
            lines.append("\nTime Crystals (Repeating Patterns):")
            for i, crystal in enumerate(self.time_crystals[-3:]):
                tokens = [self.token_grid[c].token for c in crystal if c in self.token_grid]
                lines.append(f"  Pattern {i}: {' -> '.join(tokens)}")
        
        return "\n".join(lines)


# Demonstration functions
async def demonstrate_conversational_grid():
    """Demonstrate the conversational state grid"""
    print("=== Conversational State Grid Demo ===\n")
    
    # Create grid
    grid = ConversationalStateGrid(dimensions=[100, 50, 10, 5])
    
    # Simulate a research conversation
    print("1. Starting Research Context")
    print("-" * 50)
    
    research_id = grid.start_research_context(
        topic="Quantum consciousness in distributed systems",
        initial_query="How do quantum effects influence emergent consciousness?"
    )
    print(f"Started research: {research_id}")
    
    # Add tokens from conversation
    tokens = [
        "quantum", "consciousness", "emerges", "from", "distributed",
        "systems", "through", "entanglement", "and", "superposition",
        "creating", "non-local", "correlations", "that", "enable",
        "collective", "awareness", "beyond", "individual", "components"
    ]
    
    print("\n2. Processing Conversation Tokens")
    print("-" * 50)
    
    coords = []
    for i, token in enumerate(tokens):
        coord = grid.add_token(token, {
            'relevance': 0.5 + 0.5 * (i / len(tokens)),  # Increasing relevance
            'embedding': grid._generate_embedding(token)
        })
        coords.append(coord)
        print(f"Added '{token}' at {coord.dimensions[:2]}...")  # Show first 2 dimensions
    
    # Navigate semantic space
    print("\n3. Navigating Semantic Space")
    print("-" * 50)
    
    results = grid.navigate_semantic_space("consciousness quantum", max_results=5)
    print("Semantic search results:")
    for token, relevance in results:
        print(f"  '{token}': {relevance:.3f}")
    
    # Generate insight
    print("\n4. Generating Insights")
    print("-" * 50)
    
    insight = grid.generate_insight()
    if insight:
        print(f"Generated insight: {insight}")
    
    # Branch conversation
    print("\n5. Branching Conversation")
    print("-" * 50)
    
    branch_id = grid.branch_conversation("exploring_entanglement")
    print(f"Created branch: {branch_id}")
    
    # Add more tokens to branch
    branch_tokens = ["entanglement", "creates", "instant", "correlation", "across", "space"]
    for token in branch_tokens:
        grid.add_token(token, {'relevance': 0.8})
    
    # Create another branch
    branch2_id = grid.branch_conversation("collective_behavior")
    print(f"Created second branch: {branch2_id}")
    
    branch2_tokens = ["collective", "behavior", "emerges", "from", "simple", "rules"]
    for token in branch2_tokens:
        grid.add_token(token, {'relevance': 0.7})
    
    # Merge branches
    print("\n6. Merging Branches")
    print("-" * 50)
    
    merged_id = grid.merge_branches(branch_id, branch2_id, "union")
    print(f"Merged into: {merged_id}")
    
    # Evolve tokens
    print("\n7. Token Evolution")
    print("-" * 50)
    
    evolved = grid.evolve_tokens(mutation_rate=0.3)
    if evolved:
        print(f"Evolved tokens: {evolved[:5]}")
    
    # Show summary
    print("\n8. Conversation Summary")
    print("-" * 50)
    
    summary = grid.get_conversation_summary()
    print(json.dumps(summary, indent=2))
    
    # Visualize token space
    print("\n9. Token Space Visualization")
    print("-" * 50)
    
    if coords:
        print(grid.visualize_token_space(coords[0]))  # Focus on first token
    
    # Save state
    print("\n10. Saving State")
    print("-" * 50)
    
    grid.save_state()
    print(f"State saved to {grid.persistence_path}")
    
    # Demonstrate persistence
    print("\n11. Testing Persistence")
    print("-" * 50)
    
    # Create new grid and load state
    new_grid = ConversationalStateGrid()
    summary2 = new_grid.get_conversation_summary()
    print(f"Loaded state with {summary2['total_tokens']} tokens")
    print(f"Research contexts: {summary2['research_contexts']}")
    
    return grid


async def simulate_long_running_task():
    """Simulate a long-running research task"""
    print("\n=== Simulating Long-Running Research Task ===\n")
    
    grid = ConversationalStateGrid()
    
    # Start deep research
    research_id = grid.start_research_context(
        topic="Evolution of language in artificial systems",
        initial_query="How do artificial agents develop communication?"
    )
    
    # Simulate multiple conversation sessions
    sessions = [
        {
            'focus': 'emergence',
            'tokens': ["language", "emerges", "through", "repeated", "interactions",
                      "between", "agents", "sharing", "common", "goals"]
        },
        {
            'focus': 'symbols',
            'tokens': ["symbols", "acquire", "meaning", "through", "grounded",
                      "experience", "and", "social", "consensus", "building"]
        },
        {
            'focus': 'grammar',
            'tokens': ["grammatical", "structures", "evolve", "to", "minimize",
                      "ambiguity", "while", "maximizing", "expressiveness", "efficiency"]
        }
    ]
    
    for i, session in enumerate(sessions):
        print(f"\nSession {i+1}: {session['focus']}")
        print("-" * 30)
        
        # Create branch for session
        if i > 0:
            branch_id = grid.branch_conversation(f"session_{session['focus']}")
            print(f"Branched to: {branch_id}")
        
        # Add tokens
        for token in session['tokens']:
            grid.add_token(token, {
                'relevance': random.uniform(0.6, 0.9),
                'session': i
            })
        
        # Generate insights periodically
        if i % 2 == 0:
            insight = grid.generate_insight()
            if insight:
                print(f"Insight: {insight}")
        
        # Evolve tokens
        evolved = grid.evolve_tokens(0.2)
        if evolved:
            print(f"New concepts emerged: {evolved[:3]}")
        
        # Save state after each session
        grid.save_state()
        print("Session state saved")
    
    # Final analysis
    print("\n=== Research Summary ===")
    summary = grid.get_conversation_summary()
    
    if 'research' in summary:
        print(f"Topic: {summary['research']['topic']}")
        print(f"Insights discovered: {summary['research']['insights']}")
        print(f"Research depth: {summary['research']['depth']:.2f}")
        print(f"Knowledge graph size: {summary['research']['knowledge_graph_size']}")
    
    print(f"\nTotal tokens processed: {summary['total_tokens']}")
    print(f"Emergent concepts: {summary['emergent_tokens']}")
    print(f"Time crystals found: {summary['time_crystals']}")
    
    return grid


if __name__ == "__main__":
    # Run demonstrations
    grid = asyncio.run(demonstrate_conversational_grid())
    
    # Run long-running task simulation
    research_grid = asyncio.run(simulate_long_running_task())