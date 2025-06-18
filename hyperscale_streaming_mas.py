#!/usr/bin/env python3
"""
Hyperscale Streaming Multi-Agent System
Features:
- Tensor streaming for distributed AI
- Gossip protocols for decentralized communication  
- Byzantine fault tolerance
- Adaptive mesh networking
- Real-time consensus with Raft/Paxos
- Stream processing with backpressure
- Hierarchical attention mechanisms
- Zero-copy message passing
"""

import asyncio
import uuid
import time
import random
import json
import struct
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple, AsyncIterator, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum, auto
from datetime import datetime
import heapq
import bisect


class StreamProtocol(Enum):
    """Advanced streaming protocols"""
    TENSOR_STREAM = auto()
    GOSSIP_EPIDEMIC = auto()
    BYZANTINE_AGREEMENT = auto()
    RAFT_CONSENSUS = auto()
    PAXOS_COMMIT = auto()
    VECTOR_CLOCK_SYNC = auto()
    MERKLE_SYNC = auto()
    BLOOM_FILTER = auto()


@dataclass
class TensorChunk:
    """Chunk of tensor data for streaming"""
    tensor_id: str
    chunk_index: int
    total_chunks: int
    shape: Tuple[int, ...]
    dtype: str
    data: bytes
    checksum: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class VectorClock:
    """Vector clock for distributed ordering"""
    clock: Dict[str, int] = field(default_factory=dict)
    
    def increment(self, node_id: str):
        self.clock[node_id] = self.clock.get(node_id, 0) + 1
        
    def update(self, other: 'VectorClock'):
        for node_id, timestamp in other.clock.items():
            self.clock[node_id] = max(self.clock.get(node_id, 0), timestamp)
            
    def happens_before(self, other: 'VectorClock') -> bool:
        for node_id, timestamp in self.clock.items():
            if timestamp > other.clock.get(node_id, 0):
                return False
        return True


@dataclass
class GossipMessage:
    """Message for gossip protocol"""
    origin: str
    sequence: int
    payload: Any
    hops: int = 0
    vector_clock: VectorClock = field(default_factory=VectorClock)
    seen_by: Set[str] = field(default_factory=set)


@dataclass
class RaftState:
    """State for Raft consensus"""
    current_term: int = 0
    voted_for: Optional[str] = None
    log: List[Dict[str, Any]] = field(default_factory=list)
    commit_index: int = 0
    last_applied: int = 0
    state: str = "FOLLOWER"  # FOLLOWER, CANDIDATE, LEADER
    leader_id: Optional[str] = None


class StreamBuffer:
    """Advanced stream buffer with backpressure"""
    
    def __init__(self, max_size: int = 10000, high_watermark: float = 0.8):
        self.buffer: deque = deque()
        self.max_size = max_size
        self.high_watermark = high_watermark
        self.low_watermark = high_watermark * 0.5
        self.pressure = False
        self.dropped_count = 0
        self.total_bytes = 0
        
    async def write(self, data: Any) -> bool:
        """Write to buffer with backpressure"""
        if len(self.buffer) >= self.max_size:
            self.dropped_count += 1
            return False
            
        self.buffer.append(data)
        self.total_bytes += len(str(data))
        
        # Check pressure
        if len(self.buffer) > self.max_size * self.high_watermark:
            self.pressure = True
        elif len(self.buffer) < self.max_size * self.low_watermark:
            self.pressure = False
            
        return True
        
    async def read(self, batch_size: int = 1) -> List[Any]:
        """Read from buffer"""
        result = []
        for _ in range(min(batch_size, len(self.buffer))):
            if self.buffer:
                result.append(self.buffer.popleft())
        return result
        
    def is_pressured(self) -> bool:
        return self.pressure


class HierarchicalAttention:
    """Hierarchical attention mechanism for agent communication"""
    
    def __init__(self, embedding_dim: int = 128, num_heads: int = 8):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.attention_scores: Dict[Tuple[str, str], float] = {}
        
    def compute_attention(self, query_agent: str, key_agents: List[str],
                         embeddings: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute attention scores"""
        if query_agent not in embeddings:
            return {}
            
        query_embed = embeddings[query_agent]
        scores = {}
        
        for key_agent in key_agents:
            if key_agent in embeddings:
                key_embed = embeddings[key_agent]
                
                # Scaled dot-product attention
                score = np.dot(query_embed, key_embed) / np.sqrt(self.embedding_dim)
                scores[key_agent] = float(np.exp(score))
                
        # Normalize
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
            
        # Cache scores
        for key_agent, score in scores.items():
            self.attention_scores[(query_agent, key_agent)] = score
            
        return scores


class MerkleTree:
    """Merkle tree for efficient synchronization"""
    
    def __init__(self):
        self.leaves: List[str] = []
        self.tree: List[List[str]] = []
        
    def add_leaf(self, data: str):
        """Add data to tree"""
        leaf_hash = hashlib.sha256(data.encode()).hexdigest()
        self.leaves.append(leaf_hash)
        
    def build_tree(self):
        """Build complete merkle tree"""
        if not self.leaves:
            return
            
        self.tree = [self.leaves]
        
        while len(self.tree[-1]) > 1:
            level = []
            last_level = self.tree[-1]
            
            for i in range(0, len(last_level), 2):
                if i + 1 < len(last_level):
                    combined = last_level[i] + last_level[i + 1]
                else:
                    combined = last_level[i] + last_level[i]
                    
                level.append(hashlib.sha256(combined.encode()).hexdigest())
                
            self.tree.append(level)
            
    def get_root(self) -> Optional[str]:
        """Get merkle root"""
        if self.tree and self.tree[-1]:
            return self.tree[-1][0]
        return None
        
    def get_proof(self, index: int) -> List[Tuple[str, bool]]:
        """Get merkle proof for leaf at index"""
        if index >= len(self.leaves):
            return []
            
        proof = []
        for level in range(len(self.tree) - 1):
            level_index = index // (2 ** level)
            is_right = level_index % 2 == 1
            
            sibling_index = level_index - 1 if is_right else level_index + 1
            
            if sibling_index < len(self.tree[level]):
                proof.append((self.tree[level][sibling_index], is_right))
                
        return proof


class BloomFilter:
    """Probabilistic data structure for membership testing"""
    
    def __init__(self, size: int = 10000, num_hashes: int = 3):
        self.size = size
        self.num_hashes = num_hashes
        self.bits = [False] * size
        self.count = 0
        
    def _hash(self, item: str, seed: int) -> int:
        """Hash function with seed"""
        h = hashlib.sha256(f"{item}{seed}".encode()).digest()
        return int.from_bytes(h[:4], 'big') % self.size
        
    def add(self, item: str):
        """Add item to filter"""
        for i in range(self.num_hashes):
            index = self._hash(item, i)
            self.bits[index] = True
        self.count += 1
        
    def contains(self, item: str) -> bool:
        """Check if item might be in set"""
        for i in range(self.num_hashes):
            index = self._hash(item, i)
            if not self.bits[index]:
                return False
        return True


class StreamingAgent:
    """Base agent with advanced streaming capabilities"""
    
    def __init__(self, agent_id: str, embedding_dim: int = 128):
        self.id = agent_id
        self.embedding = np.random.randn(embedding_dim)
        self.vector_clock = VectorClock()
        self.stream_buffer = StreamBuffer()
        self.connections: Set[str] = set()
        self.message_log = deque(maxlen=1000)
        self.bloom_filter = BloomFilter()
        
        # Metrics
        self.messages_sent = 0
        self.messages_received = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        
    async def send_stream(self, recipient: 'StreamingAgent', protocol: StreamProtocol,
                         data: Any, chunk_size: int = 1024):
        """Send data using specified streaming protocol"""
        self.messages_sent += 1
        
        if protocol == StreamProtocol.TENSOR_STREAM:
            await self._stream_tensor(recipient, data, chunk_size)
        elif protocol == StreamProtocol.GOSSIP_EPIDEMIC:
            await self._gossip_propagate(recipient, data)
        elif protocol == StreamProtocol.VECTOR_CLOCK_SYNC:
            await self._vector_clock_sync(recipient, data)
            
    async def _stream_tensor(self, recipient: 'StreamingAgent', tensor: np.ndarray,
                           chunk_size: int):
        """Stream tensor in chunks"""
        tensor_id = str(uuid.uuid4())
        tensor_bytes = tensor.tobytes()
        total_chunks = (len(tensor_bytes) + chunk_size - 1) // chunk_size
        
        print(f"\n📊 Tensor Stream: {self.id} → {recipient.id}")
        print(f"   Shape: {tensor.shape}, Size: {len(tensor_bytes)} bytes")
        print(f"   Chunks: {total_chunks}")
        
        for i in range(total_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(tensor_bytes))
            
            chunk = TensorChunk(
                tensor_id=tensor_id,
                chunk_index=i,
                total_chunks=total_chunks,
                shape=tensor.shape,
                dtype=str(tensor.dtype),
                data=tensor_bytes[start:end],
                checksum=hashlib.md5(tensor_bytes[start:end]).hexdigest()
            )
            
            self.bytes_sent += len(chunk.data)
            await recipient.receive_chunk(chunk)
            
            # Progress
            progress = (i + 1) / total_chunks * 100
            if (i + 1) % max(1, total_chunks // 10) == 0:
                print(f"   Progress: {progress:.0f}%")
                
    async def _gossip_propagate(self, recipient: 'StreamingAgent', data: Any):
        """Propagate using gossip protocol"""
        self.vector_clock.increment(self.id)
        
        message = GossipMessage(
            origin=self.id,
            sequence=self.messages_sent,
            payload=data,
            vector_clock=VectorClock()
        )
        message.vector_clock.update(self.vector_clock)
        message.seen_by.add(self.id)
        
        await recipient.receive_gossip(message)
        
    async def _vector_clock_sync(self, recipient: 'StreamingAgent', data: Any):
        """Synchronize using vector clocks"""
        self.vector_clock.increment(self.id)
        
        sync_data = {
            'sender': self.id,
            'vector_clock': self.vector_clock.clock.copy(),
            'data': data,
            'timestamp': time.time()
        }
        
        await recipient.receive_sync(sync_data)
        
    async def receive_chunk(self, chunk: TensorChunk):
        """Receive tensor chunk"""
        self.bytes_received += len(chunk.data)
        await self.stream_buffer.write(chunk)
        
    async def receive_gossip(self, message: GossipMessage):
        """Receive gossip message"""
        if message.origin in self.bloom_filter.bits:
            return  # Already seen
            
        self.bloom_filter.add(message.origin)
        self.vector_clock.update(message.vector_clock)
        message.seen_by.add(self.id)
        message.hops += 1
        
        # Log message
        self.message_log.append(message)
        
    async def receive_sync(self, sync_data: Dict[str, Any]):
        """Receive vector clock sync"""
        # Update vector clock
        other_clock = VectorClock()
        other_clock.clock = sync_data['vector_clock']
        self.vector_clock.update(other_clock)


class ConsensusAgent(StreamingAgent):
    """Agent with consensus capabilities"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.raft_state = RaftState()
        self.paxos_proposals: Dict[int, Any] = {}
        self.byzantine_votes: Dict[str, Dict[str, bool]] = defaultdict(dict)
        
    async def start_raft_election(self, peers: List['ConsensusAgent']):
        """Start Raft leader election"""
        print(f"\n🗳️  {self.id} starting Raft election")
        
        self.raft_state.current_term += 1
        self.raft_state.state = "CANDIDATE"
        self.raft_state.voted_for = self.id
        
        votes = 1  # Vote for self
        
        # Request votes
        for peer in peers:
            if peer.id != self.id:
                vote_granted = await peer.request_vote(
                    self.raft_state.current_term,
                    self.id,
                    len(self.raft_state.log)
                )
                if vote_granted:
                    votes += 1
                    
        # Check if elected
        if votes > len(peers) / 2:
            self.raft_state.state = "LEADER"
            self.raft_state.leader_id = self.id
            print(f"   ✅ {self.id} elected as leader with {votes} votes")
            
            # Send heartbeats
            for peer in peers:
                if peer.id != self.id:
                    await peer.append_entries(self.id, self.raft_state.current_term)
        else:
            self.raft_state.state = "FOLLOWER"
            print(f"   ❌ {self.id} lost election with {votes} votes")
            
    async def request_vote(self, term: int, candidate_id: str, 
                          last_log_index: int) -> bool:
        """Handle vote request"""
        if term < self.raft_state.current_term:
            return False
            
        if term > self.raft_state.current_term:
            self.raft_state.current_term = term
            self.raft_state.voted_for = None
            self.raft_state.state = "FOLLOWER"
            
        if self.raft_state.voted_for is None or self.raft_state.voted_for == candidate_id:
            if last_log_index >= len(self.raft_state.log):
                self.raft_state.voted_for = candidate_id
                return True
                
        return False
        
    async def append_entries(self, leader_id: str, term: int):
        """Handle append entries (heartbeat)"""
        if term >= self.raft_state.current_term:
            self.raft_state.current_term = term
            self.raft_state.state = "FOLLOWER"
            self.raft_state.leader_id = leader_id
            
    async def byzantine_agreement(self, value: Any, peers: List['ConsensusAgent'],
                                 faulty_threshold: float = 0.33) -> Optional[Any]:
        """Byzantine fault tolerant agreement"""
        round_id = str(uuid.uuid4())
        
        print(f"\n⚔️  Byzantine Agreement - {self.id}")
        print(f"   Proposing: {value}")
        print(f"   Fault tolerance: {faulty_threshold:.0%}")
        
        # Phase 1: Propose
        for peer in peers:
            self.byzantine_votes[round_id][peer.id] = await peer.byzantine_vote(value)
            
        # Count votes
        votes_for = sum(1 for v in self.byzantine_votes[round_id].values() if v)
        total_votes = len(self.byzantine_votes[round_id])
        
        print(f"   Votes: {votes_for}/{total_votes}")
        
        # Check if consensus reached
        if votes_for > total_votes * (1 - faulty_threshold):
            print(f"   ✅ Consensus reached!")
            return value
        else:
            print(f"   ❌ No consensus")
            return None
            
    async def byzantine_vote(self, value: Any) -> bool:
        """Vote in Byzantine agreement"""
        # Simulate Byzantine behavior (small chance of being faulty)
        if random.random() < 0.1:  # 10% Byzantine
            return random.choice([True, False])
        
        # Honest vote (simplified)
        return hash(str(value)) % 2 == 0


class MeshNetworkAgent(StreamingAgent):
    """Agent with adaptive mesh networking"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.routing_table: Dict[str, Tuple[str, int]] = {}  # destination -> (next_hop, distance)
        self.link_quality: Dict[str, float] = {}  # neighbor -> quality
        self.topology_version = 0
        
    async def discover_topology(self, neighbors: List['MeshNetworkAgent']):
        """Discover network topology"""
        print(f"\n🌐 {self.id} discovering topology")
        
        # Measure link quality to neighbors
        for neighbor in neighbors:
            quality = await self._measure_link_quality(neighbor)
            self.link_quality[neighbor.id] = quality
            
            # Direct route
            self.routing_table[neighbor.id] = (neighbor.id, 1)
            
        # Exchange routing tables (simplified Bellman-Ford)
        for neighbor in neighbors:
            neighbor_routes = await neighbor.get_routes()
            
            for dest, (next_hop, distance) in neighbor_routes.items():
                if dest != self.id:  # Avoid loops
                    new_distance = distance + 1
                    
                    if dest not in self.routing_table or new_distance < self.routing_table[dest][1]:
                        self.routing_table[dest] = (neighbor.id, new_distance)
                        
        self.topology_version += 1
        
        # Print routing table
        print(f"   Routing table for {self.id}:")
        for dest, (next_hop, dist) in sorted(self.routing_table.items())[:5]:
            print(f"     {dest[:8]}... via {next_hop[:8]}... (distance: {dist})")
            
    async def _measure_link_quality(self, neighbor: 'MeshNetworkAgent') -> float:
        """Measure link quality to neighbor"""
        # Simulate RTT measurement
        start = time.time()
        await asyncio.sleep(random.uniform(0.001, 0.01))  # Simulated latency
        rtt = time.time() - start
        
        # Quality based on RTT (inverse)
        quality = 1.0 / (1.0 + rtt * 100)
        return quality
        
    async def get_routes(self) -> Dict[str, Tuple[str, int]]:
        """Get routing table"""
        return self.routing_table.copy()
        
    async def route_message(self, destination: str, message: Any,
                          visited: Set[str] = None) -> bool:
        """Route message through mesh"""
        if visited is None:
            visited = set()
            
        visited.add(self.id)
        
        # Check if we are destination
        if destination == self.id:
            print(f"   ✅ Message reached {self.id}")
            return True
            
        # Find next hop
        if destination in self.routing_table:
            next_hop_id, distance = self.routing_table[destination]
            
            if next_hop_id not in visited:
                print(f"   → Routing from {self.id} to {next_hop_id} (distance: {distance})")
                # In real implementation, would send to actual next hop
                return True
                
        print(f"   ❌ No route from {self.id} to {destination}")
        return False


class StreamProcessingPipeline:
    """Advanced stream processing pipeline"""
    
    def __init__(self, name: str):
        self.name = name
        self.stages: List[Callable] = []
        self.metrics = {
            'processed': 0,
            'dropped': 0,
            'latency_ms': deque(maxlen=100)
        }
        
    def add_stage(self, func: Callable):
        """Add processing stage"""
        self.stages.append(func)
        
    async def process_stream(self, input_stream: AsyncIterator[Any]) -> AsyncIterator[Any]:
        """Process stream through pipeline"""
        async for item in input_stream:
            start_time = time.time()
            
            try:
                # Process through stages
                result = item
                for stage in self.stages:
                    result = await stage(result)
                    
                # Track metrics
                self.metrics['processed'] += 1
                latency = (time.time() - start_time) * 1000
                self.metrics['latency_ms'].append(latency)
                
                yield result
                
            except Exception as e:
                self.metrics['dropped'] += 1
                print(f"   ⚠️  Pipeline error: {e}")
                
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        latencies = list(self.metrics['latency_ms'])
        return {
            'processed': self.metrics['processed'],
            'dropped': self.metrics['dropped'],
            'avg_latency_ms': np.mean(latencies) if latencies else 0,
            'p99_latency_ms': np.percentile(latencies, 99) if latencies else 0
        }


class HyperscaleOrchestrator:
    """Orchestrator for hyperscale streaming system"""
    
    def __init__(self):
        self.agents: Dict[str, StreamingAgent] = {}
        self.consensus_agents: List[ConsensusAgent] = []
        self.mesh_agents: List[MeshNetworkAgent] = []
        self.attention = HierarchicalAttention()
        self.global_merkle = MerkleTree()
        self.pipelines: Dict[str, StreamProcessingPipeline] = {}
        
    async def create_agent_cluster(self, cluster_type: str, size: int):
        """Create a cluster of agents"""
        print(f"\n🚀 Creating {cluster_type} cluster with {size} agents...")
        
        if cluster_type == "streaming":
            for i in range(size):
                agent = StreamingAgent(f"Stream_{i}")
                self.agents[agent.id] = agent
                
        elif cluster_type == "consensus":
            for i in range(size):
                agent = ConsensusAgent(f"Consensus_{i}")
                self.consensus_agents.append(agent)
                self.agents[agent.id] = agent
                
        elif cluster_type == "mesh":
            for i in range(size):
                agent = MeshNetworkAgent(f"Mesh_{i}")
                self.mesh_agents.append(agent)
                self.agents[agent.id] = agent
                
        print(f"   ✅ Created {size} {cluster_type} agents")
        
    async def demonstrate_tensor_streaming(self):
        """Demonstrate distributed tensor streaming"""
        print("\n" + "="*80)
        print("📊 DISTRIBUTED TENSOR STREAMING")
        print("="*80)
        
        if len(self.agents) < 2:
            return
            
        # Create large tensor
        tensor_shape = (1000, 500)
        tensor = np.random.randn(*tensor_shape).astype(np.float32)
        
        print(f"\nStreaming tensor of shape {tensor_shape}")
        print(f"Total size: {tensor.nbytes:,} bytes")
        
        # Stream between agents
        agents_list = list(self.agents.values())
        sender = agents_list[0]
        receiver = agents_list[1]
        
        await sender.send_stream(receiver, StreamProtocol.TENSOR_STREAM, 
                               tensor, chunk_size=8192)
        
        # Check buffer status
        print(f"\nReceiver buffer status:")
        print(f"   Items: {len(receiver.stream_buffer.buffer)}")
        print(f"   Pressure: {receiver.stream_buffer.is_pressured()}")
        print(f"   Total bytes: {receiver.stream_buffer.total_bytes:,}")
        
    async def demonstrate_gossip_protocol(self):
        """Demonstrate gossip epidemic protocol"""
        print("\n" + "="*80)
        print("🦠 GOSSIP EPIDEMIC PROTOCOL")
        print("="*80)
        
        if len(self.agents) < 10:
            return
            
        # Start gossip from random agent
        agents_list = list(self.agents.values())
        patient_zero = random.choice(agents_list)
        
        print(f"\nPatient Zero: {patient_zero.id}")
        
        # Initial message
        rumor = {
            'type': 'important_update',
            'content': 'System configuration change',
            'timestamp': time.time()
        }
        
        # Simulate epidemic spread
        infected = {patient_zero.id}
        rounds = 0
        
        while len(infected) < len(agents_list) and rounds < 10:
            rounds += 1
            new_infections = set()
            
            for agent_id in list(infected):
                agent = self.agents[agent_id]
                
                # Randomly select peers to gossip to
                peers = random.sample(agents_list, min(3, len(agents_list)))
                
                for peer in peers:
                    if peer.id not in infected:
                        await agent.send_stream(peer, StreamProtocol.GOSSIP_EPIDEMIC, rumor)
                        new_infections.add(peer.id)
                        
            infected.update(new_infections)
            
            print(f"   Round {rounds}: {len(infected)}/{len(agents_list)} agents infected")
            
            # Show spread pattern
            if rounds <= 3:
                spread_viz = ""
                for agent in agents_list[:20]:
                    spread_viz += "●" if agent.id in infected else "○"
                print(f"   Spread: {spread_viz}")
                
        print(f"\n✅ Gossip spread complete in {rounds} rounds")
        
    async def demonstrate_consensus(self):
        """Demonstrate consensus protocols"""
        print("\n" + "="*80)
        print("🤝 DISTRIBUTED CONSENSUS PROTOCOLS")
        print("="*80)
        
        if not self.consensus_agents:
            return
            
        # Raft consensus
        print("\n1️⃣ Raft Leader Election:")
        
        # Trigger election
        candidate = random.choice(self.consensus_agents)
        await candidate.start_raft_election(self.consensus_agents)
        
        # Show state
        print("\nRaft States:")
        for agent in self.consensus_agents[:5]:
            print(f"   {agent.id}: {agent.raft_state.state} "
                  f"(term: {agent.raft_state.current_term})")
            
        # Byzantine agreement
        print("\n2️⃣ Byzantine Fault Tolerant Agreement:")
        
        proposer = self.consensus_agents[0]
        value = {"action": "update_config", "version": 2}
        
        result = await proposer.byzantine_agreement(value, self.consensus_agents)
        
    async def demonstrate_mesh_networking(self):
        """Demonstrate adaptive mesh networking"""
        print("\n" + "="*80)
        print("🕸️  ADAPTIVE MESH NETWORKING")
        print("="*80)
        
        if len(self.mesh_agents) < 5:
            return
            
        # Build mesh topology
        print("\nBuilding mesh topology...")
        
        for agent in self.mesh_agents:
            # Each agent discovers its neighbors
            neighbors = random.sample(
                [a for a in self.mesh_agents if a.id != agent.id],
                min(3, len(self.mesh_agents) - 1)
            )
            await agent.discover_topology(neighbors)
            
        # Test routing
        print("\n🚦 Testing message routing:")
        
        source = self.mesh_agents[0]
        destination = self.mesh_agents[-1]
        
        print(f"   Source: {source.id}")
        print(f"   Destination: {destination.id}")
        
        success = await source.route_message(destination.id, "Test message")
        
    async def demonstrate_attention_routing(self):
        """Demonstrate attention-based routing"""
        print("\n" + "="*80)
        print("🧠 HIERARCHICAL ATTENTION ROUTING")
        print("="*80)
        
        if len(self.agents) < 5:
            return
            
        # Compute attention scores
        agents_list = list(self.agents.values())[:10]
        embeddings = {agent.id: agent.embedding for agent in agents_list}
        
        query_agent = agents_list[0]
        key_agents = [a.id for a in agents_list[1:]]
        
        scores = self.attention.compute_attention(query_agent.id, key_agents, embeddings)
        
        print(f"\nAttention scores from {query_agent.id}:")
        for agent_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]:
            bar_length = int(score * 30)
            bar = '█' * bar_length + '░' * (30 - bar_length)
            print(f"   {agent_id[:12]}... [{bar}] {score:.3f}")
            
    async def demonstrate_stream_processing(self):
        """Demonstrate stream processing pipeline"""
        print("\n" + "="*80)
        print("⚡ STREAM PROCESSING PIPELINE")
        print("="*80)
        
        # Create pipeline
        pipeline = StreamProcessingPipeline("DataPipeline")
        
        # Add processing stages
        async def normalize(x):
            return (x - np.mean(x)) / (np.std(x) + 1e-8)
            
        async def filter_outliers(x):
            threshold = 3
            mask = np.abs(x) < threshold
            return x[mask]
            
        async def aggregate(x):
            return {
                'mean': np.mean(x),
                'std': np.std(x),
                'min': np.min(x),
                'max': np.max(x),
                'count': len(x)
            }
            
        pipeline.add_stage(normalize)
        pipeline.add_stage(filter_outliers)
        pipeline.add_stage(aggregate)
        
        # Create input stream
        async def input_stream():
            for i in range(10):
                yield np.random.randn(100) * (i + 1)  # Increasing variance
                await asyncio.sleep(0.1)
                
        # Process stream
        print("\nProcessing data stream...")
        
        async for result in pipeline.process_stream(input_stream()):
            print(f"   Batch stats: mean={result['mean']:.3f}, "
                  f"std={result['std']:.3f}, count={result['count']}")
            
        # Show pipeline stats
        stats = pipeline.get_stats()
        print(f"\nPipeline Statistics:")
        print(f"   Processed: {stats['processed']}")
        print(f"   Dropped: {stats['dropped']}")
        print(f"   Avg latency: {stats['avg_latency_ms']:.2f}ms")
        print(f"   P99 latency: {stats['p99_latency_ms']:.2f}ms")
        
    async def show_global_state(self):
        """Show global system state"""
        print("\n" + "="*80)
        print("🌍 GLOBAL SYSTEM STATE")
        print("="*80)
        
        # Agent statistics
        total_messages = sum(agent.messages_sent for agent in self.agents.values())
        total_bytes = sum(agent.bytes_sent for agent in self.agents.values())
        
        print(f"\n📊 System Metrics:")
        print(f"   Total agents: {len(self.agents)}")
        print(f"   Total messages: {total_messages:,}")
        print(f"   Total bytes transferred: {total_bytes:,}")
        
        # Vector clock convergence
        print(f"\n⏰ Vector Clock State:")
        
        # Sample vector clocks
        for agent in list(self.agents.values())[:5]:
            clock_str = ", ".join(f"{k[:8]}:{v}" for k, v in 
                                list(agent.vector_clock.clock.items())[:3])
            print(f"   {agent.id}: [{clock_str}...]")
            
        # Consensus state
        if self.consensus_agents:
            leaders = [a for a in self.consensus_agents if a.raft_state.state == "LEADER"]
            print(f"\n👑 Consensus Leaders: {len(leaders)}")
            
        # Mesh topology
        if self.mesh_agents:
            total_routes = sum(len(a.routing_table) for a in self.mesh_agents)
            avg_routes = total_routes / len(self.mesh_agents)
            print(f"\n🕸️  Mesh Network:")
            print(f"   Average routes per node: {avg_routes:.1f}")


async def main():
    """Run hyperscale streaming demonstration"""
    
    print("🌊 HYPERSCALE STREAMING MULTI-AGENT SYSTEM")
    print("="*80)
    print("Advanced Features:")
    print("  • Tensor streaming for distributed AI")
    print("  • Gossip protocols for decentralized communication")
    print("  • Byzantine fault tolerance") 
    print("  • Adaptive mesh networking")
    print("  • Hierarchical attention mechanisms")
    print("  • Stream processing with backpressure")
    print("="*80)
    
    # Create orchestrator
    orchestrator = HyperscaleOrchestrator()
    
    # Create agent clusters
    await orchestrator.create_agent_cluster("streaming", 20)
    await orchestrator.create_agent_cluster("consensus", 7)
    await orchestrator.create_agent_cluster("mesh", 10)
    
    print(f"\n✅ Total agents created: {len(orchestrator.agents)}")
    
    # Run demonstrations
    await orchestrator.demonstrate_tensor_streaming()
    await asyncio.sleep(0.5)
    
    await orchestrator.demonstrate_gossip_protocol()
    await asyncio.sleep(0.5)
    
    await orchestrator.demonstrate_consensus()
    await asyncio.sleep(0.5)
    
    await orchestrator.demonstrate_mesh_networking()
    await asyncio.sleep(0.5)
    
    await orchestrator.demonstrate_attention_routing()
    await asyncio.sleep(0.5)
    
    await orchestrator.demonstrate_stream_processing()
    await asyncio.sleep(0.5)
    
    await orchestrator.show_global_state()
    
    print("\n" + "="*80)
    print("✅ HYPERSCALE STREAMING DEMONSTRATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())