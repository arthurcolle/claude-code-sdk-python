#!/usr/bin/env python3
"""
Timeline Tools Utilities

Comprehensive utilities for:
- Retrieving tools across entire timeline history
- Inspecting token DNA and genetic evolution
- Analyzing tool mutations and heredity
- Cross-timeline tool comparison
- Temporal tool archaeology
"""

import asyncio
import json
import pickle
import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import networkx as nx
import difflib
import re
from pathlib import Path

from omniscient_conversation_matrix import (
    TokenDNA, ThoughtCrystal, RetrocausalEngine,
    OmniscientConversationMatrix
)
from ultra_advanced_grid import AutonomousTool
from dynamic_tools_framework import ToolSignature, DynamicToolFactory


@dataclass
class ToolSnapshot:
    """Snapshot of a tool at a specific point in time"""
    tool_id: str
    timestamp: datetime
    timeline_id: str
    source_code: str
    signature: ToolSignature
    execution_count: int
    success_rate: float
    mutations: List[str]
    parent_tools: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_hash(self) -> str:
        """Get unique hash of tool state"""
        data = f"{self.tool_id}:{self.timestamp}:{self.source_code}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class DNAInspectionResult:
    """Result of DNA inspection"""
    token: str
    dna: TokenDNA
    genetic_markers: Dict[str, Any]
    mutation_rate: float
    evolutionary_fitness: float
    lineage_depth: int
    genetic_diversity: float
    epigenetic_factors: Dict[str, Any]
    related_tokens: List[Tuple[str, float]]  # (token, similarity)


class TimelineToolsManager:
    """Manages tools across all timelines and realities"""
    
    def __init__(self):
        self.tool_snapshots: Dict[str, List[ToolSnapshot]] = defaultdict(list)
        self.timeline_graph = nx.DiGraph()
        self.tool_evolution_tree = nx.DiGraph()
        self.dna_database: Dict[str, TokenDNA] = {}
        self.timeline_states: Dict[str, Dict[str, Any]] = {}
        
    def capture_tool_snapshot(self, tool: Any, timeline_id: str, 
                            context: Dict[str, Any] = None) -> ToolSnapshot:
        """Capture a snapshot of a tool's current state"""
        snapshot = ToolSnapshot(
            tool_id=getattr(tool, 'tool_id', str(id(tool))),
            timestamp=datetime.now(),
            timeline_id=timeline_id,
            source_code=self._extract_source_code(tool),
            signature=self._extract_signature(tool),
            execution_count=getattr(tool, 'execution_count', 0),
            success_rate=getattr(tool, 'success_rate', 0.0),
            mutations=getattr(tool, 'mutations', []),
            parent_tools=getattr(tool, 'parent_tools', []),
            metadata=context or {}
        )
        
        self.tool_snapshots[tool.tool_id].append(snapshot)
        
        # Add to timeline graph
        self.timeline_graph.add_node(
            snapshot.get_hash(),
            snapshot=snapshot,
            timeline=timeline_id,
            timestamp=snapshot.timestamp
        )
        
        # Link to previous snapshot in same timeline
        timeline_snapshots = [
            s for s in self.tool_snapshots[tool.tool_id]
            if s.timeline_id == timeline_id
        ]
        if len(timeline_snapshots) > 1:
            prev_snapshot = timeline_snapshots[-2]
            self.timeline_graph.add_edge(
                prev_snapshot.get_hash(),
                snapshot.get_hash(),
                relationship='temporal_succession'
            )
        
        return snapshot
    
    def _extract_source_code(self, tool: Any) -> str:
        """Extract source code from tool"""
        if hasattr(tool, 'genome'):
            return tool.genome
        elif hasattr(tool, 'source_code'):
            return tool.source_code
        elif hasattr(tool, '__source__'):
            return tool.__source__
        else:
            # Try to get source from class
            import inspect
            try:
                return inspect.getsource(tool.__class__)
            except:
                return "# Source code unavailable"
    
    def _extract_signature(self, tool: Any) -> ToolSignature:
        """Extract or reconstruct tool signature"""
        if hasattr(tool, 'signature'):
            return tool.signature
        else:
            # Reconstruct signature
            return ToolSignature(
                name=getattr(tool, 'name', tool.__class__.__name__),
                parameters=getattr(tool, 'parameters', {}),
                returns=getattr(tool, 'returns', Any),
                description=getattr(tool, '__doc__', "No description")
            )
    
    def get_tool_across_timelines(self, tool_id: str) -> Dict[str, List[ToolSnapshot]]:
        """Get all versions of a tool across all timelines"""
        timeline_versions = defaultdict(list)
        
        for snapshot in self.tool_snapshots.get(tool_id, []):
            timeline_versions[snapshot.timeline_id].append(snapshot)
        
        # Sort by timestamp within each timeline
        for timeline_id in timeline_versions:
            timeline_versions[timeline_id].sort(key=lambda s: s.timestamp)
        
        return dict(timeline_versions)
    
    def get_tool_at_time(self, tool_id: str, timestamp: datetime, 
                        timeline_id: Optional[str] = None) -> Optional[ToolSnapshot]:
        """Get tool state at specific timestamp"""
        candidates = []
        
        for snapshot in self.tool_snapshots.get(tool_id, []):
            if snapshot.timestamp <= timestamp:
                if timeline_id is None or snapshot.timeline_id == timeline_id:
                    candidates.append(snapshot)
        
        if candidates:
            # Return most recent before timestamp
            return max(candidates, key=lambda s: s.timestamp)
        
        return None
    
    def trace_tool_evolution(self, tool_id: str) -> nx.DiGraph:
        """Trace the evolutionary history of a tool"""
        evolution_graph = nx.DiGraph()
        
        snapshots = self.tool_snapshots.get(tool_id, [])
        
        for snapshot in snapshots:
            evolution_graph.add_node(
                snapshot.get_hash(),
                tool_id=tool_id,
                timestamp=snapshot.timestamp,
                timeline=snapshot.timeline_id,
                mutations=len(snapshot.mutations)
            )
            
            # Link to parents
            for parent_id in snapshot.parent_tools:
                parent_snapshots = self.tool_snapshots.get(parent_id, [])
                for parent_snapshot in parent_snapshots:
                    if parent_snapshot.timestamp < snapshot.timestamp:
                        evolution_graph.add_edge(
                            parent_snapshot.get_hash(),
                            snapshot.get_hash(),
                            relationship='evolution'
                        )
        
        return evolution_graph
    
    def find_tool_divergence_points(self, tool_id: str) -> List[Dict[str, Any]]:
        """Find points where tool evolution diverged across timelines"""
        divergence_points = []
        
        timeline_versions = self.get_tool_across_timelines(tool_id)
        
        if len(timeline_versions) < 2:
            return divergence_points
        
        # Compare tools across timelines at same time points
        all_timestamps = set()
        for snapshots in timeline_versions.values():
            all_timestamps.update(s.timestamp for s in snapshots)
        
        for timestamp in sorted(all_timestamps):
            versions_at_time = {}
            
            for timeline_id, snapshots in timeline_versions.items():
                # Find snapshot closest to this timestamp
                valid_snapshots = [s for s in snapshots if s.timestamp <= timestamp]
                if valid_snapshots:
                    versions_at_time[timeline_id] = max(valid_snapshots, 
                                                      key=lambda s: s.timestamp)
            
            if len(versions_at_time) > 1:
                # Compare source codes
                source_codes = {
                    tid: snapshot.source_code 
                    for tid, snapshot in versions_at_time.items()
                }
                
                unique_sources = set(source_codes.values())
                
                if len(unique_sources) > 1:
                    divergence_points.append({
                        'timestamp': timestamp,
                        'timelines': list(versions_at_time.keys()),
                        'num_variants': len(unique_sources),
                        'snapshots': versions_at_time
                    })
        
        return divergence_points
    
    def merge_tool_histories(self, tool_id: str, timeline1: str, 
                           timeline2: str, merge_strategy: str = "union") -> ToolSnapshot:
        """Merge tool histories from different timelines"""
        versions1 = [s for s in self.tool_snapshots.get(tool_id, []) 
                    if s.timeline_id == timeline1]
        versions2 = [s for s in self.tool_snapshots.get(tool_id, []) 
                    if s.timeline_id == timeline2]
        
        if not versions1 or not versions2:
            raise ValueError("Tool not found in one or both timelines")
        
        latest1 = max(versions1, key=lambda s: s.timestamp)
        latest2 = max(versions2, key=lambda s: s.timestamp)
        
        if merge_strategy == "union":
            # Combine mutations
            merged_mutations = list(set(latest1.mutations + latest2.mutations))
            
            # Merge source code (simple line-by-line union)
            lines1 = latest1.source_code.split('\n')
            lines2 = latest2.source_code.split('\n')
            merged_lines = list(set(lines1 + lines2))
            merged_source = '\n'.join(merged_lines)
            
        elif merge_strategy == "intersection":
            # Keep only common mutations
            merged_mutations = list(set(latest1.mutations) & set(latest2.mutations))
            
            # Keep only common lines
            lines1 = set(latest1.source_code.split('\n'))
            lines2 = set(latest2.source_code.split('\n'))
            merged_lines = lines1 & lines2
            merged_source = '\n'.join(sorted(merged_lines))
            
        elif merge_strategy == "smart":
            # Use diff-based merging
            merged_source = self._smart_merge_source(
                latest1.source_code, 
                latest2.source_code
            )
            merged_mutations = self._smart_merge_mutations(
                latest1.mutations,
                latest2.mutations
            )
        
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
        
        # Create merged snapshot
        merged_snapshot = ToolSnapshot(
            tool_id=f"{tool_id}_merged",
            timestamp=datetime.now(),
            timeline_id=f"{timeline1}+{timeline2}",
            source_code=merged_source,
            signature=latest1.signature,  # Use first timeline's signature
            execution_count=latest1.execution_count + latest2.execution_count,
            success_rate=(latest1.success_rate + latest2.success_rate) / 2,
            mutations=merged_mutations,
            parent_tools=[tool_id],
            metadata={
                'merge_strategy': merge_strategy,
                'source_timelines': [timeline1, timeline2]
            }
        )
        
        self.tool_snapshots[merged_snapshot.tool_id].append(merged_snapshot)
        
        return merged_snapshot
    
    def _smart_merge_source(self, source1: str, source2: str) -> str:
        """Perform intelligent source code merging"""
        # Use difflib to find common base and merge changes
        lines1 = source1.splitlines(keepends=True)
        lines2 = source2.splitlines(keepends=True)
        
        differ = difflib.Differ()
        diff = list(differ.compare(lines1, lines2))
        
        merged = []
        for line in diff:
            if line.startswith('  '):  # Common line
                merged.append(line[2:])
            elif line.startswith('+ '):  # Line only in source2
                merged.append(line[2:])
            elif line.startswith('- '):  # Line only in source1
                # Check if there's a corresponding + line (modification)
                # For now, keep the - line
                merged.append(line[2:])
        
        return ''.join(merged)
    
    def _smart_merge_mutations(self, mutations1: List[str], 
                             mutations2: List[str]) -> List[str]:
        """Intelligently merge mutation lists"""
        # Remove duplicates while preserving order
        seen = set()
        merged = []
        
        for mutation in mutations1 + mutations2:
            if mutation not in seen:
                seen.add(mutation)
                merged.append(mutation)
        
        return merged


class DNAInspector:
    """Advanced DNA inspection and analysis utilities"""
    
    def __init__(self):
        self.dna_cache: Dict[str, TokenDNA] = {}
        self.phylogenetic_tree = nx.DiGraph()
        self.genetic_distances: Dict[Tuple[str, str], float] = {}
        
    def inspect_dna(self, token: str, dna: TokenDNA = None) -> DNAInspectionResult:
        """Perform comprehensive DNA inspection"""
        if dna is None and token in self.dna_cache:
            dna = self.dna_cache[token]
        elif dna is None:
            dna = TokenDNA(token)
            self.dna_cache[token] = dna
        
        # Extract genetic markers
        genetic_markers = self._extract_genetic_markers(dna)
        
        # Calculate mutation rate
        mutation_rate = len(dna.mutation_history) / max(1, 
            (datetime.now() - dna.mutation_history[0]['timestamp']).total_seconds() / 3600
        ) if dna.mutation_history else 0.0
        
        # Assess evolutionary fitness
        fitness = self._calculate_fitness(dna)
        
        # Determine lineage depth
        lineage_depth = len(dna.lineage)
        
        # Calculate genetic diversity
        diversity = self._calculate_genetic_diversity(dna)
        
        # Extract epigenetic factors
        epigenetic_factors = dna.epigenetic_markers.copy()
        
        # Find related tokens
        related_tokens = self._find_genetically_similar_tokens(dna)
        
        return DNAInspectionResult(
            token=token,
            dna=dna,
            genetic_markers=genetic_markers,
            mutation_rate=mutation_rate,
            evolutionary_fitness=fitness,
            lineage_depth=lineage_depth,
            genetic_diversity=diversity,
            epigenetic_factors=epigenetic_factors,
            related_tokens=related_tokens
        )
    
    def _extract_genetic_markers(self, dna: TokenDNA) -> Dict[str, Any]:
        """Extract key genetic markers from DNA"""
        markers = {}
        
        # Analyze each gene
        for gene_name, gene_data in dna.genes.items():
            if isinstance(gene_data, bytes):
                # Calculate gene statistics
                markers[gene_name] = {
                    'length': len(gene_data),
                    'entropy': self._calculate_entropy(gene_data),
                    'gc_content': self._calculate_gc_content(gene_data),
                    'unique_sequences': len(set(gene_data))
                }
        
        # Analyze chromosomes
        markers['chromosomes'] = []
        for i, chromosome in enumerate(dna.chromosomes):
            markers['chromosomes'].append({
                'index': i,
                'length': len(chromosome),
                'mutations': sum(1 for m in dna.mutation_history 
                               if m.get('chromosome') == i)
            })
        
        return markers
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of byte sequence"""
        if not data:
            return 0.0
        
        # Count byte frequencies
        freq = defaultdict(int)
        for byte in data:
            freq[byte] += 1
        
        # Calculate entropy
        entropy = 0.0
        total = len(data)
        
        for count in freq.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p) if p > 0 else 0
        
        return entropy
    
    def _calculate_gc_content(self, data: bytes) -> float:
        """Calculate GC content (for genetic similarity to biology)"""
        if not data:
            return 0.0
        
        # Count 'G' and 'C' equivalent bytes (high values)
        gc_count = sum(1 for byte in data if byte > 127)
        
        return gc_count / len(data)
    
    def _calculate_fitness(self, dna: TokenDNA) -> float:
        """Calculate evolutionary fitness score"""
        fitness = 0.5  # Base fitness
        
        # Factors that increase fitness
        if dna.lineage:
            fitness += 0.1 * min(len(dna.lineage), 5)  # Successful lineage
        
        if dna.mutation_history:
            # Moderate mutation is good
            mutation_count = len(dna.mutation_history)
            if 1 <= mutation_count <= 5:
                fitness += 0.1
            elif mutation_count > 10:
                fitness -= 0.1  # Too many mutations
        
        # Epigenetic adaptations increase fitness
        fitness += 0.05 * min(len(dna.epigenetic_markers), 5)
        
        return max(0.0, min(1.0, fitness))
    
    def _calculate_genetic_diversity(self, dna: TokenDNA) -> float:
        """Calculate genetic diversity score"""
        if not dna.chromosomes:
            return 0.0
        
        # Measure variation across chromosomes
        all_bytes = b''.join(dna.chromosomes)
        unique_bytes = len(set(all_bytes))
        total_bytes = len(all_bytes)
        
        if total_bytes == 0:
            return 0.0
        
        return unique_bytes / min(total_bytes, 256)  # Normalize to [0, 1]
    
    def _find_genetically_similar_tokens(self, dna: TokenDNA, 
                                       threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find tokens with similar DNA"""
        similar_tokens = []
        
        for token, cached_dna in self.dna_cache.items():
            if cached_dna.token != dna.token:
                similarity = dna.similarity(cached_dna)
                if similarity >= threshold:
                    similar_tokens.append((token, similarity))
        
        # Sort by similarity
        similar_tokens.sort(key=lambda x: x[1], reverse=True)
        
        return similar_tokens[:10]  # Top 10
    
    def build_phylogenetic_tree(self, tokens: List[str]) -> nx.DiGraph:
        """Build phylogenetic tree from tokens"""
        tree = nx.DiGraph()
        
        # Get DNA for all tokens
        dnas = {}
        for token in tokens:
            if token in self.dna_cache:
                dnas[token] = self.dna_cache[token]
            else:
                dnas[token] = TokenDNA(token)
                self.dna_cache[token] = dnas[token]
        
        # Build tree based on lineage
        for token, dna in dnas.items():
            tree.add_node(token, dna=dna)
            
            # Connect to parents in lineage
            for parent in dna.lineage[:2]:  # Direct parents only
                if parent in dnas:
                    tree.add_edge(parent, token)
        
        # Add similarity-based connections for orphans
        orphans = [n for n in tree.nodes() if tree.in_degree(n) == 0]
        
        for orphan in orphans:
            orphan_dna = dnas[orphan]
            best_parent = None
            best_similarity = 0.0
            
            for candidate, candidate_dna in dnas.items():
                if candidate != orphan:
                    similarity = orphan_dna.similarity(candidate_dna)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_parent = candidate
            
            if best_parent and best_similarity > 0.5:
                tree.add_edge(best_parent, orphan, weight=best_similarity)
        
        self.phylogenetic_tree = tree
        return tree
    
    def calculate_genetic_distance_matrix(self, tokens: List[str]) -> Dict[Tuple[str, str], float]:
        """Calculate pairwise genetic distances"""
        distances = {}
        
        # Ensure all tokens have DNA
        for token in tokens:
            if token not in self.dna_cache:
                self.dna_cache[token] = TokenDNA(token)
        
        # Calculate pairwise distances
        for i, token1 in enumerate(tokens):
            for j, token2 in enumerate(tokens[i+1:], i+1):
                dna1 = self.dna_cache[token1]
                dna2 = self.dna_cache[token2]
                
                # Distance = 1 - similarity
                distance = 1.0 - dna1.similarity(dna2)
                
                distances[(token1, token2)] = distance
                distances[(token2, token1)] = distance
        
        self.genetic_distances = distances
        return distances
    
    def find_common_ancestor(self, token1: str, token2: str) -> Optional[str]:
        """Find most recent common ancestor of two tokens"""
        if token1 not in self.dna_cache or token2 not in self.dna_cache:
            return None
        
        dna1 = self.dna_cache[token1]
        dna2 = self.dna_cache[token2]
        
        # Check lineages
        lineage1 = set([token1] + dna1.lineage)
        lineage2 = set([token2] + dna2.lineage)
        
        common = lineage1 & lineage2
        
        if common:
            # Find most recent (appears earliest in lineages)
            for ancestor in dna1.lineage:
                if ancestor in common:
                    return ancestor
            for ancestor in dna2.lineage:
                if ancestor in common:
                    return ancestor
        
        return None
    
    def export_genetic_report(self, token: str, output_path: str = None) -> str:
        """Generate comprehensive genetic report for a token"""
        inspection = self.inspect_dna(token)
        
        report = []
        report.append(f"=== Genetic Report for Token: '{token}' ===")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Basic Information
        report.append("1. BASIC GENETIC INFORMATION")
        report.append("-" * 40)
        report.append(f"Token: {inspection.token}")
        report.append(f"Lineage Depth: {inspection.lineage_depth}")
        report.append(f"Mutation Rate: {inspection.mutation_rate:.4f} mutations/hour")
        report.append(f"Evolutionary Fitness: {inspection.evolutionary_fitness:.2%}")
        report.append(f"Genetic Diversity: {inspection.genetic_diversity:.2%}")
        report.append("")
        
        # Genetic Markers
        report.append("2. GENETIC MARKERS")
        report.append("-" * 40)
        for gene_name, markers in inspection.genetic_markers.items():
            if gene_name != 'chromosomes':
                report.append(f"\n{gene_name}:")
                for marker, value in markers.items():
                    report.append(f"  {marker}: {value}")
        report.append("")
        
        # Chromosomes
        report.append("3. CHROMOSOME ANALYSIS")
        report.append("-" * 40)
        for chrom in inspection.genetic_markers.get('chromosomes', []):
            report.append(f"Chromosome {chrom['index']}: "
                         f"Length={chrom['length']}, "
                         f"Mutations={chrom['mutations']}")
        report.append("")
        
        # Lineage
        report.append("4. LINEAGE")
        report.append("-" * 40)
        if inspection.dna.lineage:
            for i, ancestor in enumerate(inspection.dna.lineage[:10]):
                report.append(f"  Generation -{i+1}: {ancestor}")
        else:
            report.append("  No recorded lineage")
        report.append("")
        
        # Mutations
        report.append("5. MUTATION HISTORY")
        report.append("-" * 40)
        if inspection.dna.mutation_history:
            for mutation in inspection.dna.mutation_history[-5:]:
                report.append(f"  {mutation['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - "
                            f"Chromosome {mutation.get('chromosome', '?')} - "
                            f"{mutation.get('type', 'unknown')}")
        else:
            report.append("  No mutations recorded")
        report.append("")
        
        # Epigenetic Factors
        report.append("6. EPIGENETIC MODIFICATIONS")
        report.append("-" * 40)
        if inspection.epigenetic_factors:
            for marker, data in inspection.epigenetic_factors.items():
                report.append(f"  {marker}: {data['value']} "
                            f"(added {data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            report.append("  No epigenetic modifications")
        report.append("")
        
        # Related Tokens
        report.append("7. GENETICALLY RELATED TOKENS")
        report.append("-" * 40)
        if inspection.related_tokens:
            for related_token, similarity in inspection.related_tokens[:5]:
                report.append(f"  '{related_token}': {similarity:.2%} similarity")
        else:
            report.append("  No closely related tokens found")
        report.append("")
        
        # Genetic Sequence Sample
        report.append("8. GENETIC SEQUENCE SAMPLE")
        report.append("-" * 40)
        if inspection.dna.chromosomes:
            sample = inspection.dna.chromosomes[0][:32]
            report.append(f"  First 32 bytes of Chromosome 0:")
            report.append(f"  {sample.hex()}")
        report.append("")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return report_text


class CrossTimelineToolRetriever:
    """Retrieve and analyze tools across multiple timelines"""
    
    def __init__(self, timeline_manager: TimelineToolsManager, 
                 dna_inspector: DNAInspector):
        self.timeline_manager = timeline_manager
        self.dna_inspector = dna_inspector
        
    def get_all_versions(self, tool_id: str) -> List[ToolSnapshot]:
        """Get all versions of a tool across all timelines"""
        all_versions = []
        
        timeline_versions = self.timeline_manager.get_tool_across_timelines(tool_id)
        
        for timeline_id, snapshots in timeline_versions.items():
            all_versions.extend(snapshots)
        
        # Sort by timestamp
        all_versions.sort(key=lambda s: s.timestamp)
        
        return all_versions
    
    def find_tools_by_pattern(self, pattern: str, 
                            timeline_id: Optional[str] = None) -> List[ToolSnapshot]:
        """Find tools matching a pattern in source code"""
        matching_tools = []
        regex = re.compile(pattern)
        
        for tool_id, snapshots in self.timeline_manager.tool_snapshots.items():
            for snapshot in snapshots:
                if timeline_id and snapshot.timeline_id != timeline_id:
                    continue
                
                if regex.search(snapshot.source_code):
                    matching_tools.append(snapshot)
        
        return matching_tools
    
    def get_tool_evolution_summary(self, tool_id: str) -> Dict[str, Any]:
        """Get comprehensive evolution summary for a tool"""
        summary = {
            'tool_id': tool_id,
            'total_versions': 0,
            'timelines': set(),
            'total_mutations': 0,
            'divergence_points': [],
            'evolution_tree': None,
            'time_span': None,
            'success_trend': []
        }
        
        all_versions = self.get_all_versions(tool_id)
        
        if not all_versions:
            return summary
        
        summary['total_versions'] = len(all_versions)
        summary['timelines'] = {v.timeline_id for v in all_versions}
        summary['total_mutations'] = sum(len(v.mutations) for v in all_versions)
        
        # Time span
        first_time = min(v.timestamp for v in all_versions)
        last_time = max(v.timestamp for v in all_versions)
        summary['time_span'] = {
            'start': first_time,
            'end': last_time,
            'duration': (last_time - first_time).total_seconds() / 3600  # hours
        }
        
        # Success trend
        summary['success_trend'] = [
            {
                'timestamp': v.timestamp,
                'success_rate': v.success_rate,
                'timeline': v.timeline_id
            }
            for v in all_versions
        ]
        
        # Divergence points
        summary['divergence_points'] = self.timeline_manager.find_tool_divergence_points(tool_id)
        
        # Evolution tree
        summary['evolution_tree'] = self.timeline_manager.trace_tool_evolution(tool_id)
        
        return summary
    
    def compare_tools_across_timelines(self, tool_id: str, 
                                     timestamp: datetime) -> Dict[str, Any]:
        """Compare tool versions across timelines at specific timestamp"""
        comparison = {
            'tool_id': tool_id,
            'timestamp': timestamp,
            'timeline_versions': {},
            'differences': {},
            'consensus_features': []
        }
        
        timeline_versions = self.timeline_manager.get_tool_across_timelines(tool_id)
        
        # Get version at timestamp for each timeline
        for timeline_id, snapshots in timeline_versions.items():
            version = self.timeline_manager.get_tool_at_time(
                tool_id, timestamp, timeline_id
            )
            if version:
                comparison['timeline_versions'][timeline_id] = version
        
        # Calculate differences
        if len(comparison['timeline_versions']) > 1:
            versions_list = list(comparison['timeline_versions'].values())
            base_version = versions_list[0]
            
            for i, version in enumerate(versions_list[1:], 1):
                diff_key = f"{base_version.timeline_id}_vs_{version.timeline_id}"
                
                # Source code differences
                source_diff = list(difflib.unified_diff(
                    base_version.source_code.splitlines(),
                    version.source_code.splitlines(),
                    fromfile=base_version.timeline_id,
                    tofile=version.timeline_id,
                    lineterm=''
                ))
                
                comparison['differences'][diff_key] = {
                    'source_diff_lines': len(source_diff),
                    'mutation_diff': set(version.mutations) - set(base_version.mutations),
                    'success_rate_diff': version.success_rate - base_version.success_rate
                }
        
        # Find consensus features (present in all timelines)
        if comparison['timeline_versions']:
            all_source_lines = []
            for version in comparison['timeline_versions'].values():
                all_source_lines.append(set(version.source_code.splitlines()))
            
            if all_source_lines:
                consensus_lines = set.intersection(*all_source_lines)
                comparison['consensus_features'] = list(consensus_lines)[:10]
        
        return comparison


# Demonstration functions
async def demonstrate_timeline_tools():
    """Demonstrate timeline tools utilities"""
    print("=== Timeline Tools Utilities Demo ===\n")
    
    # Initialize managers
    timeline_manager = TimelineToolsManager()
    dna_inspector = DNAInspector()
    retriever = CrossTimelineToolRetriever(timeline_manager, dna_inspector)
    
    # Create some sample tools across timelines
    print("1. Creating Sample Tools Across Timelines")
    print("-" * 50)
    
    # Timeline 1: Original
    tool1_v1 = AutonomousTool(
        tool_id="calculator_v1",
        genome="""
def calculate(x, y):
    return x + y
""",
        generation=1
    )
    
    snapshot1 = timeline_manager.capture_tool_snapshot(
        tool1_v1, "timeline_main", 
        {'context': 'original_version'}
    )
    print(f"Captured: {snapshot1.tool_id} in {snapshot1.timeline_id}")
    
    # Timeline 1: Evolved version
    await asyncio.sleep(0.1)  # Ensure different timestamp
    
    tool1_v2 = AutonomousTool(
        tool_id="calculator_v1",
        genome="""
def calculate(x, y):
    # Improved version
    result = x + y
    return result * 1.1  # 10% boost
""",
        generation=2,
        mutations=["Added result variable", "Added 10% boost"]
    )
    
    snapshot2 = timeline_manager.capture_tool_snapshot(
        tool1_v2, "timeline_main",
        {'context': 'evolved_version'}
    )
    
    # Timeline 2: Alternative evolution
    tool1_alt = AutonomousTool(
        tool_id="calculator_v1",
        genome="""
def calculate(x, y):
    # Alternative approach
    if x > y:
        return (x + y) * 1.2
    else:
        return (x + y) * 0.9
""",
        generation=2,
        mutations=["Added conditional logic", "Variable boost"]
    )
    
    snapshot3 = timeline_manager.capture_tool_snapshot(
        tool1_alt, "timeline_branch",
        {'context': 'alternative_evolution'}
    )
    
    print(f"Created 3 snapshots across 2 timelines")
    
    # 2. DNA Inspection
    print("\n2. DNA Inspection")
    print("-" * 50)
    
    # Create token DNA
    tokens = ["calculate", "function", "evolve", "calculate_v2"]
    
    # First generation
    calc_dna = TokenDNA("calculate")
    func_dna = TokenDNA("function")
    
    # Create offspring
    evolve_dna = calc_dna.crossover(func_dna)
    evolve_dna.mutate(0.1)
    
    # Inspect DNA
    inspection = dna_inspector.inspect_dna("calculate", calc_dna)
    print(f"Token: {inspection.token}")
    print(f"Genetic Diversity: {inspection.genetic_diversity:.2%}")
    print(f"Evolutionary Fitness: {inspection.evolutionary_fitness:.2%}")
    
    # 3. Tool Evolution Analysis
    print("\n3. Tool Evolution Analysis")
    print("-" * 50)
    
    evolution_summary = retriever.get_tool_evolution_summary("calculator_v1")
    print(f"Tool: {evolution_summary['tool_id']}")
    print(f"Total Versions: {evolution_summary['total_versions']}")
    print(f"Timelines: {evolution_summary['timelines']}")
    print(f"Total Mutations: {evolution_summary['total_mutations']}")
    
    # 4. Cross-Timeline Comparison
    print("\n4. Cross-Timeline Comparison")
    print("-" * 50)
    
    comparison = retriever.compare_tools_across_timelines(
        "calculator_v1",
        datetime.now()
    )
    
    print(f"Comparing at: {comparison['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Versions found: {len(comparison['timeline_versions'])}")
    
    for timeline, version in comparison['timeline_versions'].items():
        print(f"  {timeline}: {len(version.source_code)} chars, "
              f"{len(version.mutations)} mutations")
    
    # 5. Find Divergence Points
    print("\n5. Divergence Points")
    print("-" * 50)
    
    divergence_points = timeline_manager.find_tool_divergence_points("calculator_v1")
    print(f"Found {len(divergence_points)} divergence points")
    
    for point in divergence_points:
        print(f"  At {point['timestamp'].strftime('%H:%M:%S')}: "
              f"{point['num_variants']} variants across "
              f"{len(point['timelines'])} timelines")
    
    # 6. Phylogenetic Tree
    print("\n6. Building Phylogenetic Tree")
    print("-" * 50)
    
    tree = dna_inspector.build_phylogenetic_tree(["calculate", "function", "evolve"])
    print(f"Tree nodes: {tree.number_of_nodes()}")
    print(f"Tree edges: {tree.number_of_edges()}")
    
    # 7. Genetic Report
    print("\n7. Genetic Report Generation")
    print("-" * 50)
    
    report = dna_inspector.export_genetic_report("calculate", "genetic_report.txt")
    print("Report preview:")
    print("\n".join(report.split("\n")[:20]))  # First 20 lines
    
    return timeline_manager, dna_inspector


# Utility functions for external use
def save_timeline_state(timeline_manager: TimelineToolsManager, filepath: str):
    """Save timeline manager state to file"""
    state = {
        'tool_snapshots': dict(timeline_manager.tool_snapshots),
        'timeline_states': timeline_manager.timeline_states,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(state, f)


def load_timeline_state(filepath: str) -> TimelineToolsManager:
    """Load timeline manager state from file"""
    manager = TimelineToolsManager()
    
    with open(filepath, 'rb') as f:
        state = pickle.load(f)
    
    manager.tool_snapshots = defaultdict(list, state['tool_snapshots'])
    manager.timeline_states = state.get('timeline_states', {})
    
    # Rebuild graphs
    for tool_id, snapshots in manager.tool_snapshots.items():
        for snapshot in snapshots:
            manager.timeline_graph.add_node(
                snapshot.get_hash(),
                snapshot=snapshot,
                timeline=snapshot.timeline_id,
                timestamp=snapshot.timestamp
            )
    
    return manager


def visualize_tool_evolution(timeline_manager: TimelineToolsManager, 
                           tool_id: str, output_path: str = None):
    """Create visual representation of tool evolution"""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    all_versions = []
    timeline_versions = timeline_manager.get_tool_across_timelines(tool_id)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Success rate over time
    for timeline_id, snapshots in timeline_versions.items():
        timestamps = [s.timestamp for s in snapshots]
        success_rates = [s.success_rate for s in snapshots]
        
        ax1.plot(timestamps, success_rates, marker='o', label=timeline_id)
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Success Rate')
    ax1.set_title(f'Tool Evolution: {tool_id}')
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    
    # Plot 2: Mutation accumulation
    for timeline_id, snapshots in timeline_versions.items():
        timestamps = [s.timestamp for s in snapshots]
        mutation_counts = [len(s.mutations) for s in snapshots]
        
        ax2.plot(timestamps, mutation_counts, marker='s', label=timeline_id)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Total Mutations')
    ax2.set_title('Mutation Accumulation')
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


if __name__ == "__main__":
    # Run demonstration
    timeline_manager, dna_inspector = asyncio.run(demonstrate_timeline_tools())