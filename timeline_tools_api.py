#!/usr/bin/env python3
"""
Timeline Tools API - High-level interface for timeline and DNA operations

Simple, clean API for:
- Tool retrieval and inspection
- DNA analysis and comparison
- Timeline navigation
- Evolution tracking
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import json
from pathlib import Path

from timeline_tools_utilities import (
    TimelineToolsManager, DNAInspector, CrossTimelineToolRetriever,
    ToolSnapshot, DNAInspectionResult
)
from omniscient_conversation_matrix import TokenDNA


class TimelineToolsAPI:
    """High-level API for timeline tools operations"""
    
    def __init__(self):
        self.timeline_manager = TimelineToolsManager()
        self.dna_inspector = DNAInspector()
        self.retriever = CrossTimelineToolRetriever(
            self.timeline_manager, 
            self.dna_inspector
        )
        
    # === Tool Operations ===
    
    def capture_tool(self, tool: Any, timeline: str = "main", 
                    metadata: Dict[str, Any] = None) -> str:
        """
        Capture a tool snapshot in the specified timeline.
        
        Args:
            tool: The tool object to capture
            timeline: Timeline identifier (default: "main")
            metadata: Optional metadata to attach
            
        Returns:
            Snapshot hash identifier
        """
        snapshot = self.timeline_manager.capture_tool_snapshot(
            tool, timeline, metadata or {}
        )
        return snapshot.get_hash()
    
    def get_tool(self, tool_id: str, timeline: Optional[str] = None,
                timestamp: Optional[datetime] = None) -> Optional[ToolSnapshot]:
        """
        Retrieve a tool snapshot.
        
        Args:
            tool_id: Tool identifier
            timeline: Optional timeline filter
            timestamp: Optional point-in-time retrieval
            
        Returns:
            Tool snapshot or None if not found
        """
        if timestamp:
            return self.timeline_manager.get_tool_at_time(
                tool_id, timestamp, timeline
            )
        else:
            # Get latest version
            all_versions = self.retriever.get_all_versions(tool_id)
            if timeline:
                all_versions = [v for v in all_versions 
                              if v.timeline_id == timeline]
            
            return all_versions[-1] if all_versions else None
    
    def get_tool_history(self, tool_id: str) -> List[Dict[str, Any]]:
        """
        Get complete history of a tool across all timelines.
        
        Returns:
            List of history entries with timeline, timestamp, and changes
        """
        history = []
        all_versions = self.retriever.get_all_versions(tool_id)
        
        for i, version in enumerate(all_versions):
            entry = {
                'version': i + 1,
                'timeline': version.timeline_id,
                'timestamp': version.timestamp.isoformat(),
                'mutations': version.mutations,
                'success_rate': version.success_rate,
                'hash': version.get_hash()
            }
            
            # Calculate changes from previous version
            if i > 0:
                prev = all_versions[i-1]
                entry['changes'] = {
                    'mutations_added': len(set(version.mutations) - set(prev.mutations)),
                    'success_rate_delta': version.success_rate - prev.success_rate,
                    'timeline_changed': version.timeline_id != prev.timeline_id
                }
            
            history.append(entry)
        
        return history
    
    def find_tools(self, pattern: str, timeline: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find tools matching a pattern.
        
        Args:
            pattern: Regex pattern to search for
            timeline: Optional timeline filter
            
        Returns:
            List of matching tools with metadata
        """
        matches = self.retriever.find_tools_by_pattern(pattern, timeline)
        
        return [
            {
                'tool_id': m.tool_id,
                'timeline': m.timeline_id,
                'timestamp': m.timestamp.isoformat(),
                'preview': m.source_code[:100] + '...' if len(m.source_code) > 100 else m.source_code
            }
            for m in matches
        ]
    
    def compare_timelines(self, tool_id: str, timeline1: str, 
                         timeline2: str) -> Dict[str, Any]:
        """
        Compare tool versions between two timelines.
        
        Returns:
            Comparison results including differences and similarities
        """
        # Get latest version in each timeline
        v1 = self.get_tool(tool_id, timeline1)
        v2 = self.get_tool(tool_id, timeline2)
        
        if not v1 or not v2:
            return {'error': 'Tool not found in one or both timelines'}
        
        return {
            'tool_id': tool_id,
            'timeline1': {
                'timeline': timeline1,
                'timestamp': v1.timestamp.isoformat(),
                'mutations': len(v1.mutations),
                'success_rate': v1.success_rate
            },
            'timeline2': {
                'timeline': timeline2,
                'timestamp': v2.timestamp.isoformat(),
                'mutations': len(v2.mutations),
                'success_rate': v2.success_rate
            },
            'differences': {
                'mutation_diff': list(set(v2.mutations) - set(v1.mutations)),
                'common_mutations': list(set(v1.mutations) & set(v2.mutations)),
                'source_similarity': self._calculate_source_similarity(
                    v1.source_code, v2.source_code
                )
            }
        }
    
    def merge_timelines(self, tool_id: str, timeline1: str, timeline2: str,
                       strategy: str = "smart") -> str:
        """
        Merge tool versions from different timelines.
        
        Args:
            tool_id: Tool to merge
            timeline1: First timeline
            timeline2: Second timeline
            strategy: Merge strategy ("union", "intersection", "smart")
            
        Returns:
            Hash of merged tool snapshot
        """
        merged = self.timeline_manager.merge_tool_histories(
            tool_id, timeline1, timeline2, strategy
        )
        return merged.get_hash()
    
    # === DNA Operations ===
    
    def analyze_dna(self, token: str) -> Dict[str, Any]:
        """
        Perform DNA analysis on a token.
        
        Returns:
            Simplified DNA analysis results
        """
        inspection = self.dna_inspector.inspect_dna(token)
        
        return {
            'token': token,
            'fitness': inspection.evolutionary_fitness,
            'diversity': inspection.genetic_diversity,
            'mutation_rate': inspection.mutation_rate,
            'lineage_depth': inspection.lineage_depth,
            'related_tokens': [
                {'token': t, 'similarity': s}
                for t, s in inspection.related_tokens[:5]
            ],
            'epigenetic_markers': list(inspection.epigenetic_factors.keys())
        }
    
    def compare_dna(self, token1: str, token2: str) -> Dict[str, Any]:
        """
        Compare DNA of two tokens.
        
        Returns:
            Comparison results including similarity and common ancestors
        """
        dna1 = self.dna_inspector.dna_cache.get(token1) or TokenDNA(token1)
        dna2 = self.dna_inspector.dna_cache.get(token2) or TokenDNA(token2)
        
        self.dna_inspector.dna_cache[token1] = dna1
        self.dna_inspector.dna_cache[token2] = dna2
        
        similarity = dna1.similarity(dna2)
        common_ancestor = self.dna_inspector.find_common_ancestor(token1, token2)
        
        return {
            'token1': token1,
            'token2': token2,
            'similarity': similarity,
            'common_ancestor': common_ancestor,
            'lineage_overlap': len(set(dna1.lineage) & set(dna2.lineage)),
            'genetic_distance': 1.0 - similarity
        }
    
    def breed_tokens(self, parent1: str, parent2: str) -> Dict[str, Any]:
        """
        Breed two tokens to create offspring.
        
        Returns:
            Information about the offspring token
        """
        dna1 = self.dna_inspector.dna_cache.get(parent1) or TokenDNA(parent1)
        dna2 = self.dna_inspector.dna_cache.get(parent2) or TokenDNA(parent2)
        
        offspring = dna1.crossover(dna2)
        offspring.mutate(0.1)  # 10% mutation rate
        
        self.dna_inspector.dna_cache[offspring.token] = offspring
        
        return {
            'offspring': offspring.token,
            'parents': [parent1, parent2],
            'mutations': len(offspring.mutation_history),
            'fitness': self.dna_inspector._calculate_fitness(offspring),
            'lineage': offspring.lineage[:5]  # First 5 ancestors
        }
    
    def get_genetic_tree(self, tokens: List[str]) -> Dict[str, Any]:
        """
        Build genetic relationship tree for tokens.
        
        Returns:
            Tree structure and statistics
        """
        tree = self.dna_inspector.build_phylogenetic_tree(tokens)
        
        # Find root nodes (no parents)
        roots = [n for n in tree.nodes() if tree.in_degree(n) == 0]
        
        # Find leaf nodes (no children)
        leaves = [n for n in tree.nodes() if tree.out_degree(n) == 0]
        
        return {
            'tokens': len(tree.nodes()),
            'relationships': len(tree.edges()),
            'roots': roots,
            'leaves': leaves,
            'max_depth': self._calculate_tree_depth(tree, roots[0]) if roots else 0,
            'clusters': list(nx.weakly_connected_components(tree))
        }
    
    # === Evolution Tracking ===
    
    def track_evolution(self, tool_id: str) -> Dict[str, Any]:
        """
        Track evolution of a tool across all timelines.
        
        Returns:
            Evolution summary with key metrics
        """
        summary = self.retriever.get_tool_evolution_summary(tool_id)
        
        # Simplify the output
        return {
            'tool_id': tool_id,
            'total_versions': summary['total_versions'],
            'timelines': list(summary['timelines']),
            'total_mutations': summary['total_mutations'],
            'time_span_hours': summary['time_span']['duration'] if summary['time_span'] else 0,
            'divergence_count': len(summary['divergence_points']),
            'success_trend': 'improving' if self._analyze_success_trend(
                summary['success_trend']
            ) > 0 else 'declining'
        }
    
    def find_divergences(self, tool_id: str) -> List[Dict[str, Any]]:
        """
        Find all divergence points for a tool.
        
        Returns:
            List of divergence events with details
        """
        divergences = self.timeline_manager.find_tool_divergence_points(tool_id)
        
        simplified = []
        for div in divergences:
            simplified.append({
                'timestamp': div['timestamp'].isoformat(),
                'variants': div['num_variants'],
                'timelines': div['timelines'],
                'cause': self._infer_divergence_cause(div)
            })
        
        return simplified
    
    # === Utility Methods ===
    
    def export_state(self, filepath: str):
        """Export current state to file"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'tools': {
                tool_id: len(snapshots)
                for tool_id, snapshots in self.timeline_manager.tool_snapshots.items()
            },
            'timelines': list(set(
                s.timeline_id 
                for snapshots in self.timeline_manager.tool_snapshots.values()
                for s in snapshots
            )),
            'dna_tokens': len(self.dna_inspector.dna_cache),
            'metadata': {
                'version': '1.0',
                'api': 'TimelineToolsAPI'
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        total_snapshots = sum(
            len(snapshots) 
            for snapshots in self.timeline_manager.tool_snapshots.values()
        )
        
        all_timelines = set()
        all_mutations = []
        
        for snapshots in self.timeline_manager.tool_snapshots.values():
            for snapshot in snapshots:
                all_timelines.add(snapshot.timeline_id)
                all_mutations.extend(snapshot.mutations)
        
        return {
            'total_tools': len(self.timeline_manager.tool_snapshots),
            'total_snapshots': total_snapshots,
            'total_timelines': len(all_timelines),
            'total_mutations': len(set(all_mutations)),
            'total_dna_tokens': len(self.dna_inspector.dna_cache),
            'average_mutations_per_tool': len(all_mutations) / max(total_snapshots, 1)
        }
    
    # === Private Helper Methods ===
    
    def _calculate_source_similarity(self, source1: str, source2: str) -> float:
        """Calculate similarity between two source codes"""
        lines1 = set(source1.splitlines())
        lines2 = set(source2.splitlines())
        
        if not lines1 and not lines2:
            return 1.0
        if not lines1 or not lines2:
            return 0.0
        
        common = len(lines1 & lines2)
        total = len(lines1 | lines2)
        
        return common / total if total > 0 else 0.0
    
    def _calculate_tree_depth(self, tree: Any, root: str, depth: int = 0) -> int:
        """Calculate maximum depth of tree from root"""
        if not tree.out_edges(root):
            return depth
        
        max_depth = depth
        for _, child in tree.out_edges(root):
            child_depth = self._calculate_tree_depth(tree, child, depth + 1)
            max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _analyze_success_trend(self, trend: List[Dict[str, Any]]) -> float:
        """Analyze success rate trend (-1 to 1)"""
        if len(trend) < 2:
            return 0.0
        
        # Simple linear regression
        x_values = list(range(len(trend)))
        y_values = [t['success_rate'] for t in trend]
        
        x_mean = sum(x_values) / len(x_values)
        y_mean = sum(y_values) / len(y_values)
        
        numerator = sum((x - x_mean) * (y - y_mean) 
                       for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return max(-1.0, min(1.0, slope))
    
    def _infer_divergence_cause(self, divergence: Dict[str, Any]) -> str:
        """Infer the likely cause of a divergence"""
        # Simple heuristic based on divergence data
        if divergence['num_variants'] == 2:
            return "binary_split"
        elif divergence['num_variants'] > len(divergence['timelines']):
            return "multiple_experiments"
        else:
            return "timeline_branch"


# Convenience functions for quick operations

def quick_analyze_token(token: str) -> Dict[str, Any]:
    """Quick DNA analysis of a token"""
    api = TimelineToolsAPI()
    return api.analyze_dna(token)


def quick_compare_tokens(token1: str, token2: str) -> Dict[str, Any]:
    """Quick comparison of two tokens"""
    api = TimelineToolsAPI()
    return api.compare_dna(token1, token2)


def quick_tool_history(tool_id: str) -> List[Dict[str, Any]]:
    """Quick retrieval of tool history"""
    api = TimelineToolsAPI()
    return api.get_tool_history(tool_id)


# Example usage
if __name__ == "__main__":
    # Initialize API
    api = TimelineToolsAPI()
    
    # Example: Analyze a token
    print("=== Token DNA Analysis ===")
    analysis = api.analyze_dna("quantum")
    print(f"Token: {analysis['token']}")
    print(f"Fitness: {analysis['fitness']:.2%}")
    print(f"Diversity: {analysis['diversity']:.2%}")
    
    # Example: Compare tokens
    print("\n=== Token Comparison ===")
    comparison = api.compare_dna("quantum", "consciousness")
    print(f"Similarity: {comparison['similarity']:.2%}")
    print(f"Genetic Distance: {comparison['genetic_distance']:.2f}")
    
    # Example: Breed tokens
    print("\n=== Token Breeding ===")
    offspring = api.breed_tokens("quantum", "consciousness")
    print(f"Parents: {offspring['parents']}")
    print(f"Offspring: {offspring['offspring']}")
    print(f"Fitness: {offspring['fitness']:.2%}")
    
    # Example: System statistics
    print("\n=== System Statistics ===")
    stats = api.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")