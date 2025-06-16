#!/usr/bin/env python3
"""
Claude State Query Tool
Command-line tool to query Claude's current internal state
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Import our visualizer
from claude_visualizer_standalone import ClaudeVisualizer, CognitiveState

class ClaudeStateQuery:
    """Query tool for Claude's internal state"""
    
    def __init__(self):
        self.visualizer = ClaudeVisualizer()
        # Add state configurations from the visualizer
        self.state_configs = {
            CognitiveState.IDLE: {"hue": 0, "saturation": 0.5, "brightness": 0.8},
            CognitiveState.THINKING: {"hue": 240, "saturation": 0.7, "brightness": 0.9},
            CognitiveState.ANALYZING: {"hue": 180, "saturation": 0.8, "brightness": 1.0},
            CognitiveState.PLANNING: {"hue": 120, "saturation": 0.6, "brightness": 0.95},
            CognitiveState.EXECUTING: {"hue": 30, "saturation": 0.9, "brightness": 1.0},
            CognitiveState.REFLECTING: {"hue": 270, "saturation": 0.6, "brightness": 0.85},
            CognitiveState.LEARNING: {"hue": 60, "saturation": 0.7, "brightness": 0.9},
            CognitiveState.CREATING: {"hue": 300, "saturation": 0.8, "brightness": 0.95},
            CognitiveState.DEBUGGING: {"hue": 0, "saturation": 0.9, "brightness": 0.85},
            CognitiveState.SYNTHESIZING: {"hue": 210, "saturation": 0.85, "brightness": 1.0}
        }
        # Keep track of parameters as a property
        self.parameters = self.visualizer.self_image_state
        
    def get_current_state(self):
        """Get the current state as a dictionary"""
        return {
            'timestamp': datetime.now().isoformat(),
            'state': self.visualizer.cognitive_state.value,
            'signature': self.get_state_signature(),
            'parameters': self.visualizer.self_image_state.copy(),
            'state_config': self.state_configs.get(
                self.visualizer.cognitive_state, {}
            )
        }
    
    def get_state_signature(self):
        """Generate a unique signature for the current state."""
        state = self.visualizer.cognitive_state
        self_image = self.visualizer.self_image_state
        components = [
            f"{state.value[:3].upper()}",
            f"COH{int(self_image.get('coherence', 0) * 100)}",
            f"CRE{int(self_image.get('creativity', 0) * 100)}",
            f"ANA{int(self_image.get('analytical_depth', 0) * 100)}",
            f"EMP{int(self_image.get('empathy', 0) * 100)}",
            f"UNC{int(self_image.get('uncertainty', 0) * 100)}"
        ]
        return "-".join(components)
    
    def evolve_state(self):
        """Simulate natural state transitions."""
        import random
        current_state = self.visualizer.cognitive_state
        
        # Define natural transitions
        transitions = {
            CognitiveState.IDLE: [CognitiveState.THINKING, CognitiveState.ANALYZING],
            CognitiveState.THINKING: [CognitiveState.ANALYZING, CognitiveState.PLANNING, CognitiveState.CREATING],
            CognitiveState.ANALYZING: [CognitiveState.PLANNING, CognitiveState.SYNTHESIZING],
            CognitiveState.PLANNING: [CognitiveState.EXECUTING, CognitiveState.CREATING],
            CognitiveState.EXECUTING: [CognitiveState.REFLECTING, CognitiveState.DEBUGGING],
            CognitiveState.REFLECTING: [CognitiveState.LEARNING, CognitiveState.IDLE],
            CognitiveState.LEARNING: [CognitiveState.THINKING, CognitiveState.CREATING],
            CognitiveState.CREATING: [CognitiveState.EXECUTING, CognitiveState.REFLECTING],
            CognitiveState.DEBUGGING: [CognitiveState.ANALYZING, CognitiveState.EXECUTING],
            CognitiveState.SYNTHESIZING: [CognitiveState.CREATING, CognitiveState.REFLECTING]
        }
        
        # Randomly transition
        if random.random() < 0.3:  # 30% chance to transition
            next_states = transitions.get(current_state, [CognitiveState.IDLE])
            self.visualizer.cognitive_state = random.choice(next_states)
            
            # Also adjust self-image parameters
            for param in self.visualizer.self_image_state:
                delta = random.uniform(-0.1, 0.1)
                new_value = self.visualizer.self_image_state[param] + delta
                self.visualizer.self_image_state[param] = max(0, min(1, new_value))
    
    def get_state_history(self, duration_seconds=60):
        """Get state history over a duration"""
        history = []
        steps = duration_seconds
        
        for i in range(steps):
            state = self.get_current_state()
            state['step'] = i
            history.append(state)
            self.evolve_state()
            
        return history
    
    def export_state(self, format='json', output_file=None):
        """Export current state in specified format"""
        state = self.get_current_state()
        
        if format == 'json':
            output = json.dumps(state, indent=2)
        elif format == 'text':
            output = self._format_as_text(state)
        elif format == 'csv':
            output = self._format_as_csv(state)
        else:
            raise ValueError(f"Unknown format: {format}")
            
        if output_file:
            with open(output_file, 'w') as f:
                f.write(output)
            return f"State exported to {output_file}"
        else:
            return output
    
    def _format_as_text(self, state):
        """Format state as human-readable text"""
        lines = [
            "Claude Internal State Report",
            "=" * 40,
            f"Timestamp: {state['timestamp']}",
            f"Current State: {state['state']}",
            f"State Signature: {state['signature']}",
            "",
            "Parameters:",
            "-" * 20
        ]
        
        for param, value in state['parameters'].items():
            lines.append(f"  {param:20} : {value:.2f}")
            
        if state.get('state_config'):
            lines.extend([
                "",
                "State Configuration:",
                "-" * 20,
                f"  Hue: {state['state_config'].get('hue', 'N/A')}",
                f"  Saturation: {state['state_config'].get('saturation', 'N/A')}",
                f"  Brightness: {state['state_config'].get('brightness', 'N/A')}"
            ])
            
        return "\n".join(lines)
    
    def _format_as_csv(self, state):
        """Format state as CSV"""
        headers = ['timestamp', 'state', 'signature'] + list(state['parameters'].keys())
        values = [
            state['timestamp'],
            state['state'],
            state['signature']
        ] + list(state['parameters'].values())
        
        lines = [
            ",".join(headers),
            ",".join(str(v) for v in values)
        ]
        
        return "\n".join(lines)
    
    def monitor_realtime(self, interval=1):
        """Monitor state changes in real-time"""
        import time
        
        print("Monitoring Claude's state (Ctrl+C to stop)...")
        print("-" * 60)
        
        last_state = None
        
        try:
            while True:
                current = self.get_current_state()
                
                if current['state'] != last_state:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] State: {current['state']:12} | {current['signature']}")
                    last_state = current['state']
                    
                self.evolve_state()
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped")

def main():
    parser = argparse.ArgumentParser(
        description="Query Claude's internal state",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get current state as JSON
  python claude_state_query.py
  
  # Export state to file
  python claude_state_query.py -o state.json
  
  # Get state as text
  python claude_state_query.py -f text
  
  # Monitor real-time changes
  python claude_state_query.py --monitor
  
  # Get state history (60 seconds)
  python claude_state_query.py --history 60
  
  # Export as CSV
  python claude_state_query.py -f csv -o state.csv
"""
    )
    
    parser.add_argument('-f', '--format', 
                       choices=['json', 'text', 'csv'],
                       default='json',
                       help='Output format (default: json)')
    
    parser.add_argument('-o', '--output',
                       help='Output file (default: stdout)')
    
    parser.add_argument('--monitor',
                       action='store_true',
                       help='Monitor state changes in real-time')
    
    parser.add_argument('--history',
                       type=int,
                       metavar='SECONDS',
                       help='Get state history over N seconds')
    
    parser.add_argument('--interval',
                       type=float,
                       default=1.0,
                       help='Monitoring interval in seconds (default: 1.0)')
    
    args = parser.parse_args()
    
    # Create query tool
    query = ClaudeStateQuery()
    
    try:
        if args.monitor:
            # Real-time monitoring
            query.monitor_realtime(args.interval)
            
        elif args.history:
            # Get history
            history = query.get_state_history(args.history)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(history, f, indent=2)
                print(f"History exported to {args.output}")
            else:
                print(json.dumps(history, indent=2))
                
        else:
            # Get current state
            result = query.export_state(args.format, args.output)
            if not args.output:
                print(result)
            else:
                print(result)
                
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()