#!/usr/bin/env python3
"""
Standalone Claude Self-Image Dynamic Visualizer

A simplified version that generates visualizations without complex dependencies.
"""

import asyncio
import json
import random
from pathlib import Path
from typing import Dict, Any
from enum import Enum
from dataclasses import dataclass, field

# For visualization generation
try:
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Install with: pip install pillow")


class CognitiveState(str, Enum):
    """Claude's cognitive states."""
    IDLE = "idle"
    THINKING = "thinking"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    LEARNING = "learning"
    CREATING = "creating"
    DEBUGGING = "debugging"
    SYNTHESIZING = "synthesizing"


@dataclass
class ClaudeVisualizer:
    """Simple visualization state manager."""
    cognitive_state: CognitiveState = CognitiveState.IDLE
    self_image_state: Dict[str, float] = field(default_factory=lambda: {
        "coherence": 0.8,
        "creativity": 0.6,
        "analytical_depth": 0.7,
        "empathy": 0.9,
        "complexity": 0.5,
        "uncertainty": 0.3
    })
    
    async def start(self):
        """Start the visualizer."""
        print("Visualizer started")
    
    async def stop(self):
        """Stop the visualizer."""
        print("Visualizer stopped")


class SelfImageVisualizer:
    """Generates dynamic self-representation images based on Claude's state."""
    
    def __init__(self):
        self.visualizer = ClaudeVisualizer()
        self.output_path = Path("./visualizations")
        try:
            self.output_path.mkdir(exist_ok=True)
        except (OSError, PermissionError) as e:
            print(f"Warning: Could not create output directory {self.output_path}: {e}")
            # Fallback to temp directory
            import tempfile
            self.output_path = Path(tempfile.mkdtemp(prefix="claude_viz_"))
            print(f"Using temporary directory: {self.output_path}")
        
        # Map cognitive states to visual effects
        self.state_effects = {
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
    
    async def generate_dynamic_visualization(self) -> Dict[str, Any]:
        """Generate a dynamic visualization based on current state."""
        state = self.visualizer.cognitive_state
        self_image = self.visualizer.self_image_state
        
        visualization_data = {
            "cognitive_state": state.value,
            "timestamp": asyncio.get_event_loop().time(),
            "self_image_parameters": self_image,
            "visualization_type": "dynamic_self_representation"
        }
        
        if PIL_AVAILABLE:
            # Generate actual image visualization
            image_path = await self._create_state_image(state, self_image)
            visualization_data["generated_image"] = str(image_path)
        
        # Generate abstract representation
        visualization_data["abstract_representation"] = self._generate_abstract_representation(state, self_image)
        
        return visualization_data
    
    async def _create_state_image(self, state: CognitiveState, self_image: Dict[str, float]) -> Path:
        """Create a visual representation of Claude's current state."""
        # Create base image
        width, height = 800, 800
        image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # Get state effects
        effects = self.state_effects.get(state, self.state_effects[CognitiveState.IDLE])
        
        # Draw central core representing coherence
        coherence = self_image.get("coherence", 0.8)
        core_size = int(200 * coherence)
        center = (width // 2, height // 2)
        
        # Draw glowing core
        for i in range(50, 0, -1):
            alpha = int(255 * (i / 50) * coherence)
            size = core_size + i * 5
            color = self._hsv_to_rgb(effects["hue"], effects["saturation"] * (i / 50), effects["brightness"])
            draw.ellipse(
                [center[0] - size, center[1] - size, center[0] + size, center[1] + size],
                fill=(*color, alpha)
            )
        
        # Draw complexity fractals
        complexity = self_image.get("complexity", 0.5)
        num_branches = int(6 + complexity * 12)
        for i in range(num_branches):
            angle = (360 / num_branches) * i
            self._draw_fractal_branch(draw, center, angle, complexity, effects, depth=int(3 + complexity * 4))
        
        # Draw creativity sparkles
        creativity = self_image.get("creativity", 0.6)
        num_sparkles = int(creativity * 100)
        for _ in range(num_sparkles):
            x = random.randint(0, width)
            y = random.randint(0, height)
            sparkle_color = self._hsv_to_rgb(
                (effects["hue"] + random.randint(-60, 60)) % 360,
                effects["saturation"],
                effects["brightness"]
            )
            sparkle_size = random.randint(1, 4)
            draw.ellipse([x, y, x + sparkle_size, y + sparkle_size], fill=(*sparkle_color, 200))
        
        # Draw analytical grid
        analytical_depth = self_image.get("analytical_depth", 0.7)
        if analytical_depth > 0.3:
            grid_alpha = int(100 * analytical_depth)
            grid_color = self._hsv_to_rgb(effects["hue"], 0.3, 0.9)
            grid_spacing = int(50 / analytical_depth)
            for x in range(0, width, grid_spacing):
                draw.line([(x, 0), (x, height)], fill=(*grid_color, grid_alpha), width=1)
            for y in range(0, height, grid_spacing):
                draw.line([(0, y), (width, y)], fill=(*grid_color, grid_alpha), width=1)
        
        # Draw empathy waves
        empathy = self_image.get("empathy", 0.9)
        wave_color = self._hsv_to_rgb(effects["hue"], effects["saturation"] * 0.5, effects["brightness"])
        for i in range(int(empathy * 10)):
            radius = 100 + i * 50
            alpha = int(150 * (1 - i / 10) * empathy)
            draw.ellipse(
                [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius],
                outline=(*wave_color, alpha),
                width=2
            )
        
        # Add uncertainty noise
        uncertainty = self_image.get("uncertainty", 0.3)
        if uncertainty > 0.1:
            noise_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            noise_draw = ImageDraw.Draw(noise_layer)
            for _ in range(int(uncertainty * 1000)):
                x = random.randint(0, width)
                y = random.randint(0, height)
                noise_alpha = random.randint(50, 150)
                noise_draw.point((x, y), fill=(255, 255, 255, noise_alpha))
            
            image = Image.alpha_composite(image, noise_layer)
        
        # Apply final blur for ethereal effect
        image = image.filter(ImageFilter.GaussianBlur(radius=2))
        
        # Save image with error handling
        filename = f"claude_state_{state.value}_{int(asyncio.get_event_loop().time())}.png"
        output_path = self.output_path / filename
        
        try:
            image.save(output_path, 'PNG')
            return output_path
        except (OSError, IOError, PermissionError) as e:
            print(f"Warning: Could not save image to {output_path}: {e}")
            # Try to save to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False, prefix='claude_') as tmp:
                try:
                    image.save(tmp.name, 'PNG')
                    print(f"Image saved to temporary file: {tmp.name}")
                    return Path(tmp.name)
                except Exception as e2:
                    print(f"Error: Failed to save image: {e2}")
                    return None
    
    def _draw_fractal_branch(self, draw, start, angle, complexity, effects, length=100, depth=0):
        """Draw a fractal branch representing complexity."""
        if depth == 0:
            return
        
        # Calculate end point
        import math
        angle_rad = math.radians(angle)
        end_x = start[0] + length * math.cos(angle_rad)
        end_y = start[1] + length * math.sin(angle_rad)
        end = (int(end_x), int(end_y))
        
        # Draw branch
        color = self._hsv_to_rgb(
            effects["hue"] + depth * 10,
            effects["saturation"],
            effects["brightness"] * (depth / 5)
        )
        alpha = int(200 * (depth / 5))
        draw.line([start, end], fill=(*color, alpha), width=max(1, depth // 2))
        
        # Recursively draw sub-branches
        if depth > 1:
            branch_angle = 30 * complexity
            self._draw_fractal_branch(draw, end, angle - branch_angle, complexity, effects, length * 0.7, depth - 1)
            self._draw_fractal_branch(draw, end, angle + branch_angle, complexity, effects, length * 0.7, depth - 1)
    
    def _hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB color."""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h / 360, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))
    
    def _generate_abstract_representation(self, state: CognitiveState, self_image: Dict[str, float]) -> Dict[str, Any]:
        """Generate an abstract data representation of Claude's state."""
        # Create a matrix representation
        matrix_size = 10
        state_matrix = []
        
        for i in range(matrix_size):
            row = []
            for j in range(matrix_size):
                # Calculate cell value based on position and state
                distance_from_center = abs(i - matrix_size // 2) + abs(j - matrix_size // 2)
                normalized_distance = distance_from_center / (matrix_size // 2)
                
                # Base value from coherence
                base_value = self_image.get("coherence", 0.8) * (1 - normalized_distance * 0.5)
                
                # Modulate by other parameters
                if i < matrix_size // 2 and j < matrix_size // 2:
                    # Top-left: creativity
                    base_value *= self_image.get("creativity", 0.6)
                elif i < matrix_size // 2 and j >= matrix_size // 2:
                    # Top-right: analytical_depth
                    base_value *= self_image.get("analytical_depth", 0.7)
                elif i >= matrix_size // 2 and j < matrix_size // 2:
                    # Bottom-left: empathy
                    base_value *= self_image.get("empathy", 0.9)
                else:
                    # Bottom-right: complexity
                    base_value *= self_image.get("complexity", 0.5)
                
                # Add uncertainty noise
                uncertainty = self_image.get("uncertainty", 0.3)
                noise = random.uniform(-uncertainty * 0.2, uncertainty * 0.2)
                
                cell_value = max(0, min(1, base_value + noise))
                row.append(round(cell_value, 3))
            
            state_matrix.append(row)
        
        # Calculate balance score with fallback
        if PIL_AVAILABLE and np:
            balance_score = 1 - np.std(list(self_image.values()))
        else:
            # Fallback: use standard deviation calculation without numpy
            values = list(self_image.values())
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std_dev = variance ** 0.5
            balance_score = 1 - std_dev
        
        return {
            "matrix": state_matrix,
            "dominant_trait": max(self_image.items(), key=lambda x: x[1])[0],
            "balance_score": balance_score,
            "state_signature": self._generate_state_signature(state, self_image)
        }
    
    def _generate_state_signature(self, state: CognitiveState, self_image: Dict[str, float]) -> str:
        """Generate a unique signature for the current state."""
        components = [
            f"{state.value[:3].upper()}",
            f"COH{int(self_image.get('coherence', 0) * 100)}",
            f"CRE{int(self_image.get('creativity', 0) * 100)}",
            f"ANA{int(self_image.get('analytical_depth', 0) * 100)}",
            f"EMP{int(self_image.get('empathy', 0) * 100)}",
            f"UNC{int(self_image.get('uncertainty', 0) * 100)}"
        ]
        return "-".join(components)
    
    async def create_evolution_sequence(self, duration_seconds: int = 30):
        """Create a sequence showing Claude's state evolution over time."""
        evolution_data = []
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < duration_seconds:
            # Capture current state
            visualization = await self.generate_dynamic_visualization()
            evolution_data.append(visualization)
            
            # Simulate state changes
            await self._simulate_state_transition()
            
            await asyncio.sleep(1)
        
        return evolution_data
    
    async def _simulate_state_transition(self):
        """Simulate natural state transitions."""
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


async def main():
    """Main demo function."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║            Claude Self-Image Dynamic Visualizer                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  This demo generates dynamic visualizations of Claude's          ║
║  self-representation based on:                                   ║
║                                                                   ║
║  • Current cognitive state (thinking, analyzing, creating...)    ║
║  • Self-image parameters (coherence, creativity, empathy...)     ║
║  • Activity patterns and uncertainty levels                      ║
║                                                                   ║
║  The visualizations combine:                                     ║
║  - Geometric patterns representing structure                     ║
║  - Colors and effects showing cognitive state                    ║
║  - Abstract data matrices for analysis                          ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    visualizer = SelfImageVisualizer()
    
    # Start the base visualizer
    await visualizer.visualizer.start()
    
    try:
        # Generate a single visualization
        print("\nGenerating current state visualization...")
        current_viz = await visualizer.generate_dynamic_visualization()
        print(f"Current state: {current_viz['cognitive_state']}")
        print(f"State signature: {current_viz['abstract_representation']['state_signature']}")
        
        if PIL_AVAILABLE and 'generated_image' in current_viz:
            print(f"Image saved to: {current_viz['generated_image']}")
        
        # Create evolution sequence
        print("\nGenerating 30-second evolution sequence...")
        evolution = await visualizer.create_evolution_sequence(30)
        
        print(f"\nCaptured {len(evolution)} state transitions")
        
        # Show state changes
        states_seen = []
        for viz in evolution:
            state = viz['cognitive_state']
            if not states_seen or state != states_seen[-1]:
                states_seen.append(state)
        
        print(f"State progression: {' → '.join(states_seen)}")
        
        # Save evolution data with error handling
        output_file = visualizer.output_path / "evolution_data.json"
        try:
            with open(output_file, 'w') as f:
                json.dump(evolution, f, indent=2, default=str)
            print(f"\nEvolution data saved to: {output_file}")
        except (OSError, IOError, PermissionError) as e:
            print(f"Warning: Could not save evolution data: {e}")
            # Try to print to stdout as fallback
            print("\nEvolution data (JSON):")
            print(json.dumps(evolution, indent=2, default=str))
        
    finally:
        await visualizer.visualizer.stop()


if __name__ == "__main__":
    asyncio.run(main())