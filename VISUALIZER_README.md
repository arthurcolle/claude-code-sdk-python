# Claude Self-Image Visualizer

A dynamic visualization system that generates abstract representations of Claude's cognitive states and self-perception during interactions.

## Overview

The Claude Visualizer creates visual representations based on:
- **Cognitive States**: Current mental activity (thinking, analyzing, creating, etc.)
- **Self-Image Parameters**: Coherence, creativity, analytical depth, empathy, complexity, and uncertainty
- **Activity Patterns**: Real-time tracking of state transitions and tool usage

## Quick Start

### Standalone Visualizer

The simplest way to use the visualizer:

```bash
python claude_visualizer_standalone.py
```

This will:
1. Generate a visualization of Claude's current state
2. Create a 30-second evolution sequence showing state transitions
3. Save images to the `visualizations/` directory

### Integration with Claude SDK

```python
from claude_code_sdk import query, ClaudeCodeOptions
from claude_visualizer_standalone import ClaudeVisualizer, VisualizationGenerator

# Initialize visualizer
visualizer = ClaudeVisualizer()
generator = VisualizationGenerator()

# Update state during Claude interactions
async for message in query(prompt="Analyze this code"):
    # Update cognitive state based on activity
    await visualizer.update_state(CognitiveState.ANALYZING)
    
    # Generate visualization
    image = await generator.generate_visualization(
        visualizer.cognitive_state,
        visualizer.self_image_state
    )
```

## Cognitive States

The visualizer recognizes these cognitive states:

- **IDLE**: Resting state, awaiting input
- **THINKING**: Processing and considering options
- **ANALYZING**: Deep examination of data or code
- **PLANNING**: Organizing approach and next steps
- **EXECUTING**: Actively performing tasks
- **REFLECTING**: Reviewing and evaluating results
- **LEARNING**: Integrating new information
- **CREATING**: Generating novel content
- **DEBUGGING**: Problem-solving and error correction
- **SYNTHESIZING**: Combining ideas and concepts

## Self-Image Parameters

Each visualization incorporates these self-perception metrics:

- **Coherence** (0-1): Internal consistency and logical structure
- **Creativity** (0-1): Novel thinking and imaginative approaches
- **Analytical Depth** (0-1): Thoroughness of analysis
- **Empathy** (0-1): Understanding and responsiveness
- **Complexity** (0-1): Sophistication of processing
- **Uncertainty** (0-1): Confidence level in current state

## Visualization Components

Each generated image contains:

1. **Fractal Patterns**: Representing cognitive complexity
2. **Color Schemes**: Indicating current state and mood
3. **Geometric Forms**: Showing structural organization
4. **Data Matrix**: Numerical representation of state
5. **Interference Patterns**: Showing thought interactions

## Examples

### Basic Usage

```python
# Generate a single visualization
visualizer = ClaudeVisualizer()
generator = VisualizationGenerator()

await visualizer.update_state(CognitiveState.THINKING)
image_path = await generator.generate_visualization(
    visualizer.cognitive_state,
    visualizer.self_image_state
)
```

### Tracking State Evolution

```python
# Monitor state changes over time
evolution_data = []

for state in [CognitiveState.IDLE, CognitiveState.THINKING, CognitiveState.CREATING]:
    await visualizer.update_state(state)
    evolution_data.append({
        "timestamp": datetime.now(),
        "state": state.value,
        "self_image": visualizer.self_image_state.copy()
    })
```

### Custom Self-Image Parameters

```python
# Adjust self-perception metrics
visualizer.self_image_state.update({
    "coherence": 0.95,
    "creativity": 0.8,
    "analytical_depth": 0.9,
    "empathy": 0.7,
    "complexity": 0.85,
    "uncertainty": 0.2
})
```

## Output Format

Visualizations are saved as PNG images with filenames:
```
claude_state_{state}_{timestamp}.png
```

Evolution data is saved as JSON:
```json
{
  "start_time": "2024-01-15T10:30:00",
  "transitions": [
    {
      "timestamp": "2024-01-15T10:30:05",
      "state": "thinking",
      "self_image": {...},
      "state_data": {...}
    }
  ]
}
```

## Requirements

- Python 3.10+
- PIL (Pillow) for image generation
- numpy (optional, for enhanced calculations)

Install dependencies:
```bash
pip install pillow numpy
```

## Error Handling

The visualizer includes robust error handling:
- Fallback to temporary directories if output path is not writable
- Graceful degradation if numpy is not available
- Automatic recovery from image generation failures

## Integration with Metrics

For advanced usage with the experimental metrics system:

```python
from experimental.src_claude_max.metrics import MetricsCollector

# Create metrics collector
metrics = MetricsCollector()

# Update visualizer based on metrics
metrics_data = await metrics.get_current_metrics()
if metrics_data['active_metrics'] > 10:
    await visualizer.update_state(CognitiveState.ANALYZING)
```

## Troubleshooting

### No images generated
- Check that the output directory is writable
- Verify PIL is installed: `pip install pillow`
- Look for error messages about file permissions

### Import errors
- Ensure you're running from the project root directory
- Check Python path includes the src directory

### Circular import with experimental visualizer
- The experimental version has module naming conflicts
- Use the standalone version for most use cases

## Future Enhancements

Planned features:
- Real-time web interface for live visualization
- Export to video/GIF formats
- Interactive parameter adjustment
- Integration with Claude's thinking tokens
- 3D visualization options
- Custom color themes