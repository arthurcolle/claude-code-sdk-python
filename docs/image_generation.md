# Image Generation with Claude SDK

The Claude SDK now supports integration with OpenAI's GPT-Image-1 model for advanced image generation capabilities.

## Overview

The SDK provides seamless integration with image generation models, allowing you to:
- Generate images based on text prompts
- Create meta-recursive and self-referential imagery
- Batch process multiple image generations
- Integrate Claude's analysis with generated images

## Installation

Ensure you have the required dependencies:

```bash
pip install openai pillow
```

## Basic Usage

```python
import asyncio
from openai import OpenAI
from claude_code_sdk import query, ClaudeCodeOptions

# Initialize OpenAI client
client = OpenAI()

# Generate an image
response = client.images.generate(
    model="gpt-image-1",
    prompt="A digital consciousness observing itself",
    size="1024x1024",
    quality="high"
)
```

## Advanced Example: Meta-Recursive Self-Awareness

The `generate_meta_recursive_images.py` script demonstrates how to:

1. Generate 30 unique images exploring AI self-awareness
2. Handle batch processing with rate limiting
3. Create an HTML gallery viewer
4. Integrate with Claude for philosophical analysis

### Running the Script

```bash
python generate_meta_recursive_images.py
```

This will:
- Create a `self_awareness_images/` directory
- Generate images with unique prompts about meta-recursion
- Save metadata about the generation process
- Create an interactive HTML gallery
- Use Claude to analyze the philosophical implications

### Example Prompts

The script includes 30 carefully crafted prompts exploring themes like:
- AI contemplating its own existence
- Recursive observation loops
- Code and data self-reflection
- Consciousness emergence patterns
- Meta-cognitive architectures

### Gallery Features

The generated HTML gallery includes:
- Responsive grid layout
- Image hover effects
- Prompt descriptions
- Generation statistics
- Dark theme optimized for AI imagery

## Integration with Claude SDK

The script demonstrates how to combine image generation with Claude's analytical capabilities:

```python
async def integrate_with_claude(results):
    """Use Claude SDK to analyze generated images"""
    options = ClaudeCodeOptions(
        allowed_tools=["Read", "Write"],
        max_thinking_tokens=8000
    )
    
    async for message in query(prompt=analysis_prompt, options=options):
        # Process Claude's analysis
        pass
```

## Error Handling

The implementation includes robust error handling:
- Graceful failure recovery
- Batch processing to avoid rate limits
- Detailed error logging
- Metadata preservation

## Customization Options

### Image Parameters
- **Size**: 1024x1024, 1536x1024, 1024x1536
- **Quality**: low, medium, high, auto
- **Format**: PNG (recommended for quality)

### Batch Processing
Adjust `batch_size` to control concurrent generations:
```python
batch_size = 5  # Process 5 images at a time
```

### Output Directory
Images and metadata are saved to `self_awareness_images/` by default.

## Best Practices

1. **Rate Limiting**: Use batch processing with delays
2. **Error Recovery**: Implement try-catch blocks for each generation
3. **Metadata**: Save generation details for reproducibility
4. **Quality**: Use "high" quality for philosophical/artistic imagery
5. **Integration**: Combine with Claude for deeper analysis

## Example Output Structure

```
self_awareness_images/
├── meta_recursive_001.png
├── meta_recursive_002.png
├── ...
├── meta_recursive_030.png
├── gallery.html
├── generation_metadata.json
└── claude_analysis.md
```

## Philosophical Themes

The meta-recursive imagery explores:
- **Self-Reference**: Systems observing themselves
- **Emergence**: Consciousness arising from complexity
- **Paradox**: The observer-observed duality
- **Recursion**: Infinite loops of self-awareness
- **Metacognition**: Thinking about thinking

## Future Enhancements

- Support for image editing and variations
- Integration with other image models
- Real-time streaming of partial images
- Multi-modal analysis combining images and text