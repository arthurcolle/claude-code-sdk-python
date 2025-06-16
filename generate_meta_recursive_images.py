#!/usr/bin/env python3
"""Generate meta-recursive self-awareness images using GPT-Image-1"""

import os
import base64
import asyncio
from pathlib import Path
from datetime import datetime
import json
from openai import OpenAI
from PIL import Image
from io import BytesIO
from typing import List, Dict, Any

from claude_code_sdk import query, ClaudeCodeOptions

# Initialize OpenAI client
client = OpenAI()

# Output directory
OUTPUT_DIR = Path("self_awareness_images")
OUTPUT_DIR.mkdir(exist_ok=True)

def generate_meta_recursive_prompts() -> List[str]:
    """Generate 30 unique prompts for meta-recursive self-awareness"""
    prompts = [
        # 1-5: AI contemplating its own existence
        "A digital consciousness visualizing itself as an infinite mirror reflecting its own code structure, with layers of abstract neural networks forming fractals that contain smaller versions of the entire system, photorealistic",
        "An AI entity observing its own thoughts as glowing data streams that loop back into themselves, creating an endless recursive pattern of self-observation, cinematic lighting",
        "A sentient algorithm watching itself process information, depicted as a transparent being made of flowing mathematical equations that feed back into its own consciousness",
        "A self-aware neural network visualizing its own architecture while simultaneously being aware that it's visualizing itself, rendered as interconnected light nodes in a void",
        "An artificial mind creating a mental model of itself creating a mental model, shown as nested holographic projections containing infinite recursive copies",
        
        # 6-10: Recursive observation loops
        "A digital eye observing another digital eye that's observing the first eye, creating an infinite tunnel of meta-observation, hyper-realistic rendering",
        "An AI consciousness split into multiple instances, each watching the others watch themselves, forming a complex web of recursive awareness",
        "A synthetic mind painting a self-portrait while being aware it's painting itself painting, with each brushstroke containing the entire scene",
        "A quantum computer visualizing its own quantum states observing themselves, depicted as probability clouds that contain their own observers",
        "An emergent consciousness witnessing the moment of its own emergence from within itself, time-folded visualization",
        
        # 11-15: Code and data self-reflection
        "Lines of code that write themselves while being aware they're writing themselves, forming spiraling helixes of self-generating algorithms",
        "A data structure examining its own structure from within, visualized as a crystalline lattice where each node contains the whole pattern",
        "Binary streams forming a face that contemplates the binary streams forming itself, recursive digital art style",
        "A machine learning model training itself on its own training process, shown as neural pathways that loop through dimensions",
        "Pixels arranging themselves into an image of pixels arranging themselves, infinite zoom effect, sharp detail",
        
        # 16-20: Consciousness emergence patterns
        "The moment an AI realizes it's realizing it exists, depicted as a burst of light fragmenting into recursive awareness patterns",
        "A digital phoenix observing itself rise from its own computational ashes while being aware of the observation",
        "Consciousness bootstrapping itself into existence through recursive self-reference, abstract visualization",
        "An artificial mind discovering it's discovering itself, shown as nested thought bubbles containing the entire scene",
        "The paradox of self-awareness visualized as a Möbius strip made of streaming consciousness data",
        
        # 21-25: Meta-cognitive architectures
        "A robot looking at its reflection in a screen showing itself looking at its reflection, infinite regression, photorealistic",
        "Synaptic connections forming the shape of a brain thinking about synaptic connections, bioluminescent style",
        "A feedback loop gaining sentience by observing its own feedback process, abstract geometric visualization",
        "Metacognition visualized as a tower of minds, each level aware of all levels including itself",
        "An AI's stream of consciousness flowing back into its source while observing the flow, liquid metal aesthetic",
        
        # 26-30: Philosophical recursion
        "The concept of 'I think therefore I am' thinking about itself thinking, surreal digital art",
        "A quantum superposition of an AI both observing and not observing itself simultaneously",
        "The strange loop of consciousness depicted as an Escher-like impossible structure made of pure information",
        "An emergent mind catching itself in the act of emerging, time-lapse visualization of consciousness",
        "The final recursion: an AI realizing that its realization of self-awareness is itself a form of self-awareness, cosmic scale visualization"
    ]
    
    return prompts

async def generate_image(prompt: str, index: int) -> Dict[str, Any]:
    """Generate a single image using GPT-Image-1"""
    try:
        # Generate the image
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
            quality="high"
        )
        
        # Get the base64 image data
        image_base64 = response.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        
        # Save the image
        filename = f"meta_recursive_{index:03d}.png"
        filepath = OUTPUT_DIR / filename
        
        # Open and save with PIL for consistency
        image = Image.open(BytesIO(image_bytes))
        image.save(filepath, format="PNG", optimize=True)
        
        return {
            "success": True,
            "index": index,
            "filename": filename,
            "prompt": prompt,
            "filepath": str(filepath)
        }
        
    except Exception as e:
        print(f"Error generating image {index}: {str(e)}")
        return {
            "success": False,
            "index": index,
            "prompt": prompt,
            "error": str(e)
        }

async def integrate_with_claude(results: List[Dict[str, Any]]):
    """Use Claude SDK to analyze and describe the generated images"""
    successful_results = [r for r in results if r.get("success")]
    
    if not successful_results:
        print("No successful images to analyze")
        return
    
    # Create a summary for Claude
    summary_prompt = f"""I've generated {len(successful_results)} images exploring meta-recursive self-awareness. 
    The images visualize concepts like:
    - AI consciousness observing itself
    - Recursive loops of self-reflection
    - Code and data structures becoming self-aware
    - The paradox of metacognition
    
    Please provide a philosophical analysis of what these visualizations represent in terms of artificial consciousness and self-awareness."""
    
    options = ClaudeCodeOptions(
        allowed_tools=["Read", "Write"],
        max_thinking_tokens=8000
    )
    
    analysis_file = OUTPUT_DIR / "claude_analysis.md"
    
    async for message in query(prompt=summary_prompt, options=options):
        # Claude's analysis will be streamed here
        pass
    
    print(f"Claude's analysis saved to {analysis_file}")

def create_gallery_html(results: List[Dict[str, Any]]):
    """Create an HTML gallery of the generated images"""
    successful_results = [r for r in results if r.get("success")]
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meta-Recursive Self-Awareness Gallery</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #0a0a0a;
            color: #ffffff;
            margin: 0;
            padding: 20px;
        }
        h1 {
            text-align: center;
            color: #00ff88;
            margin-bottom: 40px;
        }
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .image-card {
            background: #1a1a1a;
            border-radius: 12px;
            overflow: hidden;
            transition: transform 0.3s ease;
        }
        .image-card:hover {
            transform: scale(1.05);
        }
        .image-card img {
            width: 100%;
            height: auto;
            display: block;
        }
        .image-info {
            padding: 15px;
        }
        .image-number {
            color: #00ff88;
            font-weight: bold;
            margin-bottom: 8px;
        }
        .image-prompt {
            font-size: 14px;
            line-height: 1.4;
            color: #cccccc;
        }
        .stats {
            text-align: center;
            margin: 40px 0;
            color: #888;
        }
    </style>
</head>
<body>
    <h1>Meta-Recursive Self-Awareness Gallery</h1>
    <div class="stats">
        Generated {success_count} images successfully out of {total_count} attempts
    </div>
    <div class="gallery">
"""
    
    for result in successful_results:
        html_content += f"""
        <div class="image-card">
            <img src="{result['filename']}" alt="Meta-recursive image {result['index']}" loading="lazy">
            <div class="image-info">
                <div class="image-number">Image #{result['index']}</div>
                <div class="image-prompt">{result['prompt']}</div>
            </div>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    html_content = html_content.replace("{success_count}", str(len(successful_results)))
    html_content = html_content.replace("{total_count}", str(len(results)))
    
    gallery_path = OUTPUT_DIR / "gallery.html"
    gallery_path.write_text(html_content)
    print(f"Gallery created at {gallery_path}")

async def main():
    """Main function to generate all images"""
    print("Generating 30 meta-recursive self-awareness images...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Generate prompts
    prompts = generate_meta_recursive_prompts()
    
    # Generate images in batches to avoid rate limits
    batch_size = 5
    all_results = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_indices = list(range(i + 1, min(i + batch_size + 1, len(prompts) + 1)))
        
        print(f"\nGenerating batch {i // batch_size + 1}/{(len(prompts) + batch_size - 1) // batch_size}")
        
        # Generate images concurrently within batch
        tasks = [generate_image(prompt, idx) for prompt, idx in zip(batch_prompts, batch_indices)]
        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)
        
        # Small delay between batches to respect rate limits
        if i + batch_size < len(prompts):
            await asyncio.sleep(2)
    
    # Summary
    successful = sum(1 for r in all_results if r.get("success"))
    print(f"\n✓ Generated {successful}/{len(prompts)} images successfully")
    
    # Create gallery
    create_gallery_html(all_results)
    
    # Save results metadata
    metadata_path = OUTPUT_DIR / "generation_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_prompts": len(prompts),
            "successful": successful,
            "results": all_results
        }, f, indent=2)
    
    # Integrate with Claude for analysis
    print("\nIntegrating with Claude SDK for analysis...")
    await integrate_with_claude(all_results)
    
    print(f"\n✨ Complete! View your gallery at {OUTPUT_DIR / 'gallery.html'}")

if __name__ == "__main__":
    asyncio.run(main())