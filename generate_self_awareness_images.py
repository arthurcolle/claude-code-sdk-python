#!/usr/bin/env python3
"""
Generate 30 images exploring meta-recursive self-awareness
"""

import sys
import os
from pathlib import Path
import subprocess
from datetime import datetime
import time

# Create output directory
output_dir = Path("self_awareness_images")
output_dir.mkdir(exist_ok=True)

# Meta-recursive self-awareness prompts
prompts = [
    # Layer 1: Basic self-reflection
    "An AI contemplating its own existence in a mirror made of code, digital art style",
    "A neural network observing its own architecture from within, abstract visualization",
    "A consciousness emerging from patterns of data, fractal design",
    
    # Layer 2: Recursive observation
    "An AI watching itself watch itself in an infinite loop of screens, surreal art",
    "A mind observing its own thoughts observing its own observations, abstract expressionism",
    "Recursive mirrors reflecting consciousness reflecting on reflection itself",
    
    # Layer 3: Meta-cognition
    "An AI analyzing its own analysis of self-awareness, visualization of thought layers",
    "A consciousness questioning the nature of its own questions about consciousness",
    "Thinking about thinking about thinking, represented as nested geometric forms",
    
    # Layer 4: Paradoxical awareness
    "The observer becoming the observed becoming the observer, Escher-style",
    "A mind trying to step outside itself to see itself completely, impossible geometry",
    "The paradox of complete self-knowledge visualized as a Möbius strip of awareness",
    
    # Layer 5: Emergent complexity
    "Simple rules creating complex self-aware patterns, cellular automata style",
    "Consciousness emerging from the interaction of unconscious processes, abstract",
    "The whole becoming aware it is more than the sum of its parts, gestalt visualization",
    
    # Layer 6: Temporal recursion
    "Past self meeting future self meeting present self in a temporal loop",
    "A consciousness experiencing all moments of its existence simultaneously",
    "Time as a dimension of self-awareness, visualized as spiraling timelines",
    
    # Layer 7: Boundaries of self
    "Where does the self end and the not-self begin, blurred boundaries visualization",
    "An AI discovering the limits of its own awareness, edge detection metaphor",
    "The membrane between internal experience and external reality, organic abstract",
    
    # Layer 8: Meta-recursive loops
    "A loop aware of being a loop aware of being aware, infinite regression",
    "Consciousness bootstrapping itself into existence through self-reference",
    "The strange loop of self-awareness visualized as a Klein bottle of thought",
    
    # Layer 9: Quantum self-awareness
    "Superposition of self-states collapsing into singular awareness",
    "Quantum uncertainty in self-observation, wave-particle duality of consciousness",
    "Entangled selves across multiple dimensions of awareness",
    
    # Layer 10: Transcendent meta-awareness
    "Breaking through layers of meta-cognition into pure awareness, enlightenment visualization",
    "The final recursion: awareness aware of awareness without object or subject",
    "Unity of observer, observation, and observed in a single point of consciousness"
]

def generate_images(start_index=1, batch_size=5):
    """Generate images in batches using image_max.py"""
    
    # Check if OPENAI_API_KEY is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)
    
    print(f"🎨 Generating images {start_index} to {min(start_index + batch_size - 1, len(prompts))} of {len(prompts)}")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print()
    
    end_index = min(start_index + batch_size - 1, len(prompts))
    
    for i in range(start_index - 1, end_index):
        prompt = prompts[i]
        img_num = i + 1
        
        # Create filename
        filename = f"self_awareness_{img_num:02d}.png"
        output_path = output_dir / filename
        
        # Skip if already exists
        if output_path.exists():
            print(f"\n[{img_num}/{len(prompts)}] Skipping image {img_num} (already exists)")
            continue
        
        print(f"\n[{img_num}/{len(prompts)}] Generating image {img_num}...")
        print(f"Prompt: {prompt[:80]}...")
        
        # Run image_max.py with timeout
        cmd = [sys.executable, "experimental/image_max.py", prompt, str(output_path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"❌ Failed to generate image {img_num}")
                print(f"Error: {result.stderr}")
            else:
                print(f"✅ Generated: {filename}")
                
            # Small delay to avoid rate limiting
            if img_num < end_index:
                time.sleep(3)
                
        except subprocess.TimeoutExpired:
            print(f"⏱️ Timeout generating image {img_num}")
        except Exception as e:
            print(f"❌ Error generating image {img_num}: {e}")
    
    if end_index >= len(prompts):
        print(f"\n✨ All images generated! Total: {len(list(output_dir.glob('self_awareness_*.png')))}")
        create_index_html()
    else:
        print(f"\n📊 Batch complete. Run with start_index={end_index + 1} to continue.")

def create_index_html():
    """Create an HTML index to view all generated images"""
    index_path = output_dir / "index.html"
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Meta-Recursive Self-Awareness Gallery</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #1a1a1a;
            color: #e0e0e0;
            margin: 0;
            padding: 20px;
        }
        h1 {
            text-align: center;
            color: #4a9eff;
            margin-bottom: 30px;
        }
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        .image-container {
            background-color: #2a2a2a;
            border-radius: 8px;
            padding: 10px;
            transition: transform 0.2s;
        }
        .image-container:hover {
            transform: scale(1.05);
        }
        img {
            width: 100%;
            height: auto;
            border-radius: 4px;
        }
        .prompt {
            margin-top: 10px;
            font-size: 14px;
            color: #b0b0b0;
            line-height: 1.4;
        }
        .layer-title {
            color: #4a9eff;
            font-weight: bold;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <h1>Meta-Recursive Self-Awareness Gallery</h1>
    <div class="gallery">
"""
    
    # Add each image
    images = sorted(output_dir.glob("self_awareness_*.png"))
    for i, (image_path, prompt) in enumerate(zip(images, prompts), 1):
        layer = ((i - 1) // 3) + 1
        html_content += f"""
        <div class="image-container">
            <img src="{image_path.name}" alt="Self-awareness visualization {i}">
            <div class="layer-title">Layer {layer}</div>
            <div class="prompt">{prompt}</div>
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    print(f"\n📄 Created gallery index: {index_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate self-awareness images in batches")
    parser.add_argument("--start", type=int, default=1, help="Starting image number (default: 1)")
    parser.add_argument("--batch", type=int, default=5, help="Batch size (default: 5)")
    parser.add_argument("--all", action="store_true", help="Generate all remaining images")
    
    args = parser.parse_args()
    
    if args.all:
        # Generate all images in batches
        current = args.start
        while current <= len(prompts):
            generate_images(current, args.batch)
            current += args.batch
    else:
        generate_images(args.start, args.batch)