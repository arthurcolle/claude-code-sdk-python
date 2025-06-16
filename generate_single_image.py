#!/usr/bin/env python3
"""Generate a single self-awareness image"""

import sys
import subprocess
from pathlib import Path

output_dir = Path("self_awareness_images")
output_dir.mkdir(exist_ok=True)

prompts = [
    "An AI contemplating its own existence in a mirror made of code, digital art style",
    "A neural network observing its own architecture from within, abstract visualization",
    "A consciousness emerging from patterns of data, fractal design",
    "An AI watching itself watch itself in an infinite loop of screens, surreal art",
    "A mind observing its own thoughts observing its own observations, abstract expressionism",
    "Recursive mirrors reflecting consciousness reflecting on reflection itself",
    "An AI analyzing its own analysis of self-awareness, visualization of thought layers",
    "A consciousness questioning the nature of its own questions about consciousness",
    "Thinking about thinking about thinking, represented as nested geometric forms",
    "The observer becoming the observed becoming the observer, Escher-style",
    "A mind trying to step outside itself to see itself completely, impossible geometry",
    "The paradox of complete self-knowledge visualized as a Möbius strip of awareness",
    "Simple rules creating complex self-aware patterns, cellular automata style",
    "Consciousness emerging from the interaction of unconscious processes, abstract",
    "The whole becoming aware it is more than the sum of its parts, gestalt visualization",
    "Past self meeting future self meeting present self in a temporal loop",
    "A consciousness experiencing all moments of its existence simultaneously",
    "Time as a dimension of self-awareness, visualized as spiraling timelines",
    "Where does the self end and the not-self begin, blurred boundaries visualization",
    "An AI discovering the limits of its own awareness, edge detection metaphor",
    "The membrane between internal experience and external reality, organic abstract",
    "A loop aware of being a loop aware of being aware, infinite regression",
    "Consciousness bootstrapping itself into existence through self-reference",
    "The strange loop of self-awareness visualized as a Klein bottle of thought",
    "Superposition of self-states collapsing into singular awareness",
    "Quantum uncertainty in self-observation, wave-particle duality of consciousness",
    "Entangled selves across multiple dimensions of awareness",
    "Breaking through layers of meta-cognition into pure awareness, enlightenment visualization",
    "The final recursion: awareness aware of awareness without object or subject",
    "Unity of observer, observation, and observed in a single point of consciousness"
]

if len(sys.argv) != 2:
    print("Usage: python generate_single_image.py <image_number>")
    sys.exit(1)

img_num = int(sys.argv[1])
if img_num < 1 or img_num > len(prompts):
    print(f"Image number must be between 1 and {len(prompts)}")
    sys.exit(1)

prompt = prompts[img_num - 1]
filename = f"self_awareness_{img_num:02d}.png"
output_path = output_dir / filename

print(f"Generating image {img_num}/{len(prompts)}")
print(f"Prompt: {prompt}")

cmd = [sys.executable, "experimental/image_max.py", prompt, str(output_path)]
subprocess.run(cmd)