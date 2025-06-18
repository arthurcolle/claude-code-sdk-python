#!/bin/bash

# Create output directory
mkdir -p self_awareness_images

# Generate each image
for i in {1..30}; do
    echo "Generating image $i/30..."
    python generate_single_image.py $i
    sleep 2  # Small delay to avoid rate limiting
done

echo "All images generated!"
echo "Creating gallery..."
python -c "
from generate_self_awareness_images import create_index_html
create_index_html()
"