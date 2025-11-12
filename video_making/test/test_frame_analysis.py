#!/usr/bin/env python3
"""Test script to generate and analyze a single frame."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from video_making.test.visualize_shelf0_representative import main
import subprocess

# Generate just a few frames by modifying the constants temporarily
# Actually, let's just run the full script but check a frame
print("Generating frames...")
main("0")

# Now analyze a frame that should have text
from PIL import Image
import numpy as np

# Check a frame from step 3 or later (should have artwork info)
test_frame = Path(__file__).parent / "frames" / "frame_00180.png"
if test_frame.exists():
    img = Image.open(test_frame)
    print(f"\nFrame analysis:")
    print(f"  Size: {img.size}")
    print(f"  Mode: {img.mode}")
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Check right panel area (MAP_WIDTH = 1200, CANVAS_WIDTH = 1920)
    panel_area = img_array[:, 1200:1920]
    print(f"\nRight panel area:")
    print(f"  Shape: {panel_area.shape}")
    print(f"  Min/Max pixel values: {panel_area.min()}, {panel_area.max()}")
    print(f"  Mean pixel value: {panel_area.mean():.2f}")
    print(f"  Non-zero pixels: {np.count_nonzero(panel_area)}")
    
    # Check if there are any white pixels (255, 255, 255)
    white_pixels = np.all(panel_area == [255, 255, 255], axis=2)
    print(f"  White pixels count: {white_pixels.sum()}")
    
    # Check a specific area where text should be (around y=400-600, x=1230-1890)
    text_area = img_array[400:600, 1230:1890]
    print(f"\nText area (y=400-600, x=1230-1890):")
    print(f"  Shape: {text_area.shape}")
    print(f"  Min/Max: {text_area.min()}, {text_area.max()}")
    print(f"  Mean: {text_area.mean():.2f}")
    
    # Save a zoomed view of the text area for inspection
    text_img = Image.fromarray(text_area)
    zoom_path = Path(__file__).parent / "frames" / "text_area_zoom.png"
    text_img.save(zoom_path)
    print(f"\n  Saved zoomed text area to: {zoom_path}")
    
else:
    print(f"Test frame not found: {test_frame}")

