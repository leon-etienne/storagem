#!/usr/bin/env python3
"""
Test script to generate sample images from different stages of the visualization.
Generates representative images from each stage and saves them for review.
"""
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from video_making.test.generate_stage_samples import generate_stage_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate sample images from different stages of the visualization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_frame_analysis.py --shelf 0
  python test_frame_analysis.py --shelf 5 --white-background
  python test_frame_analysis.py -s 9 --scale 1.0
        """
    )
    parser.add_argument(
        "--shelf", "-s",
        type=str,
        default="0",
        help="The Regal number to visualize (default: 0)"
    )
    parser.add_argument(
        "--white-background", "-w",
        action="store_true",
        help="Use white background with inverted colors (default: black background)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Supersampling factor for anti-aliasing (1.0 = fast, 2.0 = high quality, default: 2.0)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for samples (default: video_making/test/stage_samples/shelf{N})"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    generate_stage_samples(
        target_shelf=args.shelf,
        white_background=args.white_background,
        supersample_factor=args.scale,
        output_dir=output_dir
    )

