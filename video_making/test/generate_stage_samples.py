#!/usr/bin/env python3
"""
Generate sample images from different stages of the visualization for review.
Saves one representative image from each stage to a folder.
Uses prepare_visualization_data() from visualize_shelf0_representative.py to avoid code duplication.
"""
import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import functions from the visualization script
from video_making.test.visualize_shelf0_representative import (
    create_frame,
    prepare_visualization_data,
)

def generate_stage_samples(target_shelf: str = "0", white_background: bool = False, 
                          supersample_factor: float = 2.0, output_dir: Path = None):
    """Generate sample images from different stages.
    
    Uses prepare_visualization_data() from visualize_shelf0_representative.py
    to ensure consistency with the main visualization script.
    """
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "stage_samples" / f"shelf{target_shelf}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"Generating stage samples for Regal {target_shelf}")
    print("=" * 60)
    
    # Prepare all visualization data using the shared function from visualize_shelf0_representative.py
    data = prepare_visualization_data(target_shelf)
    
    # Extract all needed variables from the data dictionary
    df = data["df"]
    all_embeddings = data["all_embeddings"]
    all_artwork_ids = data["all_artwork_ids"]
    shelf0_mask = data["shelf0_mask"]
    all_coords_2d = data["all_coords_2d"]
    shelf0_coords_2d = data["shelf0_coords_2d"]
    centroid_coord_2d = data["centroid_coord_2d"]
    distances = data["distances"]
    top_representatives = data["top_representatives"]
    top_outliers = data["top_outliers"]
    aesthetic_representative_id = data["aesthetic_representative_id"]
    aesthetic_outlier_id = data["aesthetic_outlier_id"]
    first_idx_in_all = data["first_idx_in_all"]
    artwork_lookup = data["artwork_lookup"]
    
    # Generate sample images for each stage
    print("\n6. Generating stage samples...")
    shelf0_indices_list = list(range(len(shelf0_coords_2d)))  # Simplified - just use indices
    num_shelf0 = len(shelf0_indices_list)
    
    # Stage 1: All embeddings
    print("   Stage 1: All embeddings")
    img = create_frame("all", all_coords_2d, all_artwork_ids, shelf0_mask, all_embeddings,
                      target_shelf=target_shelf, top_representatives=top_representatives,
                      aesthetic_representative_id=aesthetic_representative_id,
                      white_background=white_background, supersample_factor=supersample_factor,
                      artwork_lookup=artwork_lookup, df=df)
    img.save(output_dir / "01_all_embeddings.png", "PNG", compress_level=1, optimize=False)
    
    # Stage 2: Identify Regal items (show about half of them)
    print("   Stage 2: Identify Regal items")
    num_shown = max(1, num_shelf0 // 2)
    # Get the actual artwork index in all_artwork_ids for the item at num_shown
    shelf0_indices_in_all = [i for i, mask in enumerate(shelf0_mask) if mask]
    current_artwork_idx = shelf0_indices_in_all[num_shown - 1] if num_shown > 0 and num_shown <= len(shelf0_indices_in_all) else shelf0_indices_in_all[0] if shelf0_indices_in_all else None
    
    if current_artwork_idx is not None:
        img = create_frame("highlight_slow", all_coords_2d, all_artwork_ids, shelf0_mask,
                          supersample_factor=supersample_factor,
                          white_background=white_background,
                          artwork_lookup=artwork_lookup, df=df,
                          all_embeddings=all_embeddings,
                          shelf0_coords=shelf0_coords_2d,
                          shelf0_coords_progressive=shelf0_coords_2d,
                          num_shelf0_shown=num_shown,
                          highlighted_artwork_idx=current_artwork_idx,
                          target_shelf=target_shelf,
                          top_representatives=top_representatives,
                          aesthetic_representative_id=aesthetic_representative_id)
        img.save(output_dir / "02_identify_regal_items.png", "PNG", compress_level=1, optimize=False)
    
    # Stage 3: Highlight with centroid and distances
    print("   Stage 3: Highlight with centroid")
    last_artwork_idx = shelf0_indices_in_all[-1] if shelf0_indices_in_all else None
    if last_artwork_idx is not None:
        img = create_frame("highlight", all_coords_2d, all_artwork_ids, shelf0_mask,
                          supersample_factor=supersample_factor,
                          white_background=white_background,
                          artwork_lookup=artwork_lookup, df=df,
                          all_embeddings=all_embeddings,
                          shelf0_coords=shelf0_coords_2d,
                          shelf0_coords_progressive=shelf0_coords_2d,
                          centroid_coord=centroid_coord_2d,
                          distances=distances,
                          num_shelf0_shown=num_shelf0,
                          highlighted_artwork_idx=last_artwork_idx,
                          target_shelf=target_shelf,
                          top_representatives=top_representatives,
                          aesthetic_representative_id=aesthetic_representative_id)
        img.save(output_dir / "03_highlight_with_centroid.png", "PNG", compress_level=1, optimize=False)
    
    # Stage 4: Finding representatives (show one representative being found)
    print("   Stage 4: Finding representatives")
    if first_idx_in_all is not None and len(shelf0_coords_2d) > 0:
        # Find the coordinate for the first representative
        shelf0_indices_in_all_list = [i for i, mask in enumerate(shelf0_mask) if mask]
        if first_idx_in_all in shelf0_indices_in_all_list:
            rep_idx_in_shelf0 = shelf0_indices_in_all_list.index(first_idx_in_all)
            if rep_idx_in_shelf0 < len(shelf0_coords_2d):
                rep_coord = shelf0_coords_2d[rep_idx_in_shelf0]
                start_coord = centroid_coord_2d
                lines_to_draw = [(start_coord, rep_coord, 1.0, False)]
                rep_distance = distances[rep_idx_in_shelf0] if rep_idx_in_shelf0 < len(distances) else 0.0
                
                img = create_frame("representative", all_coords_2d, all_artwork_ids, shelf0_mask,
                                  supersample_factor=supersample_factor,
                                  white_background=white_background,
                                  artwork_lookup=artwork_lookup, df=df,
                                  all_embeddings=all_embeddings,
                                  shelf0_coords=shelf0_coords_2d,
                                  centroid_coord=centroid_coord_2d,
                                  distances=distances,
                                  representative_idx=first_idx_in_all,
                                  highlighted_artwork_idx=first_idx_in_all,
                                  lines_to_draw=lines_to_draw,
                                  target_shelf=target_shelf,
                                  top_representatives=top_representatives,
                                  top_outliers=top_outliers,
                                  aesthetic_representative_id=aesthetic_representative_id,
                                  current_distance=rep_distance,
                                  search_mode="representative")
                img.save(output_dir / "04_finding_representatives.png", "PNG", compress_level=1, optimize=False)
    
    # Stage 5: Top 10 (5 representatives + 5 outliers)
    print("   Stage 5: Top 10 representatives and outliers")
    img = create_frame("top10", all_coords_2d, all_artwork_ids, shelf0_mask,
                      white_background=white_background,
                      supersample_factor=supersample_factor,
                      artwork_lookup=artwork_lookup, df=df,
                      all_embeddings=all_embeddings,
                      shelf0_coords=shelf0_coords_2d,
                      centroid_coord=centroid_coord_2d,
                      distances=distances,
                      target_shelf=target_shelf,
                      top_representatives=top_representatives,
                      top_outliers=top_outliers,
                      aesthetic_representative_id=aesthetic_representative_id,
                      aesthetic_outlier_id=aesthetic_outlier_id,
                      top10_reps_shown=5,
                      top10_outliers_shown=5)
    img.save(output_dir / "05_top10_representatives_outliers.png", "PNG", compress_level=1, optimize=False)
    
    # Stage 6: Ruler (measuring distances)
    print("   Stage 6: Measuring distances with ruler")
    img = create_frame("ruler", all_coords_2d, all_artwork_ids, shelf0_mask,
                      white_background=white_background,
                      supersample_factor=supersample_factor,
                      artwork_lookup=artwork_lookup, df=df,
                      all_embeddings=all_embeddings,
                      shelf0_coords=shelf0_coords_2d,
                      centroid_coord=centroid_coord_2d,
                      distances=distances,
                      representative_idx=first_idx_in_all,
                      target_shelf=target_shelf,
                      top_representatives=top_representatives,
                      top_outliers=top_outliers,
                      aesthetic_representative_id=aesthetic_representative_id,
                      aesthetic_outlier_id=aesthetic_outlier_id,
                      ruler_progress=1.0,
                      ruler_to_rep=False)
    img.save(output_dir / "06_ruler_measuring_distances.png", "PNG", compress_level=1, optimize=False)
    
    print(f"\n✓ Generated {6} stage samples in: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate sample images from different stages of the visualization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_stage_samples.py --shelf 0
  python generate_stage_samples.py --shelf 5 --white-background
  python generate_stage_samples.py -s 9 --scale 4.0  # 4K output
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
        help="Supersampling factor for anti-aliasing (2.0 = Full HD output, 4.0 = 4K output, default: 2.0)"
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
