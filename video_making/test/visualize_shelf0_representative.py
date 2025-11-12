#!/usr/bin/env python3
"""
Visualize the process of finding representatives and outliers for a given shelf.
Creates step-by-step frames showing:
1. All embeddings map
2. Identify shelf items (one by one with easing)
3. Calculate and show centroid with distances
4. Cycle through artworks showing calculations
5. Draw lines from centroid
6. Show representatives found
7. Representatives appearing one by one with images
8. Top 10 (5 representatives + 5 outliers) appearing one by one
9. Side-by-side view of selected items

Uses PIL for drawing and ffmpeg for video generation.
Follows aesthetic guidelines: clean, simple, black & white with lime green accent.

Usage:
    python visualize_shelf0_representative.py --shelf 0 --mode both
    python visualize_shelf0_representative.py --shelf 5 --mode both
"""
import argparse
import pickle
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.manifold import TSNE
from tqdm import tqdm

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available, will use t-SNE instead")

# Get script directory and project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Constants (relative to project root)
FONT_DIR = PROJECT_ROOT / "video_making/font"
FONT_REGULAR = FONT_DIR / "Neue Haas Unica W1G Regular.ttf"
FONT_LIGHT = FONT_DIR / "Neue Haas Unica W1G Light.ttf"
FONT_MEDIUM = FONT_DIR / "Neue Haas Unica W1G Medium.ttf"
FONT_THIN = FONT_DIR / "Neue Haas Unica W1G Thin.ttf"
FALLBACK_IMAGE = PROJECT_ROOT / "video_making/font/no_image.png"
CSV_PATH = PROJECT_ROOT / "artworks_with_thumbnails_ting.csv"
EMBEDDINGS_CACHE = PROJECT_ROOT / "embeddings_cache/all_embeddings.pkl"
BASE_DIR = PROJECT_ROOT / "production-export-2025-11-04t14-27-00-000z"

# Color constants (following aesthetic guidelines)
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_LIME = (0, 255, 0)  # Pure green
COLOR_GRAY_LIGHT = (200, 200, 200)  # Light gray for text on black
COLOR_GRAY_DARK = (150, 150, 150)  # Medium gray for text on black
COLOR_POINT_GRAY = (140, 140, 140)  # Gray for regular points (more gray instead of pure white)
COLOR_WHITE_LOW_OPACITY = (120, 120, 120)  # Gray for non-shelf points (more gray instead of pure black/white)
COLOR_LIME_LOW_OPACITY = (0, 150, 0)  # Simulated lower opacity for non-highlighted shelf points
COLOR_GRAY_CONNECTION = (100, 100, 100)  # Gray for connection lines

# White background mode colors
COLOR_BG_WHITE = (255, 255, 255)
COLOR_TEXT_WHITE_BG = (0, 0, 0)  # Black text on white background
COLOR_TEXT_GRAY_WHITE_BG = (100, 100, 100)  # Gray text on white background
COLOR_POINT_WHITE_BG = (50, 50, 50)  # Dark gray points on white background
COLOR_LIME_LOW_OPACITY_WHITE_BG = (0, 200, 0)  # Slightly darker green for white bg
COLOR_GRAY_CONNECTION_WHITE_BG = (180, 180, 180)  # Light gray for connection lines on white bg

# Canvas dimensions
CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080
MAP_WIDTH = 1200  # Left side for embedding map
PANEL_WIDTH = 720  # Right side for artwork info
MAP_HEIGHT = CANVAS_HEIGHT
PANEL_HEIGHT = CANVAS_HEIGHT

# Visualization parameters
POINT_SIZE = 4  # Size for regular points
POINT_SIZE_LARGE = 8  # Size for shelf 0 points
POINT_SIZE_REPRESENTATIVE = 40  # Size for representative points (with images)
LINE_WIDTH = 1
FONT_SIZE_TITLE = 48
FONT_SIZE_LABEL = 24
FONT_SIZE_INFO = 18
FONT_SIZE_SMALL = 14
FONT_SIZE_MAP_TEXT = 18  # Larger text for map labels
FIXED_IMAGE_HEIGHT = 200  # Fixed height for all images in panel

# Frame generation
FRAMES_PER_STEP = 60  # Hold each step for 60 frames (2 seconds at 30fps)
FRAMES_PER_ARTWORK = 20  # Frames per artwork when cycling through
FRAMES_PER_ADDITION = 10  # Frames for each artwork addition animation
FPS = 60  # Higher FPS for smoother animation


def load_embeddings() -> Dict:
    """Load embeddings from cache."""
    if not EMBEDDINGS_CACHE.exists():
        raise FileNotFoundError(f"Embeddings cache not found: {EMBEDDINGS_CACHE}")
    
    with open(EMBEDDINGS_CACHE, "rb") as f:
        cache_data = pickle.load(f)
    
    embeddings_dict = cache_data.get("embeddings", {})
    print(f"Loaded {len(embeddings_dict)} embeddings from cache")
    return embeddings_dict


def load_image(image_path: str) -> Optional[Image.Image]:
    """Load image from path, with fallback."""
    if pd.isna(image_path) or not image_path or image_path == "N/A":
        if FALLBACK_IMAGE.exists():
            return Image.open(FALLBACK_IMAGE).convert("RGB")
        return None
    
    # Try direct path (absolute or relative to project root)
    full_path = Path(image_path)
    if not full_path.is_absolute():
        full_path = PROJECT_ROOT / image_path
    
    if full_path.exists() and full_path.is_file():
        try:
            return Image.open(full_path).convert("RGB")
        except Exception:
            pass
    
    # Try extracting filename
    if "images/" in image_path:
        filename = image_path.split("images/")[-1]
    else:
        filename = image_path
    
    images_dir = BASE_DIR / "images"
    if images_dir.exists():
        full_path = images_dir / filename
        if full_path.exists():
            try:
                return Image.open(full_path).convert("RGB")
            except Exception:
                pass
    
    # Fallback
    if FALLBACK_IMAGE.exists():
        return Image.open(FALLBACK_IMAGE).convert("RGB")
    return None




def reduce_dimensions(embeddings: np.ndarray, method: str = "umap") -> Tuple[np.ndarray, Any]:
    """Reduce embeddings to 2D for visualization.
    
    Returns:
        Tuple of (reduced_coords, reducer) where reducer can be used to transform new points.
        For t-SNE, reducer will be None as it doesn't support transform.
    """
    print(f"Reducing dimensions using {method.upper()} from {embeddings.shape[1]}D to 2D...")
    
    if method.lower() == "umap" and UMAP_AVAILABLE:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=18,  # Increased for more global structure
            min_dist=0.0,  # Lower value to pack points closer together
            spread=12,  # Lower spread for tighter clusters
            metric='cosine',
            random_state=21
        )
        reduced = reducer.fit_transform(embeddings)
        print(f"UMAP reduction complete: {reduced.shape}")
        return reduced, reducer
    
    # Fallback to t-SNE
    reducer = TSNE(
        n_components=2,
        perplexity=20,  # Lower perplexity for tighter clusters
        learning_rate=200,  # Lower learning rate for more stable, tighter clustering
        random_state=42,
        max_iter=1000
    )
    reduced = reducer.fit_transform(embeddings)
    print(f"t-SNE reduction complete: {reduced.shape}")
    return reduced, None  # t-SNE doesn't support transform


def normalize_coords(coords: np.ndarray, margin: float = 0.1) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Normalize coordinates to fit in map area with margin."""
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)
    
    range_x = max_x - min_x
    range_y = max_y - min_y
    
    # Add margin
    margin_x = range_x * margin
    margin_y = range_y * margin
    
    # Normalize to [0, 1]
    normalized = np.zeros_like(coords)
    if range_x > 0:
        normalized[:, 0] = (coords[:, 0] - min_x + margin_x) / (range_x + 2 * margin_x)
    else:
        normalized[:, 0] = 0.5
    
    if range_y > 0:
        normalized[:, 1] = (coords[:, 1] - min_y + margin_y) / (range_y + 2 * margin_y)
    else:
        normalized[:, 1] = 0.5
    
    # Scale to map dimensions
    map_margin = 50  # Padding from edges
    map_area_width = MAP_WIDTH - 2 * map_margin
    map_area_height = MAP_HEIGHT - 2 * map_margin - 100  # Leave space for title
    
    scaled = np.zeros_like(normalized)
    scaled[:, 0] = map_margin + normalized[:, 0] * map_area_width
    scaled[:, 1] = 100 + map_margin + normalized[:, 1] * map_area_height  # Start below title
    
    bounds = (min_x - margin_x, min_y - margin_y, max_x + margin_x, max_y + margin_y)
    
    return scaled, bounds


def get_colors(white_bg: bool = False) -> Dict[str, Tuple[int, int, int]]:
    """Get color scheme based on background mode.
    
    Args:
        white_bg: If True, returns colors for white background mode
        
    Returns:
        Dictionary of color names to RGB tuples
    """
    if white_bg:
        return {
            "background": COLOR_BG_WHITE,
            "text": COLOR_TEXT_WHITE_BG,
            "text_secondary": COLOR_TEXT_GRAY_WHITE_BG,
            "point": COLOR_POINT_WHITE_BG,
            "point_low_opacity": COLOR_POINT_WHITE_BG,
            "lime": COLOR_LIME,
            "lime_low_opacity": COLOR_LIME_LOW_OPACITY_WHITE_BG,
            "connection": COLOR_GRAY_CONNECTION_WHITE_BG,
            "gray_light": COLOR_TEXT_GRAY_WHITE_BG,
            "gray_dark": COLOR_TEXT_GRAY_WHITE_BG,
        }
    else:
        return {
            "background": COLOR_BLACK,
            "text": COLOR_WHITE,
            "text_secondary": COLOR_GRAY_LIGHT,
            "point": COLOR_POINT_GRAY,
            "point_low_opacity": COLOR_WHITE_LOW_OPACITY,
            "lime": COLOR_LIME,
            "lime_low_opacity": COLOR_LIME_LOW_OPACITY,
            "connection": COLOR_GRAY_CONNECTION,
            "gray_light": COLOR_GRAY_LIGHT,
            "gray_dark": COLOR_GRAY_DARK,
        }


def get_font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    """Load font with fallback.
    
    Args:
        size: Font size in points
        weight: Font weight - "thin", "light", "regular", or "medium"
    """
    font_paths = {
        "thin": FONT_THIN,
        "light": FONT_LIGHT,
        "regular": FONT_REGULAR,
        "medium": FONT_MEDIUM,
    }
    
    # Try the requested weight first
    font_path = font_paths.get(weight.lower(), FONT_REGULAR)
    try:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
    except Exception:
        pass
    
    # Fallback to regular if requested weight not available
    if weight.lower() != "regular":
        try:
            if FONT_REGULAR.exists():
                return ImageFont.truetype(str(FONT_REGULAR), size)
        except Exception:
            pass
    
    # Final fallback to system default
    return ImageFont.load_default()


def create_frame(
    step: str,
    all_coords: np.ndarray,
    all_artwork_ids: List,
    shelf0_mask: np.ndarray,
    all_embeddings: Optional[np.ndarray] = None,
    shelf0_coords: Optional[np.ndarray] = None,
    shelf0_coords_progressive: Optional[np.ndarray] = None,
    centroid_coord: Optional[Tuple[float, float]] = None,
    distances: Optional[np.ndarray] = None,
    representative_idx: Optional[int] = None,
    df: Optional[pd.DataFrame] = None,
    highlighted_artwork_idx: Optional[int] = None,
    num_shelf0_shown: Optional[int] = None,
    lines_to_draw: Optional[List[Tuple[Tuple[float, float], Tuple[float, float], float, bool]]] = None,
    target_shelf: str = "0",
    top_representatives: Optional[List[Tuple[int, float]]] = None,
    aesthetic_representative_id: Optional[int] = None,
    color_ease_progress: Optional[float] = None,  # 0.0 to 1.0 for easing in color
    top_outliers: Optional[List[Tuple[int, float]]] = None,
    aesthetic_outlier_id: Optional[int] = None,
    top10_reps_shown: Optional[int] = None,  # Number of representatives to show in top10 (for animation)
    top10_outliers_shown: Optional[int] = None,  # Number of outliers to show in top10 (for animation)
    top10_item_ease_progress: Optional[float] = None,  # Easing progress for currently appearing top10 item
    white_background: bool = False  # Use white background instead of black
) -> Image.Image:
    """Create a single frame for the visualization."""
    # Validate representative_idx - if it's a DataFrame or invalid, set to None
    if representative_idx is not None:
        if isinstance(representative_idx, (pd.DataFrame, pd.Series)):
            representative_idx = None
        elif not isinstance(representative_idx, (int, np.integer, np.int64, np.int32)):
            # Try to convert, but if it fails, set to None
            try:
                if hasattr(representative_idx, 'item'):
                    representative_idx = int(representative_idx.item())
                else:
                    representative_idx = int(representative_idx)
            except (ValueError, TypeError, AttributeError):
                representative_idx = None
    
    # Get color scheme based on background mode
    colors = get_colors(white_background)
    
    # Create canvas with transparent background (RGBA)
    img = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), (*colors["background"], 255))
    draw = ImageDraw.Draw(img)
    
    # Load fonts with appropriate weights - Thin as standard
    font_title = get_font(FONT_SIZE_TITLE, "medium")  # Medium for titles to stand out
    font_label = get_font(FONT_SIZE_LABEL, "thin")  # Thin for labels
    font_info = get_font(FONT_SIZE_INFO, "thin")  # Thin for info text
    font_small = get_font(FONT_SIZE_SMALL, "thin")  # Thin for small text
    
    # Draw title (white text on black)
    # Always show both representatives and outliers
    title_text = f"Finding Representatives and Outliers for Shelf {target_shelf}"
    draw.text((50, 30), title_text, fill=colors["text"], font=font_title)
    
    # Draw step label
    step_labels = {
        "all": "All Embeddings",
        "highlight_slow": "Identify shelf items",
        "highlight": f"Highlighting Shelf {target_shelf}",
        "centroid": f"Highlighting Shelf {target_shelf}",
        "distances": f"Highlighting Shelf {target_shelf}",
        "representative": "Representative Found",
        "representatives": "Top 10 Representatives",
        "zoom": "Selected Representative",
        "outlier": "Outlier Found"
    }
    step_text = step_labels.get(step, step)
    draw.text((50, 90), step_text, fill=colors["text_secondary"], font=font_label)
    
    # Draw calculation info panel FIRST (so it's behind the divider line)
    # This ensures the panel background is drawn before other elements
    if highlighted_artwork_idx is not None and df is not None:
        # Draw background for panel (consistent with main canvas)
        draw.rectangle([MAP_WIDTH, 0, CANVAS_WIDTH, CANVAS_HEIGHT], fill=colors["background"])
    
    # Draw divider line - draw AFTER panel background
    draw.line([(MAP_WIDTH, 0), (MAP_WIDTH, CANVAS_HEIGHT)], fill=colors["text"], width=2)
    
    # Draw all points (fill, no stroke for non-shelf items) with lower opacity
    for i, (x, y) in enumerate(all_coords):
        if not shelf0_mask[i]:
            draw.ellipse([x - POINT_SIZE, y - POINT_SIZE, x + POINT_SIZE, y + POINT_SIZE],
                       fill=colors["point_low_opacity"], outline=None, width=0)
    
    # Draw shelf 0 points (lime green, larger) with title text
    # If progressive mode, only show up to num_shelf0_shown
    if shelf0_coords_progressive is not None and num_shelf0_shown is not None:
        # Progressive mode: show artworks being added one by one
        for i, (x, y) in enumerate(shelf0_coords_progressive):
            if i < num_shelf0_shown:
                shelf0_indices = np.where(shelf0_mask)[0]
                is_highlighted = (highlighted_artwork_idx is not None and 
                                 i < len(shelf0_indices) and 
                                 shelf0_indices[i] == highlighted_artwork_idx)
                
                # Apply color easing if this is the currently appearing item
                is_currently_appearing = (i == num_shelf0_shown - 1 and color_ease_progress is not None)
                if is_currently_appearing and color_ease_progress is not None:
                    # Ease in the color: interpolate from low opacity to full color
                    # Use ease-in-out curve
                    eased = color_ease_progress * color_ease_progress * (3 - 2 * color_ease_progress)
                    # Interpolate between low opacity green and full lime green
                    low_op = colors["lime_low_opacity"]
                    full = colors["lime"]
                    r = int(low_op[0] + (full[0] - low_op[0]) * eased)
                    g = int(low_op[1] + (full[1] - low_op[1]) * eased)
                    b = int(low_op[2] + (full[2] - low_op[2]) * eased)
                    current_color = (r, g, b)
                elif is_highlighted:
                    current_color = colors["lime"]
                else:
                    current_color = colors["lime_low_opacity"]
                
                draw.ellipse([x - POINT_SIZE_LARGE, y - POINT_SIZE_LARGE, 
                            x + POINT_SIZE_LARGE, y + POINT_SIZE_LARGE],
                           fill=current_color, outline=None, width=0)
                
                # Draw title text next to the point for all shelf items
                if df is not None:
                    try:
                        # Get artwork title for this point
                        shelf0_indices_list = list(shelf0_indices)
                        if i < len(shelf0_indices_list):
                            artwork_idx_for_text = shelf0_indices_list[i]
                            artwork_id = all_artwork_ids[artwork_idx_for_text]
                            artwork_row = df[df["id"].astype(float) == float(artwork_id)]
                            if artwork_row.empty:
                                artwork_row = df[df["id"].astype(str).str.strip() == str(artwork_id).strip()]
                            if not artwork_row.empty:
                                title = str(artwork_row.iloc[0].get("title", ""))
                                if title and title != "nan":
                                    # Draw title text to the right of the point
                                    text_x = x + POINT_SIZE_LARGE + 5
                                    text_y = y - 10
                                    font_map_text = get_font(FONT_SIZE_MAP_TEXT, "thin")
                                    draw.text((text_x, text_y), title, fill=colors["text"], font=font_map_text)
                    except Exception:
                        pass
    elif shelf0_coords is not None:
        # Normal mode: show all shelf 0 points
        for idx, (x, y) in enumerate(shelf0_coords):
            shelf0_indices = np.where(shelf0_mask)[0]
            is_highlighted = (highlighted_artwork_idx is not None and 
                            idx < len(shelf0_indices) and 
                            shelf0_indices[idx] == highlighted_artwork_idx)
            
            # Use green fill with no border for highlighted, lower opacity for others
            if is_highlighted:
                draw.ellipse([x - POINT_SIZE_LARGE, y - POINT_SIZE_LARGE, 
                            x + POINT_SIZE_LARGE, y + POINT_SIZE_LARGE],
                           fill=colors["lime"], outline=None, width=0)
            else:
                draw.ellipse([x - POINT_SIZE_LARGE, y - POINT_SIZE_LARGE, 
                            x + POINT_SIZE_LARGE, y + POINT_SIZE_LARGE],
                           fill=colors["lime_low_opacity"], outline=None, width=0)
            
            # Draw title text for all shelf items
            if df is not None:
                try:
                    # Get artwork title for this point
                    shelf0_indices_list = list(shelf0_indices)
                    if idx < len(shelf0_indices_list):
                        artwork_idx_for_text = shelf0_indices_list[idx]
                        artwork_id = all_artwork_ids[artwork_idx_for_text]
                        artwork_row = df[df["id"].astype(float) == float(artwork_id)]
                        if artwork_row.empty:
                            artwork_row = df[df["id"].astype(str).str.strip() == str(artwork_id).strip()]
                        if not artwork_row.empty:
                            title = str(artwork_row.iloc[0].get("title", ""))
                            if title and title != "nan":
                                # Draw title text to the right of the point
                                text_x = x + POINT_SIZE_LARGE + 5
                                text_y = y - 10
                                font_map_text = get_font(FONT_SIZE_MAP_TEXT, "thin")
                                draw.text((text_x, text_y), title, fill=colors["text"], font=font_map_text)
                except Exception:
                    pass
    
    # Draw connections between shelf 0 items (thin lines, gray on black - less obvious)
    # Only draw lines if step is not "highlight_slow"
    if step != "highlight_slow":
        coords_to_use = shelf0_coords_progressive if (shelf0_coords_progressive is not None and num_shelf0_shown is not None) else shelf0_coords
        if coords_to_use is not None and len(coords_to_use) > 1:
            num_to_show = num_shelf0_shown if num_shelf0_shown is not None else len(coords_to_use)
            for i in range(min(num_to_show, len(coords_to_use))):
                for j in range(i + 1, min(num_to_show, len(coords_to_use))):
                    x1, y1 = coords_to_use[i]
                    x2, y2 = coords_to_use[j]
                    draw.line([(x1, y1), (x2, y2)], fill=colors["connection"], width=LINE_WIDTH)
    
    # Extract centroid coordinates
    cx, cy = None, None
    if centroid_coord is not None:
        try:
            if isinstance(centroid_coord, np.ndarray):
                cx, cy = float(centroid_coord[0]), float(centroid_coord[1])
            elif isinstance(centroid_coord, (tuple, list)) and len(centroid_coord) >= 2:
                cx, cy = float(centroid_coord[0]), float(centroid_coord[1])
            if cx is not None and cy is not None:
                cx = max(50, min(cx, MAP_WIDTH - 50))
                cy = max(100, min(cy, MAP_HEIGHT - 50))
        except (ValueError, TypeError, IndexError, AttributeError):
            cx, cy = None, None
    
    # Draw lines from centroid to shelf 0 points (if centroid is available and not in slow step)
    if cx is not None and cy is not None and step != "highlight_slow":
        # Draw lines progressively if lines_to_draw is provided (for animation)
        if lines_to_draw is not None:
            for line_data in lines_to_draw:
                if len(line_data) == 4:  # (start, end, progress, is_closest)
                    start, end, progress, is_closest = line_data
                else:  # (start, end, progress)
                    start, end, progress = line_data
                    is_closest = False
                
                if progress > 0:
                    # Calculate intermediate point based on progress
                    x1, y1 = start
                    x2, y2 = end
                    inter_x = x1 + (x2 - x1) * progress
                    inter_y = y1 + (y2 - y1) * progress
                    # Draw line from start to intermediate point
                    # Use gray for all lines (no green line for closest)
                    line_color = colors["connection"]
                    line_width = LINE_WIDTH
                    draw.line([(x1, y1), (inter_x, inter_y)], fill=line_color, width=line_width)
        else:
            # Draw lines from centroid to shelf 0 points (gray on black - less obvious)
            # Use progressive coords if available
            coords_to_use = shelf0_coords_progressive if (shelf0_coords_progressive is not None and num_shelf0_shown is not None) else shelf0_coords
            if coords_to_use is not None:
                num_to_show = num_shelf0_shown if num_shelf0_shown is not None else len(coords_to_use)
                for i in range(min(num_to_show, len(coords_to_use))):
                    x, y = coords_to_use[i]
                    draw.line([(cx, cy), (x, y)], fill=colors["connection"], width=LINE_WIDTH)
    
    # Highlight currently displayed artwork (if different from representative)
    if highlighted_artwork_idx is not None and shelf0_coords is not None:
        # Convert to plain int
        try:
            if hasattr(highlighted_artwork_idx, 'item'):
                highlight_idx_check = int(highlighted_artwork_idx.item())
            elif isinstance(highlighted_artwork_idx, (np.integer, np.int64, np.int32)):
                highlight_idx_check = int(highlighted_artwork_idx)
            else:
                highlight_idx_check = int(highlighted_artwork_idx)
        except (ValueError, TypeError, AttributeError):
            highlight_idx_check = int(float(str(highlighted_artwork_idx)))
        
        shelf0_indices = np.where(shelf0_mask)[0]
        shelf0_indices_list = [int(x) for x in shelf0_indices.tolist()]  # Ensure all are plain ints
        if highlight_idx_check in shelf0_indices_list:
            highlight_idx_in_shelf0 = shelf0_indices_list.index(highlight_idx_check)
            if highlight_idx_in_shelf0 < len(shelf0_coords):
                hx, hy = shelf0_coords[highlight_idx_in_shelf0]
                # Draw pulsing circle around highlighted artwork
                draw.ellipse([hx - 20, hy - 20, hx + 20, hy + 20],
                            fill=None, outline=colors["text"], width=2)
    
    # Highlight representative (if different from highlighted)
    if representative_idx is not None and shelf0_coords is not None:
        # Skip if it's a DataFrame or Series (shouldn't happen, but handle gracefully)
        if isinstance(representative_idx, (pd.DataFrame, pd.Series)):
            rep_idx = None  # Skip highlighting if it's not a valid index
        else:
            # Convert to int if it's a pandas object - ensure it's a plain Python int
            try:
                if hasattr(representative_idx, 'item'):
                    rep_idx = int(representative_idx.item())
                elif isinstance(representative_idx, (np.integer, np.int64, np.int32)):
                    rep_idx = int(representative_idx)
                elif hasattr(representative_idx, '__int__'):
                    rep_idx = int(representative_idx)
                else:
                    rep_idx = int(representative_idx)
            except (ValueError, TypeError, AttributeError):
                # Last resort: try to extract value
                try:
                    rep_idx = int(float(str(representative_idx)))
                except (ValueError, TypeError):
                    rep_idx = None  # Skip if we can't convert
        
        # Convert highlighted_artwork_idx similarly
        try:
            if highlighted_artwork_idx is not None:
                if hasattr(highlighted_artwork_idx, 'item'):
                    highlight_idx = int(highlighted_artwork_idx.item())
                elif hasattr(highlighted_artwork_idx, '__int__'):
                    highlight_idx = int(highlighted_artwork_idx)
                elif isinstance(highlighted_artwork_idx, (np.integer, np.int64, np.int32)):
                    highlight_idx = int(highlighted_artwork_idx)
                else:
                    highlight_idx = int(highlighted_artwork_idx)
            else:
                highlight_idx = None
        except (ValueError, TypeError, AttributeError):
            if highlighted_artwork_idx is not None:
                highlight_idx = int(float(str(highlighted_artwork_idx)))
            else:
                highlight_idx = None
        
        shelf0_indices = np.where(shelf0_mask)[0]
        shelf0_indices_list = [int(x) for x in shelf0_indices.tolist()]  # Ensure all are plain ints
        if rep_idx is not None:
            rep_idx = int(rep_idx)  # Final safety check
            if rep_idx in shelf0_indices_list and (highlight_idx is None or rep_idx != highlight_idx):
                rep_idx_in_shelf0 = shelf0_indices_list.index(rep_idx)
                if rep_idx_in_shelf0 < len(shelf0_coords):
                    rx, ry = shelf0_coords[rep_idx_in_shelf0]
                    # Draw larger bright green circle around representative
                    draw.ellipse([rx - 15, ry - 15, rx + 15, ry + 15],
                                fill=None, outline=colors["lime"], width=3)
    
    # Mark all top 10 representatives with bright green (if not in representatives step)
    # Skip items that are already marked as the main representative to avoid duplicate circles
    if step != "representatives" and top_representatives is not None and shelf0_coords is not None:
        shelf0_indices = np.where(shelf0_mask)[0]
        # Get the representative_idx to skip it (already has a larger circle)
        rep_idx_to_skip = None
        if representative_idx is not None:
            try:
                if hasattr(representative_idx, 'item'):
                    rep_idx_to_skip = int(representative_idx.item())
                elif isinstance(representative_idx, (np.integer, np.int64, np.int32)):
                    rep_idx_to_skip = int(representative_idx)
                else:
                    rep_idx_to_skip = int(representative_idx)
            except (ValueError, TypeError, AttributeError):
                try:
                    rep_idx_to_skip = int(float(str(representative_idx)))
                except (ValueError, TypeError):
                    rep_idx_to_skip = None
        
        for rank, (artwork_idx_in_all, distance) in enumerate(top_representatives[:10]):
            # Skip if this is the main representative (already has a larger circle)
            if rep_idx_to_skip is not None and artwork_idx_in_all == rep_idx_to_skip:
                continue
                
            if artwork_idx_in_all in shelf0_indices:
                idx_in_shelf0 = list(shelf0_indices).index(artwork_idx_in_all)
                if idx_in_shelf0 < len(shelf0_coords):
                    x, y = shelf0_coords[idx_in_shelf0]
                    # Draw bright green circle around representative
                    draw.ellipse([x - 12, y - 12, x + 12, y + 12],
                                fill=None, outline=colors["lime"], width=2)
    
    # Draw calculation info panel content (text and images) - draw LAST so it's on top
    if highlighted_artwork_idx is not None and df is not None:
        # Find artwork in dataframe
        try:
            artwork_id = all_artwork_ids[highlighted_artwork_idx]
            # Try multiple ways to match the ID (handle both string and float)
            artwork_row = pd.DataFrame()
            try:
                # Try exact match with float
                artwork_row = df[df["id"].astype(float) == float(artwork_id)]
            except (ValueError, TypeError):
                pass
            
            if artwork_row.empty:
                # Try string match
                try:
                    artwork_row = df[df["id"].astype(str).str.strip() == str(artwork_id).strip()]
                except (ValueError, TypeError):
                    pass
            
            if artwork_row.empty:
                # Try removing .0 from float strings
                try:
                    artwork_row = df[df["id"].astype(str).str.replace('.0', '', regex=False).str.strip() == str(artwork_id).strip()]
                except (ValueError, TypeError):
                    pass
        except (IndexError, KeyError, TypeError) as e:
            artwork_row = pd.DataFrame()
        
        if not artwork_row.empty:
            artwork = artwork_row.iloc[0]
            # Note: Panel background already drawn earlier
            
            # Draw thumbnail at the very top first
            image = load_image(artwork.get("thumbnail", ""))
            panel_x = MAP_WIDTH + 30
            y_pos = 100  # Start from top, below the main title
            
            if image:
                # Fixed height, maintain aspect ratio
                img_w, img_h = image.size
                ratio = FIXED_IMAGE_HEIGHT / img_h
                new_w, new_h = int(img_w * ratio), FIXED_IMAGE_HEIGHT
                image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                # Draw border around image
                border_padding = 4
                img_x = MAP_WIDTH + (PANEL_WIDTH - image.width) // 2
                draw.rectangle(
                    [img_x - border_padding, y_pos - border_padding,
                     img_x + image.width + border_padding, y_pos + image.height + border_padding],
                    fill=None, outline=colors["text"], width=2
                )
                # Paste image - need to use alpha composite if image has alpha, otherwise direct paste
                if image.mode == 'RGBA':
                    # Create a white background for the image area first
                    img_area = Image.new('RGB', (image.width + 2*border_padding, image.height + 2*border_padding), colors["background"])
                    img_area.paste(image, (border_padding, border_padding), image if image.mode == 'RGBA' else None)
                    img.paste(img_area, (img_x - border_padding, y_pos - border_padding))
                else:
                    img.paste(image, (img_x, y_pos))
                y_pos += image.height + 40
            else:
                y_pos += 30
            
            # Draw divider line after image
            draw.line([(panel_x, y_pos), (CANVAS_WIDTH - 30, y_pos)], 
                     fill=colors["text"], width=1)
            y_pos += 20
            
            # Draw text info with better spacing and design
            # Title (white text on black, larger)
            title = str(artwork.get("title", "Unknown"))
            # Wrap title if too long
            max_title_width = PANEL_WIDTH - 60
            title_words = title.split()
            title_lines = []
            current_line = ""
            for word in title_words:
                test_line = current_line + (" " if current_line else "") + word
                bbox = draw.textbbox((0, 0), test_line, font=font_title)
                if bbox[2] - bbox[0] <= max_title_width:
                    current_line = test_line
                else:
                    if current_line:
                        title_lines.append(current_line)
                    current_line = word
            if current_line:
                title_lines.append(current_line)
            
            for line in title_lines[:2]:  # Max 2 lines for title
                draw.text((panel_x, y_pos), line, fill=colors["text"], font=font_title)
                y_pos += 55
            y_pos += 10
            
            # Artist with label
            artist = str(artwork.get("artist", "Unknown"))
            draw.text((panel_x, y_pos), "Artist", fill=colors["text"], font=font_small)
            y_pos += 20
            draw.text((panel_x, y_pos), artist, fill=colors["text"], font=font_label)
            y_pos += 40
            
            # Year with label
            year = str(artwork.get("year", "N/A"))
            draw.text((panel_x, y_pos), "Year", fill=colors["text"], font=font_small)
            y_pos += 20
            draw.text((panel_x, y_pos), year, fill=colors["text"], font=font_info)
            y_pos += 35
            
            # Size with label
            size = str(artwork.get("size", "N/A"))
            draw.text((panel_x, y_pos), "Size", fill=colors["text"], font=font_small)
            y_pos += 20
            draw.text((panel_x, y_pos), size, fill=colors["text"], font=font_info)
            y_pos += 35
            
            # Draw divider
            draw.line([(panel_x, y_pos), (CANVAS_WIDTH - 30, y_pos)], 
                     fill=colors["text"], width=1)
            y_pos += 20
            
            # Description (wrapped text) with label
            description = str(artwork.get("description", ""))
            if description and description != "nan" and description.strip():
                draw.text((panel_x, y_pos), "Description", fill=colors["text"], font=font_small)
                y_pos += 20
                max_width = PANEL_WIDTH - 60
                words = description.split()
                lines = []
                current_line = ""
                for word in words:
                    test_line = current_line + (" " if current_line else "") + word
                    bbox = draw.textbbox((0, 0), test_line, font=font_small)
                    if bbox[2] - bbox[0] <= max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)
                
                for line in lines[:6]:  # Limit to 6 lines
                    draw.text((panel_x, y_pos), line, fill=colors["text"], font=font_small)
                    y_pos += 18
                y_pos += 15
            
            # Internal Note (wrapped text) with label
            internal_note = str(artwork.get("internalNote", ""))
            if internal_note and internal_note != "nan" and internal_note.strip():
                draw.line([(panel_x, y_pos), (CANVAS_WIDTH - 30, y_pos)], 
                         fill=colors["text"], width=1)
                y_pos += 20
                draw.text((panel_x, y_pos), "Internal Note", fill=colors["text"], font=font_small)
                y_pos += 20
                max_width = PANEL_WIDTH - 60
                words = internal_note.split()
                lines = []
                current_line = ""
                for word in words:
                    test_line = current_line + (" " if current_line else "") + word
                    bbox = draw.textbbox((0, 0), test_line, font=font_small)
                    if bbox[2] - bbox[0] <= max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)
                
                for line in lines[:4]:  # Limit to 4 lines
                    draw.text((panel_x, y_pos), line, fill=colors["text"], font=font_small)
                    y_pos += 18
            
            # Show embedding calculation info
            y_pos += 20
            draw.line([(panel_x, y_pos), (CANVAS_WIDTH - 30, y_pos)], 
                     fill=colors["text"], width=1)
            y_pos += 25
            
            # Embedding Calculation Section
            draw.text((panel_x, y_pos), "Embedding Calculation", 
                     fill=colors["text"], font=font_label)
            y_pos += 35
            
            # Get embedding info
            if all_embeddings is not None and highlighted_artwork_idx < len(all_embeddings):
                # Embedding dimension
                emb = all_embeddings[highlighted_artwork_idx]
                emb_dim = emb.shape[0] if hasattr(emb, 'shape') else 512
                emb_norm = np.linalg.norm(emb) if hasattr(emb, '__len__') else 0.0
                emb_mean = np.mean(emb) if hasattr(emb, '__len__') else 0.0
                
                draw.text((panel_x, y_pos), "Dimension", fill=colors["text"], font=font_small)
                y_pos += 20
                draw.text((panel_x, y_pos), f"{emb_dim}", fill=colors["text"], font=font_info)
                y_pos += 30
                
                draw.text((panel_x, y_pos), "Norm", fill=colors["text"], font=font_small)
                y_pos += 20
                draw.text((panel_x, y_pos), f"{emb_norm:.4f}", fill=colors["text"], font=font_info)
                y_pos += 30
                
                draw.text((panel_x, y_pos), "Mean", fill=colors["text"], font=font_small)
                y_pos += 20
                draw.text((panel_x, y_pos), f"{emb_mean:.4f}", fill=colors["text"], font=font_info)
                y_pos += 30
            
            # Distance to centroid if available
            if distances is not None and highlighted_artwork_idx is not None:
                # Convert to plain int
                try:
                    if hasattr(highlighted_artwork_idx, 'item'):
                        highlight_idx_dist = int(highlighted_artwork_idx.item())
                    elif isinstance(highlighted_artwork_idx, (np.integer, np.int64, np.int32)):
                        highlight_idx_dist = int(highlighted_artwork_idx)
                    else:
                        highlight_idx_dist = int(highlighted_artwork_idx)
                except (ValueError, TypeError, AttributeError):
                    highlight_idx_dist = int(float(str(highlighted_artwork_idx)))
                
                shelf0_indices = np.where(shelf0_mask)[0]
                shelf0_indices_int = np.array([int(x) for x in shelf0_indices])  # Ensure all are plain ints
                if highlight_idx_dist in shelf0_indices_int:
                    dist_idx = np.where(shelf0_indices_int == highlight_idx_dist)[0][0]
                    if dist_idx < len(distances):
                        distance = distances[dist_idx]
                        draw.text((panel_x, y_pos), "Distance to Centroid", 
                                 fill=colors["text"], font=font_small)
                        y_pos += 20
                        draw.text((panel_x, y_pos), f"{distance:.4f}", 
                                 fill=colors["lime"], font=font_label)
                        y_pos += 30
                        
                        # Show if this is the representative - convert both to plain ints for comparison
                        if representative_idx is not None:
                            try:
                                if hasattr(representative_idx, 'item'):
                                    rep_idx_compare = int(representative_idx.item())
                                elif isinstance(representative_idx, (np.integer, np.int64, np.int32)):
                                    rep_idx_compare = int(representative_idx)
                                else:
                                    rep_idx_compare = int(representative_idx)
                            except (ValueError, TypeError, AttributeError):
                                rep_idx_compare = int(float(str(representative_idx)))
                            
                            if highlight_idx_dist == rep_idx_compare:
                                y_pos += 10
                                draw.text((panel_x, y_pos), "Representative", 
                                         fill=colors["lime"], font=font_label)
    
    # Draw top 10 representatives with images on circles (with easing support)
    if step == "representatives" and top_representatives is not None and shelf0_coords is not None:
        shelf0_indices = np.where(shelf0_mask)[0]
        for rank, (artwork_idx_in_all, distance) in enumerate(top_representatives[:10]):
            if artwork_idx_in_all in shelf0_indices:
                idx_in_shelf0 = list(shelf0_indices).index(artwork_idx_in_all)
                if idx_in_shelf0 < len(shelf0_coords):
                    x, y = shelf0_coords[idx_in_shelf0]
                    
                    # Check if this is the aesthetic representative
                    is_aesthetic_rep = False
                    if aesthetic_representative_id is not None:
                        try:
                            aid = all_artwork_ids[artwork_idx_in_all]
                            if (float(aid) == float(aesthetic_representative_id) or 
                                str(aid) == str(aesthetic_representative_id) or
                                str(aid).replace('.0', '') == str(aesthetic_representative_id).replace('.0', '')):
                                is_aesthetic_rep = True
                        except (ValueError, TypeError):
                            if str(all_artwork_ids[artwork_idx_in_all]) == str(aesthetic_representative_id):
                                is_aesthetic_rep = True
                    
                    # Apply easing for the currently appearing item (last in the list)
                    is_currently_appearing = (rank == len(top_representatives) - 1 and color_ease_progress is not None)
                    opacity = 1.0
                    if is_currently_appearing and color_ease_progress is not None:
                        # Ease in opacity
                        eased = color_ease_progress * color_ease_progress * (3 - 2 * color_ease_progress)
                        opacity = eased
                    
                    # Get artwork image
                    if df is not None:
                        try:
                            artwork_id = all_artwork_ids[artwork_idx_in_all]
                            artwork_row = df[df["id"].astype(float) == float(artwork_id)]
                            if artwork_row.empty:
                                artwork_row = df[df["id"].astype(str).str.strip() == str(artwork_id).strip()]
                            
                            if not artwork_row.empty:
                                artwork = artwork_row.iloc[0]
                                thumbnail = load_image(artwork.get("thumbnail", ""))
                                
                                if thumbnail:
                                    # Resize image to fit circle
                                    size = POINT_SIZE_REPRESENTATIVE * 2
                                    thumbnail_resized = thumbnail.resize((size, size), Image.Resampling.LANCZOS)
                                    
                                    # Apply opacity if easing
                                    if opacity < 1.0:
                                        # Create alpha channel for opacity
                                        alpha = Image.new("L", (size, size), int(255 * opacity))
                                        thumbnail_resized.putalpha(alpha)
                                    
                                    # Create circular mask
                                    mask = Image.new("L", (size, size), 0)
                                    mask_draw = ImageDraw.Draw(mask)
                                    mask_draw.ellipse([0, 0, size, size], fill=255)
                                    
                                    # Apply opacity to mask if easing
                                    if opacity < 1.0:
                                        mask_array = np.array(mask, dtype=np.float32)
                                        mask_array *= opacity
                                        mask = Image.fromarray(mask_array.astype(np.uint8))
                                    
                                    # Paste image with circular mask
                                    img.paste(thumbnail_resized, 
                                            (int(x - POINT_SIZE_REPRESENTATIVE), 
                                             int(y - POINT_SIZE_REPRESENTATIVE)), 
                                            mask)
                                    
                                    # Mark aesthetic representative with bright green circle (only if not already marked as main representative)
                                    if is_aesthetic_rep:
                                        # Check if this is the main representative to avoid duplicate circles
                                        is_main_rep = False
                                        if representative_idx is not None:
                                            try:
                                                if hasattr(representative_idx, 'item'):
                                                    main_rep_idx = int(representative_idx.item())
                                                elif isinstance(representative_idx, (np.integer, np.int64, np.int32)):
                                                    main_rep_idx = int(representative_idx)
                                                else:
                                                    main_rep_idx = int(representative_idx)
                                                if artwork_idx_in_all == main_rep_idx:
                                                    is_main_rep = True
                                            except (ValueError, TypeError, AttributeError):
                                                pass
                                        
                                        # Only draw circle if not already the main representative (which has a larger circle)
                                        if not is_main_rep:
                                            # Draw bright green circle around aesthetic representative (with opacity)
                                            outline_color = colors["lime"]
                                            if opacity < 1.0:
                                                outline_color = tuple(int(c * opacity) for c in colors["lime"])
                                            draw.ellipse([int(x - POINT_SIZE_REPRESENTATIVE - 5), 
                                                        int(y - POINT_SIZE_REPRESENTATIVE - 5),
                                                        int(x + POINT_SIZE_REPRESENTATIVE + 5), 
                                                        int(y + POINT_SIZE_REPRESENTATIVE + 5)],
                                                       fill=None, outline=outline_color, width=3)
                        except Exception:
                            pass
    
    
    # Draw centroid with crosshair design (more elegant than big circle)
    if cx is not None and cy is not None:
        # Draw a small crosshair at centroid
        crosshair_size = 12
        line_length = 20
        # Draw horizontal line
        draw.line([(int(cx - line_length), int(cy)), (int(cx + line_length), int(cy))],
                 fill=colors["text"], width=2)
        # Draw vertical line
        draw.line([(int(cx), int(cy - line_length)), (int(cx), int(cy + line_length))],
                 fill=colors["text"], width=2)
        # Draw small circle at center
        draw.ellipse([int(cx - crosshair_size), int(cy - crosshair_size), 
                     int(cx + crosshair_size), int(cy + crosshair_size)],
                    fill=None, outline=colors["text"], width=2)
    
    # Draw table on right side for representatives step
    if step == "representatives" and top_representatives is not None and df is not None:
        panel_x = MAP_WIDTH + 30
        y_pos = 100
        
        # Title - determine based on average distance
        if top_representatives is not None and len(top_representatives) > 0:
            avg_distance = sum(d for _, d in top_representatives[:5]) / min(5, len(top_representatives))
            table_title = "Top 10 Outliers" if avg_distance > 0.5 else "Top 10 Representatives"
        else:
            table_title = "Top 10 Representatives"
        draw.text((panel_x, y_pos), table_title, fill=colors["text"], font=font_title)
        y_pos += 60
        
        # Draw table header
        draw.text((panel_x, y_pos), "Rank", fill=colors["text"], font=font_label)
        draw.text((panel_x + 60, y_pos), "Title", fill=colors["text"], font=font_label)
        draw.text((panel_x + 300, y_pos), "Artist", fill=colors["text"], font=font_label)
        draw.text((panel_x + 500, y_pos), "Distance", fill=colors["text"], font=font_label)
        y_pos += 40
        
        # Draw divider
        draw.line([(panel_x, y_pos), (CANVAS_WIDTH - 30, y_pos)], fill=colors["text"], width=1)
        y_pos += 20
        
        # Draw table rows
        for rank, (artwork_idx_in_all, distance) in enumerate(top_representatives[:10]):
            try:
                artwork_id = all_artwork_ids[artwork_idx_in_all]
                artwork_row = df[df["id"].astype(float) == float(artwork_id)]
                if artwork_row.empty:
                    artwork_row = df[df["id"].astype(str).str.strip() == str(artwork_id).strip()]
                
                if not artwork_row.empty:
                    artwork = artwork_row.iloc[0]
                    
                    # Check if this is the aesthetic representative
                    is_aesthetic_rep = False
                    if aesthetic_representative_id is not None:
                        try:
                            if (float(artwork_id) == float(aesthetic_representative_id) or 
                                str(artwork_id) == str(aesthetic_representative_id) or
                                str(artwork_id).replace('.0', '') == str(aesthetic_representative_id).replace('.0', '')):
                                is_aesthetic_rep = True
                        except (ValueError, TypeError):
                            if str(artwork_id) == str(aesthetic_representative_id):
                                is_aesthetic_rep = True
                    
                    # Row color - green for aesthetic rep, white for others
                    row_color = colors["lime"] if is_aesthetic_rep else colors["text"]
                    
                    # Rank
                    draw.text((panel_x, y_pos), f"{rank + 1}", fill=row_color, font=font_info)
                    
                    # Title (truncate if too long)
                    title = str(artwork.get("title", "Unknown"))[:30]
                    draw.text((panel_x + 60, y_pos), title, fill=row_color, font=font_small)
                    
                    # Artist (truncate if too long)
                    artist = str(artwork.get("artist", "Unknown"))[:20]
                    draw.text((panel_x + 300, y_pos), artist, fill=row_color, font=font_small)
                    
                    # Distance
                    draw.text((panel_x + 500, y_pos), f"{distance:.4f}", fill=row_color, font=font_small)
                    
                    # Small thumbnail
                    thumbnail = load_image(artwork.get("thumbnail", ""))
                    if thumbnail:
                        thumb_size = 30
                        thumbnail_resized = thumbnail.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                        img.paste(thumbnail_resized, (panel_x + 420, y_pos - 5))
                    
                    y_pos += 50
            except Exception:
                pass
    
    # Draw green lines between top 10 items (5 representatives + 5 outliers)
    if step == "top10" and top_representatives is not None and top_outliers is not None and shelf0_coords is not None:
        shelf0_indices = np.where(shelf0_mask)[0]
        # Get all top 10 items (5 reps + 5 outliers)
        all_top10_items = []
        num_reps_to_show = top10_reps_shown if top10_reps_shown is not None else len(top_representatives[:5])
        num_outliers_to_show = top10_outliers_shown if top10_outliers_shown is not None else len(top_outliers[:5])
        
        # Add representatives - always include all shown reps
        for rank, (artwork_idx_in_all, distance) in enumerate(top_representatives[:5]):
            if rank < num_reps_to_show and artwork_idx_in_all in shelf0_indices:
                idx_in_shelf0 = list(shelf0_indices).index(artwork_idx_in_all)
                if idx_in_shelf0 < len(shelf0_coords):
                    all_top10_items.append(shelf0_coords[idx_in_shelf0])
        
        # Add outliers - include shown outliers
        for rank, (artwork_idx_in_all, distance) in enumerate(top_outliers[:5]):
            if rank < num_outliers_to_show and artwork_idx_in_all in shelf0_indices:
                idx_in_shelf0 = list(shelf0_indices).index(artwork_idx_in_all)
                if idx_in_shelf0 < len(shelf0_coords):
                    all_top10_items.append(shelf0_coords[idx_in_shelf0])
        
        # Draw green lines between all pairs of top 10 items
        # Only draw if we have at least 2 items
        if len(all_top10_items) >= 2:
            for i in range(len(all_top10_items)):
                for j in range(i + 1, len(all_top10_items)):
                    x1, y1 = all_top10_items[i]
                    x2, y2 = all_top10_items[j]
                    draw.line([(x1, y1), (x2, y2)], fill=colors["lime"], width=1)
    
    # Draw top 10 table (5 representatives + 5 outliers) on right side
    if step == "top10" and top_representatives is not None and top_outliers is not None and df is not None:
        panel_x = MAP_WIDTH + 30
        y_pos = 100
        
        # Title - shorter, better spacing
        draw.text((panel_x, y_pos), "Representatives", fill=colors["text"], font=font_title)
        y_pos += 80  # Increased spacing
        
        # Representatives section header
        draw.text((panel_x, y_pos), "Rank", fill=colors["text"], font=font_small)
        draw.text((panel_x + 60, y_pos), "Title", fill=colors["text"], font=font_small)
        draw.text((panel_x + 250, y_pos), "Artist", fill=colors["text"], font=font_small)
        draw.text((panel_x + 400, y_pos), "Distance", fill=colors["text"], font=font_small)
        y_pos += 30
        
        # Draw representatives rows (progressive with easing)
        num_reps_to_show = top10_reps_shown if top10_reps_shown is not None else len(top_representatives[:5])
        for rank, (artwork_idx_in_all, distance) in enumerate(top_representatives[:5]):
            if rank >= num_reps_to_show:
                break
            
            try:
                artwork_id = all_artwork_ids[artwork_idx_in_all]
                artwork_row = df[df["id"].astype(float) == float(artwork_id)]
                if artwork_row.empty:
                    artwork_row = df[df["id"].astype(str).str.strip() == str(artwork_id).strip()]
                
                if not artwork_row.empty:
                    artwork = artwork_row.iloc[0]
                    is_selected = (aesthetic_representative_id is not None and 
                                  (float(artwork_id) == float(aesthetic_representative_id) or 
                                   str(artwork_id) == str(aesthetic_representative_id)))
                    
                    # Apply easing for the currently appearing item
                    # Only apply easing if we're currently showing representatives (not outliers)
                    is_currently_appearing = (rank == num_reps_to_show - 1 and 
                                            top10_item_ease_progress is not None and 
                                            (top10_outliers_shown is None or top10_outliers_shown == 0))
                    if is_currently_appearing and top10_item_ease_progress is not None:
                        # Ease in opacity: interpolate from transparent to full
                        eased = top10_item_ease_progress * top10_item_ease_progress * (3 - 2 * top10_item_ease_progress)
                        base_color = colors["lime"] if is_selected else colors["text"]
                        # Interpolate from black (invisible) to base color
                        r = int(base_color[0] * eased)
                        g = int(base_color[1] * eased)
                        b = int(base_color[2] * eased)
                        row_color = (r, g, b)
                    else:
                        row_color = colors["lime"] if is_selected else colors["text"]
                    
                    # Draw tiny thumbnail on the left
                    thumbnail = load_image(artwork.get("thumbnail", ""))
                    if thumbnail:
                        thumb_size = 20  # Tiny thumbnail size
                        thumbnail_resized = thumbnail.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                        # Position thumbnail before rank number
                        thumb_x = panel_x - 30
                        thumb_y = y_pos - 2
                        img.paste(thumbnail_resized, (thumb_x, thumb_y))
                    
                    draw.text((panel_x, y_pos), f"{rank + 1}", fill=row_color, font=font_small)
                    title = str(artwork.get("title", "Unknown"))[:25]
                    draw.text((panel_x + 60, y_pos), title, fill=row_color, font=font_small)
                    artist = str(artwork.get("artist", "Unknown"))[:20]
                    draw.text((panel_x + 250, y_pos), artist, fill=row_color, font=font_small)
                    draw.text((panel_x + 400, y_pos), f"{distance:.4f}", fill=row_color, font=font_small)
                    y_pos += 35
            except Exception:
                pass
        
        y_pos += 20
        draw.line([(panel_x, y_pos), (CANVAS_WIDTH - 30, y_pos)], fill=colors["text"], width=1)
        y_pos += 30
        
        # Outliers section title
        draw.text((panel_x, y_pos), "Outliers", fill=colors["text"], font=font_title)
        y_pos += 80  # Increased spacing
        
        # Outliers section header
        draw.text((panel_x, y_pos), "Rank", fill=colors["text"], font=font_small)
        draw.text((panel_x + 60, y_pos), "Title", fill=colors["text"], font=font_small)
        draw.text((panel_x + 250, y_pos), "Artist", fill=colors["text"], font=font_small)
        draw.text((panel_x + 400, y_pos), "Distance", fill=colors["text"], font=font_small)
        y_pos += 30
        
        # Draw outliers rows (progressive with easing)
        num_outliers_to_show = top10_outliers_shown if top10_outliers_shown is not None else len(top_outliers[:5])
        for rank, (artwork_idx_in_all, distance) in enumerate(top_outliers[:5]):
            if rank >= num_outliers_to_show:
                break
            
            try:
                artwork_id = all_artwork_ids[artwork_idx_in_all]
                artwork_row = df[df["id"].astype(float) == float(artwork_id)]
                if artwork_row.empty:
                    artwork_row = df[df["id"].astype(str).str.strip() == str(artwork_id).strip()]
                
                if not artwork_row.empty:
                    artwork = artwork_row.iloc[0]
                    is_selected = (aesthetic_outlier_id is not None and 
                                  (float(artwork_id) == float(aesthetic_outlier_id) or 
                                   str(artwork_id) == str(aesthetic_outlier_id)))
                    
                    # Apply easing for the currently appearing item (last outlier being shown)
                    is_currently_appearing = (rank == num_outliers_to_show - 1 and top10_item_ease_progress is not None)
                    if is_currently_appearing and top10_item_ease_progress is not None:
                        # Ease in opacity
                        eased = top10_item_ease_progress * top10_item_ease_progress * (3 - 2 * top10_item_ease_progress)
                        base_color = colors["lime"] if is_selected else colors["text"]
                        r = int(base_color[0] * eased)
                        g = int(base_color[1] * eased)
                        b = int(base_color[2] * eased)
                        row_color = (r, g, b)
                    else:
                        row_color = colors["lime"] if is_selected else colors["text"]
                    
                    # Draw tiny thumbnail on the left
                    thumbnail = load_image(artwork.get("thumbnail", ""))
                    if thumbnail:
                        thumb_size = 20  # Tiny thumbnail size
                        thumbnail_resized = thumbnail.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                        # Position thumbnail before rank number
                        thumb_x = panel_x - 30
                        thumb_y = y_pos - 2
                        img.paste(thumbnail_resized, (thumb_x, thumb_y))
                    
                    draw.text((panel_x, y_pos), f"{rank + 1}", fill=row_color, font=font_small)
                    title = str(artwork.get("title", "Unknown"))[:25]
                    draw.text((panel_x + 60, y_pos), title, fill=row_color, font=font_small)
                    artist = str(artwork.get("artist", "Unknown"))[:20]
                    draw.text((panel_x + 250, y_pos), artist, fill=row_color, font=font_small)
                    draw.text((panel_x + 400, y_pos), f"{distance:.4f}", fill=row_color, font=font_small)
                    y_pos += 35
            except Exception:
                pass
    
    # Draw side-by-side view of selected items on left screen, rank table on right
    if step == "side_by_side" and df is not None:
        # Clear left screen area (map area)
        draw.rectangle([0, 0, MAP_WIDTH, MAP_HEIGHT], fill=colors["background"])
        
        # Find representative and outlier artworks
        rep_artwork = None
        outlier_artwork = None
        rep_idx = None
        outlier_idx = None
        
        if aesthetic_representative_id is not None:
            for i, aid in enumerate(all_artwork_ids):
                try:
                    if float(aid) == float(aesthetic_representative_id) or str(aid) == str(aesthetic_representative_id):
                        rep_idx = i
                        rep_id = aid
                        rep_row = df[df["id"].astype(float) == float(rep_id)]
                        if rep_row.empty:
                            rep_row = df[df["id"].astype(str).str.strip() == str(rep_id).strip()]
                        if not rep_row.empty:
                            rep_artwork = rep_row.iloc[0]
                        break
                except (ValueError, TypeError):
                    if str(aid) == str(aesthetic_representative_id):
                        rep_idx = i
                        rep_id = aid
                        rep_row = df[df["id"].astype(str).str.strip() == str(rep_id).strip()]
                        if not rep_row.empty:
                            rep_artwork = rep_row.iloc[0]
                        break
        
        if aesthetic_outlier_id is not None:
            for i, aid in enumerate(all_artwork_ids):
                try:
                    if float(aid) == float(aesthetic_outlier_id) or str(aid) == str(aesthetic_outlier_id):
                        outlier_idx = i
                        outlier_id = aid
                        outlier_row = df[df["id"].astype(float) == float(outlier_id)]
                        if outlier_row.empty:
                            outlier_row = df[df["id"].astype(str).str.strip() == str(outlier_id).strip()]
                        if not outlier_row.empty:
                            outlier_artwork = outlier_row.iloc[0]
                        break
                except (ValueError, TypeError):
                    if str(aid) == str(aesthetic_outlier_id):
                        outlier_idx = i
                        outlier_id = aid
                        outlier_row = df[df["id"].astype(str).str.strip() == str(outlier_id).strip()]
                        if not outlier_row.empty:
                            outlier_artwork = outlier_row.iloc[0]
                        break
        
        # Draw both items on left side, stacked vertically
        left_x = 50
        y_start = 100
        item_width = MAP_WIDTH - 100  # Use most of left side width
        item_height = (MAP_HEIGHT - 150) // 2  # Split height for two items
        
        # Representative (top half of left side)
        if rep_artwork is not None:
            rep_y = y_start
            
            # Image - fixed height (smaller to leave more space for info)
            rep_image = load_image(rep_artwork.get("thumbnail", ""))
            if rep_image:
                img_w, img_h = rep_image.size
                # Use smaller fixed height to leave more space for text
                fixed_img_height = min(item_height - 250, 200)  # Leave more space for text
                ratio = fixed_img_height / img_h
                new_w, new_h = int(img_w * ratio), fixed_img_height
                if new_w > item_width - 40:
                    ratio = (item_width - 40) / img_w
                    new_w, new_h = int(img_w * ratio), int(img_h * ratio)
                rep_image = rep_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                img_x = left_x + (item_width - rep_image.width) // 2
                img.paste(rep_image, (img_x, rep_y))
                rep_y += fixed_img_height + 20
            
            # Title - use green for label, consistent font size
            title = str(rep_artwork.get("title", "Unknown"))
            draw.text((left_x, rep_y), "Representative", fill=colors["lime"], font=font_label)
            rep_y += 35
            # Wrap title if needed
            max_title_width = item_width - 20
            title_words = title.split()
            title_lines = []
            current_line = ""
            for word in title_words:
                test_line = current_line + (" " if current_line else "") + word
                bbox = draw.textbbox((0, 0), test_line, font=font_info)
                if bbox[2] - bbox[0] <= max_title_width:
                    current_line = test_line
                else:
                    if current_line:
                        title_lines.append(current_line)
                    current_line = word
            if current_line:
                title_lines.append(current_line)
            
            for line in title_lines[:2]:  # Max 2 lines
                draw.text((left_x, rep_y), line, fill=colors["text"], font=font_info)
                rep_y += 25
            rep_y += 10
            
            # Artist
            artist = str(rep_artwork.get("artist", "Unknown"))
            draw.text((left_x, rep_y), "Artist", fill=colors["text"], font=font_small)
            rep_y += 18
            draw.text((left_x, rep_y), artist, fill=colors["text"], font=font_info)
            rep_y += 25
            
            # Year
            year = str(rep_artwork.get("year", "N/A"))
            draw.text((left_x, rep_y), "Year", fill=colors["text"], font=font_small)
            rep_y += 18
            draw.text((left_x, rep_y), year, fill=colors["text"], font=font_info)
            rep_y += 25
            
            # Size
            size = str(rep_artwork.get("size", "N/A"))
            if size and size != "N/A" and size != "nan":
                draw.text((left_x, rep_y), "Size", fill=colors["text"], font=font_small)
                rep_y += 18
                draw.text((left_x, rep_y), size, fill=colors["text"], font=font_info)
                rep_y += 25
            
            # Distance to centroid
            if distances is not None and rep_idx is not None:
                shelf0_indices = np.where(shelf0_mask)[0]
                if rep_idx in shelf0_indices:
                    shelf0_idx = list(shelf0_indices).index(rep_idx)
                    if shelf0_idx < len(distances):
                        distance = distances[shelf0_idx]
                        draw.text((left_x, rep_y), "Distance to Centroid", fill=colors["text"], font=font_small)
                        rep_y += 18
                        draw.text((left_x, rep_y), f"{distance:.4f}", fill=colors["lime"], font=font_info)
                        rep_y += 25
            
            # Description (wrapped text)
            description = str(rep_artwork.get("description", ""))
            if description and description != "nan" and description.strip():
                draw.line([(left_x, rep_y), (MAP_WIDTH - 50, rep_y)], fill=colors["text"], width=1)
                rep_y += 15
                draw.text((left_x, rep_y), "Description", fill=colors["text"], font=font_small)
                rep_y += 18
                max_width = item_width - 20
                words = description.split()
                lines = []
                current_line = ""
                for word in words:
                    test_line = current_line + (" " if current_line else "") + word
                    bbox = draw.textbbox((0, 0), test_line, font=font_small)
                    if bbox[2] - bbox[0] <= max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)
                
                for line in lines[:3]:  # Limit to 3 lines to fit
                    draw.text((left_x, rep_y), line, fill=colors["text"], font=font_small)
                    rep_y += 16
        
        # Draw divider between representative and outlier on left
        divider_y = y_start + item_height
        draw.line([(left_x, divider_y), (MAP_WIDTH - 50, divider_y)], fill=colors["text"], width=1)
        
        # Outlier (bottom half of left side)
        if outlier_artwork is not None:
            outlier_y = divider_y + 20
            
            # Image - fixed height (smaller to leave more space for info)
            outlier_image = load_image(outlier_artwork.get("thumbnail", ""))
            if outlier_image:
                img_w, img_h = outlier_image.size
                # Use smaller fixed height to leave more space for text
                fixed_img_height = min(item_height - 250, 200)  # Leave more space for text
                ratio = fixed_img_height / img_h
                new_w, new_h = int(img_w * ratio), fixed_img_height
                if new_w > item_width - 40:
                    ratio = (item_width - 40) / img_w
                    new_w, new_h = int(img_w * ratio), int(img_h * ratio)
                outlier_image = outlier_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                img_x = left_x + (item_width - outlier_image.width) // 2
                img.paste(outlier_image, (img_x, outlier_y))
                outlier_y += fixed_img_height + 20
            
            # Title - use green for label, consistent font size
            title = str(outlier_artwork.get("title", "Unknown"))
            draw.text((left_x, outlier_y), "Outlier", fill=colors["lime"], font=font_label)
            outlier_y += 35
            # Wrap title if needed
            max_title_width = item_width - 20
            title_words = title.split()
            title_lines = []
            current_line = ""
            for word in title_words:
                test_line = current_line + (" " if current_line else "") + word
                bbox = draw.textbbox((0, 0), test_line, font=font_info)
                if bbox[2] - bbox[0] <= max_title_width:
                    current_line = test_line
                else:
                    if current_line:
                        title_lines.append(current_line)
                    current_line = word
            if current_line:
                title_lines.append(current_line)
            
            for line in title_lines[:2]:  # Max 2 lines
                draw.text((left_x, outlier_y), line, fill=colors["text"], font=font_info)
                outlier_y += 25
            outlier_y += 10
            
            # Artist
            artist = str(outlier_artwork.get("artist", "Unknown"))
            draw.text((left_x, outlier_y), "Artist", fill=colors["text"], font=font_small)
            outlier_y += 18
            draw.text((left_x, outlier_y), artist, fill=colors["text"], font=font_info)
            outlier_y += 25
            
            # Year
            year = str(outlier_artwork.get("year", "N/A"))
            draw.text((left_x, outlier_y), "Year", fill=colors["text"], font=font_small)
            outlier_y += 18
            draw.text((left_x, outlier_y), year, fill=colors["text"], font=font_info)
            outlier_y += 25
            
            # Size
            size = str(outlier_artwork.get("size", "N/A"))
            if size and size != "N/A" and size != "nan":
                draw.text((left_x, outlier_y), "Size", fill=colors["text"], font=font_small)
                outlier_y += 18
                draw.text((left_x, outlier_y), size, fill=colors["text"], font=font_info)
                outlier_y += 25
            
            # Distance to centroid
            if distances is not None and outlier_idx is not None:
                shelf0_indices = np.where(shelf0_mask)[0]
                if outlier_idx in shelf0_indices:
                    shelf0_idx = list(shelf0_indices).index(outlier_idx)
                    if shelf0_idx < len(distances):
                        distance = distances[shelf0_idx]
                        draw.text((left_x, outlier_y), "Distance to Centroid", fill=colors["text"], font=font_small)
                        outlier_y += 18
                        draw.text((left_x, outlier_y), f"{distance:.4f}", fill=colors["lime"], font=font_info)
                        outlier_y += 25
            
            # Description (wrapped text)
            description = str(outlier_artwork.get("description", ""))
            if description and description != "nan" and description.strip():
                draw.line([(left_x, outlier_y), (MAP_WIDTH - 50, outlier_y)], fill=colors["text"], width=1)
                outlier_y += 15
                draw.text((left_x, outlier_y), "Description", fill=colors["text"], font=font_small)
                outlier_y += 18
                max_width = item_width - 20
                words = description.split()
                lines = []
                current_line = ""
                for word in words:
                    test_line = current_line + (" " if current_line else "") + word
                    bbox = draw.textbbox((0, 0), test_line, font=font_small)
                    if bbox[2] - bbox[0] <= max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)
                
                for line in lines[:3]:  # Limit to 3 lines to fit
                    draw.text((left_x, outlier_y), line, fill=colors["text"], font=font_small)
                    outlier_y += 16
        
        # Draw rank table on right side (keep existing top10 table code)
        panel_x = MAP_WIDTH + 30
        y_pos = 100
        
        # Representatives section
        draw.text((panel_x, y_pos), "Representatives", fill=colors["text"], font=font_title)
        y_pos += 80  # Match top10 spacing
        draw.text((panel_x, y_pos), "Rank", fill=colors["text"], font=font_small)
        draw.text((panel_x + 60, y_pos), "Title", fill=colors["text"], font=font_small)
        draw.text((panel_x + 250, y_pos), "Artist", fill=colors["text"], font=font_small)
        draw.text((panel_x + 400, y_pos), "Distance", fill=colors["text"], font=font_small)
        y_pos += 30
        
        # Draw representatives rows
        if top_representatives is not None:
            for rank, (artwork_idx_in_all, distance) in enumerate(top_representatives[:5]):
                try:
                    artwork_id = all_artwork_ids[artwork_idx_in_all]
                    artwork_row = df[df["id"].astype(float) == float(artwork_id)]
                    if artwork_row.empty:
                        artwork_row = df[df["id"].astype(str).str.strip() == str(artwork_id).strip()]
                    
                    if not artwork_row.empty:
                        artwork = artwork_row.iloc[0]
                        is_selected = (aesthetic_representative_id is not None and 
                                      (float(artwork_id) == float(aesthetic_representative_id) or 
                                       str(artwork_id) == str(aesthetic_representative_id)))
                        row_color = colors["lime"] if is_selected else colors["text"]
                        
                        draw.text((panel_x, y_pos), f"{rank + 1}", fill=row_color, font=font_small)
                        title = str(artwork.get("title", "Unknown"))[:25]
                        draw.text((panel_x + 60, y_pos), title, fill=row_color, font=font_small)
                        artist = str(artwork.get("artist", "Unknown"))[:20]
                        draw.text((panel_x + 250, y_pos), artist, fill=row_color, font=font_small)
                        draw.text((panel_x + 400, y_pos), f"{distance:.4f}", fill=row_color, font=font_small)
                        y_pos += 35
                except Exception:
                    pass
        
        y_pos += 20
        draw.line([(panel_x, y_pos), (CANVAS_WIDTH - 30, y_pos)], fill=colors["text"], width=1)
        y_pos += 30
        
        # Outliers section
        draw.text((panel_x, y_pos), "Outliers", fill=colors["text"], font=font_title)
        y_pos += 80  # Match top10 spacing
        draw.text((panel_x, y_pos), "Rank", fill=colors["text"], font=font_small)
        draw.text((panel_x + 60, y_pos), "Title", fill=colors["text"], font=font_small)
        draw.text((panel_x + 250, y_pos), "Artist", fill=colors["text"], font=font_small)
        draw.text((panel_x + 400, y_pos), "Distance", fill=colors["text"], font=font_small)
        y_pos += 30
        
        # Draw outliers rows
        if top_outliers is not None:
            for rank, (artwork_idx_in_all, distance) in enumerate(top_outliers[:5]):
                try:
                    artwork_id = all_artwork_ids[artwork_idx_in_all]
                    artwork_row = df[df["id"].astype(float) == float(artwork_id)]
                    if artwork_row.empty:
                        artwork_row = df[df["id"].astype(str).str.strip() == str(artwork_id).strip()]
                    
                    if not artwork_row.empty:
                        artwork = artwork_row.iloc[0]
                        is_selected = (aesthetic_outlier_id is not None and 
                                      (float(artwork_id) == float(aesthetic_outlier_id) or 
                                       str(artwork_id) == str(aesthetic_outlier_id)))
                        row_color = colors["lime"] if is_selected else colors["text"]
                        
                        draw.text((panel_x, y_pos), f"{rank + 1}", fill=row_color, font=font_small)
                        title = str(artwork.get("title", "Unknown"))[:25]
                        draw.text((panel_x + 60, y_pos), title, fill=row_color, font=font_small)
                        artist = str(artwork.get("artist", "Unknown"))[:20]
                        draw.text((panel_x + 250, y_pos), artist, fill=row_color, font=font_small)
                        draw.text((panel_x + 400, y_pos), f"{distance:.4f}", fill=row_color, font=font_small)
                        y_pos += 35
                except Exception:
                    pass
    
    return img


def main(target_shelf: str = "0", mode: str = "both", white_background: bool = False):
    """Main function to generate visualization frames and video.
    
    Args:
        target_shelf: The shelf number to visualize (as string, e.g., "0", "5")
        mode: "representative", "outlier", or "both" - which type to visualize (default: "both")
        white_background: If True, use white background with inverted colors (default: False)
    """
    print("=" * 60)
    if mode == "both":
        print(f"Visualizing Shelf {target_shelf} Representatives and Outliers")
    else:
        mode_title = "Representative" if mode == "representative" else "Outlier"
        print(f"Visualizing Shelf {target_shelf} {mode_title} Finding Process")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(str(CSV_PATH))
    print(f"   Loaded {len(df)} artworks from CSV")
    
    embeddings_dict = load_embeddings()
    
    # Parse shelf numbers - handle both float and string formats
    def parse_shelf_numbers_flexible(shelf_value):
        """Parse shelfNo value handling both float and string formats."""
        if pd.isna(shelf_value) or shelf_value is None:
            return []
        # Convert to string first, handling float values
        if isinstance(shelf_value, (int, float)):
            shelf_str = str(int(shelf_value))
        else:
            shelf_str = str(shelf_value).strip()
        if not shelf_str:
            return []
        # Split by semicolon or comma
        shelf_numbers = []
        for part in shelf_str.replace(",", ";").split(";"):
            part = part.strip()
            if part:
                try:
                    # Normalize to string representation of integer
                    normalized = str(int(float(part)))
                    shelf_numbers.append(normalized)
                except (ValueError, TypeError):
                    shelf_numbers.append(part)
        return shelf_numbers
    
    df["shelf_list"] = df["shelfNo"].apply(parse_shelf_numbers_flexible)
    df = df[df["shelf_list"].apply(len) > 0].copy()
    df = df.explode("shelf_list", ignore_index=True)
    df["shelfNo"] = df["shelf_list"].astype(str)
    df = df.drop(columns=["shelf_list"])
    
    # Filter for target shelf - check both string and numeric formats
    target_shelf_str = str(target_shelf)
    try:
        target_shelf_num = int(float(target_shelf))
    except (ValueError, TypeError):
        target_shelf_num = None
    
    df_shelf0 = df[
        (df["shelfNo"] == target_shelf_str) | 
        (df["shelfNo"].astype(str) == target_shelf_str)
    ].copy()
    
    # Also check numeric if target_shelf is numeric
    if target_shelf_num is not None:
        df_shelf0 = pd.concat([
            df_shelf0,
            df[pd.to_numeric(df["shelfNo"], errors='coerce') == target_shelf_num]
        ]).drop_duplicates()
    
    print(f"   Found {len(df_shelf0)} artworks on shelf {target_shelf}")
    
    # Match embeddings and load thumbnails
    print("\n2. Matching embeddings and loading thumbnails...")
    all_embeddings = []
    all_indices = []
    all_artwork_ids = []
    all_thumbnails = []
    
    # Get all artworks with embeddings
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading embeddings and thumbnails"):
        raw_id = row.get("id", idx)
        try:
            artwork_id = int(float(raw_id)) if pd.notna(raw_id) else idx
        except (ValueError, TypeError):
            artwork_id = raw_id if pd.notna(raw_id) else idx
        
        emb = None
        if artwork_id in embeddings_dict:
            emb = embeddings_dict[artwork_id]
        elif str(artwork_id) in embeddings_dict:
            emb = embeddings_dict[str(artwork_id)]
        
        if emb is not None:
            all_embeddings.append(emb)
            all_indices.append(idx)
            all_artwork_ids.append(artwork_id)
            # Load thumbnail
            thumbnail_path = row.get("thumbnail", "")
            thumb_img = load_image(thumbnail_path)
            all_thumbnails.append(thumb_img)
    
    all_embeddings = np.array(all_embeddings, dtype=np.float32)
    print(f"   Matched {len(all_embeddings)} embeddings")
    
    # Get shelf 0 embeddings
    shelf0_indices_in_all = []
    for idx, row in df_shelf0.iterrows():
        raw_id = row.get("id", idx)
        try:
            artwork_id = int(float(raw_id)) if pd.notna(raw_id) else idx
        except (ValueError, TypeError):
            artwork_id = raw_id if pd.notna(raw_id) else idx
        
        if artwork_id in all_artwork_ids:
            shelf0_indices_in_all.append(all_artwork_ids.index(artwork_id))
    
    shelf0_mask = np.zeros(len(all_embeddings), dtype=bool)
    shelf0_mask[shelf0_indices_in_all] = True
    shelf0_embeddings = all_embeddings[shelf0_mask]
    
    print(f"   Found {len(shelf0_embeddings)} shelf {target_shelf} embeddings")
    
    # Reduce dimensions
    print("\n3. Reducing dimensions...")
    all_coords_2d_raw, reducer = reduce_dimensions(all_embeddings)
    all_coords_2d, bounds = normalize_coords(all_coords_2d_raw)
    shelf0_coords_2d = all_coords_2d[shelf0_mask]
    
    # Calculate centroid
    print("\n4. Calculating centroid...")
    # Calculate centroid as mean of embeddings (don't normalize - we'll use it for distance calculation)
    centroid = np.mean(shelf0_embeddings, axis=0)
    
    # Find distances (using cosine distance or Euclidean - let's use Euclidean)
    distances = np.linalg.norm(shelf0_embeddings - centroid, axis=1)
    
    # Get aesthetic IDs from aesthetic.md - these are the curated selections
    REPRESENTATIVES = {
        0: 464, 1: 152, 2: 161, 3: 376, 4: 450,
        5: 360, 6: 468, 7: 107, 8: 389, 9: 185
    }
    OUTLIERS = {
        0: 479, 1: 386, 2: 334, 3: 82, 4: 424,
        5: 310, 6: 93, 7: 96, 8: 343, 9: 441
    }
    target_shelf_int = int(target_shelf) if target_shelf.isdigit() else 0
    aesthetic_representative_id = REPRESENTATIVES.get(target_shelf_int, None)
    aesthetic_outlier_id = OUTLIERS.get(target_shelf_int, None)
    
    # Find the aesthetic representative and outlier in shelf0 data
    # Note: distances array index matches df_shelf0.iloc position
    aesthetic_rep_idx_in_shelf0 = None
    aesthetic_outlier_idx_in_shelf0 = None
    
    if aesthetic_representative_id is not None:
        for pos, (idx, row) in enumerate(df_shelf0.iterrows()):
            artwork_id = row.get("id")
            try:
                if (float(artwork_id) == float(aesthetic_representative_id) or 
                    str(artwork_id) == str(aesthetic_representative_id) or
                    str(artwork_id).replace('.0', '') == str(aesthetic_representative_id).replace('.0', '')):
                    aesthetic_rep_idx_in_shelf0 = pos  # Position in df_shelf0 (matches distances array)
                    break
            except (ValueError, TypeError):
                if str(artwork_id) == str(aesthetic_representative_id):
                    aesthetic_rep_idx_in_shelf0 = pos
                    break
    
    if aesthetic_outlier_id is not None:
        for pos, (idx, row) in enumerate(df_shelf0.iterrows()):
            artwork_id = row.get("id")
            try:
                if (float(artwork_id) == float(aesthetic_outlier_id) or 
                    str(artwork_id) == str(aesthetic_outlier_id) or
                    str(artwork_id).replace('.0', '') == str(aesthetic_outlier_id).replace('.0', '')):
                    aesthetic_outlier_idx_in_shelf0 = pos  # Position in df_shelf0 (matches distances array)
                    break
            except (ValueError, TypeError):
                if str(artwork_id) == str(aesthetic_outlier_id):
                    aesthetic_outlier_idx_in_shelf0 = pos
                    break
    
    # Calculate top 5 by distance (for display)
    top_5_representatives_indices = np.argsort(distances)[:5]
    top_5_outliers_indices = np.argsort(distances)[-5:][::-1]  # Reverse to get farthest first
    
    # Ensure aesthetic IDs are included in the lists
    # If aesthetic rep is not in top 5, replace the last one
    if aesthetic_rep_idx_in_shelf0 is not None:
        # Convert to position in distances array (which matches df_shelf0 order)
        if aesthetic_rep_idx_in_shelf0 not in top_5_representatives_indices:
            # Replace the last (5th) item with the aesthetic one
            top_5_representatives_indices = np.append(top_5_representatives_indices[:4], aesthetic_rep_idx_in_shelf0)
            # Re-sort to maintain distance order
            top_5_representatives_indices = top_5_representatives_indices[np.argsort(distances[top_5_representatives_indices])]
    
    # If aesthetic outlier is not in top 5, replace the last one
    if aesthetic_outlier_idx_in_shelf0 is not None:
        if aesthetic_outlier_idx_in_shelf0 not in top_5_outliers_indices:
            # Replace the last (5th) item with the aesthetic one
            top_5_outliers_indices = np.append(top_5_outliers_indices[:4], aesthetic_outlier_idx_in_shelf0)
            # Re-sort to maintain distance order (farthest first)
            top_5_outliers_indices = top_5_outliers_indices[np.argsort(distances[top_5_outliers_indices])[::-1]]
    
    # Build list of top 5 representatives with their indices in all_artwork_ids
    top_representatives = []  # List of (artwork_idx_in_all, distance) tuples
    for idx_in_shelf0 in top_5_representatives_indices:
        artwork_id = df_shelf0.iloc[idx_in_shelf0].get("id")
        distance = distances[idx_in_shelf0]
        
        # Find this artwork in all_artwork_ids
        try:
            rep_id_float = float(artwork_id)
        except (ValueError, TypeError):
            rep_id_float = None
        rep_id_str = str(artwork_id)
        
        for i, aid in enumerate(all_artwork_ids):
            # Try multiple comparison methods
            if rep_id_float is not None:
                try:
                    if float(aid) == rep_id_float:
                        top_representatives.append((int(i), distance))
                        break
                except (ValueError, TypeError):
                    pass
            
            # Try string comparison
            if str(aid) == rep_id_str or str(aid).replace('.0', '') == rep_id_str.replace('.0', ''):
                top_representatives.append((int(i), distance))
                break
    
    # Build list of top 5 outliers with their indices in all_artwork_ids
    top_outliers = []  # List of (artwork_idx_in_all, distance) tuples
    for idx_in_shelf0 in top_5_outliers_indices:
        artwork_id = df_shelf0.iloc[idx_in_shelf0].get("id")
        distance = distances[idx_in_shelf0]
        
        # Find this artwork in all_artwork_ids
        try:
            outlier_id_float = float(artwork_id)
        except (ValueError, TypeError):
            outlier_id_float = None
        outlier_id_str = str(artwork_id)
        
        for i, aid in enumerate(all_artwork_ids):
            # Try multiple comparison methods
            if outlier_id_float is not None:
                try:
                    if float(aid) == outlier_id_float:
                        top_outliers.append((int(i), distance))
                        break
                except (ValueError, TypeError):
                    pass
            
            # Try string comparison
            if str(aid) == outlier_id_str or str(aid).replace('.0', '') == outlier_id_str.replace('.0', ''):
                top_outliers.append((int(i), distance))
                break
    
    # For backward compatibility, use top_representatives as top_items
    top_items = top_representatives
    
    # Get the first representative for backward compatibility
    first_idx_in_shelf0 = top_5_representatives_indices[0] if len(top_5_representatives_indices) > 0 else None
    first_artwork_id = df_shelf0.iloc[first_idx_in_shelf0].get("id") if first_idx_in_shelf0 is not None else None
    first_idx_in_all = top_representatives[0][0] if top_representatives else None
    
    # Verify aesthetic IDs are included
    if aesthetic_representative_id is not None:
        rep_ids_in_top5 = [df_shelf0.iloc[i].get("id") for i in top_5_representatives_indices]
        print(f"   Aesthetic representative ID {aesthetic_representative_id} {'found' if any(str(aid) == str(aesthetic_representative_id) or float(aid) == float(aesthetic_representative_id) for aid in rep_ids_in_top5) else 'NOT FOUND'} in top 5 representatives")
    
    if aesthetic_outlier_id is not None:
        outlier_ids_in_top5 = [df_shelf0.iloc[i].get("id") for i in top_5_outliers_indices]
        print(f"   Aesthetic outlier ID {aesthetic_outlier_id} {'found' if any(str(aid) == str(aesthetic_outlier_id) or float(aid) == float(aesthetic_outlier_id) for aid in outlier_ids_in_top5) else 'NOT FOUND'} in top 5 outliers")
    
    # Project centroid to 2D
    # Use the mean of 2D coordinates directly - this is the most accurate approach
    # because:
    # 1. The 2D coordinates already correctly represent the shelf items
    # 2. The mean of 2D coords will be in the center of the shelf cluster
    # 3. UMAP/t-SNE transform for new points (like the centroid) can be unreliable
    centroid_coord_2d = np.mean(shelf0_coords_2d, axis=0)
    print(f"   Centroid 2D position calculated as mean of shelf {target_shelf} coordinates")
    
    # Validate centroid coordinate
    if len(shelf0_coords_2d) > 0:
        print(f"   Centroid 2D position: ({centroid_coord_2d[0]:.2f}, {centroid_coord_2d[1]:.2f})")
        print(f"   Shelf {target_shelf} coords range: x=[{shelf0_coords_2d[:, 0].min():.2f}, {shelf0_coords_2d[:, 0].max():.2f}], y=[{shelf0_coords_2d[:, 1].min():.2f}, {shelf0_coords_2d[:, 1].max():.2f}]")
    else:
        print(f"   Warning: No shelf {target_shelf} coordinates found!")
        centroid_coord_2d = np.array([MAP_WIDTH // 2, MAP_HEIGHT // 2])  # Default to center
    
    print(f"   Representative artwork ID: {first_artwork_id}")
    if first_idx_in_shelf0 is not None:
        print(f"   Distance to centroid: {distances[first_idx_in_shelf0]:.4f}")
    
    # Generate frames
    print("\n5. Generating frames...")
    frames_dir = SCRIPT_DIR / "frames" / f"shelf{target_shelf}_both"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    
    # Step 1: Show all embeddings
    print(f"   Generating step 1: All embeddings...")
    for i in tqdm(range(FRAMES_PER_STEP), desc="Step 1 frames"):
        img = create_frame("all", all_coords_2d, all_artwork_ids, shelf0_mask, all_embeddings, 
                          target_shelf=target_shelf, top_representatives=top_items,
                          aesthetic_representative_id=aesthetic_representative_id,
                          white_background=white_background)
        img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG")
        frame_count += 1
    
    # Step 2: Show shelf items one by one slowly (NO LINES, just identification) with color easing
    print(f"   Generating step 2: Identifying shelf {target_shelf} items (slow, no lines)...")
    shelf0_indices_list = np.where(shelf0_mask)[0].tolist()
    num_shelf0 = len(shelf0_indices_list)
    
    # Show each artwork one by one, more slowly with color easing
    FRAMES_PER_ARTWORK_SLOW = 30  # More frames per artwork for slower pace
    EASE_IN_FRAMES = 15  # Frames for color easing animation
    for artwork_num in range(1, num_shelf0 + 1):
        # Get the artwork index being added
        current_artwork_idx_in_all = shelf0_indices_list[artwork_num - 1]
        
        # Generate frames for this artwork - NO lines, NO centroid, just showing the item
        for i in range(FRAMES_PER_ARTWORK_SLOW):
            # Calculate color ease progress for the currently appearing item
            if i < EASE_IN_FRAMES:
                color_ease = (i + 1) / EASE_IN_FRAMES
            else:
                color_ease = None  # Fully appeared, no easing needed
            
            img = create_frame("highlight_slow", all_coords_2d, all_artwork_ids, shelf0_mask,
                              white_background=white_background, 
                              all_embeddings=all_embeddings,
                              shelf0_coords=shelf0_coords_2d, 
                              shelf0_coords_progressive=shelf0_coords_2d,
                              num_shelf0_shown=artwork_num, 
                              df=df,
                              highlighted_artwork_idx=current_artwork_idx_in_all,
                              target_shelf=target_shelf,
                              top_representatives=top_items,
                              aesthetic_representative_id=aesthetic_representative_id,
                              color_ease_progress=color_ease)
            img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG")
            frame_count += 1
    
    # Hold final state for a bit - show all shelf items
    if num_shelf0 > 0:
        last_artwork_idx = shelf0_indices_list[-1]
        for i in range(FRAMES_PER_STEP // 2):
            img = create_frame("highlight_slow", all_coords_2d, all_artwork_ids, shelf0_mask,
                              white_background=white_background, 
                              all_embeddings=all_embeddings,
                              shelf0_coords=shelf0_coords_2d, 
                              shelf0_coords_progressive=shelf0_coords_2d,
                              num_shelf0_shown=num_shelf0,
                              df=df,
                              highlighted_artwork_idx=last_artwork_idx,
                              target_shelf=target_shelf,
                              top_representatives=top_items,
                              aesthetic_representative_id=aesthetic_representative_id)
            img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG")
            frame_count += 1
    
    # Step 3: Highlight shelf with centroid and distances (combined step) with easing
    print(f"   Generating step 3: Highlighting shelf {target_shelf} with centroid and distances (with easing)...")
    
    # Animate adding each artwork one by one with lines, centroid, and distances
    EASE_IN_FRAMES_STEP3 = 10  # Frames for easing in each item
    for artwork_num in range(1, num_shelf0 + 1):
        # Get the artwork index being added
        current_artwork_idx_in_all = shelf0_indices_list[artwork_num - 1]
        
        # Calculate dynamic centroid based on artworks added so far
        shelf0_embeddings_progressive = shelf0_embeddings[:artwork_num]
        centroid_progressive = np.mean(shelf0_embeddings_progressive, axis=0)
        # Don't normalize - use raw mean for centroid
        # Project to 2D (use average of shown coordinates)
        centroid_coord_2d_progressive = np.mean(shelf0_coords_2d[:artwork_num], axis=0)
        
        # Calculate distances for progressive set
        distances_progressive = np.linalg.norm(shelf0_embeddings_progressive - centroid_progressive, axis=1)
        
        # Generate frames for this addition - show text for the current artwork with centroid and distances
        for i in range(FRAMES_PER_ADDITION):
            # Calculate color ease progress for the currently appearing item
            if i < EASE_IN_FRAMES_STEP3:
                color_ease = (i + 1) / EASE_IN_FRAMES_STEP3
            else:
                color_ease = None  # Fully appeared
            
            img = create_frame("highlight", all_coords_2d, all_artwork_ids, shelf0_mask,
                              white_background=white_background, 
                              all_embeddings=all_embeddings,
                              shelf0_coords=shelf0_coords_2d, 
                              shelf0_coords_progressive=shelf0_coords_2d,
                              centroid_coord=centroid_coord_2d_progressive,
                              distances=distances_progressive,
                              num_shelf0_shown=artwork_num, 
                              df=df,
                              highlighted_artwork_idx=current_artwork_idx_in_all,
                              target_shelf=target_shelf,
                              top_representatives=top_items,
                              aesthetic_representative_id=aesthetic_representative_id,
                              color_ease_progress=color_ease)
            img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG")
            frame_count += 1
    
    # Hold final state for a bit - show all items with final centroid and distances
    if num_shelf0 > 0:
        last_artwork_idx = shelf0_indices_list[-1]
        for i in range(FRAMES_PER_STEP):
            img = create_frame("highlight", all_coords_2d, all_artwork_ids, shelf0_mask,
                              white_background=white_background, 
                              all_embeddings=all_embeddings,
                              shelf0_coords=shelf0_coords_2d, 
                              shelf0_coords_progressive=shelf0_coords_2d,
                              centroid_coord=centroid_coord_2d,
                              distances=distances,
                              num_shelf0_shown=num_shelf0,
                              df=df,
                              highlighted_artwork_idx=last_artwork_idx,
                              target_shelf=target_shelf,
                              top_representatives=top_items,
                              aesthetic_representative_id=aesthetic_representative_id)
            img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG")
            frame_count += 1
    
    # Step 4: Cycle through each shelf artwork showing calculations
    print(f"   Generating step 4: Cycling through shelf {target_shelf} artworks...")
    shelf0_indices_list = np.where(shelf0_mask)[0].tolist()
    # Sort by distance to centroid for better visualization
    # Map shelf 0 indices to their positions in the distances array
    shelf0_distances_sorted = []
    for idx_in_all in shelf0_indices_list:
        idx_in_shelf0 = shelf0_indices_list.index(idx_in_all)
        if idx_in_shelf0 < len(distances):
            shelf0_distances_sorted.append((idx_in_all, distances[idx_in_shelf0]))
    
    # Sort by distance (closest first)
    shelf0_distances_sorted = sorted(shelf0_distances_sorted, key=lambda x: x[1])
    
    for artwork_idx, dist in tqdm(shelf0_distances_sorted, desc="Artworks"):
        for i in range(FRAMES_PER_ARTWORK):
            img = create_frame("representative", all_coords_2d, all_artwork_ids, shelf0_mask,
                              white_background=white_background,
                              all_embeddings=all_embeddings,
                              shelf0_coords=shelf0_coords_2d,
                              centroid_coord=centroid_coord_2d,
                              distances=distances,
                              representative_idx=first_idx_in_all,
                              df=df,
                              highlighted_artwork_idx=artwork_idx,
                              target_shelf=target_shelf,
                              top_representatives=top_items,
                              aesthetic_representative_id=aesthetic_representative_id)
            img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG")
            frame_count += 1
    
    # Step 5: Draw lines spreading out from centroid (frame by frame)
    print("   Generating step 5: Drawing lines from centroid...")
    shelf0_indices_list = np.where(shelf0_mask)[0].tolist()
    
    # Sort artworks by distance to centroid (closest first)
    shelf0_distances_with_indices = []
    for idx_in_all in shelf0_indices_list:
        idx_in_shelf0 = shelf0_indices_list.index(idx_in_all)
        if idx_in_shelf0 < len(distances):
            shelf0_distances_with_indices.append((idx_in_all, distances[idx_in_shelf0], idx_in_shelf0))
    
    shelf0_distances_sorted = sorted(shelf0_distances_with_indices, key=lambda x: x[1])
    
    # Animate drawing lines one by one
    FRAMES_PER_LINE = 15  # Frames to draw each line
    PAUSE_AFTER_CLOSEST = 60  # Frames to pause after drawing line to closest
    
    lines_drawn = []
    for line_idx, (artwork_idx, distance, shelf0_idx) in enumerate(shelf0_distances_sorted):
        # Get coordinates for this artwork
        artwork_coord = shelf0_coords_2d[shelf0_idx]
        start_coord = centroid_coord_2d
        
        # Animate drawing this line
        for frame_in_line in range(FRAMES_PER_LINE):
            progress = (frame_in_line + 1) / FRAMES_PER_LINE
            
            # Build list of lines to draw (all previous + current in progress)
            lines_to_draw = []
            for prev_idx, (prev_artwork_idx, _, prev_shelf0_idx) in enumerate(shelf0_distances_sorted[:line_idx]):
                prev_coord = shelf0_coords_2d[prev_shelf0_idx]
                is_prev_closest = (prev_idx == 0)
                lines_to_draw.append((start_coord, prev_coord, 1.0, is_prev_closest))  # Fully drawn
            
            # Current line (in progress) - mark as closest if it's the first one
            is_current_closest = (line_idx == 0)
            lines_to_draw.append((start_coord, artwork_coord, progress, is_current_closest))
            
            # Determine which artwork to highlight on right side
            # If this is the closest (first in sorted list) and line is complete, show it
            highlight_idx = None
            if line_idx == 0 and progress >= 1.0:
                highlight_idx = artwork_idx
            
            img = create_frame("representative", all_coords_2d, all_artwork_ids, shelf0_mask,
                              white_background=white_background,
                              all_embeddings=all_embeddings,
                              shelf0_coords=shelf0_coords_2d,
                              centroid_coord=centroid_coord_2d,
                              distances=distances,
                              representative_idx=first_idx_in_all,
                              df=df,
                              highlighted_artwork_idx=highlight_idx or artwork_idx,
                              lines_to_draw=lines_to_draw,
                              target_shelf=target_shelf,
                              top_representatives=top_items,
                              aesthetic_representative_id=aesthetic_representative_id)
            img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG")
            frame_count += 1
        
        # If this is the closest (representative), pause and show it on right side
        if line_idx == 0:
            for pause_frame in range(PAUSE_AFTER_CLOSEST):
                # All lines drawn so far (including the closest, fully drawn)
                lines_to_draw = []
                for prev_idx, (prev_artwork_idx, _, prev_shelf0_idx) in enumerate(shelf0_distances_sorted[:line_idx + 1]):
                    prev_coord = shelf0_coords_2d[prev_shelf0_idx]
                    is_prev_closest = (prev_idx == 0)
                    lines_to_draw.append((start_coord, prev_coord, 1.0, is_prev_closest))
                
                img = create_frame("representative", all_coords_2d, all_artwork_ids, shelf0_mask,
                              white_background=white_background,
                                  all_embeddings=all_embeddings,
                                  shelf0_coords=shelf0_coords_2d,
                                  centroid_coord=centroid_coord_2d,
                                  distances=distances,
                                  representative_idx=first_idx_in_all,
                                  df=df,
                                  highlighted_artwork_idx=artwork_idx,
                                  lines_to_draw=lines_to_draw,
                                  target_shelf=target_shelf,
                                  top_representatives=top_items,
                                  aesthetic_representative_id=aesthetic_representative_id)
                img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG")
                frame_count += 1
    
    # Step 6: Show top 10 (5 representatives + 5 outliers) appearing one by one with simultaneous line drawing
    print(f"   Generating step 6: Top 10 (5 Representatives + 5 Outliers) appearing one by one...")
    
    # Constants for animation
    FRAMES_PER_TOP10_ITEM = 20  # Frames per item
    EASE_IN_FRAMES_TOP10 = 12  # Frames for easing in each item
    
    # First show all 5 representatives one by one
    for rep_num in range(1, 6):  # 1 to 5
        for frame_in_item in range(FRAMES_PER_TOP10_ITEM):
            # Calculate easing progress for currently appearing item
            if frame_in_item < EASE_IN_FRAMES_TOP10:
                ease_progress = (frame_in_item + 1) / EASE_IN_FRAMES_TOP10
            else:
                ease_progress = None  # Fully appeared
            
            img = create_frame("top10", all_coords_2d, all_artwork_ids, shelf0_mask,
                              white_background=white_background,
                              all_embeddings=all_embeddings,
                              shelf0_coords=shelf0_coords_2d,
                              centroid_coord=centroid_coord_2d,
                              distances=distances,
                              df=df,
                              target_shelf=target_shelf,
                              top_representatives=top_representatives,
                              top_outliers=top_outliers,
                              aesthetic_representative_id=aesthetic_representative_id,
                              aesthetic_outlier_id=aesthetic_outlier_id,
                              top10_reps_shown=rep_num,
                              top10_outliers_shown=0,  # Not showing outliers yet
                              top10_item_ease_progress=ease_progress)
            img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG")
            frame_count += 1
    
    # Hold all representatives shown for a moment
    for i in range(FRAMES_PER_STEP // 2):
        img = create_frame("top10", all_coords_2d, all_artwork_ids, shelf0_mask,
                          white_background=white_background,
                          all_embeddings=all_embeddings,
                          shelf0_coords=shelf0_coords_2d,
                          centroid_coord=centroid_coord_2d,
                          distances=distances,
                          df=df,
                          target_shelf=target_shelf,
                          top_representatives=top_representatives,
                          top_outliers=top_outliers,
                          aesthetic_representative_id=aesthetic_representative_id,
                          aesthetic_outlier_id=aesthetic_outlier_id,
                          top10_reps_shown=5,
                          top10_outliers_shown=0)
        img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG")
        frame_count += 1
    
    # Then show all 5 outliers one by one
    for outlier_num in range(1, 6):  # 1 to 5
        for frame_in_item in range(FRAMES_PER_TOP10_ITEM):
            # Calculate easing progress for currently appearing item
            if frame_in_item < EASE_IN_FRAMES_TOP10:
                ease_progress = (frame_in_item + 1) / EASE_IN_FRAMES_TOP10
            else:
                ease_progress = None  # Fully appeared
            
            img = create_frame("top10", all_coords_2d, all_artwork_ids, shelf0_mask,
                              white_background=white_background,
                              all_embeddings=all_embeddings,
                              shelf0_coords=shelf0_coords_2d,
                              centroid_coord=centroid_coord_2d,
                              distances=distances,
                              df=df,
                              target_shelf=target_shelf,
                              top_representatives=top_representatives,
                              top_outliers=top_outliers,
                              aesthetic_representative_id=aesthetic_representative_id,
                              aesthetic_outlier_id=aesthetic_outlier_id,
                              top10_reps_shown=5,  # All reps shown
                              top10_outliers_shown=outlier_num,
                              top10_item_ease_progress=ease_progress)
            img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG")
            frame_count += 1
    
    # Hold final state with all items shown
    for i in range(FRAMES_PER_STEP):
        img = create_frame("top10", all_coords_2d, all_artwork_ids, shelf0_mask,
                          white_background=white_background,
                          all_embeddings=all_embeddings,
                          shelf0_coords=shelf0_coords_2d,
                          centroid_coord=centroid_coord_2d,
                          distances=distances,
                          df=df,
                          target_shelf=target_shelf,
                          top_representatives=top_representatives,
                          top_outliers=top_outliers,
                          aesthetic_representative_id=aesthetic_representative_id,
                          aesthetic_outlier_id=aesthetic_outlier_id,
                          top10_reps_shown=5,
                          top10_outliers_shown=5)
        img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG")
        frame_count += 1
    
    # Step 7: Show both selected items side by side with details on left screen, rank table on right
    print(f"   Generating step 7: Selected items side by side with rank table...")
    for i in tqdm(range(FRAMES_PER_STEP * 2), desc="Step 7 frames"):  # Hold longer
        img = create_frame("side_by_side", all_coords_2d, all_artwork_ids, shelf0_mask,
                          all_embeddings=all_embeddings,
                          shelf0_coords=shelf0_coords_2d,
                          centroid_coord=centroid_coord_2d,
                          distances=distances,
                          df=df,
                          target_shelf=target_shelf,
                          top_representatives=top_representatives,
                          top_outliers=top_outliers,
                          aesthetic_representative_id=aesthetic_representative_id,
                          aesthetic_outlier_id=aesthetic_outlier_id,
                          white_background=white_background)
        img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG")
        frame_count += 1
    
    print(f"   Generated {frame_count} frames")
    
    # Create video with ffmpeg
    print("\n6. Creating video with ffmpeg...")
    output_video = frames_dir.parent / f"shelf{target_shelf}_both.mp4"
    
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        str(output_video)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"   Video created: {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"   Error creating video: {e}")
        print(f"   stderr: {e.stderr.decode() if e.stderr else 'N/A'}")
    except FileNotFoundError:
        print("   ffmpeg not found. Please install ffmpeg to create video.")
        print(f"   Frames are saved in: {frames_dir}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize the process of finding representatives for a given shelf.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_shelf0_representative.py --shelf 0
  python visualize_shelf0_representative.py --shelf 5
  python visualize_shelf0_representative.py -s 9
        """
    )
    parser.add_argument(
        "--shelf", "-s",
        type=str,
        default="0",
        help="The shelf number to visualize (default: 0)"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="both",
        choices=["representative", "outlier", "both"],
        help="Visualization mode: 'representative', 'outlier', or 'both' (default: both)"
    )
    parser.add_argument(
        "--white-background", "-w",
        action="store_true",
        help="Use white background with inverted colors (default: black background)"
    )
    
    args = parser.parse_args()
    main(target_shelf=args.shelf, mode=args.mode, white_background=args.white_background)

