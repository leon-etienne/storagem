#!/usr/bin/env python3
"""
Visualize the process of finding representatives and outliers for a given Regal.
Creates step-by-step frames showing:
1. All embeddings map
2. Identify Regal items (one by one with easing)
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
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from sklearn.manifold import TSNE
from tqdm import tqdm
import warnings

# Suppress PIL DecompressionBombWarning for large images
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb check

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available, will use t-SNE instead")

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("imageio not available, virtual render mode will not work. Install with: pip install imageio imageio-ffmpeg")

# Get script directory and project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Constants (relative to project root)
FONT_DIR = PROJECT_ROOT / "video_making/font"
FONT_REGULAR = FONT_DIR / "Neue Haas Unica W1G Regular.ttf"
FONT_LIGHT = FONT_DIR / "Neue Haas Unica W1G Light.ttf"
FONT_MEDIUM = FONT_DIR / "Neue Haas Unica W1G Medium.ttf"
FONT_THIN = FONT_DIR / "Neue Haas Unica W1G Thin.ttf"
FONT_MONO = FONT_DIR / "UbuntuMono-Regular.ttf"  # Monofont for normal text
FALLBACK_IMAGE = PROJECT_ROOT / "video_making/font/no_image.png"
MAP_DIR = FONT_DIR / "map"  # Floor plan images directory (for black background)
MAP_DIR_WHITE = FONT_DIR / "map_white"  # Floor plan images directory (for white background)
CSV_PATH = PROJECT_ROOT / "artworks_with_thumbnails_final.csv"
EMBEDDINGS_CACHE = PROJECT_ROOT / "embeddings_cache/all_embeddings.pkl"
BASE_DIR = PROJECT_ROOT / "production-export-2025-11-14t12-58-07-436z"

# Color constants (following aesthetic guidelines)
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_LIME = (0, 255, 0)  # Pure green
COLOR_GRAY_LIGHT = (200, 200, 200)  # Light gray for text on black
COLOR_GRAY_DARK = (150, 150, 150)  # Medium gray for text on black
COLOR_POINT_GRAY = (90, 90, 90)  # Gray for regular points (more gray instead of pure white)
COLOR_WHITE_LOW_OPACITY = (180, 180, 180)  # Whiter gray for dots (whiter than gray lines)
COLOR_CIRCLE_GRAY = (150, 150, 150)  # Lighter gray for selection circles on black background (better visibility)
COLOR_LIME_LOW_OPACITY = (0, 150, 0)  # Simulated lower opacity for non-highlighted Regal points
COLOR_GRAY_CONNECTION = (100, 100, 100)  # Gray for connection lines (darker than dots)

# White background mode colors
COLOR_BG_WHITE = (255, 255, 255)
COLOR_TEXT_WHITE_BG = (0, 0, 0)  # Black text on white background
COLOR_TEXT_GRAY_WHITE_BG = (100, 100, 100)  # Gray text on white background
COLOR_POINT_WHITE_BG = (120, 120, 120)  # More grey (darker) points on white background
COLOR_LIME_LOW_OPACITY_WHITE_BG = (0, 200, 0)  # Slightly darker green for white bg
COLOR_GRAY_CONNECTION_WHITE_BG = (180, 180, 180)  # Light gray for connection lines on white bg
COLOR_CIRCLE_GRAY_WHITE_BG = (80, 80, 80)  # Darker gray for selection circles on white background

# Canvas dimensions (default: Full HD)
CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080
MAP_WIDTH = 1200  # Left side for embedding map
PANEL_WIDTH = 720  # Right side for artwork info
MAP_HEIGHT = CANVAS_HEIGHT
PANEL_HEIGHT = CANVAS_HEIGHT

# 4K dimensions
CANVAS_WIDTH_4K = 3840
CANVAS_HEIGHT_4K = 2160
MAP_WIDTH_4K = 2400  # Left side for embedding map (2x)
PANEL_WIDTH_4K = 1440  # Right side for artwork info (2x)

# Anti-aliasing: Use supersampling (render at higher resolution, then downscale)
SUPERSAMPLE_FACTOR = 2  # Render at 2x resolution for better anti-aliasing
SUPERSAMPLED_WIDTH = CANVAS_WIDTH * SUPERSAMPLE_FACTOR
SUPERSAMPLED_HEIGHT = CANVAS_HEIGHT * SUPERSAMPLE_FACTOR

# Visualization parameters
POINT_SIZE = 4  # Size for regular points
POINT_SIZE_LARGE = 8  # Size for Regal 0 points
POINT_SIZE_REPRESENTATIVE = 40  # Size for representative points (with images)
LINE_WIDTH = 1
FONT_SIZE_TITLE = 48
FONT_SIZE_LABEL = 28  # Slightly bigger subtitle
FONT_SIZE_INFO = 18
FONT_SIZE_SMALL = 16  # Increased from 14 for better readability in right panel
FONT_SIZE_MAP_TEXT = 12  # Reduced from 18 - smaller text for map labels on left side
FONT_SIZE_TABLE = 18  # Increased from 16 - larger font for top10 and side-by-side tables
FIXED_IMAGE_HEIGHT = 150  # Fixed height for all images in panel (standardized across all stages)
FIXED_IMAGE_HEIGHT_LARGE = 280  # Larger height for non-ruler steps (increased from 250)

# Frame generation
FRAMES_PER_STEP = 60  # Hold each step for 60 frames (1 second at 60fps)
FRAMES_PER_ARTWORK = 20  # Frames per artwork when cycling through
FRAMES_PER_ADDITION = 20  # Frames for each artwork addition animation (slower)
FPS = 60  # Higher FPS for smoother animation
EXTRA_HOLD_FRAMES = 120  # Additional 2 seconds (120 frames at 60fps) to hold at end of each stage

# Centralized subtitle labels for easy management
# This dictionary maps step names to their default subtitle text
# Use get_subtitle_label() function to retrieve labels with optional formatting
STEP_LABELS = {
    "all": "All Embeddings",
    "highlight_slow": "Identify Regal items",
    "highlight": "Highlighting Regal {shelf}",  # {shelf} will be replaced with target_shelf
    "centroid": "Highlighting Regal {shelf}",
    "distances": "Highlighting Regal {shelf}",
    "representative": "Calculating Distance",  # Can be overridden dynamically
    "ruler": "Measuring Distances",
    "representatives": "Top 10 Representatives",
    "zoom": "Selected Representative",
    "outlier": "Object Selections",
    "top10": "Top 10 Representatives and Outliers",  # Default, can be customized
}


def get_subtitle_label(step: str, target_shelf: Optional[str] = None, **kwargs) -> str:
    """
    Get subtitle label for a given step with optional formatting.
    
    Args:
        step: The step name (e.g., "all", "highlight", "representative")
        target_shelf: Optional shelf number to format into label (replaces {shelf} placeholder)
        **kwargs: Additional formatting parameters (e.g., search_mode, current_distance)
    
    Returns:
        Formatted subtitle string
    
    Examples:
        get_subtitle_label("all")  # Returns "All Embeddings"
        get_subtitle_label("highlight", target_shelf="5")  # Returns "Highlighting Regal 5"
        get_subtitle_label("representative", search_mode="outlier")  # Returns "Calculating Distance"
    """
    # Get base label from dictionary
    base_label = STEP_LABELS.get(step, step)
    
    # Replace {shelf} placeholder if target_shelf is provided
    if target_shelf is not None and "{shelf}" in base_label:
        base_label = base_label.replace("{shelf}", str(target_shelf))
    
    # Handle special cases with dynamic content
    # For representative step, always use "Calculating Distance" regardless of search_mode
    if step == "representative":
        base_label = "Calculating Distance"
    
    # Handle top10 step with custom label
    if step == "top10" and "custom_label" in kwargs:
        base_label = kwargs["custom_label"]
    
    return base_label


def get_final_subtitle_text(
    step: str,
    target_shelf: Optional[str] = None,
    search_mode: Optional[str] = None,
    current_distance: Optional[float] = None,
    highlighted_artwork_idx: Optional[int] = None,
    top_representatives: Optional[List[Tuple[int, float]]] = None,
    top_outliers: Optional[List[Tuple[int, float]]] = None,
    top10_reps_shown: Optional[int] = None,
    top10_outliers_shown: Optional[int] = None,
    centroid_coord: Optional[Tuple[float, float]] = None,
) -> str:
    """
    Get the final subtitle text with all dynamic modifications applied.
    This is the actual text that will be displayed, including distances, centroid info, etc.
    Use this function to track the actual displayed subtitle for easing transitions.
    
    Args:
        step: The step name
        target_shelf: Optional shelf number
        search_mode: Optional search mode ("representative" or "outlier")
        current_distance: Optional current distance value
        highlighted_artwork_idx: Optional highlighted artwork index
        top_representatives: Optional list of top representatives
        top_outliers: Optional list of top outliers
        top10_reps_shown: Optional number of top10 reps shown
        top10_outliers_shown: Optional number of top10 outliers shown
        centroid_coord: Optional centroid coordinates
    
    Returns:
        Final subtitle string with all modifications applied
    """
    # Get base label
    step_text = get_subtitle_label(step, target_shelf=target_shelf, search_mode=search_mode)
    
    # Add distance display for top10 step
    if step == "top10":
        current_display_distance = None
        if top10_reps_shown is not None and top10_reps_shown > 0 and top_representatives is not None:
            if top10_reps_shown <= len(top_representatives):
                current_display_distance = top_representatives[top10_reps_shown - 1][1]
        elif top10_outliers_shown is not None and top10_outliers_shown > 0 and top_outliers is not None:
            if top10_outliers_shown <= len(top_outliers):
                current_display_distance = top_outliers[top10_outliers_shown - 1][1]
        
        if current_display_distance is not None:
            step_text += f" | Distance: {current_display_distance:.4f}"
    
    # Add search mode and distance to subtitle for representative step
    if step == "representative":
        if search_mode is not None:
            # When actively calculating, show "Calculating Distance"
            step_text = "Calculating Distance"
            if current_distance is not None:
                step_text += f" | Distance: {current_distance:.4f}"
        else:
            # When an object is selected/found, show "Object Selections"
            step_text = "Object Selections"
    
    # Add centroid position to subtitle in highlight steps
    if step in ["highlight", "centroid", "distances"] and centroid_coord is not None:
        try:
            if isinstance(centroid_coord, np.ndarray):
                cx_val, cy_val = float(centroid_coord[0]), float(centroid_coord[1])
            elif isinstance(centroid_coord, (tuple, list)) and len(centroid_coord) >= 2:
                cx_val, cy_val = float(centroid_coord[0]), float(centroid_coord[1])
            else:
                cx_val, cy_val = None, None
            if cx_val is not None and cy_val is not None:
                step_text += f" | Centroid: ({cx_val:.1f}, {cy_val:.1f})"
        except (ValueError, TypeError, IndexError, AttributeError):
            pass
    
    return step_text


def ascii_text_ease(original_text: str, target_text: str, progress: float) -> str:
    """
    Interpolate between two ASCII strings character by character.
    
    Args:
        original_text: The starting text string
        target_text: The ending text string
        progress: Interpolation progress (0.0 = original_text, 1.0 = target_text)
    
    Returns:
        The interpolated text string
    """
    # Clamp progress
    progress = max(0.0, min(1.0, progress))
    
    # Handle empty strings
    if not original_text:
        return target_text if progress >= 1.0 else ""
    if not target_text:
        return original_text if progress <= 0.0 else ""
    
    # Use target length (simple approach)
    max_len = max(len(original_text), len(target_text))
    original_padded = original_text.ljust(max_len)
    target_padded = target_text.ljust(max_len)
    
    # Interpolate character by character
    result_chars = []
    for i in range(max_len):
        orig_ascii = ord(original_padded[i])
        target_ascii = ord(target_padded[i])
        interpolated_ascii = int(orig_ascii + (target_ascii - orig_ascii) * progress)
        interpolated_ascii = max(32, min(126, interpolated_ascii))  # Clamp to printable ASCII
        result_chars.append(chr(interpolated_ascii))
    
    result = ''.join(result_chars)
    
    # Trim to target length if needed
    if progress >= 1.0:
        result = result[:len(target_text)]
    elif progress <= 0.0:
        result = result[:len(original_text)]
    else:
        # Interpolate length
        current_len = int(len(original_text) + (len(target_text) - len(original_text)) * progress)
        result = result[:current_len]
    
    return result


def build_artwork_lookup(df: pd.DataFrame) -> Dict:
    """Build a fast lookup dictionary for artwork IDs to row indices.
    
    Creates mappings for both string and float representations of IDs.
    Returns dict mapping: id (str/float) -> row_index (int)
    """
    lookup = {}
    for idx, row in df.iterrows():
        artwork_id = row.get("id")
        if pd.isna(artwork_id):
            continue
        
        # Add string representation
        id_str = str(artwork_id).strip()
        lookup[id_str] = idx
        
        # Add normalized string (remove .0)
        id_str_normalized = id_str.replace('.0', '').strip()
        if id_str_normalized != id_str:
            lookup[id_str_normalized] = idx
        
        # Add float representation if possible
        try:
            id_float = float(artwork_id)
            lookup[id_float] = idx
        except (ValueError, TypeError):
            pass
    
    return lookup


def load_embeddings() -> Dict:
    """Load embeddings from cache."""
    if not EMBEDDINGS_CACHE.exists():
        raise FileNotFoundError(f"Embeddings cache not found: {EMBEDDINGS_CACHE}")
    
    with open(EMBEDDINGS_CACHE, "rb") as f:
        cache_data = pickle.load(f)
    
    embeddings_dict = cache_data.get("embeddings", {})
    print(f"Loaded {len(embeddings_dict)} embeddings from cache")
    return embeddings_dict


# Image cache for performance
_image_cache: Dict[str, Optional[Image.Image]] = {}

# Floor plan cache for performance (key: (shelf, scale))
_floor_plan_cache: Dict[Tuple[str, float], Optional[Image.Image]] = {}

def load_image(image_path: str) -> Optional[Image.Image]:
    """Load image from path, with fallback and caching."""
    # Normalize path for cache key
    cache_key = str(image_path) if image_path else "N/A"
    
    # Check cache first
    if cache_key in _image_cache:
        cached_img = _image_cache[cache_key]
        # Return a copy since images might be modified (resized, etc.)
        if cached_img is not None:
            return cached_img.copy()
        return None
    
    result = None
    
    if pd.isna(image_path) or not image_path or image_path == "N/A":
        if FALLBACK_IMAGE.exists():
            # Keep transparency for fallback image (it's a transparent PNG)
            fallback_img = Image.open(FALLBACK_IMAGE)
            if fallback_img.mode == "RGBA":
                result = fallback_img  # Keep RGBA for transparent images
            else:
                result = fallback_img.convert("RGB")
        else:
            result = None
    else:
        # Try direct path (absolute or relative to project root)
        full_path = Path(image_path)
        if not full_path.is_absolute():
            full_path = PROJECT_ROOT / image_path
        
        if full_path.exists() and full_path.is_file():
            try:
                loaded_img = Image.open(full_path)
                # Keep transparency if it's RGBA, otherwise convert to RGB
                if loaded_img.mode == "RGBA":
                    result = loaded_img
                else:
                    result = loaded_img.convert("RGB")
            except Exception:
                pass
        
        # Try extracting filename
        if result is None:
            if "images/" in image_path:
                filename = image_path.split("images/")[-1]
            else:
                filename = image_path
            
            images_dir = BASE_DIR / "images"
            if images_dir.exists():
                full_path = images_dir / filename
                if full_path.exists():
                    try:
                        loaded_img = Image.open(full_path)
                        # Keep transparency if it's RGBA, otherwise convert to RGB
                        if loaded_img.mode == "RGBA":
                            result = loaded_img
                        else:
                            result = loaded_img.convert("RGB")
                    except Exception:
                        pass
        
        # Fallback
        if result is None and FALLBACK_IMAGE.exists():
            # Keep transparency for fallback image (it's a transparent PNG)
            fallback_img = Image.open(FALLBACK_IMAGE)
            if fallback_img.mode == "RGBA":
                result = fallback_img  # Keep RGBA for transparent images
            else:
                result = fallback_img.convert("RGB")
    
    # Cache the result
    _image_cache[cache_key] = result
    # Return a copy since images might be modified (resized, etc.)
    if result is not None:
        return result.copy()
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
            n_neighbors=20,  # Increased for more global structure
            min_dist=0.1,  # Lower value to pack points closer together
            spread=24,  # Lower spread for tighter clusters
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
            "circle_gray": COLOR_CIRCLE_GRAY_WHITE_BG,  # Use darker gray for white background
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
            "circle_gray": COLOR_CIRCLE_GRAY,
        }


# Font cache for performance
_font_cache: Dict[Tuple[int, str], ImageFont.FreeTypeFont] = {}

def get_font(size: int, weight: str = "regular", mono: bool = False) -> ImageFont.FreeTypeFont:
    """Load font with fallback and caching.
    
    Args:
        size: Font size in points
        weight: Font weight - "thin", "light", "regular", or "medium"
        mono: If True, use monofont (UbuntuMono) instead of Neue Haas Unica
    """
    # Check cache first
    cache_key = (size, weight.lower(), mono)
    if cache_key in _font_cache:
        return _font_cache[cache_key]
    
    # Use monofont if requested
    if mono:
        font = None
        try:
            if FONT_MONO.exists():
                font = ImageFont.truetype(str(FONT_MONO), size)
        except Exception:
            pass
        if font is None:
            font = ImageFont.load_default()
        _font_cache[cache_key] = font
        return font
    
    font_paths = {
        "thin": FONT_THIN,
        "light": FONT_LIGHT,
        "regular": FONT_REGULAR,
        "medium": FONT_MEDIUM,
    }
    
    # Try the requested weight first
    font_path = font_paths.get(weight.lower(), FONT_REGULAR)
    font = None
    try:
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), size)
    except Exception:
        pass
    
    # Fallback to regular if requested weight not available
    if font is None and weight.lower() != "regular":
        try:
            if FONT_REGULAR.exists():
                font = ImageFont.truetype(str(FONT_REGULAR), size)
        except Exception:
            pass
    
    # Final fallback to system default
    if font is None:
        font = ImageFont.load_default()
    
    # Cache the font
    _font_cache[cache_key] = font
    return font


def draw_artwork_title(draw, title: str, panel_x: int, y_pos: int, max_width: int,
                       colors: Dict, font_title_large: ImageFont.FreeTypeFont, scale: float = 1.0,
                       previous_title: Optional[str] = None, ease_progress: Optional[float] = None) -> int:
    """Draw artwork title (large, no label).
    
    Args:
        previous_title: Previous title for ASCII easing
        ease_progress: Progress for ASCII easing (0.0 to 1.0)
    
    Returns:
        Updated y_pos after drawing title
    """
    # Apply ASCII easing if provided
    display_title = title
    if previous_title is not None and ease_progress is not None:
        display_title = ascii_text_ease(previous_title, title, ease_progress)
    
    title_words = display_title.split()
    title_lines = []
    current_line = ""
    for word in title_words:
        test_line = current_line + (" " if current_line else "") + word
        bbox = draw.textbbox((0, 0), test_line, font=font_title_large)
        if bbox[2] - bbox[0] <= max_width:
            current_line = test_line
        else:
            if current_line:
                title_lines.append(current_line)
            current_line = word
    if current_line:
        title_lines.append(current_line)
    
    for line in title_lines[:2]:
        draw.text((panel_x, y_pos), line, fill=colors["text"], font=font_title_large)
        y_pos += int(22 * scale)
    y_pos += int(10 * scale)
    return y_pos


def draw_artist_year(draw, artist: str, year: str, panel_x: int, y_pos: int, max_width: int,
                     colors: Dict, font_side: ImageFont.FreeTypeFont, scale: float = 1.0) -> int:
    """Draw artist and year (no labels).
    
    Returns:
        Updated y_pos after drawing artist and year
    """
    # Artist (no label)
    artist_bbox = draw.textbbox((0, 0), artist, font=font_side)
    if artist_bbox[2] - artist_bbox[0] > max_width:
        while artist and draw.textbbox((0, 0), artist + "...", font=font_side)[2] > max_width:
            artist = artist[:-1]
        artist = artist + "..." if artist else "..."
    draw.text((panel_x, y_pos), artist, fill=colors["text"], font=font_side)
    y_pos += int(22 * scale)
    
    # Year (no label)
    draw.text((panel_x, y_pos), year, fill=colors["text"], font=font_side)
    y_pos += int(30 * scale)
    return y_pos


def safe_int_convert(value: Any) -> Optional[int]:
    """Safely convert various types to int, handling numpy types and pandas objects."""
    if value is None:
        return None
    try:
        if isinstance(value, (pd.DataFrame, pd.Series)):
            return None
        if hasattr(value, 'item'):
            return int(value.item())
        if isinstance(value, (np.integer, np.int64, np.int32)):
            return int(value)
        if hasattr(value, '__int__'):
            return int(value)
        return int(float(str(value)))
    except (ValueError, TypeError, AttributeError):
        return None


def extract_centroid_coords(centroid_coord: Optional[Any], scale: float = 1.0, 
                           map_width: float = None, map_height: float = None) -> Tuple[Optional[float], Optional[float]]:
    """Extract and validate centroid coordinates."""
    if centroid_coord is None:
        return None, None
    try:
        if isinstance(centroid_coord, np.ndarray):
            cx, cy = float(centroid_coord[0]), float(centroid_coord[1])
        elif isinstance(centroid_coord, (tuple, list)) and len(centroid_coord) >= 2:
            cx, cy = float(centroid_coord[0]), float(centroid_coord[1])
        else:
            return None, None
        
        if cx is not None and cy is not None and map_width is not None and map_height is not None:
            cx = max(50 * scale, min(cx, map_width - 50 * scale))
            cy = max(100 * scale, min(cy, map_height - 50 * scale))
        return cx, cy
    except (ValueError, TypeError, IndexError, AttributeError):
        return None, None


def find_artwork_by_id(artwork_id: Any, df: pd.DataFrame, artwork_lookup: Optional[Dict] = None) -> Optional[pd.Series]:
    """Find artwork row in DataFrame by ID, using lookup if available."""
    if df is None or artwork_id is None:
        return None
    
    # Use pre-computed lookup if available
    if artwork_lookup is not None:
        artwork_id_str = str(artwork_id).strip()
        artwork_id_normalized = artwork_id_str.replace('.0', '').strip()
        artwork_id_float = None
        try:
            artwork_id_float = float(artwork_id)
        except (ValueError, TypeError):
            pass
        
        lookup_idx = None
        if artwork_id_str in artwork_lookup:
            lookup_idx = artwork_lookup[artwork_id_str]
        elif artwork_id_normalized in artwork_lookup:
            lookup_idx = artwork_lookup[artwork_id_normalized]
        elif artwork_id_float is not None and artwork_id_float in artwork_lookup:
            lookup_idx = artwork_lookup[artwork_id_float]
        
        if lookup_idx is not None:
            return df.iloc[lookup_idx]
    
    # Fallback to original lookup method
    artwork_row = pd.DataFrame()
    try:
        artwork_row = df[df["id"].astype(float) == float(artwork_id)]
    except (ValueError, TypeError):
        pass
    
    if artwork_row.empty:
        try:
            artwork_row = df[df["id"].astype(str).str.strip() == str(artwork_id).strip()]
        except (ValueError, TypeError):
            pass
    
    if artwork_row.empty:
        try:
            artwork_row = df[df["id"].astype(str).str.replace('.0', '', regex=False).str.strip() == str(artwork_id).strip()]
        except (ValueError, TypeError):
            pass
    
    return artwork_row.iloc[0] if not artwork_row.empty else None


def matches_artwork_id(artwork_id: Any, target_id: Any) -> bool:
    """Check if artwork_id matches target_id using multiple comparison methods."""
    if artwork_id is None or target_id is None:
        return False
    try:
        if float(artwork_id) == float(target_id):
            return True
        if str(artwork_id) == str(target_id):
            return True
        if str(artwork_id).replace('.0', '') == str(target_id).replace('.0', ''):
            return True
    except (ValueError, TypeError):
        if str(artwork_id) == str(target_id):
            return True
    return False


def blend_color_with_opacity(color: Tuple[int, int, int], background: Tuple[int, int, int], 
                             opacity: float) -> Tuple[int, int, int]:
    """Blend a color with background based on opacity (0.0 = background, 1.0 = color)."""
    opacity = max(0.0, min(1.0, opacity))
    r = int(color[0] * opacity + background[0] * (1 - opacity))
    g = int(color[1] * opacity + background[1] * (1 - opacity))
    b = int(color[2] * opacity + background[2] * (1 - opacity))
    return (r, g, b)


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.ImageDraw) -> List[str]:
    """Wrap text to fit within max_width, returning list of lines."""
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines


def draw_line_with_opacity(img: Image.Image, draw: ImageDraw.ImageDraw, 
                          start: Tuple[float, float], end: Tuple[float, float],
                          color: Tuple[int, int, int], width: int, opacity: float = 1.0):
    """Draw a line with transparency using alpha channel.
    
    Args:
        img: The RGBA image to draw on
        draw: The ImageDraw object
        start: Start coordinates (x, y)
        end: End coordinates (x, y)
        color: RGB color tuple
        width: Line width
        opacity: Opacity value (0.0 to 1.0, where 1.0 is fully opaque)
    """
    if opacity <= 0.0:
        return  # Don't draw if fully transparent
    if opacity >= 1.0:
        # Fully opaque, draw directly
        draw.line([start, end], fill=color, width=width)
        return
    
    # For transparency, create a temporary RGBA image for the line
    # Calculate bounding box for the line with padding for width
    x1, y1 = start
    x2, y2 = end
    padding = width + 2
    min_x = int(min(x1, x2) - padding)
    min_y = int(min(y1, y2) - padding)
    max_x = int(max(x1, x2) + padding)
    max_y = int(max(y1, y2) + padding)
    
    # Ensure coordinates are within image bounds
    img_width, img_height = img.size
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(img_width, max_x)
    max_y = min(img_height, max_y)
    
    if max_x <= min_x or max_y <= min_y:
        return
    
    # Create temporary RGBA image for the line
    temp_img = Image.new("RGBA", (max_x - min_x, max_y - min_y), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp_img)
    
    # Draw line on temporary image (relative coordinates) in base color
    temp_draw.line([(x1 - min_x, y1 - min_y), (x2 - min_x, y2 - min_y)], 
                   fill=color, width=width)
    
    # Apply opacity by modifying the alpha channel
    if opacity < 1.0:
        alpha = temp_img.split()[3]  # Get alpha channel
        alpha = alpha.point(lambda p: int(p * opacity))  # Multiply by opacity
        temp_img.putalpha(alpha)
    
    # Composite onto main image
    img.paste(temp_img, (min_x, min_y), temp_img)


def draw_two_column_fields(draw, artwork, panel_x: int, y_pos: int, max_width: int,
                          colors: Dict, font_small: ImageFont.FreeTypeFont, 
                          font_side: ImageFont.FreeTypeFont,
                          all_embeddings: Optional[np.ndarray] = None,
                          artwork_idx: Optional[int] = None,
                          distances: Optional[np.ndarray] = None,
                          shelf0_mask: Optional[np.ndarray] = None,
                          shelf0_indices: Optional[np.ndarray] = None,
                          scale: float = 1.0,
                          show_raw_embedding: bool = True) -> int:
    """Draw remaining fields in two-column layout.
    
    Left column: ID, Size, Handling Status, Raw Embedding (optional)
    Right column: Delivery Date, Weight, Distance to Centroid
    
    Args:
        max_width: Maximum width available for the panel (e.g., PANEL_WIDTH_SCALED - 60 or CANVAS_WIDTH_SCALED - panel_x - 20)
        scale: Scaling factor for supersampling
        show_raw_embedding: If False, skip drawing Raw Embedding field
    
    Returns:
        Updated y_pos after drawing all fields
    """
    col1_x = panel_x
    col2_x = panel_x + max_width // 2 + int(10 * scale)
    col_y_start = y_pos
    col_y = col_y_start
    max_col_width = max_width // 2 - int(10 * scale)
    
    # Left column
    # Use whiter color for labels, monofont for content
    font_label_white = get_font(int(FONT_SIZE_SMALL * scale), "thin")  # Label font (not mono, whiter)
    font_content = get_font(int(FONT_SIZE_TABLE * scale), "light", mono=False)  # Content font (light, no mono)
    
    # ID
    try:
        artwork_id_int = int(float(artwork.get("id", "N/A")))
        draw.text((col1_x, col_y), "ID", fill=colors["text"], font=font_label_white)  # Whiter label
        col_y += int(18 * scale)
        # Format as 4 digits (e.g., 0013)
        draw.text((col1_x, col_y), f"{artwork_id_int:04d}", fill=colors["text"], font=font_content)  # Monofont content
        col_y += int(25 * scale)
    except (ValueError, TypeError):
        pass
    
    # Size
    size = str(artwork.get("size", "N/A"))
    if size and size != "N/A" and size != "nan":
        draw.text((col1_x, col_y), "Size", fill=colors["text"], font=font_label_white)  # Whiter label
        col_y += int(18 * scale)
        size_bbox = draw.textbbox((0, 0), size, font=font_content)
        if size_bbox[2] - size_bbox[0] > max_col_width:
            while size and draw.textbbox((0, 0), size + "...", font=font_content)[2] > max_col_width:
                size = size[:-1]
            size = size + "..." if size else "..."
        draw.text((col1_x, col_y), size, fill=colors["text"], font=font_content)  # Monofont content
        col_y += int(25 * scale)
    
    # Handling Status
    handling_status = str(artwork.get("handling_status", "N/A"))
    if handling_status and handling_status != "N/A" and handling_status != "nan":
        draw.text((col1_x, col_y), "Handling Status", fill=colors["text"], font=font_label_white)  # Whiter label
        col_y += int(18 * scale)
        draw.text((col1_x, col_y), handling_status, fill=colors["text"], font=font_content)  # Monofont content
        col_y += int(25 * scale)
    
    # Raw Embedding (only show if show_raw_embedding is True)
    if show_raw_embedding and all_embeddings is not None and artwork_idx is not None and artwork_idx < len(all_embeddings):
        emb = all_embeddings[artwork_idx]
        draw.text((col1_x, col_y), "Raw Embedding", fill=colors["text"], font=font_label_white)  # Whiter label
        col_y += int(18 * scale)
        emb_preview = emb[:8] if len(emb) > 8 else emb
        emb_str = ", ".join([f"{v:.3f}" for v in emb_preview])
        if len(emb) > 8:
            emb_str += f" ... ({len(emb)} dims)"
        emb_words = emb_str.split(", ")
        emb_lines = []
        current_line = ""
        for word in emb_words:
            test_line = current_line + (", " if current_line else "") + word
            bbox = draw.textbbox((0, 0), test_line, font=font_content)
            if bbox[2] - bbox[0] <= max_col_width:
                current_line = test_line
            else:
                if current_line:
                    emb_lines.append(current_line)
                current_line = word
        if current_line:
            emb_lines.append(current_line)
        for line in emb_lines[:2]:  # Max 2 lines
            draw.text((col1_x, col_y), line, fill=colors["text"], font=font_content)  # Monofont content
            col_y += int(18 * scale)
        col_y += int(8 * scale)
    
    # Right column
    col_y_right = col_y_start
    
    # Delivery Date
    delivery_date = str(artwork.get("deliveryDate", "N/A"))
    if delivery_date and delivery_date != "N/A" and delivery_date != "nan":
        draw.text((col2_x, col_y_right), "Delivery Date", fill=colors["text"], font=font_label_white)  # Whiter label
        col_y_right += int(18 * scale)
        draw.text((col2_x, col_y_right), delivery_date, fill=colors["text"], font=font_content)  # Monofont content
        col_y_right += int(25 * scale)
    
    # Weight
    weight = str(artwork.get("weight", "N/A"))
    if weight and weight != "N/A" and weight != "nan":
        draw.text((col2_x, col_y_right), "Weight", fill=colors["text"], font=font_label_white)  # Whiter label
        col_y_right += int(18 * scale)
        draw.text((col2_x, col_y_right), weight, fill=colors["text"], font=font_content)  # Monofont content
        col_y_right += int(25 * scale)
    
    # Distance
    if distances is not None and artwork_idx is not None and shelf0_mask is not None:
        shelf0_indices_arr = np.where(shelf0_mask)[0] if shelf0_indices is None else shelf0_indices
        if artwork_idx in shelf0_indices_arr:
            shelf0_idx = list(shelf0_indices_arr).index(artwork_idx)
            if shelf0_idx < len(distances):
                distance = distances[shelf0_idx]
                draw.text((col2_x, col_y_right), "Distance to Centroid", fill=colors["text"], font=font_label_white)  # Whiter label
                col_y_right += int(18 * scale)
                draw.text((col2_x, col_y_right), f"{distance:.4f}", fill=colors["lime"], font=font_content)  # Monofont content
                col_y_right += int(25 * scale)
    
    # Return max of both columns
    return max(col_y, col_y_right) + int(10 * scale)


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
    white_background: bool = False,  # Use white background instead of black
    circle_expand_progress: Optional[float] = None,  # 0.0 to 1.0 for expanding circle animation (for highlight_slow step)
    ruler_progress: Optional[float] = None,  # 0.0 to 1.0 for ruler line animation progress
    ruler_to_rep: bool = True,  # True for drawing to representative, False for outlier
    connection_lines_opacity: Optional[float] = None,  # 0.0 to 1.0 for fading connection lines
    current_distance: Optional[float] = None,  # Current distance being displayed
    search_mode: Optional[str] = None,  # "representative" or "outlier"
    supersample_factor: float = 2.0,  # Supersampling factor for anti-aliasing (2.0 = Full HD output, 4.0 = 4K output)
    artwork_lookup: Optional[Dict] = None,  # Pre-computed lookup dict for artwork data (id -> row index or row data)
    top10_blink_progress: Optional[float] = None,  # 0.0 to 1.0 for blink effect after top10 (0=gray, 1=green)
    ruler_circle_expand_progress: Optional[float] = None,  # 0.0 to 1.0 for circle expansion in ruler step
    ruler_text_fade_progress: Optional[float] = None,  # 0.0 to 1.0 for text fade in during ruler step
    panel_border_progress: Optional[float] = None,  # 0.0 to 1.0 for panel border animation (smooth fade-in)
    highlight_circle_expand_progress: Optional[float] = None,  # 0.0 to 1.0 for grey circle expansion in representative step
    green_dots_dim_progress: Optional[float] = None,  # 0.0 to 1.0 for dimming green dots (0=full bright, 1=dimmed)
    title_ease_progress: Optional[float] = None,  # 0.0 to 1.0 for title text easing animation
    previous_title: Optional[str] = None,  # Previous title text for easing
    subtitle_ease_progress: Optional[float] = None,  # 0.0 to 1.0 for subtitle text easing animation
    previous_subtitle: Optional[str] = None,  # Previous subtitle text for easing
    info_title_ease_progress: Optional[float] = None,  # 0.0 to 1.0 for info panel title text easing
    previous_info_title: Optional[str] = None,  # Previous info panel title for easing
    info_artist_ease_progress: Optional[float] = None,  # 0.0 to 1.0 for info panel artist text easing
    previous_info_artist: Optional[str] = None,  # Previous info panel artist for easing
) -> Image.Image:
    """Create a single frame for the visualization."""
    # Validate representative_idx - if it's a DataFrame or invalid, set to None
    representative_idx = safe_int_convert(representative_idx)
    
    # Get color scheme based on background mode
    colors = get_colors(white_background)
    
    # Determine target resolution and render strategy
    # For 4K: render directly at 4K (no downscaling needed)
    # For 1080p: render at 2x (2160p) and downscale for anti-aliasing
    if supersample_factor == 4.0:
        # 4K mode: render directly at 4K
        target_width = CANVAS_WIDTH_4K
        target_height = CANVAS_HEIGHT_4K
        # Use 4K dimensions directly (already 2x base size)
        map_width = MAP_WIDTH_4K
        panel_width = PANEL_WIDTH_4K
        render_width = target_width
        render_height = target_height
        # Coordinates are normalized to base canvas (1920x1080), need 2x scale for 4K
        # But map_width/panel_width are already 2x, so don't scale them again
        coord_scale = 2.0  # For coordinates
        dim_scale = 1.0  # For dimensions (already at 4K size)
        needs_downscale = False
    else:
        # 1080p mode: render at 2x and downscale
        target_width = CANVAS_WIDTH
        target_height = CANVAS_HEIGHT
        map_width = MAP_WIDTH
        panel_width = PANEL_WIDTH
        render_width = int(target_width * 2.0)  # Render at 2x for anti-aliasing
        render_height = int(target_height * 2.0)
        coord_scale = 2.0  # For coordinates
        dim_scale = 2.0  # For dimensions (scale from base to 2x)
        needs_downscale = True
    
    # Create canvas at render resolution
    img = Image.new("RGBA", (render_width, render_height), (*colors["background"], 255))
    draw = ImageDraw.Draw(img)
    
    # For backward compatibility, use coord_scale as scale for visual elements
    # (fonts, sizes, positions all scale with coordinates)
    scale = coord_scale
    
    # Scale coordinate arrays (coordinates are normalized to base canvas)
    all_coords = all_coords * coord_scale
    if shelf0_coords is not None:
        shelf0_coords = shelf0_coords * coord_scale
    if shelf0_coords_progressive is not None:
        shelf0_coords_progressive = shelf0_coords_progressive * coord_scale
    if centroid_coord is not None:
        centroid_coord = (centroid_coord[0] * coord_scale, centroid_coord[1] * coord_scale) if isinstance(centroid_coord, (tuple, list)) else centroid_coord * coord_scale
    
    # Scale lines_to_draw coordinates (they're in the same coordinate space as all_coords_2d)
    if lines_to_draw is not None:
        scaled_lines_to_draw = []
        for line_data in lines_to_draw:
            if len(line_data) == 4:  # (start, end, progress, is_closest)
                start, end, progress, is_closest = line_data
                # Scale start and end coordinates - convert to tuples for consistency
                if isinstance(start, np.ndarray):
                    start = (float(start[0] * coord_scale), float(start[1] * coord_scale))
                elif isinstance(start, (tuple, list)) and len(start) >= 2:
                    start = (float(start[0] * coord_scale), float(start[1] * coord_scale))
                if isinstance(end, np.ndarray):
                    end = (float(end[0] * coord_scale), float(end[1] * coord_scale))
                elif isinstance(end, (tuple, list)) and len(end) >= 2:
                    end = (float(end[0] * coord_scale), float(end[1] * coord_scale))
                scaled_lines_to_draw.append((start, end, progress, is_closest))
            else:  # (start, end, progress)
                start, end, progress = line_data
                # Scale start and end coordinates - convert to tuples for consistency
                if isinstance(start, np.ndarray):
                    start = (float(start[0] * coord_scale), float(start[1] * coord_scale))
                elif isinstance(start, (tuple, list)) and len(start) >= 2:
                    start = (float(start[0] * coord_scale), float(start[1] * coord_scale))
                if isinstance(end, np.ndarray):
                    end = (float(end[0] * coord_scale), float(end[1] * coord_scale))
                elif isinstance(end, (tuple, list)) and len(end) >= 2:
                    end = (float(end[0] * coord_scale), float(end[1] * coord_scale))
                scaled_lines_to_draw.append((start, end, progress))
        lines_to_draw = scaled_lines_to_draw
    
    # Scale size constants (use coord_scale for visual elements, dim_scale for dimensions)
    POINT_SIZE_SCALED = POINT_SIZE * coord_scale
    POINT_SIZE_LARGE_SCALED = POINT_SIZE_LARGE * coord_scale
    POINT_SIZE_REPRESENTATIVE_SCALED = POINT_SIZE_REPRESENTATIVE * coord_scale
    LINE_WIDTH_SCALED = LINE_WIDTH * coord_scale
    RULER_DOT_SIZE_SCALED = 16 * coord_scale if step == "ruler" else None
    
    # Scale dimension constants (dimensions are already at target size in 4K mode)
    MAP_WIDTH_SCALED = map_width * dim_scale
    PANEL_WIDTH_SCALED = panel_width * dim_scale
    MAP_HEIGHT_SCALED = target_height * dim_scale
    PANEL_HEIGHT_SCALED = target_height * dim_scale
    CANVAS_WIDTH_SCALED = render_width
    CANVAS_HEIGHT_SCALED = render_height
    
    # Compute shelf0_indices once per frame (used multiple times)
    shelf0_indices = np.where(shelf0_mask)[0]
    shelf0_indices_list = shelf0_indices.tolist()
    
    # Draw floor plan image at top left, rotated 90 degrees (draw early so it's in background)
    # Cache floor plan per (shelf, scale, white_bg) combination for efficiency
    floor_plan_cache_key = (target_shelf, scale, white_background)
    if floor_plan_cache_key in _floor_plan_cache:
        floor_plan_img = _floor_plan_cache[floor_plan_cache_key]
    else:
        floor_plan_img = None
        # Use map_white folder for white background, map folder for black background
        map_dir = MAP_DIR_WHITE if white_background else MAP_DIR
        floor_plan_path = map_dir / f"shelf_{target_shelf}.png"
        if floor_plan_path.exists():
            try:
                floor_plan_img = Image.open(floor_plan_path)
                # Rotate 90 degrees counter-clockwise (the other way)
                floor_plan_img = floor_plan_img.rotate(90, expand=True)
                # Resize to reasonable size (max 350px width/height after rotation - increased from 300)
                max_size = int(350 * scale)
                if floor_plan_img.width > max_size or floor_plan_img.height > max_size:
                    ratio = min(max_size / floor_plan_img.width, max_size / floor_plan_img.height)
                    new_width = int(floor_plan_img.width * ratio)
                    new_height = int(floor_plan_img.height * ratio)
                    floor_plan_img = floor_plan_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                _floor_plan_cache[floor_plan_cache_key] = floor_plan_img
            except Exception:
                _floor_plan_cache[floor_plan_cache_key] = None
    
    if floor_plan_img is not None:
        # Position at top left with some padding (moved more to top-left)
        floor_plan_x = 20 * scale
        floor_plan_y = 30 * scale
        
        # Paste the floor plan image
        if floor_plan_img.mode == 'RGBA':
            img.paste(floor_plan_img, (int(floor_plan_x), int(floor_plan_y)), floor_plan_img)
        else:
            img.paste(floor_plan_img, (int(floor_plan_x), int(floor_plan_y)))
    
    # Load fonts with appropriate weights - scale font sizes for supersampled canvas
    font_title = get_font(int(FONT_SIZE_TITLE * scale), "medium")  # Medium for titles to stand out
    font_label = get_font(int(FONT_SIZE_LABEL * scale), "thin")  # Thin for labels
    font_info = get_font(int(FONT_SIZE_INFO * scale), "thin", mono=True)  # Monofont for info text
    font_small = get_font(int(FONT_SIZE_SMALL * scale), "thin", mono=True)  # Monofont for small text
    
    # Draw step label (subtitle) at top - use centralized subtitle system
    # Get final subtitle text with all dynamic modifications applied
    step_text = get_final_subtitle_text(
        step=step,
        target_shelf=target_shelf,
        search_mode=search_mode,
        current_distance=current_distance,
        highlighted_artwork_idx=highlighted_artwork_idx,
        top_representatives=top_representatives,
        top_outliers=top_outliers,
        top10_reps_shown=top10_reps_shown,
        top10_outliers_shown=top10_outliers_shown,
        centroid_coord=centroid_coord,
    )
    
    # Draw title at left side, higher up
    # Format shelf number as 2 digits (e.g., 08)
    target_shelf_formatted = f"{int(target_shelf):02d}" if target_shelf.isdigit() else target_shelf
    current_title = f"Regal {target_shelf_formatted}"
    
    # Apply ASCII text easing if previous title and progress are provided
    if previous_title is not None and title_ease_progress is not None:
        title_text = ascii_text_ease(previous_title, current_title, title_ease_progress)
    else:
        title_text = current_title
    
    title_x = 50 * scale  # Left side
    title_y = CANVAS_HEIGHT_SCALED - 100 * scale  # Higher up from bottom
    draw.text((title_x, title_y), title_text, fill=colors["text"], font=font_title)
    
    # Draw subtitle (step label) on the right side of the title, vertically centered
    # Apply ASCII text easing if previous subtitle and progress are provided
    if previous_subtitle is not None and subtitle_ease_progress is not None:
        display_subtitle = ascii_text_ease(previous_subtitle, step_text, subtitle_ease_progress)
    else:
        display_subtitle = step_text
    
    title_bbox = draw.textbbox((title_x, title_y), title_text, font=font_title)
    title_width = title_bbox[2] - title_bbox[0]
    title_center_y = (title_bbox[1] + title_bbox[3]) / 2  # Vertical center of title bounding box
    
    # Get subtitle bounding box relative to baseline at (0, 0)
    subtitle_bbox = draw.textbbox((0, 0), display_subtitle, font=font_label)
    subtitle_center_offset = (subtitle_bbox[1] + subtitle_bbox[3]) / 2  # Center offset from baseline
    
    subtitle_x = title_x + title_width + 20 * scale  # 20px spacing after title
    # Center subtitle vertically with title by aligning their center points
    # Calculate y position (baseline) so subtitle center aligns with title center
    subtitle_y = title_center_y - subtitle_center_offset
    # Subtitle should be white (not grey)
    draw.text((subtitle_x, subtitle_y), display_subtitle, fill=colors["text"], font=font_label)
    
    # Draw divider line first
    draw.line([(MAP_WIDTH_SCALED, 0), (MAP_WIDTH_SCALED, CANVAS_HEIGHT_SCALED)], fill=colors["text"], width=int(LINE_WIDTH_SCALED * 2))
    
    # Draw lines (connections between items) - FIRST layer
    # Draw connections between Regal 0 items (thin lines, gray on black - less obvious)
    # Only draw lines if step is not "highlight_slow" and not "representative" and not after highlight stage
    # Apply opacity if connection_lines_opacity is provided (for fade out animation)
    # Don't draw grey lines after finding centroid (only in highlight stage)
    if step != "highlight_slow" and step != "representative" and step not in ["ruler", "top10", "side_by_side"]:
        # Only draw in highlight/centroid/distances steps
        if step in ["highlight", "centroid", "distances"]:
            coords_to_use = shelf0_coords_progressive if (shelf0_coords_progressive is not None and num_shelf0_shown is not None) else shelf0_coords
            if coords_to_use is not None and len(coords_to_use) > 1:
                num_to_show = num_shelf0_shown if num_shelf0_shown is not None else len(coords_to_use)
                # Draw lines with transparency
                base_color = colors["connection"]
                opacity = connection_lines_opacity if connection_lines_opacity is not None else 1.0
                opacity = max(0.0, min(1.0, opacity))
                
                for i in range(min(num_to_show, len(coords_to_use))):
                    for j in range(i + 1, min(num_to_show, len(coords_to_use))):
                        x1, y1 = coords_to_use[i]
                        x2, y2 = coords_to_use[j]
                        draw_line_with_opacity(img, draw, (x1, y1), (x2, y2), base_color, int(LINE_WIDTH_SCALED), opacity)
    
    # Draw all points (fill, no stroke for non-Regal items) with lower opacity
    # Dots (gray dots) are drawn AFTER connection lines but BEFORE lines from centroid
    # In "all" step, draw ALL points as gray (including Regal ones)
    # In "highlight_slow" step, draw Regal points as gray first (they'll transition to green)
    # In other steps, only draw non-Regal points as gray (Regal ones will be drawn as green later)
    for i, (x, y) in enumerate(all_coords):
        if step == "all":
            # In "all" step, draw everything as gray
            draw.ellipse([x - POINT_SIZE_SCALED, y - POINT_SIZE_SCALED, x + POINT_SIZE_SCALED, y + POINT_SIZE_SCALED],
                       fill=colors["point_low_opacity"], outline=None, width=0)
        elif step == "highlight_slow" and shelf0_mask[i]:
            # In highlight_slow step, draw Regal points as gray first (they'll transition to green)
            # Get the index in shelf0_indices to check if this should be shown as gray
            if i in shelf0_indices:
                idx_in_shelf0 = shelf0_indices_list.index(i)
                # Draw as gray if not yet shown (beyond num_shelf0_shown)
                # The green dots will be drawn later in the code, so we need to draw gray for all items
                # that haven't been identified yet
                if num_shelf0_shown is None or idx_in_shelf0 >= num_shelf0_shown:
                    # Draw as gray (not yet identified - will transition to green)
                    draw.ellipse([x - POINT_SIZE_SCALED, y - POINT_SIZE_SCALED, x + POINT_SIZE_SCALED, y + POINT_SIZE_SCALED],
                               fill=colors["point_low_opacity"], outline=None, width=0)
        elif not shelf0_mask[i]:
            # In other steps, only draw non-Regal points as gray
            draw.ellipse([x - POINT_SIZE_SCALED, y - POINT_SIZE_SCALED, x + POINT_SIZE_SCALED, y + POINT_SIZE_SCALED],
                       fill=colors["point_low_opacity"], outline=None, width=0)
    
    # Draw text labels for Regal items (after grey dots, before green lines and green dots)
    if step != "all" and shelf0_coords is not None and df is not None:
        shelf0_indices_list = list(np.where(shelf0_mask)[0])
        # Determine how many items to show text for
        if step == "highlight_slow" and num_shelf0_shown is not None:
            num_text_to_show = num_shelf0_shown
            text_indices_to_show = list(range(num_text_to_show))
        elif step == "highlight":
            num_text_to_show = len(shelf0_coords)
            text_indices_to_show = list(range(num_text_to_show))
        else:
            num_text_to_show = len(shelf0_coords)
            text_indices_to_show = list(range(num_text_to_show))
        
        for i, (x, y) in enumerate(shelf0_coords):
            if step == "highlight_slow":
                if i >= num_text_to_show:
                    continue
            elif i >= num_text_to_show:
                continue
            try:
                if i < len(shelf0_indices_list):
                    artwork_idx_for_text = shelf0_indices_list[i]
                    artwork_id = all_artwork_ids[artwork_idx_for_text]
                    artwork = find_artwork_by_id(artwork_id, df, artwork_lookup)
                    if artwork is not None:
                        title = str(artwork.get("title", ""))
                        if title and title != "nan":
                            text_x = x + POINT_SIZE_LARGE_SCALED + 5 * scale
                            text_y = y - 15 * scale
                            font_map_text = get_font(int(FONT_SIZE_MAP_TEXT * scale), "thin")
                            draw.text((int(text_x), int(text_y)), title, fill=colors["text"], font=font_map_text)
            except Exception:
                pass
    
    # Extract centroid coordinates (needed for green lines from centroid)
    cx, cy = extract_centroid_coords(centroid_coord, scale, MAP_WIDTH_SCALED, MAP_HEIGHT_SCALED)
    
    # Draw green lines from centroid to Regal 0 points (after text, before green dots)
    # In "representative" step, only draw green lines (from lines_to_draw), no grey lines
    if cx is not None and cy is not None and step != "highlight_slow":
        # Draw lines progressively if lines_to_draw is provided (for animation)
        if lines_to_draw is not None:
            for line_idx, line_data in enumerate(lines_to_draw):
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
                    # In "representative" step, use green for lines being drawn
                    if step == "representative":
                        line_color = colors["lime"]
                        line_width = int(3 * scale)  # Thicker line for representative step, scaled
                    else:
                        line_color = colors["connection"]
                        line_width = int(LINE_WIDTH_SCALED)
                    draw.line([(x1, y1), (inter_x, inter_y)], fill=line_color, width=line_width)
                    
                    # Add small numbers on the side of the line (for representative step)
                    if step == "representative" and progress > 0.3:  # Only show when line is partially drawn
                        # Calculate midpoint of visible line
                        mid_x = x1 + (inter_x - x1) * 0.5
                        mid_y = y1 + (inter_y - y1) * 0.5
                        
                        # Calculate perpendicular offset for number placement
                        dx = inter_x - x1
                        dy = inter_y - y1
                        length = np.sqrt(dx*dx + dy*dy)
                        if length > 0:
                            perp_x = -dy / length * 15 * scale  # 15px offset, scaled
                            perp_y = dx / length * 15 * scale
                            num_x = int(mid_x + perp_x)
                            num_y = int(mid_y + perp_y)
                            
                            # Draw distance number in small font (properly scaled)
                            if current_distance is not None:
                                font_small_num = get_font(int(12 * scale), "light", mono=False)  # Light font for numbers
                                num_text = f"{current_distance:.3f}"
                                
                                # Only draw background for the last measure distance (last line in lines_to_draw)
                                is_last_measure = (line_idx == len(lines_to_draw) - 1)
                                
                                if is_last_measure:
                                    # Draw with background for readability (properly scaled)
                                    bbox = draw.textbbox((int(num_x), int(num_y)), num_text, font=font_small_num)
                                    padding = int(2 * scale)
                                    draw.rectangle(
                                        [int(bbox[0] - padding), int(bbox[1] - padding),
                                         int(bbox[2] + padding), int(bbox[3] + padding)],
                                        fill=colors["background"], outline=None
                                    )
                                
                                draw.text((int(num_x), int(num_y)), num_text, fill=colors["lime"], font=font_small_num)
        else:
            # Draw lines from centroid to Regal 0 points (green lines in representative step, grey otherwise)
            # Only draw in highlight/centroid/distances steps, not after
            if step in ["highlight", "centroid", "distances"]:
                # Use progressive coords if available
                # Apply opacity if connection_lines_opacity is provided (for fade out animation)
                coords_to_use = shelf0_coords_progressive if (shelf0_coords_progressive is not None and num_shelf0_shown is not None) else shelf0_coords
                if coords_to_use is not None:
                    num_to_show = num_shelf0_shown if num_shelf0_shown is not None else len(coords_to_use)
                    # Draw lines with transparency
                    # Use green for representative step, grey for other steps
                    base_color = colors["lime"] if step == "representative" else colors["connection"]
                    opacity = connection_lines_opacity if connection_lines_opacity is not None else 1.0
                    opacity = max(0.0, min(1.0, opacity))
                    
                    for idx in range(min(num_to_show, len(coords_to_use))):
                        x, y = coords_to_use[idx]
                        draw_line_with_opacity(img, draw, (cx, cy), (x, y), base_color, int(LINE_WIDTH_SCALED), opacity)
    
    # Draw Regal 0 points (lime green, larger) with title text
    # If progressive mode, only show up to num_shelf0_shown
    # EXCEPTION: In "highlight" step, show ALL items (from shelf0_coords) regardless of num_shelf0_shown
    # IMPORTANT: In "all" step, DO NOT draw shelf0 points here (they're already drawn as grey above)
    if step != "all" and shelf0_coords_progressive is not None and num_shelf0_shown is not None:
        # In highlight step, show all items from shelf0_coords (all identified items)
        # In highlight_slow step, ONLY show items up to num_shelf0_shown (progressive)
        # In other steps, use progressive mode
        if step == "highlight":
            coords_for_dots = shelf0_coords if shelf0_coords is not None else shelf0_coords_progressive
            num_dots_to_show = len(coords_for_dots)
        else:
            # For highlight_slow and other steps, use progressive coords
            coords_for_dots = shelf0_coords_progressive
            num_dots_to_show = num_shelf0_shown
        
        # Progressive mode: show artworks being added one by one
        # In highlight_slow step, show all items up to num_shelf0_shown (they stay visible once identified)
        # In highlight step, show all items that have been identified
        # IMPORTANT: Only iterate up to num_dots_to_show to prevent showing too many items
        for i, (x, y) in enumerate(coords_for_dots[:num_dots_to_show]):
            # In highlight_slow step, show all items up to num_shelf0_shown (they accumulate and stay green)
            if step == "highlight_slow":
                # Show all items that have been identified so far (up to num_shelf0_shown)
                if i >= num_shelf0_shown:
                    continue  # Skip items not yet identified
            elif i >= num_dots_to_show:
                continue  # Skip items beyond what should be shown (shouldn't happen due to slice, but keep for safety)
            
            # Check if this is a selected item during blink phase (skip regular dot)
            skip_dot = False
            if step == "top10" and top10_blink_progress is not None and df is not None:
                if i < len(shelf0_indices):
                    artwork_idx_for_check = shelf0_indices[i]
                    artwork_id = all_artwork_ids[artwork_idx_for_check]
                    is_selected_rep = (aesthetic_representative_id is not None and 
                                      (float(artwork_id) == float(aesthetic_representative_id) or 
                                       str(artwork_id) == str(aesthetic_representative_id)))
                    is_selected_outlier = (aesthetic_outlier_id is not None and 
                                          (float(artwork_id) == float(aesthetic_outlier_id) or 
                                           str(artwork_id) == str(aesthetic_outlier_id)))
                    if is_selected_rep or is_selected_outlier:
                        skip_dot = True  # Skip regular dot, will be drawn larger later
            
            if skip_dot:
                continue  # Skip drawing regular dot for selected items during blink
            
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
                # In "highlight" step, ALL previously identified items should stay full green
                # In "highlight_slow" step, ALL identified items (i < num_shelf0_shown) should stay full green
                # This includes the currently appearing item after easing is done
                if step == "highlight":
                    # In highlight step, show all items that have been identified (all should be full green)
                    current_color = colors["lime"]
                elif step == "highlight_slow" and i < num_shelf0_shown:
                    # In highlight_slow step, ALL identified items (including current one after easing) stay fully green
                    current_color = colors["lime"]
                else:
                    # Item not yet identified
                    current_color = colors["lime_low_opacity"]
            
            # Apply dimming effect if specified (for transition between highlight and finding representative stages)
            if green_dots_dim_progress is not None and current_color == colors["lime"]:
                # Dim from full bright green to a medium green (between bright and low opacity)
                # Use a dimmed green color (midway between bright and low opacity)
                dimmed_green = (
                    int(colors["lime"][0] * 0.6),  # 60% of bright green
                    int(colors["lime"][1] * 0.6),
                    int(colors["lime"][2] * 0.6)
                )
                # Interpolate between full bright and dimmed based on progress
                dim_progress = max(0.0, min(1.0, green_dots_dim_progress))
                r = int(colors["lime"][0] * (1 - dim_progress) + dimmed_green[0] * dim_progress)
                g = int(colors["lime"][1] * (1 - dim_progress) + dimmed_green[1] * dim_progress)
                b = int(colors["lime"][2] * (1 - dim_progress) + dimmed_green[2] * dim_progress)
                current_color = (r, g, b)
            
            draw.ellipse([x - POINT_SIZE_LARGE_SCALED, y - POINT_SIZE_LARGE_SCALED, 
                        x + POINT_SIZE_LARGE_SCALED, y + POINT_SIZE_LARGE_SCALED],
                       fill=current_color, outline=None, width=0)
        
    elif step != "all" and shelf0_coords is not None:
        # Normal mode: show all Regal 0 points (but not in "all" step)
        # Skip selected items during blink phase (they'll be drawn larger later)
        for idx, (x, y) in enumerate(shelf0_coords):
            # Check if this is a selected item during blink phase
            skip_dot = False
            if step == "top10" and top10_blink_progress is not None and df is not None:
                if idx < len(shelf0_indices):
                    artwork_idx_for_check = shelf0_indices[idx]
                    artwork_id = all_artwork_ids[artwork_idx_for_check]
                    is_selected_rep = (aesthetic_representative_id is not None and 
                                      (float(artwork_id) == float(aesthetic_representative_id) or 
                                       str(artwork_id) == str(aesthetic_representative_id)))
                    is_selected_outlier = (aesthetic_outlier_id is not None and 
                                          (float(artwork_id) == float(aesthetic_outlier_id) or 
                                           str(artwork_id) == str(aesthetic_outlier_id)))
                    if is_selected_rep or is_selected_outlier:
                        skip_dot = True  # Skip regular dot, will be drawn larger later
            
            if skip_dot:
                continue  # Skip drawing regular dot for selected items during blink
            
            is_highlighted = (highlighted_artwork_idx is not None and 
                            idx < len(shelf0_indices) and 
                            shelf0_indices[idx] == highlighted_artwork_idx)
            
            # Use green fill with no border for highlighted, lower opacity for others
            if is_highlighted:
                draw.ellipse([x - POINT_SIZE_LARGE_SCALED, y - POINT_SIZE_LARGE_SCALED, 
                            x + POINT_SIZE_LARGE_SCALED, y + POINT_SIZE_LARGE_SCALED],
                           fill=colors["lime"], outline=None, width=0)
            else:
                draw.ellipse([x - POINT_SIZE_LARGE_SCALED, y - POINT_SIZE_LARGE_SCALED, 
                            x + POINT_SIZE_LARGE_SCALED, y + POINT_SIZE_LARGE_SCALED],
                           fill=colors["lime_low_opacity"], outline=None, width=0)
    
    # Draw expanding gray circle for highlight_slow and highlight steps (AFTER green dots)
    # This circle expands around the currently highlighted artwork to draw attention
    if step in ["highlight_slow", "highlight"] and highlighted_artwork_idx is not None and shelf0_coords is not None and circle_expand_progress is not None:
        highlight_idx_check = safe_int_convert(highlighted_artwork_idx)
        shelf0_indices_list_int = [int(x) for x in shelf0_indices_list]
        if highlight_idx_check is not None and highlight_idx_check in shelf0_indices_list_int:
            highlight_idx_in_shelf0 = shelf0_indices_list_int.index(highlight_idx_check)
            if highlight_idx_in_shelf0 < len(shelf0_coords):
                hx, hy = shelf0_coords[highlight_idx_in_shelf0]
                # Expanding circle animation: start from small (5px) to full size (20px radius)
                # Use scaled values for consistency with other circle expansions
                min_radius = 5 * scale
                max_radius = 20 * scale
                # Use ease-in-out curve for smooth expansion
                eased = circle_expand_progress * circle_expand_progress * (3 - 2 * circle_expand_progress)
                current_radius = min_radius + (max_radius - min_radius) * eased
                # Draw gray expanding circle around green dot (decorative highlight)
                draw.ellipse([int(hx - current_radius), int(hy - current_radius), 
                            int(hx + current_radius), int(hy + current_radius)],
                            fill=None, outline=colors["circle_gray"], width=int(2 * scale))
    
    # Highlight currently displayed artwork (if different from representative)
    # Draw gray circle AFTER green circle (so it appears on top but behind other elements)
    if highlighted_artwork_idx is not None and shelf0_coords is not None:
        highlight_idx_check = safe_int_convert(highlighted_artwork_idx)
        
        shelf0_indices_list_int = [int(x) for x in shelf0_indices_list]  # Ensure all are plain ints
        if highlight_idx_check in shelf0_indices_list_int:
            highlight_idx_in_shelf0 = shelf0_indices_list_int.index(highlight_idx_check)
            if highlight_idx_in_shelf0 < len(shelf0_coords):
                hx, hy = shelf0_coords[highlight_idx_in_shelf0]
                # Draw gray circle around highlighted artwork with smooth expansion animation
                # Use same smooth animation method as other circle expansions
                if highlight_circle_expand_progress is not None:
                    # Start from small radius, expand to full size
                    min_radius = 5 * scale
                    max_radius = 20 * scale
                    # Use ease-in-out curve for smooth expansion (same as circle_expand_progress)
                    eased = highlight_circle_expand_progress * highlight_circle_expand_progress * (3 - 2 * highlight_circle_expand_progress)
                    circle_radius = min_radius + (max_radius - min_radius) * eased
                else:
                    # Fully expanded if no animation
                    circle_radius = int(20 * scale)
                
                draw.ellipse([int(hx - circle_radius), int(hy - circle_radius), 
                            int(hx + circle_radius), int(hy + circle_radius)],
                            fill=None, outline=colors["circle_gray"], width=int(2 * scale))
    
    # Highlight representative (if different from highlighted)
    # IMPORTANT: Do NOT draw green circle in "highlight_slow" (identify) step - only in later steps
    if representative_idx is not None and shelf0_coords is not None and step != "highlight_slow":
        rep_idx = safe_int_convert(representative_idx)
        highlight_idx = safe_int_convert(highlighted_artwork_idx)
        
        shelf0_indices_list_int = [int(x) for x in shelf0_indices_list]  # Ensure all are plain ints
        if rep_idx is not None:
            rep_idx = int(rep_idx)  # Final safety check
            if rep_idx in shelf0_indices_list_int and (highlight_idx is None or rep_idx != highlight_idx):
                rep_idx_in_shelf0 = shelf0_indices_list_int.index(rep_idx)
                if rep_idx_in_shelf0 < len(shelf0_coords):
                    rx, ry = shelf0_coords[rep_idx_in_shelf0]
                    # Draw larger bright green circle around representative - properly scaled
                    circle_radius = int(15 * scale)
                    draw.ellipse([int(rx - circle_radius), int(ry - circle_radius), 
                                int(rx + circle_radius), int(ry + circle_radius)],
                                fill=None, outline=colors["lime"], width=int(3 * scale))
    
    # Mark all top 10 representatives with bright green (if not in representatives step)
    # Skip items that are already marked as the main representative to avoid duplicate circles
    if step != "representatives" and top_representatives is not None and shelf0_coords is not None:
        # Get the representative_idx to skip it (already has a larger circle)
        rep_idx_to_skip = safe_int_convert(representative_idx)
        
        for rank, (artwork_idx_in_all, distance) in enumerate(top_representatives[:10]):
            # Skip if this is the main representative (already has a larger circle)
            if rep_idx_to_skip is not None and artwork_idx_in_all == rep_idx_to_skip:
                continue
                
            if artwork_idx_in_all in shelf0_indices:
                idx_in_shelf0 = list(shelf0_indices).index(artwork_idx_in_all)
                if idx_in_shelf0 < len(shelf0_coords):
                    x, y = shelf0_coords[idx_in_shelf0]
                    # Draw bright green circle around representative - properly scaled
                    circle_radius = int(12 * scale)
                    draw.ellipse([int(x - circle_radius), int(y - circle_radius), 
                                int(x + circle_radius), int(y + circle_radius)],
                                fill=None, outline=colors["lime"], width=int(2 * scale))
    
    # Draw panel background AFTER all left-side content (lines, dots, labels)
    # This overlays any text that might overflow from the left side into the panel area
    draw.rectangle([MAP_WIDTH_SCALED, 0, CANVAS_WIDTH_SCALED, CANVAS_HEIGHT_SCALED], fill=colors["background"])
    
    # Draw calculation info panel content (text and images) - draw LAST so it's on top
    if highlighted_artwork_idx is not None and df is not None:
        # Find artwork in dataframe
        try:
            artwork_id = all_artwork_ids[highlighted_artwork_idx]
            artwork = find_artwork_by_id(artwork_id, df, artwork_lookup)
        except (IndexError, KeyError, TypeError):
            artwork = None
        
        if artwork is not None:
            # Note: Panel background already drawn earlier
            
            # Draw thumbnail at the very top first - standardized layout
            image = load_image(artwork.get("thumbnail", ""))
            panel_x = MAP_WIDTH_SCALED + 30 * scale
            y_pos = 60 * scale  # Standardized start position
            
            if image:
                # Use larger image for non-ruler steps, standard size for ruler step
                # Use high-quality LANCZOS resampling for better image quality
                # Scale image height for supersampled canvas
                img_w, img_h = image.size
                base_image_height = FIXED_IMAGE_HEIGHT_LARGE if step != "ruler" else FIXED_IMAGE_HEIGHT
                image_height = int(base_image_height * scale)
                ratio = image_height / img_h
                new_w, new_h = int(img_w * ratio), image_height
                # Use LANCZOS for best quality (already set, but ensure it's used)
                image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                # Paste image - need to use alpha composite if image has alpha, otherwise direct paste
                img_x = MAP_WIDTH_SCALED + (PANEL_WIDTH_SCALED - image.width) // 2
                if image.mode == 'RGBA':
                    img.paste(image, (int(img_x), int(y_pos)), image)
                else:
                    img.paste(image, (int(img_x), int(y_pos)))
                
                # Draw smooth black border animation around artwork image
                # Use same smooth animation method as circle expansion (ease-in-out)
                if panel_border_progress is not None:
                    # Use ease-in-out curve for smooth animation
                    eased = panel_border_progress * panel_border_progress * (3 - 2 * panel_border_progress)
                    # Border opacity fades in from 0 to 1
                    border_opacity = eased
                    # Border color: black (or white if white background)
                    border_color = colors["text"] if not white_background else colors["background"]
                    # Apply opacity to border color
                    if border_opacity < 1.0:
                        # Blend with background
                        bg_color = colors["background"] if not white_background else colors["text"]
                        border_r = int(border_color[0] * border_opacity + bg_color[0] * (1 - border_opacity))
                        border_g = int(border_color[1] * border_opacity + bg_color[1] * (1 - border_opacity))
                        border_b = int(border_color[2] * border_opacity + bg_color[2] * (1 - border_opacity))
                        border_color = (border_r, border_g, border_b)
                else:
                    # Fully visible border if no animation
                    border_color = colors["text"] if not white_background else colors["background"]
                
                # Draw border rectangle around image
                border_width = int(2 * scale)
                border_rect = [int(img_x) - border_width, int(y_pos) - border_width,
                              int(img_x) + image.width + border_width, int(y_pos) + image.height + border_width]
                draw.rectangle(border_rect, fill=None, outline=border_color, width=border_width)
                
                y_pos += image.height + 20 * scale  # Standardized spacing
            else:
                y_pos += 20 * scale
            
            # Draw divider line after image - standardized spacing
            draw.line([(panel_x, y_pos), (CANVAS_WIDTH_SCALED - 30 * scale, y_pos)], 
                     fill=colors["text"], width=int(LINE_WIDTH_SCALED))
            y_pos += 20 * scale  # Standardized spacing
            
            # Standardized font sizes
            font_section = get_font(int(FONT_SIZE_TABLE * scale), "thin")
            
            # Title - standardized size (slightly larger)
            font_title_large = get_font(int(22 * scale), "medium")  # Increased from 20pt to 22pt Medium
            title = str(artwork.get("title", "Unknown"))
            max_title_width = PANEL_WIDTH_SCALED - 60 * scale
            y_pos = draw_artwork_title(draw, title, panel_x, y_pos, max_title_width, colors, font_title_large, scale,
                                      previous_title=previous_info_title, ease_progress=info_title_ease_progress)
            
            # Artist and Year - standardized sizes
            artist = str(artwork.get("artist", "Unknown"))
            # Apply ASCII easing to artist if provided
            if previous_info_artist is not None and info_artist_ease_progress is not None:
                artist = ascii_text_ease(previous_info_artist, artist, info_artist_ease_progress)
            year = str(artwork.get("year", "N/A"))
            max_artist_width = PANEL_WIDTH_SCALED - 60 * scale
            # Draw artist with standardized font
            font_artist = get_font(int(16 * scale), "medium")  # Standardized: 16pt Medium
            artist_bbox = draw.textbbox((0, 0), artist, font=font_artist)
            if artist_bbox[2] - artist_bbox[0] > max_artist_width:
                while artist and draw.textbbox((0, 0), artist + "...", font=font_artist)[2] > max_artist_width:
                    artist = artist[:-1]
                artist = artist + "..." if artist else "..."
            draw.text((panel_x, y_pos), artist, fill=colors["text"], font=font_artist)
            y_pos += 22 * scale  # Standardized spacing
            # Year with standardized font (thin, no mono)
            font_year = get_font(int(FONT_SIZE_TABLE * scale), "thin", mono=False)  # Thin font for year
            draw.text((panel_x, y_pos), year, fill=colors["text"], font=font_year)
            y_pos += 25 * scale  # Standardized spacing
            
            # Divider before two-column fields - skip in ruler step (less separation lines)
            if step != "ruler":
                draw.line([(panel_x, y_pos), (CANVAS_WIDTH_SCALED - 30 * scale, y_pos)], 
                         fill=colors["text"], width=int(LINE_WIDTH_SCALED))
                y_pos += 20 * scale  # Standardized spacing
            else:
                y_pos += 10 * scale  # Less spacing in ruler step
            
            # Two column layout for remaining fields - standardized layout (distance included here)
            max_width = PANEL_WIDTH_SCALED - 60 * scale
            y_pos = draw_two_column_fields(
                draw, artwork, panel_x, y_pos, max_width,
                colors, font_small, font_section,
                scale=scale,
                all_embeddings=all_embeddings,
                artwork_idx=highlighted_artwork_idx,
                distances=distances,
                shelf0_mask=shelf0_mask
            )
            
            # Description (wrapped text) with label - standardized spacing
            description = str(artwork.get("description", ""))
            if description and description != "nan" and description.strip():
                y_pos += 10 * scale  # Standardized spacing
                # Skip divider line in ruler step (less separation lines)
                if step != "ruler":
                    draw.line([(panel_x, y_pos), (CANVAS_WIDTH_SCALED - 30 * scale, y_pos)], 
                             fill=colors["text"], width=int(LINE_WIDTH_SCALED))
                    y_pos += 20 * scale  # Standardized spacing
                else:
                    y_pos += 10 * scale  # Less spacing in ruler step
                # Label should be whiter, content should be monofont
                font_label_desc = get_font(int(FONT_SIZE_SMALL * scale), "thin")  # Label (not mono, whiter)
                font_content_desc = get_font(int(FONT_SIZE_SMALL * scale), "thin", mono=True)  # Content (mono)
                draw.text((panel_x, y_pos), "Description", fill=colors["text"], font=font_label_desc)
                y_pos += 20 * scale  # Standardized spacing
                max_width = PANEL_WIDTH_SCALED - 60 * scale
                words = description.split()
                lines = []
                current_line = ""
                for word in words:
                    test_line = current_line + (" " if current_line else "") + word
                    bbox = draw.textbbox((0, 0), test_line, font=font_content_desc)
                    if bbox[2] - bbox[0] <= max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)
                
                for line in lines[:6]:  # Limit to 6 lines
                    draw.text((panel_x, y_pos), line, fill=colors["text"], font=font_content_desc)
                    y_pos += 18 * scale  # Standardized spacing
                y_pos += 15 * scale  # Standardized spacing
            
            # Internal Note (wrapped text) with label - standardized spacing
            internal_note = str(artwork.get("internalNote", ""))
            if internal_note and internal_note != "nan" and internal_note.strip():
                draw.line([(panel_x, y_pos), (CANVAS_WIDTH_SCALED - 30 * scale, y_pos)], 
                         fill=colors["text"], width=int(LINE_WIDTH_SCALED))
                y_pos += 20 * scale  # Standardized spacing
                font_label_note = get_font(int(FONT_SIZE_SMALL * scale), "thin")  # Label (not mono, whiter)
                font_content_note = get_font(int(FONT_SIZE_SMALL * scale), "thin", mono=True)  # Content (mono)
                draw.text((panel_x, y_pos), "Internal Note", fill=colors["text"], font=font_label_note)
                y_pos += 20 * scale  # Standardized spacing
                max_width = PANEL_WIDTH_SCALED - 60 * scale
                words = internal_note.split()
                lines = []
                current_line = ""
                for word in words:
                    test_line = current_line + (" " if current_line else "") + word
                    bbox = draw.textbbox((0, 0), test_line, font=font_content_note)
                    if bbox[2] - bbox[0] <= max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)
                
                for line in lines[:4]:  # Limit to 4 lines
                    draw.text((panel_x, y_pos), line, fill=colors["text"], font=font_content_note)
                    y_pos += 18 * scale  # Standardized spacing
    
    # Draw top 10 representatives with images on circles (with easing support)
    if step == "representatives" and top_representatives is not None and shelf0_coords is not None:
        for rank, (artwork_idx_in_all, distance) in enumerate(top_representatives[:10]):
            if artwork_idx_in_all in shelf0_indices:
                idx_in_shelf0 = shelf0_indices_list.index(artwork_idx_in_all)
                if idx_in_shelf0 < len(shelf0_coords):
                    x, y = shelf0_coords[idx_in_shelf0]
                    
                    # Check if this is the aesthetic representative
                    is_aesthetic_rep = False
                    if aesthetic_representative_id is not None:
                        try:
                            aid = all_artwork_ids[artwork_idx_in_all]
                            if matches_artwork_id(aid, aesthetic_representative_id):
                                is_aesthetic_rep = True
                        except (ValueError, TypeError):
                            if matches_artwork_id(all_artwork_ids[artwork_idx_in_all], aesthetic_representative_id):
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
                            artwork = find_artwork_by_id(artwork_id, df, artwork_lookup)
                            
                            if artwork is not None:
                                thumbnail = load_image(artwork.get("thumbnail", ""))
                                
                                if thumbnail:
                                    # Resize image to fit circle
                                    size = POINT_SIZE_REPRESENTATIVE_SCALED * 2
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
                                    # Reposition slightly for better clarity (offset by 5px up and right)
                                    offset_x = 5
                                    offset_y = -5
                                    img.paste(thumbnail_resized, 
                                            (int(x - POINT_SIZE_REPRESENTATIVE_SCALED + offset_x), 
                                             int(y - POINT_SIZE_REPRESENTATIVE_SCALED + offset_y)), 
                                            mask)
                                    
                                    # Mark aesthetic representative with bright green circle (only if not already marked as main representative)
                                    # IMPORTANT: Do NOT draw green circle in "highlight_slow" (identify) step - only in later steps
                                    if is_aesthetic_rep and step != "highlight_slow":
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
                                            # Account for thumbnail offset
                                            offset_x = 5
                                            offset_y = -5
                                            outline_color = colors["lime"]
                                            if opacity < 1.0:
                                                outline_color = tuple(int(c * opacity) for c in colors["lime"])
                                            draw.ellipse([int(x - POINT_SIZE_REPRESENTATIVE_SCALED - 5 + offset_x), 
                                                        int(y - POINT_SIZE_REPRESENTATIVE_SCALED - 5 + offset_y),
                                                        int(x + POINT_SIZE_REPRESENTATIVE_SCALED + 5 + offset_x), 
                                                        int(y + POINT_SIZE_REPRESENTATIVE_SCALED + 5 + offset_y)],
                                                       fill=None, outline=outline_color, width=int(3 * scale))
                        except Exception:
                            pass
    
    
    # Draw ruler lines from centroid to representative and outlier (ruler step)
    if step == "ruler" and cx is not None and cy is not None and ruler_progress is not None and shelf0_coords is not None:
        # Find representative and outlier coordinates
        rep_coord = None
        outlier_coord = None
        rep_distance = None
        outlier_distance = None
        
        shelf0_indices_list_int = [int(x) for x in shelf0_indices_list]
        
        if aesthetic_representative_id is not None:
            for i, aid in enumerate(all_artwork_ids):
                try:
                    if (float(aid) == float(aesthetic_representative_id) or 
                        str(aid) == str(aesthetic_representative_id) or
                        str(aid).replace('.0', '') == str(aesthetic_representative_id).replace('.0', '')):
                        if i in shelf0_indices_list:
                            rep_idx_in_shelf0 = shelf0_indices_list.index(i)
                            if rep_idx_in_shelf0 < len(shelf0_coords):
                                rep_coord = shelf0_coords[rep_idx_in_shelf0]
                                if rep_idx_in_shelf0 < len(distances):
                                    rep_distance = distances[rep_idx_in_shelf0]
                        break
                except (ValueError, TypeError):
                    if str(aid) == str(aesthetic_representative_id):
                        if i in shelf0_indices_list:
                            rep_idx_in_shelf0 = shelf0_indices_list.index(i)
                            if rep_idx_in_shelf0 < len(shelf0_coords):
                                rep_coord = shelf0_coords[rep_idx_in_shelf0]
                                if rep_idx_in_shelf0 < len(distances):
                                    rep_distance = distances[rep_idx_in_shelf0]
                        break
        
        if aesthetic_outlier_id is not None:
            for i, aid in enumerate(all_artwork_ids):
                try:
                    if (float(aid) == float(aesthetic_outlier_id) or 
                        str(aid) == str(aesthetic_outlier_id) or
                        str(aid).replace('.0', '') == str(aesthetic_outlier_id).replace('.0', '')):
                        if i in shelf0_indices_list:
                            outlier_idx_in_shelf0 = shelf0_indices_list.index(i)
                            if outlier_idx_in_shelf0 < len(shelf0_coords):
                                outlier_coord = shelf0_coords[outlier_idx_in_shelf0]
                                if outlier_idx_in_shelf0 < len(distances):
                                    outlier_distance = distances[outlier_idx_in_shelf0]
                        break
                except (ValueError, TypeError):
                    if str(aid) == str(aesthetic_outlier_id):
                        if i in shelf0_indices_list:
                            outlier_idx_in_shelf0 = shelf0_indices_list.index(i)
                            if outlier_idx_in_shelf0 < len(shelf0_coords):
                                outlier_coord = shelf0_coords[outlier_idx_in_shelf0]
                                if outlier_idx_in_shelf0 < len(distances):
                                    outlier_distance = distances[outlier_idx_in_shelf0]
                        break
        
        # Draw ruler line(s) - both representative and outlier draw simultaneously
        targets_to_draw = []
        # Always draw both lines simultaneously with the same progress
        if rep_coord is not None and rep_distance is not None:
            targets_to_draw.append((rep_coord, rep_distance, True))  # (coord, distance, is_rep)
        if outlier_coord is not None and outlier_distance is not None:
            targets_to_draw.append((outlier_coord, outlier_distance, False))  # (coord, distance, is_rep)
        
        for target_coord, target_distance, is_rep in targets_to_draw:
            # Both lines use the same progress (draw simultaneously)
            line_progress = ruler_progress
            
            # Calculate intermediate point based on progress
            tx, ty = target_coord
            inter_x = cx + (tx - cx) * line_progress
            inter_y = cy + (ty - cy) * line_progress
            
            # Calculate line direction vector
            dx = tx - cx
            dy = ty - cy
            length = np.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                # Normalize direction
                dir_x = dx / length
                dir_y = dy / length
                perp_x = -dir_y  # Perpendicular vector (rotate 90 degrees)
                perp_y = dir_x
                
                # Draw main ruler line (thicker, more prominent)
                line_width = 3
                draw.line([(int(cx), int(cy)), (int(inter_x), int(inter_y))], 
                         fill=colors["lime"], width=line_width)
                
                # Draw tick marks along the line (ruler style with major/minor ticks)
                num_ticks = 20  # More ticks for better precision
                major_tick_interval = 5  # Every 5th tick is major
                for tick in range(1, num_ticks + 1):
                    tick_progress = (tick / (num_ticks + 1)) * line_progress
                    if tick_progress <= line_progress:
                        tick_x = cx + (tx - cx) * tick_progress
                        tick_y = cy + (ty - cy) * tick_progress
                        
                        # Major ticks are longer, minor ticks are shorter
                        is_major = (tick % major_tick_interval == 0)
                        tick_length = 12 if is_major else 6
                        tick_width = 2 if is_major else 1
                        
                        tick_start_x = tick_x - perp_x * tick_length
                        tick_start_y = tick_y - perp_y * tick_length
                        tick_end_x = tick_x + perp_x * tick_length
                        tick_end_y = tick_y + perp_y * tick_length
                        draw.line([(int(tick_start_x), int(tick_start_y)), 
                                  (int(tick_end_x), int(tick_end_y))], 
                                 fill=colors["lime"], width=tick_width)
                
                # Draw arrowhead at the end of the line (when line is complete or nearly complete)
                if line_progress > 0.8:
                    arrow_size = 12
                    arrow_x = inter_x
                    arrow_y = inter_y
                    
                    # Arrowhead points backward along the line
                    arrow_back_x = arrow_x - dir_x * arrow_size
                    arrow_back_y = arrow_y - dir_y * arrow_size
                    
                    # Arrowhead wings (perpendicular to line)
                    wing_length = arrow_size * 0.6
                    wing1_x = arrow_back_x + perp_x * wing_length
                    wing1_y = arrow_back_y + perp_y * wing_length
                    wing2_x = arrow_back_x - perp_x * wing_length
                    wing2_y = arrow_back_y - perp_y * wing_length
                    
                    # Draw arrowhead as filled triangle
                    arrow_points = [
                        (int(arrow_x), int(arrow_y)),
                        (int(wing1_x), int(wing1_y)),
                        (int(wing2_x), int(wing2_y))
                    ]
                    draw.polygon(arrow_points, fill=colors["lime"])
                
                # Draw distance label at midpoint of visible line (in green, with better styling)
                # Animate the appearance of the text box (fade in)
                if line_progress > 0.3:  # Only show label when line is partially drawn
                    # Use fixed midpoint of the full line (not animated midpoint)
                    label_x = cx + (tx - cx) * 0.5
                    label_y = cy + (ty - cy) * 0.5
                    
                    # Offset label perpendicular to line (below the line)
                    label_offset = 35 * scale
                    label_x += perp_x * label_offset
                    label_y += perp_y * label_offset
                    
                    # Draw distance text in green with better styling
                    distance_text = f"{target_distance:.4f}"
                    font_ruler = get_font(int(FONT_SIZE_INFO * scale), "medium")  # Smaller font for centroid numbers
                    
                    # Animate text box appearance (fade in from 0.3 to 0.5 progress)
                    fade_start = 0.3
                    fade_end = 0.5
                    if line_progress < fade_end:
                        # Calculate fade progress (0.0 to 1.0)
                        fade_progress = (line_progress - fade_start) / (fade_end - fade_start)
                        fade_progress = max(0.0, min(1.0, fade_progress))
                        # Use ease-in-out curve
                        fade_progress = fade_progress * fade_progress * (3 - 2 * fade_progress)
                    else:
                        fade_progress = 1.0
                    
                    # Draw text with rounded background for better readability
                    # First get bbox to center the text properly
                    bbox = draw.textbbox((0, 0), distance_text, font=font_ruler)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # Center the text on label_x, label_y
                    text_x = int(label_x - text_width / 2)
                    text_y = int(label_y - text_height / 2)
                    
                    # Recalculate bbox with centered position
                    bbox = draw.textbbox((text_x, text_y), distance_text, font=font_ruler)
                    padding = int(8 * scale)
                    # Draw background rectangle with rounded corners effect (using multiple rectangles)
                    bg_rect = [bbox[0] - padding, bbox[1] - padding, 
                              bbox[2] + padding, bbox[3] + padding]
                    
                    # Apply fade to background and outline
                    bg_alpha = int(255 * fade_progress)
                    outline_alpha = int(255 * fade_progress)
                    
                    # Blend background color with fade
                    bg_r = int(colors["background"][0] * fade_progress + colors["background"][0] * (1 - fade_progress))
                    bg_g = int(colors["background"][1] * fade_progress + colors["background"][1] * (1 - fade_progress))
                    bg_b = int(colors["background"][2] * fade_progress + colors["background"][2] * (1 - fade_progress))
                    
                    # For outline, blend lime with background
                    outline_r = int(colors["lime"][0] * fade_progress + colors["background"][0] * (1 - fade_progress))
                    outline_g = int(colors["lime"][1] * fade_progress + colors["background"][1] * (1 - fade_progress))
                    outline_b = int(colors["lime"][2] * fade_progress + colors["background"][2] * (1 - fade_progress))
                    
                    draw.rectangle(bg_rect, fill=(bg_r, bg_g, bg_b), outline=(outline_r, outline_g, outline_b), width=int(2 * scale))
                    
                    # Draw text on top with fade (centered)
                    text_r = int(colors["lime"][0] * fade_progress + colors["background"][0] * (1 - fade_progress))
                    text_g = int(colors["lime"][1] * fade_progress + colors["background"][1] * (1 - fade_progress))
                    text_b = int(colors["lime"][2] * fade_progress + colors["background"][2] * (1 - fade_progress))
                    draw.text((text_x, text_y), distance_text, 
                             fill=(text_r, text_g, text_b), font=font_ruler)
    
    # Make representative and outlier dots and text more visible during ruler step
    if step == "ruler" and shelf0_coords is not None and df is not None:
        shelf0_indices_list_int = [int(x) for x in shelf0_indices_list]
        
        # Find representative and outlier indices
        rep_idx_in_shelf0 = None
        outlier_idx_in_shelf0 = None
        
        if aesthetic_representative_id is not None:
            for i, aid in enumerate(all_artwork_ids):
                try:
                    if (float(aid) == float(aesthetic_representative_id) or 
                        str(aid) == str(aesthetic_representative_id) or
                        str(aid).replace('.0', '') == str(aesthetic_representative_id).replace('.0', '')):
                        if i in shelf0_indices_list:
                            rep_idx_in_shelf0 = shelf0_indices_list.index(i)
                        break
                except (ValueError, TypeError):
                    if str(aid) == str(aesthetic_representative_id):
                        if i in shelf0_indices_list:
                            rep_idx_in_shelf0 = shelf0_indices_list.index(i)
                        break
        
        if aesthetic_outlier_id is not None:
            for i, aid in enumerate(all_artwork_ids):
                try:
                    if (float(aid) == float(aesthetic_outlier_id) or 
                        str(aid) == str(aesthetic_outlier_id) or
                        str(aid).replace('.0', '') == str(aesthetic_outlier_id).replace('.0', '')):
                        if i in shelf0_indices_list:
                            outlier_idx_in_shelf0 = shelf0_indices_list.index(i)
                        break
                except (ValueError, TypeError):
                    if str(aid) == str(aesthetic_outlier_id):
                        if i in shelf0_indices_list:
                            outlier_idx_in_shelf0 = shelf0_indices_list.index(i)
                        break
        
        # Draw larger, brighter dots for representative and outlier
        # Use gradual expansion if ruler_circle_expand_progress is provided
        for idx_in_shelf0, coord in enumerate(shelf0_coords):
            if idx_in_shelf0 == rep_idx_in_shelf0 or idx_in_shelf0 == outlier_idx_in_shelf0:
                x, y = coord
                
                # Calculate circle size based on expansion progress
                if ruler_circle_expand_progress is not None:
                    # Start from blink end size (2.5x POINT_SIZE_LARGE_SCALED), expand to RULER_DOT_SIZE_SCALED
                    # This creates a smooth transition from the blink phase
                    blink_end_size = POINT_SIZE_LARGE_SCALED * 2.5
                    min_size = blink_end_size
                    max_size = RULER_DOT_SIZE_SCALED
                    # Use ease-in-out curve for smooth expansion
                    eased = ruler_circle_expand_progress * ruler_circle_expand_progress * (3 - 2 * ruler_circle_expand_progress)
                    # If RULER_DOT_SIZE is smaller than blink end size, shrink smoothly
                    # If larger, expand smoothly
                    current_size = min_size + (max_size - min_size) * eased
                else:
                    current_size = RULER_DOT_SIZE_SCALED
                
                # Draw larger bright green dot
                draw.ellipse([x - current_size, y - current_size, 
                            x + current_size, y + current_size],
                           fill=colors["lime"], outline=None, width=0)
                # Draw bright green circle around it (also scale with expansion)
                circle_outline_size = current_size + 5 * scale
                draw.ellipse([x - circle_outline_size, y - circle_outline_size, 
                            x + circle_outline_size, y + circle_outline_size],
                           fill=None, outline=colors["lime"], width=int(3 * scale))
        
        # Draw larger, more visible text labels for representative and outlier
        if rep_idx_in_shelf0 is not None and rep_idx_in_shelf0 < len(shelf0_coords):
            x, y = shelf0_coords[rep_idx_in_shelf0]
            try:
                artwork_idx_for_text = shelf0_indices_list[rep_idx_in_shelf0]
                artwork_id = all_artwork_ids[artwork_idx_for_text]
                artwork = find_artwork_by_id(artwork_id, df, artwork_lookup)
                if artwork is not None:
                    title = str(artwork.get("title", ""))
                    if title and title != "nan":
                        # Draw title text with larger font and background
                        # Use fade in effect if ruler_text_fade_progress is provided
                        text_x = x + RULER_DOT_SIZE_SCALED + 10 * scale
                        text_y = y - 15 * scale
                        font_ruler_text = get_font(int(18 * scale), "medium")  # Smaller font for ruler step
                        
                        # Calculate fade progress
                        fade_opacity = 1.0
                        if ruler_text_fade_progress is not None:
                            fade_opacity = max(0.0, min(1.0, ruler_text_fade_progress))
                            # Use ease-in-out curve
                            fade_opacity = fade_opacity * fade_opacity * (3 - 2 * fade_opacity)
                        
                        # Draw background for text (with fade)
                        bbox = draw.textbbox((text_x, text_y), title, font=font_ruler_text)
                        padding = int(6 * scale)
                        bg_rect = [bbox[0] - padding, bbox[1] - padding, 
                                  bbox[2] + padding, bbox[3] + padding]
                        
                        # Blend background and outline colors with fade
                        bg_r = int(colors["background"][0] * fade_opacity + colors["background"][0] * (1 - fade_opacity))
                        bg_g = int(colors["background"][1] * fade_opacity + colors["background"][1] * (1 - fade_opacity))
                        bg_b = int(colors["background"][2] * fade_opacity + colors["background"][2] * (1 - fade_opacity))
                        outline_r = int(colors["lime"][0] * fade_opacity + colors["background"][0] * (1 - fade_opacity))
                        outline_g = int(colors["lime"][1] * fade_opacity + colors["background"][1] * (1 - fade_opacity))
                        outline_b = int(colors["lime"][2] * fade_opacity + colors["background"][2] * (1 - fade_opacity))
                        text_r = int(colors["lime"][0] * fade_opacity + colors["background"][0] * (1 - fade_opacity))
                        text_g = int(colors["lime"][1] * fade_opacity + colors["background"][1] * (1 - fade_opacity))
                        text_b = int(colors["lime"][2] * fade_opacity + colors["background"][2] * (1 - fade_opacity))
                        
                        draw.rectangle(bg_rect, fill=(bg_r, bg_g, bg_b), outline=(outline_r, outline_g, outline_b), width=int(2 * scale))
                        draw.text((text_x, text_y), title, fill=(text_r, text_g, text_b), font=font_ruler_text)
            except Exception:
                pass
        
        if outlier_idx_in_shelf0 is not None and outlier_idx_in_shelf0 < len(shelf0_coords):
            x, y = shelf0_coords[outlier_idx_in_shelf0]
            try:
                artwork_idx_for_text = shelf0_indices_list[outlier_idx_in_shelf0]
                artwork_id = all_artwork_ids[artwork_idx_for_text]
                artwork_row = df[df["id"].astype(float) == float(artwork_id)]
                if artwork_row.empty:
                    artwork_row = df[df["id"].astype(str).str.strip() == str(artwork_id).strip()]
                if not artwork_row.empty:
                    title = str(artwork_row.iloc[0].get("title", ""))
                    if title and title != "nan":
                        # Draw title text with larger font and background
                        # Use fade in effect if ruler_text_fade_progress is provided
                        text_x = x + RULER_DOT_SIZE_SCALED + 10 * scale
                        text_y = y - 15 * scale
                        font_ruler_text = get_font(int(18 * scale), "medium")  # Smaller font for ruler step
                        
                        # Calculate fade progress
                        fade_opacity = 1.0
                        if ruler_text_fade_progress is not None:
                            fade_opacity = max(0.0, min(1.0, ruler_text_fade_progress))
                            # Use ease-in-out curve
                            fade_opacity = fade_opacity * fade_opacity * (3 - 2 * fade_opacity)
                        
                        # Draw background for text (with fade)
                        bbox = draw.textbbox((text_x, text_y), title, font=font_ruler_text)
                        padding = int(6 * scale)
                        bg_rect = [bbox[0] - padding, bbox[1] - padding, 
                                  bbox[2] + padding, bbox[3] + padding]
                        
                        # Blend background and outline colors with fade
                        bg_r = int(colors["background"][0] * fade_opacity + colors["background"][0] * (1 - fade_opacity))
                        bg_g = int(colors["background"][1] * fade_opacity + colors["background"][1] * (1 - fade_opacity))
                        bg_b = int(colors["background"][2] * fade_opacity + colors["background"][2] * (1 - fade_opacity))
                        outline_r = int(colors["lime"][0] * fade_opacity + colors["background"][0] * (1 - fade_opacity))
                        outline_g = int(colors["lime"][1] * fade_opacity + colors["background"][1] * (1 - fade_opacity))
                        outline_b = int(colors["lime"][2] * fade_opacity + colors["background"][2] * (1 - fade_opacity))
                        text_r = int(colors["lime"][0] * fade_opacity + colors["background"][0] * (1 - fade_opacity))
                        text_g = int(colors["lime"][1] * fade_opacity + colors["background"][1] * (1 - fade_opacity))
                        text_b = int(colors["lime"][2] * fade_opacity + colors["background"][2] * (1 - fade_opacity))
                        
                        draw.rectangle(bg_rect, fill=(bg_r, bg_g, bg_b), outline=(outline_r, outline_g, outline_b), width=int(2 * scale))
                        draw.text((text_x, text_y), title, fill=(text_r, text_g, text_b), font=font_ruler_text)
            except Exception:
                pass
    
    # Draw two selected works info on right side during ruler step - standardized layout
    if step == "ruler" and df is not None:
        panel_x = MAP_WIDTH_SCALED + 30 * scale
        y_pos = 60 * scale  # Standardized start position
        
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
                        # Use lookup if available
                        if artwork_lookup is not None:
                            rep_id_str = str(rep_id).strip()
                            rep_id_normalized = rep_id_str.replace('.0', '').strip()
                            lookup_idx = None
                            try:
                                rep_id_float = float(rep_id)
                                if rep_id_float in artwork_lookup:
                                    lookup_idx = artwork_lookup[rep_id_float]
                            except (ValueError, TypeError):
                                pass
                            if lookup_idx is None and rep_id_str in artwork_lookup:
                                lookup_idx = artwork_lookup[rep_id_str]
                            if lookup_idx is None and rep_id_normalized in artwork_lookup:
                                lookup_idx = artwork_lookup[rep_id_normalized]
                            if lookup_idx is not None:
                                rep_artwork = df.iloc[lookup_idx]
                        else:
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
                        # Use lookup if available
                        if artwork_lookup is not None:
                            rep_id_str = str(rep_id).strip()
                            rep_id_normalized = rep_id_str.replace('.0', '').strip()
                            lookup_idx = None
                            if rep_id_str in artwork_lookup:
                                lookup_idx = artwork_lookup[rep_id_str]
                            if lookup_idx is None and rep_id_normalized in artwork_lookup:
                                lookup_idx = artwork_lookup[rep_id_normalized]
                            if lookup_idx is not None:
                                rep_artwork = df.iloc[lookup_idx]
                        else:
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
                        # Use lookup if available
                        if artwork_lookup is not None:
                            outlier_id_str = str(outlier_id).strip()
                            outlier_id_normalized = outlier_id_str.replace('.0', '').strip()
                            lookup_idx = None
                            try:
                                outlier_id_float = float(outlier_id)
                                if outlier_id_float in artwork_lookup:
                                    lookup_idx = artwork_lookup[outlier_id_float]
                            except (ValueError, TypeError):
                                pass
                            if lookup_idx is None and outlier_id_str in artwork_lookup:
                                lookup_idx = artwork_lookup[outlier_id_str]
                            if lookup_idx is None and outlier_id_normalized in artwork_lookup:
                                lookup_idx = artwork_lookup[outlier_id_normalized]
                            if lookup_idx is not None:
                                outlier_artwork = df.iloc[lookup_idx]
                        else:
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
                        # Use lookup if available
                        if artwork_lookup is not None:
                            outlier_id_str = str(outlier_id).strip()
                            outlier_id_normalized = outlier_id_str.replace('.0', '').strip()
                            lookup_idx = None
                            if outlier_id_str in artwork_lookup:
                                lookup_idx = artwork_lookup[outlier_id_str]
                            if lookup_idx is None and outlier_id_normalized in artwork_lookup:
                                lookup_idx = artwork_lookup[outlier_id_normalized]
                            if lookup_idx is not None:
                                outlier_artwork = df.iloc[lookup_idx]
                        else:
                            outlier_row = df[df["id"].astype(str).str.strip() == str(outlier_id).strip()]
                            if not outlier_row.empty:
                                outlier_artwork = outlier_row.iloc[0]
                        break
        
        # Representative section (top half of right side) - standardized layout
        # Always show representative section if available (regardless of ruler_to_rep)
        if rep_artwork is not None:
            font_side = get_font(int(FONT_SIZE_TABLE * scale), "thin")
            font_title_large = get_font(int(20 * scale), "medium")  # Standardized: 20pt Medium
            
            # Start at 60px from top - standardized
            y_pos_rep = 60 * scale
            
            # Draw thumbnail at the very top first - standardized size
            rep_image = load_image(rep_artwork.get("thumbnail", ""))
            if rep_image:
                # Standardized fixed height, maintain aspect ratio (scale for supersampling)
                img_w, img_h = rep_image.size
                image_height_scaled = FIXED_IMAGE_HEIGHT * scale
                ratio = image_height_scaled / img_h
                new_w, new_h = int(img_w * ratio), int(image_height_scaled)
                rep_image = rep_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                # No border - paste image directly
                img_x = MAP_WIDTH_SCALED + (PANEL_WIDTH_SCALED - rep_image.width) // 2
                # Paste image - need to use alpha composite if image has alpha, otherwise direct paste
                if rep_image.mode == 'RGBA':
                    img.paste(rep_image, (int(img_x), int(y_pos_rep)), rep_image)
                else:
                    img.paste(rep_image, (int(img_x), int(y_pos_rep)))
                y_pos_rep += rep_image.height + 20 * scale  # Standardized spacing
            else:
                y_pos_rep += 20 * scale
            
            # Draw divider line after image - standardized spacing
            draw.line([(panel_x, y_pos_rep), (CANVAS_WIDTH_SCALED - 30 * scale, y_pos_rep)], 
                     fill=colors["text"], width=int(LINE_WIDTH_SCALED))
            y_pos_rep += 20 * scale  # Standardized spacing
            
            # Section label - standardized size
            font_section_label = get_font(int(20 * scale), "medium")  # Standardized: 20pt Medium
            draw.text((panel_x, y_pos_rep), "Closest to Center", fill=colors["lime"], font=font_section_label)
            y_pos_rep += 30 * scale  # Standardized spacing
            
            # Title - standardized size (with proper spacing)
            title = str(rep_artwork.get("title", "Unknown"))
            max_title_width = CANVAS_WIDTH_SCALED - panel_x - 20 * scale
            y_pos_rep = draw_artwork_title(draw, title, panel_x, y_pos_rep, max_title_width, colors, font_title_large, scale)
            
            # Artist and Year - standardized sizes (with proper spacing)
            artist = str(rep_artwork.get("artist", "Unknown"))
            year = str(rep_artwork.get("year", "N/A"))
            max_artist_width = CANVAS_WIDTH_SCALED - panel_x - 20 * scale
            # Draw artist with standardized font
            font_artist = get_font(int(16 * scale), "medium")  # Standardized: 16pt Medium
            artist_bbox = draw.textbbox((0, 0), artist, font=font_artist)
            if artist_bbox[2] - artist_bbox[0] > max_artist_width:
                while artist and draw.textbbox((0, 0), artist + "...", font=font_artist)[2] > max_artist_width:
                    artist = artist[:-1]
                artist = artist + "..." if artist else "..."
            draw.text((panel_x, y_pos_rep), artist, fill=colors["text"], font=font_artist)
            y_pos_rep += int(22 * scale)  # Standardized spacing (ensure int)
            # Year with standardized font (thin, no mono)
            font_year = get_font(int(FONT_SIZE_TABLE * scale), "thin", mono=False)  # Thin font for year
            draw.text((panel_x, y_pos_rep), year, fill=colors["text"], font=font_year)
            y_pos_rep += 25 * scale  # Standardized spacing
            
            # Divider before details - standardized layout
            draw.line([(panel_x, y_pos_rep), (CANVAS_WIDTH_SCALED - 30 * scale, y_pos_rep)], 
                     fill=colors["text"], width=int(LINE_WIDTH_SCALED))
            y_pos_rep += 20 * scale  # Standardized spacing
            
            # Two column layout for remaining fields (distance will be included here)
            max_width = CANVAS_WIDTH_SCALED - panel_x - 20 * scale  # Fixed: use scale
            y_pos_rep = draw_two_column_fields(
                draw, rep_artwork, panel_x, y_pos_rep, max_width,
                colors, font_small, font_side,
                scale=scale,
                all_embeddings=all_embeddings,
                artwork_idx=rep_idx,
                distances=distances,  # Show distance in two-column section
                shelf0_mask=shelf0_mask,
                show_raw_embedding=False  # Don't show Raw Embedding in ruler step
            )
        
        # Calculate divider position - split panel in half if both sections exist
        if rep_artwork is not None and outlier_artwork is not None:
            # Split panel vertically - representative gets top half, outlier gets bottom half
            divider_y = CANVAS_HEIGHT_SCALED // 2
            draw.line([(panel_x, divider_y), (CANVAS_WIDTH_SCALED - 30 * scale, divider_y)], fill=colors["text"], width=int(LINE_WIDTH_SCALED))
            y_pos_outlier = divider_y + 20 * scale  # Start outlier section below divider
        elif rep_artwork is not None:
            # Only representative - use normal spacing
            y_pos_outlier = y_pos_rep + 20 * scale
            draw.line([(panel_x, y_pos_outlier), (CANVAS_WIDTH_SCALED - 30 * scale, y_pos_outlier)], fill=colors["text"], width=int(LINE_WIDTH_SCALED))
            y_pos_outlier += 20 * scale
        else:
            # Only outlier or neither - start at top
            y_pos_outlier = 60 * scale
        
        # Outlier section (bottom half of right side) - standardized layout
        # Always show outlier section if available
        if outlier_artwork is not None:
            font_side = get_font(int(FONT_SIZE_TABLE * scale), "thin")
            font_title_large = get_font(int(20 * scale), "medium")  # Standardized: 20pt Medium
            
            # Use calculated y_pos_outlier from above
            y_pos = y_pos_outlier
            
            # Draw thumbnail at the very top first - standardized size
            outlier_image = load_image(outlier_artwork.get("thumbnail", ""))
            if outlier_image:
                # Standardized fixed height, maintain aspect ratio (scale for supersampling)
                img_w, img_h = outlier_image.size
                image_height_scaled = FIXED_IMAGE_HEIGHT * scale
                ratio = image_height_scaled / img_h
                new_w, new_h = int(img_w * ratio), int(image_height_scaled)
                outlier_image = outlier_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                # No border - paste image directly
                img_x = MAP_WIDTH_SCALED + (PANEL_WIDTH_SCALED - outlier_image.width) // 2
                # Paste image - need to use alpha composite if image has alpha, otherwise direct paste
                if outlier_image.mode == 'RGBA':
                    img.paste(outlier_image, (int(img_x), int(y_pos)), outlier_image)
                else:
                    img.paste(outlier_image, (int(img_x), int(y_pos)))
                y_pos += outlier_image.height + 20 * scale  # Standardized spacing
            else:
                y_pos += 20 * scale
            
            # Draw divider line after image - standardized spacing
            draw.line([(panel_x, y_pos), (CANVAS_WIDTH_SCALED - 30 * scale, y_pos)], 
                     fill=colors["text"], width=int(LINE_WIDTH_SCALED))
            y_pos += 20 * scale  # Standardized spacing
            
            # Section label - standardized size
            font_section_label = get_font(int(20 * scale), "medium")  # Standardized: 20pt Medium
            draw.text((panel_x, y_pos), "Furthest from Center", fill=colors["lime"], font=font_section_label)
            y_pos += 30 * scale  # Standardized spacing
            
            # Title - standardized size (with proper spacing)
            title = str(outlier_artwork.get("title", "Unknown"))
            max_title_width = CANVAS_WIDTH_SCALED - panel_x - 20 * scale
            y_pos = draw_artwork_title(draw, title, panel_x, y_pos, max_title_width, colors, font_title_large, scale)
            
            # Artist and Year - standardized sizes (with proper spacing)
            artist = str(outlier_artwork.get("artist", "Unknown"))
            year = str(outlier_artwork.get("year", "N/A"))
            max_artist_width = CANVAS_WIDTH_SCALED - panel_x - 20 * scale
            # Draw artist with standardized font
            font_artist = get_font(int(16 * scale), "medium")  # Standardized: 16pt Medium
            artist_bbox = draw.textbbox((0, 0), artist, font=font_artist)
            if artist_bbox[2] - artist_bbox[0] > max_artist_width:
                while artist and draw.textbbox((0, 0), artist + "...", font=font_artist)[2] > max_artist_width:
                    artist = artist[:-1]
                artist = artist + "..." if artist else "..."
            draw.text((panel_x, y_pos), artist, fill=colors["text"], font=font_artist)
            y_pos += int(22 * scale)  # Standardized spacing (ensure int)
            # Year with standardized font (thin, no mono)
            font_year = get_font(int(FONT_SIZE_TABLE * scale), "thin", mono=False)  # Thin font for year
            draw.text((panel_x, y_pos), year, fill=colors["text"], font=font_year)
            y_pos += 25 * scale  # Standardized spacing
            
            # Divider before details - standardized layout
            draw.line([(panel_x, y_pos), (CANVAS_WIDTH_SCALED - 30 * scale, y_pos)], 
                     fill=colors["text"], width=int(LINE_WIDTH_SCALED))
            y_pos += 20 * scale  # Standardized spacing
            
            # Two column layout for remaining fields (distance will be included here)
            max_width = CANVAS_WIDTH_SCALED - panel_x - 20 * scale
            y_pos = draw_two_column_fields(
                draw, outlier_artwork, panel_x, y_pos, max_width,
                colors, font_small, font_side,
                scale=scale,
                all_embeddings=all_embeddings,
                artwork_idx=outlier_idx,
                distances=distances,  # Show distance in two-column section
                shelf0_mask=shelf0_mask,
                show_raw_embedding=False  # Don't show Raw Embedding in ruler step
            )
    
    # Draw centroid with crosshair design (more elegant than big circle)
    if cx is not None and cy is not None:
        # Draw a small crosshair at centroid
        crosshair_size = 12 * scale
        line_length = 20 * scale
        # Draw horizontal line
        draw.line([(int(cx - line_length), int(cy)), (int(cx + line_length), int(cy))],
                 fill=colors["text"], width=int(2 * scale))
        # Draw vertical line
        draw.line([(int(cx), int(cy - line_length)), (int(cx), int(cy + line_length))],
                 fill=colors["text"], width=int(2 * scale))
        # Draw small circle at center
        draw.ellipse([int(cx - crosshair_size), int(cy - crosshair_size), 
                     int(cx + crosshair_size), int(cy + crosshair_size)],
                    fill=None, outline=colors["text"], width=int(2 * scale))
    
    # Draw table on right side for representatives step - standardized layout
    if step == "representatives" and top_representatives is not None and df is not None:
        panel_x = MAP_WIDTH_SCALED + 30 * scale
        y_pos = 60 * scale  # Standardized start position
        
        # Title - determine based on average distance
        if top_representatives is not None and len(top_representatives) > 0:
            avg_distance = sum(d for _, d in top_representatives[:5]) / min(5, len(top_representatives))
            table_title = "Top 10 Outliers" if avg_distance > 0.5 else "Top 10 Representatives"
        else:
            table_title = "Top 10 Representatives"
        draw.text((panel_x, y_pos), table_title, fill=colors["text"], font=font_title)
        y_pos += int(60 * scale)
        
        # Draw table header
        draw.text((panel_x, y_pos), "Rank", fill=colors["text"], font=font_label)
        draw.text((panel_x + int(60 * scale), y_pos), "Title", fill=colors["text"], font=font_label)
        draw.text((panel_x + int(300 * scale), y_pos), "Artist", fill=colors["text"], font=font_label)
        draw.text((panel_x + int(500 * scale), y_pos), "Distance", fill=colors["text"], font=font_label)
        y_pos += int(40 * scale)
        
        # Draw divider
        draw.line([(panel_x, y_pos), (CANVAS_WIDTH_SCALED - 30 * scale, y_pos)], fill=colors["text"], width=int(LINE_WIDTH_SCALED))
        y_pos += int(20 * scale)
        
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
                    is_aesthetic_rep = matches_artwork_id(artwork_id, aesthetic_representative_id) if aesthetic_representative_id is not None else False
                    
                    # Row color - green for aesthetic rep, white for others
                    row_color = colors["lime"] if is_aesthetic_rep else colors["text"]
                    
                    # Rank
                    draw.text((panel_x, y_pos), f"{rank + 1}", fill=row_color, font=font_info)
                    
                    # Title (truncate if too long)
                    title = str(artwork.get("title", "Unknown"))[:30]
                    draw.text((panel_x + int(60 * scale), y_pos), title, fill=row_color, font=font_small)
                    
                    # Artist (truncate if too long)
                    artist = str(artwork.get("artist", "Unknown"))[:20]
                    draw.text((panel_x + int(300 * scale), y_pos), artist, fill=row_color, font=font_small)
                    
                    # Distance
                    draw.text((panel_x + int(500 * scale), y_pos), f"{distance:.4f}", fill=row_color, font=font_small)
                    
                    # Small thumbnail
                    thumbnail = load_image(artwork.get("thumbnail", ""))
                    if thumbnail:
                        thumb_size = int(30 * scale)
                        thumbnail_resized = thumbnail.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                        img.paste(thumbnail_resized, (panel_x + int(420 * scale), int(y_pos - 5 * scale)))
                    
                    y_pos += int(50 * scale)
            except Exception:
                pass
    
    # Draw green lines between top 10 items (5 representatives + 5 outliers)
    if step == "top10" and top_representatives is not None and top_outliers is not None and shelf0_coords is not None:
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
                    draw.line([(x1, y1), (x2, y2)], fill=colors["lime"], width=int(LINE_WIDTH_SCALED))
        
        # Draw larger circles for selected items (representative and outlier) during blink
        # They should grow larger on the left side (map) during blink
        if top10_blink_progress is not None and df is not None:
            # Find selected representative and outlier coordinates
            selected_rep_coord = None
            selected_outlier_coord = None
            
            # Find representative
            if aesthetic_representative_id is not None:
                for rank, (artwork_idx_in_all, distance) in enumerate(top_representatives[:5]):
                    if artwork_idx_in_all in shelf0_indices:
                        try:
                            artwork_id = all_artwork_ids[artwork_idx_in_all]
                            if (float(artwork_id) == float(aesthetic_representative_id) or 
                                str(artwork_id) == str(aesthetic_representative_id)):
                                idx_in_shelf0 = list(shelf0_indices).index(artwork_idx_in_all)
                                if idx_in_shelf0 < len(shelf0_coords):
                                    selected_rep_coord = shelf0_coords[idx_in_shelf0]
                                    break
                        except (ValueError, TypeError):
                            pass
            
            # Find outlier
            if aesthetic_outlier_id is not None:
                for rank, (artwork_idx_in_all, distance) in enumerate(top_outliers[:5]):
                    if artwork_idx_in_all in shelf0_indices:
                        try:
                            artwork_id = all_artwork_ids[artwork_idx_in_all]
                            if (float(artwork_id) == float(aesthetic_outlier_id) or 
                                str(artwork_id) == str(aesthetic_outlier_id)):
                                idx_in_shelf0 = list(shelf0_indices).index(artwork_idx_in_all)
                                if idx_in_shelf0 < len(shelf0_coords):
                                    selected_outlier_coord = shelf0_coords[idx_in_shelf0]
                                    break
                        except (ValueError, TypeError):
                            pass
            
            # Draw larger circles for selected items with growth animation
            # Color: oscillates gray->green->gray->green, ending at green (1.0)
            # Size: grows continuously from base to max, stays at max at end
            blink_cycle = (np.sin(top10_blink_progress * np.pi * 4) + 1) / 2  # 0 to 1, oscillates
            # Ensure it ends at green (1.0) - at progress=1.0, sin(4π)=0, so cycle=0.5
            # Adjust: use last quarter of blink to fade to green
            if top10_blink_progress > 0.75:
                # Fade to green in last quarter
                fade_to_green = (top10_blink_progress - 0.75) / 0.25  # 0 to 1
                blink_cycle = 0.5 + 0.5 * fade_to_green  # 0.5 to 1.0
            
            # Size grows continuously and stays at max
            base_size = POINT_SIZE_LARGE_SCALED
            max_size = POINT_SIZE_LARGE_SCALED * 2.5  # 2.5x larger
            # Use eased growth: start slow, end fast, stay at max
            size_progress = min(1.0, top10_blink_progress * 1.2)  # Slightly faster growth
            size_progress = size_progress * size_progress * (3 - 2 * size_progress)  # Ease in-out
            current_size = base_size + (max_size - base_size) * size_progress
            
            # Draw for representative
            if selected_rep_coord is not None:
                x, y = selected_rep_coord
                # Draw larger green dot
                draw.ellipse([x - current_size, y - current_size, 
                            x + current_size, y + current_size],
                           fill=colors["lime"], outline=None, width=0)
                # Draw bright green circle around it
                circle_outline_size = current_size + 5 * scale
                draw.ellipse([x - circle_outline_size, y - circle_outline_size, 
                            x + circle_outline_size, y + circle_outline_size],
                           fill=None, outline=colors["lime"], width=int(3 * scale))
            
            # Draw for outlier
            if selected_outlier_coord is not None:
                x, y = selected_outlier_coord
                # Draw larger green dot
                draw.ellipse([x - current_size, y - current_size, 
                            x + current_size, y + current_size],
                           fill=colors["lime"], outline=None, width=0)
                # Draw bright green circle around it
                circle_outline_size = current_size + 5 * scale
                draw.ellipse([x - circle_outline_size, y - circle_outline_size, 
                            x + circle_outline_size, y + circle_outline_size],
                           fill=None, outline=colors["lime"], width=int(3 * scale))
    
    # Draw top 10 table (5 representatives + 5 outliers) on right side
    if step == "top10" and top_representatives is not None and top_outliers is not None and df is not None:
        panel_x = MAP_WIDTH_SCALED + 30 * scale
        y_pos = 60 * scale  # Higher up
        
        # Title - larger, more prominent with medium weight (changed from "Top 10")
        font_section_title = get_font(int(32 * scale), "medium")  # Larger, medium weight for prominence
        draw.text((panel_x, y_pos), "Closest to Center", fill=colors["lime"], font=font_section_title)
        y_pos += int(50 * scale)  # Less spacing
        
        # Representatives section header (use monofont for labels)
        font_table_label = get_font(int(FONT_SIZE_TABLE * scale), "thin")  # Labels
        font_table_content = get_font(int(FONT_SIZE_TABLE * scale), "light", mono=False)  # Content (light, no mono)
        draw.text((panel_x + int(80 * scale), y_pos), "Rank", fill=colors["text"], font=font_table_label)
        draw.text((panel_x + int(140 * scale), y_pos), "Title", fill=colors["text"], font=font_table_label)
        draw.text((panel_x + int(480 * scale), y_pos), "Artist", fill=colors["text"], font=font_table_label)
        draw.text((panel_x + int(600 * scale), y_pos), "Distance", fill=colors["text"], font=font_table_label)
        y_pos += int(40 * scale)
        
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
                    # Check if this is the selected representative OR outlier
                    is_selected_rep = matches_artwork_id(artwork_id, aesthetic_representative_id) if aesthetic_representative_id is not None else False
                    is_selected_outlier = matches_artwork_id(artwork_id, aesthetic_outlier_id) if aesthetic_outlier_id is not None else False
                    is_selected = is_selected_rep or is_selected_outlier
                    
                    # Apply blink effect for selected items if in blink phase
                    if is_selected and top10_blink_progress is not None:
                        # Blink between gray and green (0.0 = gray, 1.0 = green)
                        # Use sine wave for smooth blink: 0 -> 1 -> 0 -> 1, ending at green
                        blink_cycle = (np.sin(top10_blink_progress * np.pi * 4) + 1) / 2  # 0 to 1, oscillates
                        # Ensure it ends at green (1.0) - fade to green in last quarter
                        if top10_blink_progress > 0.75:
                            fade_to_green = (top10_blink_progress - 0.75) / 0.25  # 0 to 1
                            blink_cycle = 0.5 + 0.5 * fade_to_green  # 0.5 to 1.0
                        # Interpolate between text color (gray) and lime green
                        text_r = int(colors["text"][0] * (1 - blink_cycle) + colors["lime"][0] * blink_cycle)
                        text_g = int(colors["text"][1] * (1 - blink_cycle) + colors["lime"][1] * blink_cycle)
                        text_b = int(colors["text"][2] * (1 - blink_cycle) + colors["lime"][2] * blink_cycle)
                        row_color = (text_r, text_g, text_b)
                    else:
                        # Selected items use same color as others (no green highlighting)
                        row_color = colors["text"]
                    
                    # Draw thumbnail on the left (bigger size)
                    thumbnail = load_image(artwork.get("thumbnail", ""))
                    if thumbnail:
                        thumb_size = int(60 * scale)  # Bigger thumbnail size
                        thumbnail_resized = thumbnail.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                        # Position thumbnail before rank number
                        thumb_x = panel_x
                        thumb_y = int(y_pos - 5 * scale)
                        # Handle transparent images
                        if thumbnail_resized.mode == "RGBA":
                            img.paste(thumbnail_resized, (int(thumb_x), thumb_y), thumbnail_resized)
                        else:
                            img.paste(thumbnail_resized, (int(thumb_x), thumb_y))
                    
                    # Use monofont for content
                    draw.text((panel_x + int(80 * scale), y_pos), f"{rank + 1}", fill=row_color, font=font_table_content)
                    
                    # Title with text wrapping (constraint box: from 140 to 480)
                    title = str(artwork.get("title", "Unknown"))
                    title_x = panel_x + int(140 * scale)
                    # Calculate width correctly: end position (480) - start position (140) - margin (10)
                    title_max_width = int((480 - 140 - 10) * scale)  # 330px width for title column
                    title_words = title.split()
                    title_lines = []
                    current_line = ""
                    for word in title_words:
                        test_line = current_line + (" " if current_line else "") + word
                        bbox = draw.textbbox((0, 0), test_line, font=font_table_content)
                        if bbox[2] - bbox[0] <= title_max_width:
                            current_line = test_line
                        else:
                            # If current_line is empty, the word itself is too long - truncate it
                            if not current_line:
                                # Measure single word and truncate if needed
                                word_bbox = draw.textbbox((0, 0), word, font=font_table_content)
                                if word_bbox[2] - word_bbox[0] > title_max_width:
                                    # Truncate long word
                                    truncated = word
                                    while truncated and draw.textbbox((0, 0), truncated + "...", font=font_table_content)[2] > title_max_width:
                                        truncated = truncated[:-1]
                                    current_line = truncated + "..." if truncated else "..."
                                    title_lines.append(current_line)
                                    current_line = ""
                                else:
                                    current_line = word
                            else:
                                # Save current line and start new one with this word
                                title_lines.append(current_line)
                                current_line = word
                    if current_line:
                        title_lines.append(current_line)
                    # Draw title (max 2 lines, but allow more if title is very long)
                    title_y = y_pos
                    max_title_lines = 3 if len(title_lines) > 2 else 2  # Allow 3 lines if needed
                    for line in title_lines[:max_title_lines]:
                        draw.text((title_x, title_y), line, fill=row_color, font=font_table_content)
                        title_y += int(18 * scale)
                    
                    # Artist with text wrapping (constraint box: from 480 to 600)
                    artist = str(artwork.get("artist", "Unknown"))
                    artist_x = panel_x + int(480 * scale)
                    # Calculate width correctly: end position (600) - start position (480) - margin (10)
                    artist_max_width = int((600 - 480 - 10) * scale)  # 110px width for artist column
                    artist_words = artist.split()
                    artist_lines = []
                    current_line = ""
                    for word in artist_words:
                        test_line = current_line + (" " if current_line else "") + word
                        bbox = draw.textbbox((0, 0), test_line, font=font_table_content)
                        if bbox[2] - bbox[0] <= artist_max_width:
                            current_line = test_line
                        else:
                            if current_line:
                                artist_lines.append(current_line)
                            current_line = word
                    if current_line:
                        artist_lines.append(current_line)
                    # Draw artist (max 2 lines)
                    artist_y = y_pos
                    for line in artist_lines[:2]:
                        draw.text((artist_x, artist_y), line, fill=row_color, font=font_table_content)
                        artist_y += int(18 * scale)
                    
                    # Distance
                    draw.text((panel_x + int(600 * scale), y_pos), f"{distance:.4f}", fill=row_color, font=font_table_content)
                    
                    # Calculate row height based on max lines (title or artist)
                    max_lines = max(len(title_lines[:2]), len(artist_lines[:2]), 1)
                    y_pos += int(max(70 * scale, max_lines * 18 * scale + 10 * scale))  # At least 70px, or based on content
            except Exception:
                pass
        
        y_pos += int(20 * scale)
        draw.line([(panel_x, y_pos), (CANVAS_WIDTH_SCALED - 30 * scale, y_pos)], fill=colors["text"], width=int(LINE_WIDTH_SCALED))
        y_pos += int(30 * scale)
        
        # Outliers section title - larger, more prominent with medium weight (changed from "Outliers")
        font_section_title = get_font(int(32 * scale), "medium")  # Larger, medium weight for prominence
        draw.text((panel_x, y_pos), "Furthest from Center", fill=colors["lime"], font=font_section_title)
        y_pos += int(50 * scale)  # Less spacing
        
        # Outliers section header (use monofont for labels)
        font_table_label = get_font(int(FONT_SIZE_TABLE * scale), "thin")  # Labels
        font_table_content = get_font(int(FONT_SIZE_TABLE * scale), "light", mono=False)  # Content (light, no mono)
        draw.text((panel_x + int(80 * scale), y_pos), "Rank", fill=colors["text"], font=font_table_label)
        draw.text((panel_x + int(140 * scale), y_pos), "Title", fill=colors["text"], font=font_table_label)
        draw.text((panel_x + int(480 * scale), y_pos), "Artist", fill=colors["text"], font=font_table_label)
        draw.text((panel_x + int(600 * scale), y_pos), "Distance", fill=colors["text"], font=font_table_label)
        y_pos += int(40 * scale)
        
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
                    # Check if this is the selected representative OR outlier
                    is_selected_rep = matches_artwork_id(artwork_id, aesthetic_representative_id) if aesthetic_representative_id is not None else False
                    is_selected_outlier = matches_artwork_id(artwork_id, aesthetic_outlier_id) if aesthetic_outlier_id is not None else False
                    is_selected = is_selected_rep or is_selected_outlier
                    
                    # Apply blink effect for selected items if in blink phase
                    if is_selected and top10_blink_progress is not None:
                        # Blink between gray and green (0.0 = gray, 1.0 = green)
                        # Use sine wave for smooth blink: 0 -> 1 -> 0 -> 1, ending at green
                        blink_cycle = (np.sin(top10_blink_progress * np.pi * 4) + 1) / 2  # 0 to 1, oscillates
                        # Ensure it ends at green (1.0) - fade to green in last quarter
                        if top10_blink_progress > 0.75:
                            fade_to_green = (top10_blink_progress - 0.75) / 0.25  # 0 to 1
                            blink_cycle = 0.5 + 0.5 * fade_to_green  # 0.5 to 1.0
                        # Interpolate between text color (gray) and lime green
                        text_r = int(colors["text"][0] * (1 - blink_cycle) + colors["lime"][0] * blink_cycle)
                        text_g = int(colors["text"][1] * (1 - blink_cycle) + colors["lime"][1] * blink_cycle)
                        text_b = int(colors["text"][2] * (1 - blink_cycle) + colors["lime"][2] * blink_cycle)
                        row_color = (text_r, text_g, text_b)
                    else:
                        # Selected items use same color as others (no green highlighting)
                        row_color = colors["text"]
                    
                    # Draw thumbnail on the left (bigger size)
                    thumbnail = load_image(artwork.get("thumbnail", ""))
                    if thumbnail:
                        thumb_size = int(60 * scale)  # Bigger thumbnail size
                        thumbnail_resized = thumbnail.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                        # Position thumbnail before rank number
                        thumb_x = panel_x
                        thumb_y = int(y_pos - 5 * scale)
                        # Handle transparent images
                        if thumbnail_resized.mode == "RGBA":
                            img.paste(thumbnail_resized, (int(thumb_x), thumb_y), thumbnail_resized)
                        else:
                            img.paste(thumbnail_resized, (int(thumb_x), thumb_y))
                    
                    # Use monofont for content
                    draw.text((panel_x + int(80 * scale), y_pos), f"{rank + 1}", fill=row_color, font=font_table_content)
                    
                    # Title with text wrapping (constraint box: from 140 to 480)
                    title = str(artwork.get("title", "Unknown"))
                    title_x = panel_x + int(140 * scale)
                    # Calculate width correctly: end position (480) - start position (140) - margin (10)
                    title_max_width = int((480 - 140 - 10) * scale)  # 330px width for title column
                    title_words = title.split()
                    title_lines = []
                    current_line = ""
                    for word in title_words:
                        test_line = current_line + (" " if current_line else "") + word
                        bbox = draw.textbbox((0, 0), test_line, font=font_table_content)
                        if bbox[2] - bbox[0] <= title_max_width:
                            current_line = test_line
                        else:
                            # If current_line is empty, the word itself is too long - truncate it
                            if not current_line:
                                # Measure single word and truncate if needed
                                word_bbox = draw.textbbox((0, 0), word, font=font_table_content)
                                if word_bbox[2] - word_bbox[0] > title_max_width:
                                    # Truncate long word
                                    truncated = word
                                    while truncated and draw.textbbox((0, 0), truncated + "...", font=font_table_content)[2] > title_max_width:
                                        truncated = truncated[:-1]
                                    current_line = truncated + "..." if truncated else "..."
                                    title_lines.append(current_line)
                                    current_line = ""
                                else:
                                    current_line = word
                            else:
                                # Save current line and start new one with this word
                                title_lines.append(current_line)
                                current_line = word
                    if current_line:
                        title_lines.append(current_line)
                    # Draw title (max 2 lines, but allow more if title is very long)
                    title_y = y_pos
                    max_title_lines = 3 if len(title_lines) > 2 else 2  # Allow 3 lines if needed
                    for line in title_lines[:max_title_lines]:
                        draw.text((title_x, title_y), line, fill=row_color, font=font_table_content)
                        title_y += int(18 * scale)
                    
                    # Artist with text wrapping (constraint box: from 480 to 600)
                    artist = str(artwork.get("artist", "Unknown"))
                    artist_x = panel_x + int(480 * scale)
                    # Calculate width correctly: end position (600) - start position (480) - margin (10)
                    artist_max_width = int((600 - 480 - 10) * scale)  # 110px width for artist column
                    artist_words = artist.split()
                    artist_lines = []
                    current_line = ""
                    for word in artist_words:
                        test_line = current_line + (" " if current_line else "") + word
                        bbox = draw.textbbox((0, 0), test_line, font=font_table_content)
                        if bbox[2] - bbox[0] <= artist_max_width:
                            current_line = test_line
                        else:
                            if current_line:
                                artist_lines.append(current_line)
                            current_line = word
                    if current_line:
                        artist_lines.append(current_line)
                    # Draw artist (max 2 lines)
                    artist_y = y_pos
                    for line in artist_lines[:2]:
                        draw.text((artist_x, artist_y), line, fill=row_color, font=font_table_content)
                        artist_y += int(18 * scale)
                    
                    # Distance
                    draw.text((panel_x + int(600 * scale), y_pos), f"{distance:.4f}", fill=row_color, font=font_table_content)
                    
                    # Calculate row height based on max lines (title or artist)
                    max_lines = max(len(title_lines[:2]), len(artist_lines[:2]), 1)
                    y_pos += int(max(70 * scale, max_lines * 18 * scale + 10 * scale))  # At least 70px, or based on content
            except Exception:
                pass
    
        # Draw side-by-side view of selected items on left screen, rank table on right - standardized layout
        if step == "side_by_side" and df is not None:
            # Clear left screen area (map area)
            draw.rectangle([0, 0, MAP_WIDTH_SCALED, MAP_HEIGHT_SCALED], fill=colors["background"])
            
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
            
            # Draw both items on left side, stacked vertically, each with horizontal layout (image left, info right)
            left_x = 50 * scale
            y_start = 100 * scale
            item_width = MAP_WIDTH_SCALED - 100 * scale  # Use most of left side width
            item_height = (MAP_HEIGHT_SCALED - 150 * scale) // 2  # Split height for two items
            
            # Representative (top half of left side) - horizontal layout
            if rep_artwork is not None:
                rep_y = y_start
                img_x = left_x
                info_x = left_x + 300 * scale  # Image takes ~300px, info starts after (scaled)
                
                # Image on the left - bigger size
                rep_image = load_image(rep_artwork.get("thumbnail", ""))
                if rep_image:
                    img_w, img_h = rep_image.size
                    fixed_img_height = min(int(item_height - 40 * scale), int(300 * scale))  # Bigger image (scaled)
                    ratio = fixed_img_height / img_h
                    new_w, new_h = int(img_w * ratio), fixed_img_height
                    if new_w > 280 * scale:  # Limit image width (scaled)
                        ratio = (280 * scale) / img_w
                        new_w, new_h = int(img_w * ratio), int(img_h * ratio)
                    rep_image = rep_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    # Handle transparent images
                    if rep_image.mode == "RGBA":
                        img.paste(rep_image, (int(img_x), int(rep_y)), rep_image)
                    else:
                        img.paste(rep_image, (int(img_x), int(rep_y)))
                
                # Info on the right side of image
                info_y = rep_y
                
                # Title - use green for label
                title = str(rep_artwork.get("title", "Unknown"))
                draw.text((info_x, info_y), "Closest to Center", fill=colors["lime"], font=font_label)
                info_y += int(35 * scale)
                # Wrap title if needed - constrain to MAP_WIDTH_SCALED
                max_title_width = MAP_WIDTH_SCALED - info_x - 20 * scale
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
                
                # Use smaller font for side-by-side view
                font_side = get_font(int(FONT_SIZE_TABLE * scale), "thin")
                for line in title_lines[:2]:  # Max 2 lines
                    draw.text((info_x, info_y), line, fill=colors["text"], font=font_side)
                    info_y += int(25 * scale)
                info_y += int(10 * scale)
                
                # Artist - truncate if too long (use smaller font)
                artist = str(rep_artwork.get("artist", "Unknown"))
                draw.text((info_x, info_y), "Artist", fill=colors["text"], font=font_small)
                info_y += int(18 * scale)
                max_artist_width = MAP_WIDTH_SCALED - info_x - 20 * scale
                artist_bbox = draw.textbbox((0, 0), artist, font=font_side)
                if artist_bbox[2] - artist_bbox[0] > max_artist_width:
                    # Truncate artist name
                    while artist and draw.textbbox((0, 0), artist + "...", font=font_side)[2] > max_artist_width:
                        artist = artist[:-1]
                    artist = artist + "..." if artist else "..."
                draw.text((info_x, info_y), artist, fill=colors["text"], font=font_side)
                info_y += int(25 * scale)
                
                # Year
                year = str(rep_artwork.get("year", "N/A"))
                draw.text((info_x, info_y), "Year", fill=colors["text"], font=font_small)
                info_y += int(18 * scale)
                draw.text((info_x, info_y), year, fill=colors["text"], font=font_side)
                info_y += int(25 * scale)
                
                # Size - truncate if too long
                size = str(rep_artwork.get("size", "N/A"))
                if size and size != "N/A" and size != "nan":
                    draw.text((info_x, info_y), "Size", fill=colors["text"], font=font_small)
                    info_y += int(18 * scale)
                    max_size_width = MAP_WIDTH_SCALED - info_x - 20 * scale
                    size_bbox = draw.textbbox((0, 0), size, font=font_side)
                    if size_bbox[2] - size_bbox[0] > max_size_width:
                        # Truncate size
                        while size and draw.textbbox((0, 0), size + "...", font=font_side)[2] > max_size_width:
                            size = size[:-1]
                        size = size + "..." if size else "..."
                    draw.text((info_x, info_y), size, fill=colors["text"], font=font_side)
                    info_y += int(25 * scale)
                
                # Distance to centroid
                if distances is not None and rep_idx is not None:
                    if rep_idx in shelf0_indices:
                        shelf0_idx = shelf0_indices_list.index(rep_idx)
                        if shelf0_idx < len(distances):
                            distance = distances[shelf0_idx]
                            draw.text((info_x, info_y), "Distance to Centroid", fill=colors["text"], font=font_small)
                            info_y += int(18 * scale)
                            draw.text((info_x, info_y), f"{distance:.4f}", fill=colors["lime"], font=font_side)
                            info_y += int(25 * scale)
                
                # Description (wrapped text)
                description = str(rep_artwork.get("description", ""))
                if description and description != "nan" and description.strip():
                    draw.line([(info_x, info_y), (MAP_WIDTH_SCALED - 50 * scale, info_y)], fill=colors["text"], width=int(LINE_WIDTH_SCALED))
                    info_y += int(15 * scale)
                    draw.text((info_x, info_y), "Description", fill=colors["text"], font=font_small)
                    info_y += int(18 * scale)
                    max_width = MAP_WIDTH_SCALED - info_x - 20 * scale
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
                        draw.text((info_x, info_y), line, fill=colors["text"], font=font_small)
                        info_y += int(16 * scale)
            
            # Draw divider between representative and outlier on left (aligned to exact center)
            # Calculate exact center of left side area
            divider_y = MAP_HEIGHT_SCALED // 2  # Exact center of canvas height
            divider_start_x = left_x
            divider_end_x = MAP_WIDTH_SCALED - 50 * scale
            draw.line([(divider_start_x, divider_y), (divider_end_x, divider_y)], fill=colors["text"], width=int(LINE_WIDTH_SCALED))
            
            # Outlier (bottom half of left side) - horizontal layout
            if outlier_artwork is not None:
                outlier_y = divider_y + 20 * scale
                img_x = left_x
                info_x = left_x + 300 * scale  # Image takes ~300px, info starts after (scaled)
                
                # Image on the left - bigger size
                outlier_image = load_image(outlier_artwork.get("thumbnail", ""))
                if outlier_image:
                    img_w, img_h = outlier_image.size
                    fixed_img_height = min(int(item_height - 40 * scale), int(300 * scale))  # Bigger image (scaled)
                    ratio = fixed_img_height / img_h
                    new_w, new_h = int(img_w * ratio), fixed_img_height
                    if new_w > 280 * scale:  # Limit image width (scaled)
                        ratio = (280 * scale) / img_w
                        new_w, new_h = int(img_w * ratio), int(img_h * ratio)
                    outlier_image = outlier_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    # Handle transparent images
                    if outlier_image.mode == "RGBA":
                        img.paste(outlier_image, (int(img_x), int(outlier_y)), outlier_image)
                    else:
                        img.paste(outlier_image, (int(img_x), int(outlier_y)))
                
                # Info on the right side of image
                info_y = outlier_y
                
                # Title - use green for label
                title = str(outlier_artwork.get("title", "Unknown"))
                draw.text((info_x, info_y), "Furthest from Center", fill=colors["lime"], font=font_label)
                info_y += int(35 * scale)
                # Wrap title if needed - constrain to MAP_WIDTH_SCALED
                max_title_width = MAP_WIDTH_SCALED - info_x - 20 * scale
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
                
                # Use smaller font for side-by-side view
                font_side = get_font(int(FONT_SIZE_TABLE * scale), "thin")
                for line in title_lines[:2]:  # Max 2 lines
                    draw.text((info_x, info_y), line, fill=colors["text"], font=font_side)
                    info_y += int(25 * scale)
                info_y += int(10 * scale)
                
                # Artist - truncate if too long (use smaller font)
                artist = str(outlier_artwork.get("artist", "Unknown"))
                draw.text((info_x, info_y), "Artist", fill=colors["text"], font=font_small)
                info_y += int(18 * scale)
                max_artist_width = MAP_WIDTH_SCALED - info_x - 20 * scale
                artist_bbox = draw.textbbox((0, 0), artist, font=font_side)
                if artist_bbox[2] - artist_bbox[0] > max_artist_width:
                    # Truncate artist name
                    while artist and draw.textbbox((0, 0), artist + "...", font=font_side)[2] > max_artist_width:
                        artist = artist[:-1]
                    artist = artist + "..." if artist else "..."
                draw.text((info_x, info_y), artist, fill=colors["text"], font=font_side)
                info_y += int(25 * scale)
                
                # Year
                year = str(outlier_artwork.get("year", "N/A"))
                draw.text((info_x, info_y), "Year", fill=colors["text"], font=font_small)
                info_y += int(18 * scale)
                draw.text((info_x, info_y), year, fill=colors["text"], font=font_side)
                info_y += int(25 * scale)
                
                # Size - truncate if too long
                size = str(outlier_artwork.get("size", "N/A"))
                if size and size != "N/A" and size != "nan":
                    draw.text((info_x, info_y), "Size", fill=colors["text"], font=font_small)
                    info_y += int(18 * scale)
                    max_size_width = MAP_WIDTH_SCALED - info_x - 20 * scale
                    size_bbox = draw.textbbox((0, 0), size, font=font_side)
                    if size_bbox[2] - size_bbox[0] > max_size_width:
                        # Truncate size
                        while size and draw.textbbox((0, 0), size + "...", font=font_side)[2] > max_size_width:
                            size = size[:-1]
                        size = size + "..." if size else "..."
                    draw.text((info_x, info_y), size, fill=colors["text"], font=font_side)
                    info_y += int(25 * scale)
                
                # Distance to centroid
                if distances is not None and outlier_idx is not None:
                    if outlier_idx in shelf0_indices:
                        shelf0_idx = shelf0_indices_list.index(outlier_idx)
                        if shelf0_idx < len(distances):
                            distance = distances[shelf0_idx]
                            draw.text((info_x, info_y), "Distance to Centroid", fill=colors["text"], font=font_small)
                            info_y += int(18 * scale)
                            draw.text((info_x, info_y), f"{distance:.4f}", fill=colors["lime"], font=font_side)
                            info_y += int(25 * scale)
                
                # Description (wrapped text)
                description = str(outlier_artwork.get("description", ""))
                if description and description != "nan" and description.strip():
                    draw.line([(info_x, info_y), (MAP_WIDTH_SCALED - 50 * scale, info_y)], fill=colors["text"], width=int(LINE_WIDTH_SCALED))
                    info_y += int(15 * scale)
                    draw.text((info_x, info_y), "Description", fill=colors["text"], font=font_small)
                    info_y += int(18 * scale)
                    max_width = MAP_WIDTH_SCALED - info_x - 20 * scale
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
                        draw.text((info_x, info_y), line, fill=colors["text"], font=font_small)
                        info_y += int(16 * scale)
            
            # Draw rank table on right side (match top10 structure with thumbnails)
            panel_x = MAP_WIDTH_SCALED + 30 * scale
            y_pos = 100 * scale
            
            # Representatives section
            draw.text((panel_x, y_pos), "Representatives", fill=colors["text"], font=font_title)
            y_pos += int(80 * scale)  # Match top10 spacing
            font_table = get_font(int(FONT_SIZE_TABLE * scale), "thin")
            draw.text((panel_x + 80, y_pos), "Rank", fill=colors["text"], font=font_table)
            draw.text((panel_x + 140, y_pos), "Title", fill=colors["text"], font=font_table)
            draw.text((panel_x + 350, y_pos), "Artist", fill=colors["text"], font=font_table)
            draw.text((panel_x + 520, y_pos), "Distance", fill=colors["text"], font=font_table)
            y_pos += int(40 * scale)
            
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
                            
                            # Draw thumbnail on the left (bigger size)
                            thumbnail = load_image(artwork.get("thumbnail", ""))
                            if thumbnail:
                                thumb_size = 60  # Bigger thumbnail size
                                thumbnail_resized = thumbnail.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                                # Position thumbnail before rank number
                                thumb_x = panel_x
                                thumb_y = y_pos - 5
                                # Handle transparent images
                                if thumbnail_resized.mode == "RGBA":
                                    img.paste(thumbnail_resized, (int(thumb_x), int(thumb_y)), thumbnail_resized)
                                else:
                                    img.paste(thumbnail_resized, (int(thumb_x), int(thumb_y)))
                            
                            draw.text((panel_x + 80, y_pos), f"{rank + 1}", fill=row_color, font=font_table)
                            title = str(artwork.get("title", "Unknown"))[:30]
                            draw.text((panel_x + 140, y_pos), title, fill=row_color, font=font_table)
                            artist = str(artwork.get("artist", "Unknown"))[:25]
                            draw.text((panel_x + 350, y_pos), artist, fill=row_color, font=font_table)
                            draw.text((panel_x + 520, y_pos), f"{distance:.4f}", fill=row_color, font=font_table)
                            y_pos += int(70 * scale)
                    except Exception:
                        pass
            
            y_pos += 20 * scale
            draw.line([(panel_x, y_pos), (CANVAS_WIDTH_SCALED - 30 * scale, y_pos)], fill=colors["text"], width=int(LINE_WIDTH_SCALED))
            y_pos += 30 * scale
            
            # Outliers section
            draw.text((panel_x, y_pos), "Outliers", fill=colors["text"], font=font_title)
            y_pos += int(80 * scale)  # Match top10 spacing
            font_table = get_font(int(FONT_SIZE_TABLE * scale), "thin")
            draw.text((panel_x + 80, y_pos), "Rank", fill=colors["text"], font=font_table)
            draw.text((panel_x + 140, y_pos), "Title", fill=colors["text"], font=font_table)
            draw.text((panel_x + 350, y_pos), "Artist", fill=colors["text"], font=font_table)
            draw.text((panel_x + 520, y_pos), "Distance", fill=colors["text"], font=font_table)
            y_pos += int(40 * scale)
            
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
                            
                            # Draw thumbnail on the left (bigger size)
                            thumbnail = load_image(artwork.get("thumbnail", ""))
                            if thumbnail:
                                thumb_size = 60  # Bigger thumbnail size
                                thumbnail_resized = thumbnail.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                                # Position thumbnail before rank number
                                thumb_x = panel_x
                                thumb_y = y_pos - 5
                                # Handle transparent images
                                if thumbnail_resized.mode == "RGBA":
                                    img.paste(thumbnail_resized, (int(thumb_x), int(thumb_y)), thumbnail_resized)
                                else:
                                    img.paste(thumbnail_resized, (int(thumb_x), int(thumb_y)))
                            
                            draw.text((panel_x + 80, y_pos), f"{rank + 1}", fill=row_color, font=font_table)
                            title = str(artwork.get("title", "Unknown"))[:30]
                            draw.text((panel_x + 140, y_pos), title, fill=row_color, font=font_table)
                            artist = str(artwork.get("artist", "Unknown"))[:25]
                            draw.text((panel_x + 350, y_pos), artist, fill=row_color, font=font_table)
                            draw.text((panel_x + 520, y_pos), f"{distance:.4f}", fill=row_color, font=font_table)
                            y_pos += int(70 * scale)
                    except Exception:
                        pass
    
    # Downscale only if needed (1080p mode renders at 2x and needs downscaling)
    # 4K mode renders directly at target resolution, no downscaling needed
    if needs_downscale:
        # Downscale from 2x render resolution to final 1080p for smooth anti-aliasing
        # Use LANCZOS resampling for highest quality downscaling
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    return img


def prepare_visualization_data(target_shelf: str = "0"):
    """Prepare all data needed for visualization.
    
    Returns a dictionary with all the prepared data:
    - df: Full dataframe
    - df_shelf0: Filtered dataframe for target shelf
    - all_embeddings: All embeddings array
    - all_artwork_ids: List of artwork IDs matching embeddings
    - shelf0_mask: Boolean mask for shelf0 items
    - shelf0_embeddings: Embeddings for shelf0 items
    - all_coords_2d: 2D coordinates for all items
    - shelf0_coords_2d: 2D coordinates for shelf0 items
    - centroid: Centroid of shelf0 embeddings
    - distances: Distances from centroid for shelf0 items
    - top_representatives: List of (artwork_idx_in_all, distance) tuples
    - top_outliers: List of (artwork_idx_in_all, distance) tuples
    - aesthetic_representative_id: Aesthetic representative ID
    - aesthetic_outlier_id: Aesthetic outlier ID
    - first_idx_in_all: Index of first representative in all_artwork_ids
    - artwork_lookup: Lookup dictionary for artwork data
    """
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
    
    print(f"   Found {len(df_shelf0)} artworks on Regal {target_shelf}")
    
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
    
    print(f"   Found {len(shelf0_embeddings)} Regal {target_shelf} embeddings")
    
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
    # Matches aesthetic.md curation (video_making/font/aesthetic.md lines 150-152)
    REPRESENTATIVES = {
        0: 464, 1: 152, 2: 161, 3: 119, 4: 454,
        5: 360, 6: 468, 7: 107, 8: 385, 9: 185
    }
    OUTLIERS = {
        0: 479, 1: 386, 2: 326, 3: 82, 4: 424,
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
    
    # Project centroid to 2D
    # Use the mean of 2D coordinates directly - this is the most accurate approach
    centroid_coord_2d = np.mean(shelf0_coords_2d, axis=0)
    
    # Build artwork lookup dictionary for fast DataFrame access
    print("\n5. Building artwork lookup...")
    artwork_lookup = build_artwork_lookup(df)
    
    return {
        "df": df,
        "df_shelf0": df_shelf0,
        "all_embeddings": all_embeddings,
        "all_artwork_ids": all_artwork_ids,
        "shelf0_mask": shelf0_mask,
        "shelf0_embeddings": shelf0_embeddings,
        "all_coords_2d": all_coords_2d,
        "shelf0_coords_2d": shelf0_coords_2d,
        "centroid": centroid,
        "centroid_coord_2d": centroid_coord_2d,
        "distances": distances,
        "top_representatives": top_representatives,
        "top_outliers": top_outliers,
        "top_items": top_items,
        "aesthetic_representative_id": aesthetic_representative_id,
        "aesthetic_outlier_id": aesthetic_outlier_id,
        "first_idx_in_all": first_idx_in_all,
        "artwork_lookup": artwork_lookup,
    }


def main(target_shelf: str = "0", mode: str = "both", white_background: bool = False, supersample_factor: float = 2.0, virtual_render: bool = False):
    """Main function to generate visualization frames and video.
    
    Args:
        target_shelf: The Regal number to visualize (as string, e.g., "0", "5")
        mode: "representative", "outlier", or "both" - which type to visualize (default: "both")
        white_background: If True, use white background with inverted colors (default: False)
        supersample_factor: Supersampling factor for anti-aliasing (2.0 = Full HD output, 4.0 = 4K output, default: 2.0)
        virtual_render: If True, store frames in memory and write directly to video without saving individual frames (default: False)
    """
    print("=" * 60)
    if mode == "both":
        print(f"Visualizing Regal {target_shelf} Representatives and Outliers")
    else:
        mode_title = "Representative" if mode == "representative" else "Outlier"
        print(f"Visualizing Regal {target_shelf} {mode_title} Finding Process")
    print("=" * 60)
    
    # Prepare all visualization data using the shared function
    data = prepare_visualization_data(target_shelf)
    
    # Extract all needed variables from the data dictionary
    df = data["df"]
    df_shelf0 = data["df_shelf0"]
    all_embeddings = data["all_embeddings"]
    all_artwork_ids = data["all_artwork_ids"]
    shelf0_mask = data["shelf0_mask"]
    shelf0_embeddings = data["shelf0_embeddings"]
    all_coords_2d = data["all_coords_2d"]
    shelf0_coords_2d = data["shelf0_coords_2d"]
    centroid = data["centroid"]
    centroid_coord_2d = data["centroid_coord_2d"]
    distances = data["distances"]
    top_representatives = data["top_representatives"]
    top_outliers = data["top_outliers"]
    top_items = data["top_items"]
    aesthetic_representative_id = data["aesthetic_representative_id"]
    aesthetic_outlier_id = data["aesthetic_outlier_id"]
    first_idx_in_all = data["first_idx_in_all"]
    artwork_lookup = data["artwork_lookup"]
    
    # Get additional info for logging
    first_idx_in_shelf0 = np.argsort(distances)[0] if len(distances) > 0 else None
    first_artwork_id = df_shelf0.iloc[first_idx_in_shelf0].get("id") if first_idx_in_shelf0 is not None else None
    
    # Verify aesthetic IDs are included
    top_5_representatives_indices = np.argsort(distances)[:5]
    top_5_outliers_indices = np.argsort(distances)[-5:][::-1]
    if aesthetic_representative_id is not None:
        rep_ids_in_top5 = [df_shelf0.iloc[i].get("id") for i in top_5_representatives_indices]
        print(f"   Aesthetic representative ID {aesthetic_representative_id} {'found' if any(str(aid) == str(aesthetic_representative_id) or float(aid) == float(aesthetic_representative_id) for aid in rep_ids_in_top5) else 'NOT FOUND'} in top 5 representatives")
    
    if aesthetic_outlier_id is not None:
        outlier_ids_in_top5 = [df_shelf0.iloc[i].get("id") for i in top_5_outliers_indices]
        print(f"   Aesthetic outlier ID {aesthetic_outlier_id} {'found' if any(str(aid) == str(aesthetic_outlier_id) or float(aid) == float(aesthetic_outlier_id) for aid in outlier_ids_in_top5) else 'NOT FOUND'} in top 5 outliers")
    
    # Validate centroid coordinate
    if len(shelf0_coords_2d) > 0:
        print(f"   Centroid 2D position: ({centroid_coord_2d[0]:.2f}, {centroid_coord_2d[1]:.2f})")
        print(f"   Regal {target_shelf} coords range: x=[{shelf0_coords_2d[:, 0].min():.2f}, {shelf0_coords_2d[:, 0].max():.2f}], y=[{shelf0_coords_2d[:, 1].min():.2f}, {shelf0_coords_2d[:, 1].max():.2f}]")
    else:
        print(f"   Warning: No Regal {target_shelf} coordinates found!")
        centroid_coord_2d = np.array([MAP_WIDTH // 2, MAP_HEIGHT // 2])  # Default to center
    
    print(f"   Representative artwork ID: {first_artwork_id}")
    if first_idx_in_shelf0 is not None:
        print(f"   Distance to centroid: {distances[first_idx_in_shelf0]:.4f}")
    
    print(f"   Built lookup for {len(artwork_lookup)} artwork ID mappings")
    
    # Generate frames
    print("\n6. Generating frames...")
    folder_suffix = "_white" if white_background else ""
    
    # Initialize frame storage
    if virtual_render:
        if not IMAGEIO_AVAILABLE:
            print("   ERROR: virtual_render requires imageio. Install with: pip install imageio imageio-ffmpeg")
            print("   Falling back to normal render mode (saving frames to disk)")
            virtual_render = False
        else:
            print("   Using virtual render mode (frames stored in memory)")
            frames_list = []  # Store frames in memory
    else:
        # Try to use external drive, fall back to default if not found
        external_drive = Path("/Volumes/NO NAME/storageMuseum")
        if external_drive.exists() and external_drive.is_dir():
            base_output_dir = external_drive / "frames"
            print(f"   Using external drive: {base_output_dir}")
        else:
            base_output_dir = SCRIPT_DIR / "frames"
            print(f"   Using default location: {base_output_dir}")
        
        frames_dir = base_output_dir / f"shelf{target_shelf}_both{folder_suffix}"
        frames_dir.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    
    # Helper function to save or store frame
    def save_or_store_frame(img: Image.Image, frame_num: int):
        """Save frame to disk or store in memory based on virtual_render mode."""
        if virtual_render:
            # Convert PIL Image to numpy array for imageio
            frames_list.append(np.array(img))
        else:
            img.save(frames_dir / f"frame_{frame_num:05d}.png", "PNG", compress_level=1, optimize=False)
    
    # Track previous subtitle for easing transitions
    previous_subtitle = None
    # Use hold frames (2 seconds) for subtitle morphing
    SUBTITLE_EASE_FRAMES = EXTRA_HOLD_FRAMES  # 120 frames = 2 seconds for subtitle morphing
    
    # Step 1: Show all embeddings
    print(f"   Generating step 1: All embeddings...")
    current_subtitle = get_subtitle_label("all")
    for i in tqdm(range(FRAMES_PER_STEP + EXTRA_HOLD_FRAMES), desc="Step 1 frames"):
        # Animate subtitle transition using hold frames at the beginning
        subtitle_ease = None
        if previous_subtitle is not None and i < SUBTITLE_EASE_FRAMES:
            subtitle_ease = (i + 1) / SUBTITLE_EASE_FRAMES
        else:
            subtitle_ease = None
        
        img = create_frame("all", all_coords_2d, all_artwork_ids, shelf0_mask, all_embeddings, 
                          target_shelf=target_shelf, top_representatives=top_items,
                          aesthetic_representative_id=aesthetic_representative_id,
                          white_background=white_background, supersample_factor=supersample_factor,
                          artwork_lookup=artwork_lookup, df=df,
                          previous_subtitle=previous_subtitle,
                          subtitle_ease_progress=subtitle_ease)
        # Save with high quality PNG settings
        save_or_store_frame(img, frame_count)
        frame_count += 1
    
    # Update previous subtitle for next step - use final subtitle text for proper easing
    previous_subtitle = get_final_subtitle_text(
        step="all",
        target_shelf=target_shelf,
    )
    
    # Step 2: Show Regal items one by one slowly (NO LINES, just identification) with color easing
    print(f"   Generating step 2: Identifying Regal {target_shelf} items (slow, no lines)...")
    shelf0_indices_list = np.where(shelf0_mask)[0].tolist()
    num_shelf0 = len(shelf0_indices_list)
    
    # Show each artwork one by one, more slowly with color easing
    # IMPORTANT: Only show ONE item at a time - each item should appear individually
    FRAMES_PER_ARTWORK_SLOW = 30  # More frames per artwork for slower pace
    EASE_IN_FRAMES = 15  # Frames for color easing animation
    current_subtitle = get_subtitle_label("highlight_slow")
    for artwork_num in tqdm(range(1, num_shelf0 + 1), desc="Step 2: Identifying items"):
        # Get the artwork index being added
        current_artwork_idx_in_all = shelf0_indices_list[artwork_num - 1]
        
        # Generate frames for this artwork - NO lines, NO centroid, just showing the item
        CIRCLE_EXPAND_FRAMES = 20  # Frames for circle expansion animation
        CIRCLE_EXPAND_DELAY = 5  # Delay before circle starts expanding (after green dot appears)
        PANEL_BORDER_FRAMES = 20  # Frames for panel border animation (same as circle expansion)
        for i in range(FRAMES_PER_ARTWORK_SLOW):
            # Calculate color ease progress for the currently appearing item
            if i < EASE_IN_FRAMES:
                color_ease = (i + 1) / EASE_IN_FRAMES
            else:
                color_ease = None  # Fully appeared, no easing needed
            
            # Calculate circle expansion progress (only for the currently appearing item)
            # Start expanding slightly after green dot appears for better visual flow
            circle_expand = None
            if i >= CIRCLE_EXPAND_DELAY and i < CIRCLE_EXPAND_DELAY + CIRCLE_EXPAND_FRAMES:
                circle_expand = (i - CIRCLE_EXPAND_DELAY + 1) / CIRCLE_EXPAND_FRAMES
            elif i >= CIRCLE_EXPAND_DELAY + CIRCLE_EXPAND_FRAMES:
                circle_expand = 1.0  # Fully expanded
            
            # Calculate panel border progress (smooth fade-in, same method as circle expansion)
            panel_border = None
            if i < PANEL_BORDER_FRAMES:
                panel_border = (i + 1) / PANEL_BORDER_FRAMES
            
            # Animate subtitle transition only for first artwork of step
            subtitle_ease = None
            if artwork_num == 1 and previous_subtitle is not None and i < SUBTITLE_EASE_FRAMES:
                subtitle_ease = (i + 1) / SUBTITLE_EASE_FRAMES
            
            img = create_frame("highlight_slow", all_coords_2d, all_artwork_ids, shelf0_mask,
                              supersample_factor=supersample_factor,
                              white_background=white_background,
                              artwork_lookup=artwork_lookup, df=df, 
                              all_embeddings=all_embeddings,
                              shelf0_coords=shelf0_coords_2d, 
                              shelf0_coords_progressive=shelf0_coords_2d,
                              num_shelf0_shown=artwork_num,
                              highlighted_artwork_idx=current_artwork_idx_in_all,
                              target_shelf=target_shelf,
                              top_representatives=top_items,
                              aesthetic_representative_id=aesthetic_representative_id,
                              color_ease_progress=color_ease,
                              circle_expand_progress=circle_expand,
                              panel_border_progress=panel_border,
                              previous_subtitle=previous_subtitle if artwork_num == 1 else None,
                              subtitle_ease_progress=subtitle_ease)
            # Save with high quality PNG settings
            save_or_store_frame(img, frame_count)
            frame_count += 1
    
    # Hold final state for a bit - show all Regal items
    if num_shelf0 > 0:
        last_artwork_idx = shelf0_indices_list[-1]
        for i in tqdm(range(FRAMES_PER_STEP // 2 + EXTRA_HOLD_FRAMES), desc="Step 2: Hold final"):
            img = create_frame("highlight_slow", all_coords_2d, all_artwork_ids, shelf0_mask,
                              supersample_factor=supersample_factor,
                              white_background=white_background,
                              artwork_lookup=artwork_lookup, df=df, 
                              all_embeddings=all_embeddings,
                              shelf0_coords=shelf0_coords_2d, 
                              shelf0_coords_progressive=shelf0_coords_2d,
                              num_shelf0_shown=num_shelf0,
                              highlighted_artwork_idx=last_artwork_idx,
                              target_shelf=target_shelf,
                              top_representatives=top_items,
                              aesthetic_representative_id=aesthetic_representative_id)
            # Save with high quality PNG settings
            save_or_store_frame(img, frame_count)
            frame_count += 1
    
    # Update previous subtitle for next step - use final subtitle text for proper easing
    previous_subtitle = get_final_subtitle_text(
        step="highlight_slow",
        target_shelf=target_shelf,
    )
    
    # Step 3: Highlight Regal with centroid and distances (combined step) with easing
    print(f"   Generating step 3: Highlighting Regal {target_shelf} with centroid and distances (with easing)...")
    current_subtitle = get_subtitle_label("highlight", target_shelf=target_shelf)
    
    # IMPORTANT: All green dots from "identify regal items" step should already be shown
    # We show all items (num_shelf0_shown=num_shelf0) but animate lines/centroid progressively
    # Animate adding each artwork one by one with lines, centroid, and distances
    for artwork_num in tqdm(range(1, num_shelf0 + 1), desc="Step 3: Adding centroid"):
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
        # Use consistent frame counts with other steps
        CIRCLE_EXPAND_FRAMES_STEP3 = 20  # Frames for circle expansion animation (consistent with other steps)
        CIRCLE_EXPAND_DELAY_STEP3 = 5  # Delay before circle starts expanding (after green dot appears)
        EASE_IN_FRAMES_STEP3 = 15  # Frames for easing in each item (consistent with other steps)
        for i in range(FRAMES_PER_ADDITION):
            # Calculate color ease progress for the currently appearing item
            if i < EASE_IN_FRAMES_STEP3:
                color_ease = (i + 1) / EASE_IN_FRAMES_STEP3
            else:
                color_ease = None  # Fully appeared
            
            # Calculate circle expansion progress (only for the currently appearing item)
            # Start expanding slightly after green dot appears for better visual flow
            circle_expand = None
            if i >= CIRCLE_EXPAND_DELAY_STEP3 and i < CIRCLE_EXPAND_DELAY_STEP3 + CIRCLE_EXPAND_FRAMES_STEP3:
                circle_expand = (i - CIRCLE_EXPAND_DELAY_STEP3 + 1) / CIRCLE_EXPAND_FRAMES_STEP3
            elif i >= CIRCLE_EXPAND_DELAY_STEP3 + CIRCLE_EXPAND_FRAMES_STEP3:
                circle_expand = 1.0  # Fully expanded
            
            # Animate subtitle transition only for first artwork of step
            subtitle_ease = None
            if artwork_num == 1 and previous_subtitle is not None and i < SUBTITLE_EASE_FRAMES:
                subtitle_ease = (i + 1) / SUBTITLE_EASE_FRAMES
            
            # Show ALL items (num_shelf0) as green dots from step 2, but only animate lines/centroid for progressive set
            # The drawing code now handles this: in "highlight" step, it shows all dots from shelf0_coords
            # but lines are still drawn progressively based on num_shelf0_shown
            # IMPORTANT: Pass num_shelf0_shown=num_shelf0 to ensure all green dots are visible (from step 2)
            img = create_frame("highlight", all_coords_2d, all_artwork_ids, shelf0_mask,
                              supersample_factor=supersample_factor,
                              white_background=white_background,
                              artwork_lookup=artwork_lookup, df=df, 
                              all_embeddings=all_embeddings,
                              shelf0_coords=shelf0_coords_2d, 
                              shelf0_coords_progressive=shelf0_coords_2d[:artwork_num],  # Progressive for lines only
                              centroid_coord=centroid_coord_2d_progressive,
                              distances=distances_progressive,
                              num_shelf0_shown=num_shelf0,  # Show ALL green dots from step 2 (not progressive)
                              highlighted_artwork_idx=current_artwork_idx_in_all,
                              target_shelf=target_shelf,
                              top_representatives=top_items,
                              aesthetic_representative_id=aesthetic_representative_id,
                              color_ease_progress=None,  # No color easing - all items already green
                              circle_expand_progress=circle_expand,
                              previous_subtitle=previous_subtitle if artwork_num == 1 else None,
                              subtitle_ease_progress=subtitle_ease)
            # Save with high quality PNG settings
            save_or_store_frame(img, frame_count)
            frame_count += 1
    
    # Update previous subtitle for next step - use final subtitle text for proper easing
    previous_subtitle = get_final_subtitle_text(
        step="highlight",
        target_shelf=target_shelf,
        centroid_coord=centroid_coord_2d if 'centroid_coord_2d' in locals() else None,
    )
    
    # Hold final state for a bit - show all items with final centroid and distances
    # Increase hold frames for centroid highlighting (more still frames)
    if num_shelf0 > 0:
        last_artwork_idx = shelf0_indices_list[-1]
        for i in tqdm(range(FRAMES_PER_STEP * 2 + EXTRA_HOLD_FRAMES), desc="Step 3: Hold final"):  # Double the hold frames + extra 3 seconds
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
                              top_representatives=top_items,
                              aesthetic_representative_id=aesthetic_representative_id)
            # Save with high quality PNG settings
            save_or_store_frame(img, frame_count)
            frame_count += 1
    
    # Step 3.5: Animate grey connection lines disappearing and dim green dots
    print("   Generating step 3.5: Fading out connection lines and dimming green dots...")
    FRAMES_FADE_OUT = 30  # Frames to fade out lines
    for fade_frame in tqdm(range(FRAMES_FADE_OUT), desc="Step 3.5: Fade out"):
        fade_progress = 1.0 - (fade_frame + 1) / FRAMES_FADE_OUT  # 1.0 to 0.0
        # Dim green dots as lines fade out (dim progress goes from 0.0 to 1.0)
        dim_progress = fade_progress  # Dim as lines fade (1.0 = fully dimmed when lines are gone)
        img = create_frame("highlight", all_coords_2d, all_artwork_ids, shelf0_mask,
                          white_background=white_background,
                          supersample_factor=supersample_factor,
                          artwork_lookup=artwork_lookup, df=df, 
                          all_embeddings=all_embeddings,
                          shelf0_coords=shelf0_coords_2d, 
                          shelf0_coords_progressive=shelf0_coords_2d,
                          centroid_coord=centroid_coord_2d,
                          distances=distances,
                          num_shelf0_shown=num_shelf0,
                          highlighted_artwork_idx=last_artwork_idx if num_shelf0 > 0 else None,
                          target_shelf=target_shelf,
                          top_representatives=top_items,
                          aesthetic_representative_id=aesthetic_representative_id,
                          connection_lines_opacity=fade_progress,
                          green_dots_dim_progress=dim_progress)
        # Save with high quality PNG settings
        save_or_store_frame(img, frame_count)
        frame_count += 1
    
    # Hold final state for Step 3.5 - show lines fully faded out, green dots dimmed
    if num_shelf0 > 0:
        last_artwork_idx = shelf0_indices_list[-1] if shelf0_indices_list else None
        for i in tqdm(range(EXTRA_HOLD_FRAMES), desc="Step 3.5: Hold final"):
            img = create_frame("highlight", all_coords_2d, all_artwork_ids, shelf0_mask,
                              white_background=white_background,
                              supersample_factor=supersample_factor,
                              artwork_lookup=artwork_lookup, df=df, 
                              all_embeddings=all_embeddings,
                              shelf0_coords=shelf0_coords_2d, 
                              shelf0_coords_progressive=shelf0_coords_2d,
                              centroid_coord=centroid_coord_2d,
                              distances=distances,
                              num_shelf0_shown=num_shelf0,
                              highlighted_artwork_idx=last_artwork_idx,
                              target_shelf=target_shelf,
                              top_representatives=top_items,
                              aesthetic_representative_id=aesthetic_representative_id,
                              connection_lines_opacity=0.0,  # Fully faded out
                              green_dots_dim_progress=1.0)  # Fully dimmed
            # Save with high quality PNG settings
            save_or_store_frame(img, frame_count)
            frame_count += 1
    
    # Update previous subtitle for next step - use final subtitle text for proper easing
    previous_subtitle = get_final_subtitle_text(
        step="highlight",
        target_shelf=target_shelf,
        centroid_coord=centroid_coord_2d if 'centroid_coord_2d' in locals() else None,
    )
    
    # Step 4: Cycle through each Regal artwork showing calculations
    # Do representatives first, then outliers
    print(f"   Generating step 4: Cycling through Regal {target_shelf} artworks (representatives first, then outliers)...")
    current_subtitle = get_subtitle_label("representative", search_mode="representative")  # Will change dynamically during step
    shelf0_indices_list = np.where(shelf0_mask)[0].tolist()
    # Sort by distance to centroid for better visualization
    # Map shelf 0 indices to their positions in the distances array
    shelf0_distances_sorted = []
    for idx_in_all in shelf0_indices_list:
        idx_in_shelf0 = shelf0_indices_list.index(idx_in_all)
        if idx_in_shelf0 < len(distances):
            shelf0_distances_sorted.append((idx_in_all, distances[idx_in_shelf0], idx_in_shelf0))
    
    # Sort by distance (closest first)
    shelf0_distances_sorted = sorted(shelf0_distances_sorted, key=lambda x: x[1])
    
    # Separate into representatives (closest) and outliers (farthest)
    # Use median distance as threshold
    median_distance = np.median([d for _, d, _ in shelf0_distances_sorted])
    representatives = [(idx, dist, shelf0_idx) for idx, dist, shelf0_idx in shelf0_distances_sorted if dist <= median_distance]
    outliers = [(idx, dist, shelf0_idx) for idx, dist, shelf0_idx in shelf0_distances_sorted if dist > median_distance]
    # Sort representatives closest first, outliers farthest first
    representatives = sorted(representatives, key=lambda x: x[1])
    outliers = sorted(outliers, key=lambda x: x[1], reverse=True)
    
    # First show all representatives
    # Skip drawing lines for the last 2 representatives (no lines in final steps)
    num_reps_to_show_lines = max(0, len(representatives) - 2) if len(representatives) > 2 else 0
    # Gradually brighten green dots during first few representatives (fade in from dimmed to bright)
    BRIGHTEN_FRAMES = 60  # Frames to brighten green dots (first 60 frames of step 4)
    total_frames_so_far = 0
    # Track previous info panel text for easing
    previous_info_title = None
    previous_info_artist = None
    for rep_idx, (artwork_idx, dist, shelf0_idx) in enumerate(tqdm(representatives, desc="Representatives")):
        # Draw line from centroid to this artwork (animated)
        artwork_coord = shelf0_coords_2d[shelf0_idx]
        start_coord = centroid_coord_2d
        
        # Skip drawing line for last 2 representatives
        should_draw_line = rep_idx < num_reps_to_show_lines
        
        # Animate line drawing for this artwork
        FRAMES_PER_REP_LINE = 25  # Frames to draw line for each representative (slower)
        PANEL_BORDER_FRAMES = 20  # Frames for panel border animation (same as identify stage)
        HIGHLIGHT_CIRCLE_EXPAND_FRAMES = 20  # Frames for grey circle expansion animation
        HIGHLIGHT_CIRCLE_DELAY = 5  # Delay before circle starts expanding (after line starts drawing)
        for frame_in_line in range(FRAMES_PER_REP_LINE):
            line_progress = (frame_in_line + 1) / FRAMES_PER_REP_LINE if should_draw_line else 0.0
            lines_to_draw = [(start_coord, artwork_coord, line_progress, False)] if should_draw_line else []
            
            # Calculate panel border progress (smooth fade-in, same method as identify stage)
            panel_border = None
            if frame_in_line < PANEL_BORDER_FRAMES:
                panel_border = (frame_in_line + 1) / PANEL_BORDER_FRAMES
            
            # Calculate highlight circle expansion progress (smooth expansion, same method as other circles)
            # Start expanding slightly after line starts drawing for better visual flow
            highlight_circle_expand = None
            if frame_in_line >= HIGHLIGHT_CIRCLE_DELAY and frame_in_line < HIGHLIGHT_CIRCLE_DELAY + HIGHLIGHT_CIRCLE_EXPAND_FRAMES:
                highlight_circle_expand = (frame_in_line - HIGHLIGHT_CIRCLE_DELAY + 1) / HIGHLIGHT_CIRCLE_EXPAND_FRAMES
            elif frame_in_line >= HIGHLIGHT_CIRCLE_DELAY + HIGHLIGHT_CIRCLE_EXPAND_FRAMES:
                highlight_circle_expand = 1.0  # Fully expanded
            
            # Calculate green dots brightening progress (fade in from dimmed to bright)
            green_dots_dim = None
            if total_frames_so_far < BRIGHTEN_FRAMES:
                # Brighten from dimmed (1.0) to bright (0.0) over BRIGHTEN_FRAMES
                brighten_progress = total_frames_so_far / BRIGHTEN_FRAMES  # 0.0 to 1.0
                green_dots_dim = 1.0 - brighten_progress  # 1.0 (dimmed) to 0.0 (bright)
            else:
                green_dots_dim = None  # Fully bright after BRIGHTEN_FRAMES
            
            # Animate subtitle transition only for first representative
            subtitle_ease = None
            if rep_idx == 0 and previous_subtitle is not None and frame_in_line < min(FRAMES_PER_REP_LINE, SUBTITLE_EASE_FRAMES):
                subtitle_ease = (frame_in_line + 1) / min(FRAMES_PER_REP_LINE, SUBTITLE_EASE_FRAMES)
            
            # Get current artwork info for easing
            current_info_title = None
            current_info_artist = None
            info_title_ease = None
            info_artist_ease = None
            if df is not None:
                try:
                    artwork_id = all_artwork_ids[artwork_idx]
                    artwork_row = df[df["id"].astype(float) == float(artwork_id)]
                    if artwork_row.empty:
                        artwork_row = df[df["id"].astype(str).str.strip() == str(artwork_id).strip()]
                    if not artwork_row.empty:
                        current_info_title = str(artwork_row.iloc[0].get("title", "Unknown"))
                        current_info_artist = str(artwork_row.iloc[0].get("artist", "Unknown"))
                        # Animate info panel text when artwork changes
                        # For first artwork of step: animate from previous step's artwork
                        # For subsequent artworks: animate from previous artwork
                        if frame_in_line < min(FRAMES_PER_REP_LINE, SUBTITLE_EASE_FRAMES):
                            if previous_info_title is not None and previous_info_title != current_info_title:
                                info_title_ease = (frame_in_line + 1) / min(FRAMES_PER_REP_LINE, SUBTITLE_EASE_FRAMES)
                            if previous_info_artist is not None and previous_info_artist != current_info_artist:
                                info_artist_ease = (frame_in_line + 1) / min(FRAMES_PER_REP_LINE, SUBTITLE_EASE_FRAMES)
                except Exception:
                    pass
            
            img = create_frame("representative", all_coords_2d, all_artwork_ids, shelf0_mask,
                              supersample_factor=supersample_factor,
                              white_background=white_background,
                              artwork_lookup=artwork_lookup, df=df,
                              all_embeddings=all_embeddings,
                              shelf0_coords=shelf0_coords_2d,
                              centroid_coord=centroid_coord_2d,
                              distances=distances,
                              representative_idx=first_idx_in_all,
                              highlighted_artwork_idx=artwork_idx,
                              lines_to_draw=lines_to_draw,
                              target_shelf=target_shelf,
                              top_representatives=top_items,
                              top_outliers=top_outliers,
                              aesthetic_representative_id=aesthetic_representative_id,
                              current_distance=dist,
                              search_mode="representative",
                              panel_border_progress=panel_border,
                              highlight_circle_expand_progress=highlight_circle_expand,
                              green_dots_dim_progress=green_dots_dim,
                              previous_subtitle=previous_subtitle if rep_idx == 0 else None,
                              subtitle_ease_progress=subtitle_ease,
                              previous_info_title=previous_info_title if (previous_info_title is not None and previous_info_title != current_info_title) else None,
                              info_title_ease_progress=info_title_ease,
                              previous_info_artist=previous_info_artist if (previous_info_artist is not None and previous_info_artist != current_info_artist) else None,
                              info_artist_ease_progress=info_artist_ease)
            # Save with high quality PNG settings
            save_or_store_frame(img, frame_count)
            frame_count += 1
            total_frames_so_far += 1
            
            # Update previous info panel text after first frame
            if rep_idx == 0 and frame_in_line == 0 and current_info_title is not None:
                previous_info_title = current_info_title
                previous_info_artist = current_info_artist
            elif rep_idx > 0 and frame_in_line == 0 and current_info_title is not None:
                # Update for next artwork
                previous_info_title = current_info_title
                previous_info_artist = current_info_artist
        
        # Hold the line and show artwork info (border and circle fully visible)
        for i in range(FRAMES_PER_ARTWORK - FRAMES_PER_REP_LINE):
            # Update previous info panel text for next artwork
            if i == 0 and current_info_title is not None:
                previous_info_title = current_info_title
                previous_info_artist = current_info_artist
            # Calculate green dots brightening progress for hold frames too
            green_dots_dim = None
            if total_frames_so_far < BRIGHTEN_FRAMES:
                brighten_progress = total_frames_so_far / BRIGHTEN_FRAMES
                green_dots_dim = 1.0 - brighten_progress
            else:
                green_dots_dim = None
            
            lines_to_draw = [(start_coord, artwork_coord, 1.0, False)] if should_draw_line else []  # Fully drawn or no line
            img = create_frame("representative", all_coords_2d, all_artwork_ids, shelf0_mask,
                              supersample_factor=supersample_factor,
                              white_background=white_background,
                              artwork_lookup=artwork_lookup, df=df,
                              all_embeddings=all_embeddings,
                              shelf0_coords=shelf0_coords_2d,
                              centroid_coord=centroid_coord_2d,
                              distances=distances,
                              representative_idx=first_idx_in_all,
                              highlighted_artwork_idx=artwork_idx,
                              lines_to_draw=lines_to_draw,
                              target_shelf=target_shelf,
                              top_representatives=top_items,
                              top_outliers=top_outliers,
                              aesthetic_representative_id=aesthetic_representative_id,
                              current_distance=dist,
                              search_mode="representative",
                              panel_border_progress=1.0,  # Fully visible border
                              highlight_circle_expand_progress=1.0,  # Fully expanded circle
                              green_dots_dim_progress=green_dots_dim,
                              previous_info_title=previous_info_title if (previous_info_title is not None and current_info_title is not None and previous_info_title != current_info_title) else None,
                              info_title_ease_progress=1.0 if (previous_info_title is not None and current_info_title is not None and previous_info_title != current_info_title) else None,
                              previous_info_artist=previous_info_artist if (previous_info_artist is not None and current_info_artist is not None and previous_info_artist != current_info_artist) else None,
                              info_artist_ease_progress=1.0 if (previous_info_artist is not None and current_info_artist is not None and previous_info_artist != current_info_artist) else None)
            # Save with high quality PNG settings
            save_or_store_frame(img, frame_count)
            frame_count += 1
            total_frames_so_far += 1
    
    # Then show all outliers
    # Reset previous info for outliers (they're different artworks)
    previous_info_title = None
    previous_info_artist = None
    for outlier_idx, (artwork_idx, dist, shelf0_idx) in enumerate(tqdm(outliers, desc="Outliers")):
        # Draw line from centroid to this artwork (animated)
        artwork_coord = shelf0_coords_2d[shelf0_idx]
        start_coord = centroid_coord_2d
        
        # Get current artwork info for easing
        current_info_title = None
        current_info_artist = None
        info_title_ease = None
        info_artist_ease = None
        if df is not None:
            try:
                artwork_id = all_artwork_ids[artwork_idx]
                artwork_row = df[df["id"].astype(float) == float(artwork_id)]
                if artwork_row.empty:
                    artwork_row = df[df["id"].astype(str).str.strip() == str(artwork_id).strip()]
                if not artwork_row.empty:
                    current_info_title = str(artwork_row.iloc[0].get("title", "Unknown"))
                    current_info_artist = str(artwork_row.iloc[0].get("artist", "Unknown"))
            except Exception:
                pass
        
        # Animate line drawing for this artwork
        FRAMES_PER_REP_LINE = 25  # Frames to draw line for each representative (slower)
        PANEL_BORDER_FRAMES = 20  # Frames for panel border animation (same as identify stage)
        HIGHLIGHT_CIRCLE_EXPAND_FRAMES = 20  # Frames for grey circle expansion animation
        HIGHLIGHT_CIRCLE_DELAY = 5  # Delay before circle starts expanding (after line starts drawing)
        for frame_in_line in range(FRAMES_PER_REP_LINE):
            line_progress = (frame_in_line + 1) / FRAMES_PER_REP_LINE
            lines_to_draw = [(start_coord, artwork_coord, line_progress, False)]
            
            # Calculate panel border progress (smooth fade-in, same method as identify stage)
            panel_border = None
            if frame_in_line < PANEL_BORDER_FRAMES:
                panel_border = (frame_in_line + 1) / PANEL_BORDER_FRAMES
            
            # Calculate highlight circle expansion progress (smooth expansion, same method as other circles)
            # Start expanding slightly after line starts drawing for better visual flow
            highlight_circle_expand = None
            if frame_in_line >= HIGHLIGHT_CIRCLE_DELAY and frame_in_line < HIGHLIGHT_CIRCLE_DELAY + HIGHLIGHT_CIRCLE_EXPAND_FRAMES:
                highlight_circle_expand = (frame_in_line - HIGHLIGHT_CIRCLE_DELAY + 1) / HIGHLIGHT_CIRCLE_EXPAND_FRAMES
            elif frame_in_line >= HIGHLIGHT_CIRCLE_DELAY + HIGHLIGHT_CIRCLE_EXPAND_FRAMES:
                highlight_circle_expand = 1.0  # Fully expanded
            
            # Animate info panel text when artwork changes
            if frame_in_line < min(FRAMES_PER_REP_LINE, SUBTITLE_EASE_FRAMES):
                if previous_info_title is not None and current_info_title is not None and previous_info_title != current_info_title:
                    info_title_ease = (frame_in_line + 1) / min(FRAMES_PER_REP_LINE, SUBTITLE_EASE_FRAMES)
                if previous_info_artist is not None and current_info_artist is not None and previous_info_artist != current_info_artist:
                    info_artist_ease = (frame_in_line + 1) / min(FRAMES_PER_REP_LINE, SUBTITLE_EASE_FRAMES)
            
            img = create_frame("representative", all_coords_2d, all_artwork_ids, shelf0_mask,
                              supersample_factor=supersample_factor,
                              white_background=white_background,
                              artwork_lookup=artwork_lookup, df=df,
                              all_embeddings=all_embeddings,
                              shelf0_coords=shelf0_coords_2d,
                              centroid_coord=centroid_coord_2d,
                              distances=distances,
                              representative_idx=first_idx_in_all,
                              highlighted_artwork_idx=artwork_idx,
                              lines_to_draw=lines_to_draw,
                              target_shelf=target_shelf,
                              top_representatives=top_items,
                              top_outliers=top_outliers,
                              aesthetic_representative_id=aesthetic_representative_id,
                              current_distance=dist,
                              search_mode="outlier",
                              panel_border_progress=panel_border,
                              highlight_circle_expand_progress=highlight_circle_expand,
                              previous_info_title=previous_info_title if (previous_info_title is not None and current_info_title is not None and previous_info_title != current_info_title) else None,
                              info_title_ease_progress=info_title_ease,
                              previous_info_artist=previous_info_artist if (previous_info_artist is not None and current_info_artist is not None and previous_info_artist != current_info_artist) else None,
                              info_artist_ease_progress=info_artist_ease)
            # Save with high quality PNG settings
            save_or_store_frame(img, frame_count)
            frame_count += 1
            
            # Update previous info panel text after first frame
            if frame_in_line == 0 and current_info_title is not None:
                previous_info_title = current_info_title
                previous_info_artist = current_info_artist
        
        # Hold the line and show artwork info (border and circle fully visible)
        for i in range(FRAMES_PER_ARTWORK - FRAMES_PER_REP_LINE):
            # Update previous info panel text for next artwork
            if i == 0 and current_info_title is not None:
                previous_info_title = current_info_title
                previous_info_artist = current_info_artist
            lines_to_draw = [(start_coord, artwork_coord, 1.0, False)]  # Fully drawn
            img = create_frame("representative", all_coords_2d, all_artwork_ids, shelf0_mask,
                              supersample_factor=supersample_factor,
                              white_background=white_background,
                              artwork_lookup=artwork_lookup, df=df,
                              all_embeddings=all_embeddings,
                              shelf0_coords=shelf0_coords_2d,
                              centroid_coord=centroid_coord_2d,
                              distances=distances,
                              representative_idx=first_idx_in_all,
                              highlighted_artwork_idx=artwork_idx,
                              lines_to_draw=lines_to_draw,
                              target_shelf=target_shelf,
                              top_representatives=top_items,
                              top_outliers=top_outliers,
                              aesthetic_representative_id=aesthetic_representative_id,
                              current_distance=dist,
                              search_mode="outlier",
                              panel_border_progress=1.0,  # Fully visible border
                              highlight_circle_expand_progress=1.0,  # Fully expanded circle
                              previous_info_title=previous_info_title if (previous_info_title is not None and current_info_title is not None and previous_info_title != current_info_title) else None,
                              info_title_ease_progress=1.0 if (previous_info_title is not None and current_info_title is not None and previous_info_title != current_info_title) else None,
                              previous_info_artist=previous_info_artist if (previous_info_artist is not None and current_info_artist is not None and previous_info_artist != current_info_artist) else None,
                              info_artist_ease_progress=1.0 if (previous_info_artist is not None and current_info_artist is not None and previous_info_artist != current_info_artist) else None)
            # Save with high quality PNG settings
            save_or_store_frame(img, frame_count)
            frame_count += 1
    
    # Hold final state for Step 4 - show last artwork with line
    if len(outliers) > 0:
        last_outlier_idx, last_outlier_dist, last_outlier_shelf0_idx = outliers[-1]
        last_outlier_coord = shelf0_coords_2d[last_outlier_shelf0_idx]
        start_coord = centroid_coord_2d
        lines_to_draw = [(start_coord, last_outlier_coord, 1.0, False)]
        
        for i in tqdm(range(EXTRA_HOLD_FRAMES), desc="Step 4: Hold final"):
            img = create_frame("representative", all_coords_2d, all_artwork_ids, shelf0_mask,
                              supersample_factor=supersample_factor,
                              white_background=white_background,
                              artwork_lookup=artwork_lookup, df=df,
                              all_embeddings=all_embeddings,
                              shelf0_coords=shelf0_coords_2d,
                              centroid_coord=centroid_coord_2d,
                              distances=distances,
                              representative_idx=first_idx_in_all,
                              highlighted_artwork_idx=last_outlier_idx,
                              lines_to_draw=lines_to_draw,
                              target_shelf=target_shelf,
                              top_representatives=top_items,
                              top_outliers=top_outliers,
                              aesthetic_representative_id=aesthetic_representative_id,
                              current_distance=last_outlier_dist,
                              search_mode="outlier")
            # Save with high quality PNG settings
            save_or_store_frame(img, frame_count)
            frame_count += 1
    
    # Update previous subtitle for next step - use final subtitle text for proper easing
    # Use the last displayed subtitle from step 4 (could be representative or outlier)
    previous_subtitle = get_final_subtitle_text(
        step="representative",
        target_shelf=target_shelf,
        search_mode="outlier",  # Last one shown was outlier
        current_distance=last_outlier_dist if 'last_outlier_dist' in locals() else None,
    )
    
    # Step 5: Draw lines spreading out from centroid (frame by frame)
    print("   Generating step 5: Drawing lines from centroid...")
    current_subtitle = get_subtitle_label("representative", search_mode="representative")  # Same as step 4
    shelf0_indices_list = np.where(shelf0_mask)[0].tolist()
    
    # Sort artworks by distance to centroid (closest first)
    shelf0_distances_with_indices = []
    for idx_in_all in shelf0_indices_list:
        idx_in_shelf0 = shelf0_indices_list.index(idx_in_all)
        if idx_in_shelf0 < len(distances):
            shelf0_distances_with_indices.append((idx_in_all, distances[idx_in_shelf0], idx_in_shelf0))
    
    shelf0_distances_sorted = sorted(shelf0_distances_with_indices, key=lambda x: x[1])
    
    # Animate drawing lines one by one
    FRAMES_PER_LINE = 30  # Frames to draw each line (longer duration)
    PAUSE_AFTER_CLOSEST = 60  # Frames to pause after drawing line to closest
    
    lines_drawn = []
    for line_idx, (artwork_idx, distance, shelf0_idx) in tqdm(enumerate(shelf0_distances_sorted), desc="Step 5: Drawing lines", total=len(shelf0_distances_sorted)):
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
                              supersample_factor=supersample_factor,
                              white_background=white_background,
                              artwork_lookup=artwork_lookup, df=df,
                              all_embeddings=all_embeddings,
                              shelf0_coords=shelf0_coords_2d,
                              centroid_coord=centroid_coord_2d,
                              distances=distances,
                              representative_idx=first_idx_in_all,
                              highlighted_artwork_idx=highlight_idx or artwork_idx,
                              lines_to_draw=lines_to_draw,
                              target_shelf=target_shelf,
                              top_representatives=top_items,
                              top_outliers=top_outliers,
                              aesthetic_representative_id=aesthetic_representative_id)
            # Save with high quality PNG settings
            save_or_store_frame(img, frame_count)
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
                              supersample_factor=supersample_factor,
                              white_background=white_background,
                              artwork_lookup=artwork_lookup, df=df,
                                  all_embeddings=all_embeddings,
                                  shelf0_coords=shelf0_coords_2d,
                                  centroid_coord=centroid_coord_2d,
                                  distances=distances,
                                  representative_idx=first_idx_in_all,
                                  highlighted_artwork_idx=artwork_idx,
                                  lines_to_draw=lines_to_draw,
                                  target_shelf=target_shelf,
                                  top_representatives=top_items,
                                  top_outliers=top_outliers,
                                  aesthetic_representative_id=aesthetic_representative_id)
                # Save with high quality PNG settings
                save_or_store_frame(img, frame_count)
                frame_count += 1
    
    # Hold final state for Step 5 - show all lines drawn
    if len(shelf0_distances_sorted) > 0:
        last_artwork_idx, _, _ = shelf0_distances_sorted[-1]
        start_coord = centroid_coord_2d
        # Build all lines fully drawn
        all_lines_drawn = []
        for prev_idx, (prev_artwork_idx, _, prev_shelf0_idx) in enumerate(shelf0_distances_sorted):
            prev_coord = shelf0_coords_2d[prev_shelf0_idx]
            is_prev_closest = (prev_idx == 0)
            all_lines_drawn.append((start_coord, prev_coord, 1.0, is_prev_closest))
        
        for i in tqdm(range(EXTRA_HOLD_FRAMES), desc="Step 5: Hold final"):
            img = create_frame("representative", all_coords_2d, all_artwork_ids, shelf0_mask,
                              supersample_factor=supersample_factor,
                              white_background=white_background,
                              artwork_lookup=artwork_lookup, df=df,
                              all_embeddings=all_embeddings,
                              shelf0_coords=shelf0_coords_2d,
                              centroid_coord=centroid_coord_2d,
                              distances=distances,
                              representative_idx=first_idx_in_all,
                              highlighted_artwork_idx=last_artwork_idx,
                              lines_to_draw=all_lines_drawn,
                              target_shelf=target_shelf,
                              top_representatives=top_items,
                              top_outliers=top_outliers,
                              aesthetic_representative_id=aesthetic_representative_id)
            # Save with high quality PNG settings
            save_or_store_frame(img, frame_count)
            frame_count += 1
    
    # Update previous subtitle for next step - use final subtitle text for proper easing
    previous_subtitle = get_final_subtitle_text(
        step="representative",
        target_shelf=target_shelf,
        search_mode="representative",
    )
    
    # Step 6: Show top 10 (5 representatives + 5 outliers) appearing one by one with simultaneous line drawing
    print(f"   Generating step 6: Top 10 (5 Representatives + 5 Outliers) appearing one by one...")
    current_subtitle = get_subtitle_label("top10")
    
    # Constants for animation
    FRAMES_PER_TOP10_ITEM = 40  # Frames per item (slower)
    EASE_IN_FRAMES_TOP10 = 15  # Frames for easing in each item
    
    # First show all 5 representatives one by one
    for rep_num in tqdm(range(1, 6), desc="Step 6: Representatives"):  # 1 to 5
        for frame_in_item in range(FRAMES_PER_TOP10_ITEM):
            # Calculate easing progress for currently appearing item
            if frame_in_item < EASE_IN_FRAMES_TOP10:
                ease_progress = (frame_in_item + 1) / EASE_IN_FRAMES_TOP10
            else:
                ease_progress = None
            
            # Animate subtitle transition only for first item
            subtitle_ease = None
            if rep_num == 1 and previous_subtitle is not None and frame_in_item < min(FRAMES_PER_TOP10_ITEM, SUBTITLE_EASE_FRAMES):
                subtitle_ease = (frame_in_item + 1) / min(FRAMES_PER_TOP10_ITEM, SUBTITLE_EASE_FRAMES)  # Fully appeared
            
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
                              top10_reps_shown=rep_num,
                              top10_outliers_shown=0,  # Not showing outliers yet
                              top10_item_ease_progress=ease_progress,
                              previous_subtitle=previous_subtitle if rep_num == 1 else None,
                              subtitle_ease_progress=subtitle_ease)
            # Save with high quality PNG settings
            save_or_store_frame(img, frame_count)
            frame_count += 1
    
    # Hold all representatives shown for a moment
    for i in tqdm(range(FRAMES_PER_STEP // 2 + EXTRA_HOLD_FRAMES), desc="Step 6: Hold reps"):
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
                              top10_outliers_shown=0,
                              previous_subtitle=None,  # No subtitle change in hold frames
                              subtitle_ease_progress=None)
        # Save with high quality PNG settings
        save_or_store_frame(img, frame_count)
        frame_count += 1
    
    # Then show all 5 outliers one by one
    for outlier_num in tqdm(range(1, 6), desc="Step 6: Outliers"):  # 1 to 5
        for frame_in_item in range(FRAMES_PER_TOP10_ITEM):
            # Calculate easing progress for currently appearing item
            if frame_in_item < EASE_IN_FRAMES_TOP10:
                ease_progress = (frame_in_item + 1) / EASE_IN_FRAMES_TOP10
            else:
                ease_progress = None  # Fully appeared
            
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
                              top10_reps_shown=5,  # All reps shown
                              top10_outliers_shown=outlier_num,
                              top10_item_ease_progress=ease_progress)
            # Save with high quality PNG settings
            save_or_store_frame(img, frame_count)
            frame_count += 1
    
    # Hold final state with all items shown
    for i in tqdm(range(FRAMES_PER_STEP + EXTRA_HOLD_FRAMES), desc="Step 6: Hold final"):
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
                          top10_outliers_shown=5,
                          previous_subtitle=None,
                          subtitle_ease_progress=None)
        # Save with high quality PNG settings
        save_or_store_frame(img, frame_count)
        frame_count += 1
    
    # Step 6.5: Blink effect on selected items (2 seconds = 120 frames)
    print("   Generating step 6.5: Blink effect on selected items...")
    BLINK_FRAMES = 120  # 2 seconds at 60fps
    # Subtitle stays the same for step 6.5, no easing needed
    for blink_frame in tqdm(range(BLINK_FRAMES), desc="Step 6.5: Blink"):
        blink_progress = blink_frame / BLINK_FRAMES  # 0.0 to 1.0
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
                          top10_outliers_shown=5,
                          top10_blink_progress=blink_progress)
        # Save with high quality PNG settings
        save_or_store_frame(img, frame_count)
        frame_count += 1
    
    # Hold final blink state (items at green and max size) for smooth transition
    print("   Generating step 6.6: Hold final blink state...")
    HOLD_BLINK_FRAMES = 30  # 0.5 seconds at 60fps
    for i in tqdm(range(HOLD_BLINK_FRAMES), desc="Step 6.6: Hold blink end"):
        # Use blink_progress=1.0 to show final state (green, max size)
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
                          top10_outliers_shown=5,
                          top10_blink_progress=1.0)  # Final state: green, max size
        # Save with high quality PNG settings
        save_or_store_frame(img, frame_count)
        frame_count += 1
    
    # Update previous subtitle for next step - use final subtitle text for proper easing
    previous_subtitle = get_final_subtitle_text(
        step="top10",
        target_shelf=target_shelf,
        top10_reps_shown=5,
        top10_outliers_shown=5,
        top_representatives=top_representatives if 'top_representatives' in locals() else None,
        top_outliers=top_outliers if 'top_outliers' in locals() else None,
    )
    
    # Step 7: Draw ruler lines from centroid to representative and outlier (simultaneously)
    print("   Generating step 7: Drawing ruler lines to representative and outlier simultaneously...")
    current_subtitle = get_subtitle_label("ruler")
    FRAMES_PER_RULER = 50  # Frames to draw each ruler line (slower)
    FRAMES_CIRCLE_EXPAND = 30  # Frames for circle expansion animation
    FRAMES_TEXT_FADE = 40  # Frames for text fade in animation
    
    # Draw both lines simultaneously with gradual circle expansion and text fade
    for frame_num in tqdm(range(FRAMES_PER_RULER), desc="Step 7: Ruler lines (both simultaneously)"):
        progress = (frame_num + 1) / FRAMES_PER_RULER
        
        # Animate subtitle transition at the beginning
        subtitle_ease = None
        if previous_subtitle is not None and frame_num < min(FRAMES_PER_RULER, SUBTITLE_EASE_FRAMES):
            subtitle_ease = (frame_num + 1) / min(FRAMES_PER_RULER, SUBTITLE_EASE_FRAMES)
        
        # Calculate circle expansion progress (first 30 frames)
        circle_expand = None
        if frame_num < FRAMES_CIRCLE_EXPAND:
            circle_expand = (frame_num + 1) / FRAMES_CIRCLE_EXPAND
        
        # Calculate text fade progress (first 40 frames)
        text_fade = None
        if frame_num < FRAMES_TEXT_FADE:
            text_fade = (frame_num + 1) / FRAMES_TEXT_FADE
        
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
                          ruler_progress=progress,
                          ruler_to_rep=True,  # Parameter kept for compatibility, but both lines draw simultaneously now
                              ruler_circle_expand_progress=circle_expand,
                              ruler_text_fade_progress=text_fade,
                              previous_subtitle=previous_subtitle,
                              subtitle_ease_progress=subtitle_ease)
        # Save with high quality PNG settings
        save_or_store_frame(img, frame_count)
        frame_count += 1
    
    # Hold both rulers visible with info on right side
    for i in tqdm(range(FRAMES_PER_STEP * 2 + EXTRA_HOLD_FRAMES), desc="Step 7: Hold final"):  # Hold for 4 seconds + extra 3 seconds
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
                          ruler_to_rep=True)  # Parameter kept for compatibility
        # Save with high quality PNG settings
        save_or_store_frame(img, frame_count)
        frame_count += 1
    
    print(f"   Generated {frame_count} frames")
    
    # Create video
    print("\n7. Creating video...")
    video_suffix = "_white" if white_background else ""
    
    if virtual_render:
        # Write video directly from memory using imageio
        try:
            # Determine output directory
            external_drive = Path("/Volumes/NO NAME/storageMuseum")
            if external_drive.exists() and external_drive.is_dir():
                base_output_dir = external_drive / "frames"
            else:
                base_output_dir = SCRIPT_DIR / "frames"
            
            output_video = base_output_dir / f"shelf{target_shelf}_both{video_suffix}.mp4"
            output_video.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"   Writing {len(frames_list)} frames directly to video...")
            # Use imageio to write video with high quality settings
            imageio.mimwrite(
                str(output_video),
                frames_list,
                fps=FPS,
                codec='libx264',
                quality=8,  # 0-10 scale, 8 is high quality
                pixelformat='yuv420p',
                macro_block_size=None  # Let imageio choose optimal block size
            )
            print(f"   Video created: {output_video}")
            print(f"   Memory usage: ~{len(frames_list) * frames_list[0].nbytes / (1024**3):.2f} GB for frames")
        except Exception as e:
            print(f"   Error creating video with imageio: {e}")
            print("   Try installing imageio-ffmpeg: pip install imageio-ffmpeg")
    else:
        # Use ffmpeg subprocess (original method)
        output_video = frames_dir.parent / f"shelf{target_shelf}_both{video_suffix}.mp4"
        
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(FPS),
            "-i", str(frames_dir / "frame_%05d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",  # Lower CRF = higher quality (18 is high quality, 23 is default)
            "-preset", "slow",  # Slower encoding = better quality
            "-tune", "stillimage",  # Optimize for still images
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
            if not virtual_render:
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
  python visualize_shelf0_representative.py -s 9 --scale 4.0  # 4K output
        """
    )
    parser.add_argument(
        "--shelf", "-s",
        type=str,
        default="0",
        help="The Regal number to visualize (default: 0)"
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
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Supersampling factor for anti-aliasing (2.0 = Full HD output, 4.0 = 4K output, default: 2.0)"
    )
    parser.add_argument(
        "--virtual-render", "-v",
        action="store_true",
        help="Virtual render mode: store frames in memory and write directly to video without saving individual frames (requires imageio)"
    )
    
    args = parser.parse_args()
    main(target_shelf=args.shelf, mode=args.mode, white_background=args.white_background, supersample_factor=args.scale, virtual_render=args.virtual_render)

