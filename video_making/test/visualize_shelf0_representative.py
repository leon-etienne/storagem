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
FONT_MONO = FONT_DIR / "UbuntuMono-Regular.ttf"  # Monofont for normal text
FALLBACK_IMAGE = PROJECT_ROOT / "video_making/font/no_image.png"
CSV_PATH = PROJECT_ROOT / "artworks_with_thumbnails_ting.csv"
EMBEDDINGS_CACHE = PROJECT_ROOT / "embeddings_cache/all_embeddings.pkl"
BASE_DIR = PROJECT_ROOT / "production-export-2025-11-13t13-42-48-005z"

# Color constants (following aesthetic guidelines)
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_LIME = (0, 255, 0)  # Pure green
COLOR_GRAY_LIGHT = (200, 200, 200)  # Light gray for text on black
COLOR_GRAY_DARK = (150, 150, 150)  # Medium gray for text on black
COLOR_POINT_GRAY = (90, 90, 90)  # Gray for regular points (more gray instead of pure white)
COLOR_WHITE_LOW_OPACITY = (80, 80, 80)  # Gray for non-Regal points (more gray instead of pure black/white)
COLOR_CIRCLE_GRAY = (100, 100, 100)  # Gray for selection circles (instead of black)
COLOR_LIME_LOW_OPACITY = (0, 150, 0)  # Simulated lower opacity for non-highlighted Regal points
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
FIXED_IMAGE_HEIGHT_LARGE = 250  # Larger height for non-ruler steps (to make images bigger)

# Frame generation
FRAMES_PER_STEP = 60  # Hold each step for 60 frames (2 seconds at 30fps)
FRAMES_PER_ARTWORK = 20  # Frames per artwork when cycling through
FRAMES_PER_ADDITION = 20  # Frames for each artwork addition animation (slower)
FPS = 60  # Higher FPS for smoother animation


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
            "circle_gray": COLOR_CIRCLE_GRAY,
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
                       colors: Dict, font_title_large: ImageFont.FreeTypeFont, scale: float = 1.0) -> int:
    """Draw artwork title (large, no label).
    
    Returns:
        Updated y_pos after drawing title
    """
    title_words = title.split()
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
    font_content = get_font(int(FONT_SIZE_TABLE * scale), "thin", mono=True)  # Content font (mono)
    
    # ID
    try:
        artwork_id_int = int(float(artwork.get("id", "N/A")))
        draw.text((col1_x, col_y), "ID", fill=colors["text"], font=font_label_white)  # Whiter label
        col_y += int(18 * scale)
        draw.text((col1_x, col_y), f"{artwork_id_int}", fill=colors["text"], font=font_content)  # Monofont content
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
    supersample_factor: float = 2.0,  # Supersampling factor for anti-aliasing (1.0 = no supersampling, 2.0 = 2x resolution)
    artwork_lookup: Optional[Dict] = None,  # Pre-computed lookup dict for artwork data (id -> row index or row data)
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
    
    # Create canvas at supersampled resolution for better anti-aliasing
    # We'll render at higher resolution and then downscale for smooth edges
    SUPERSAMPLED_WIDTH_LOCAL = int(CANVAS_WIDTH * supersample_factor)
    SUPERSAMPLED_HEIGHT_LOCAL = int(CANVAS_HEIGHT * supersample_factor)
    img = Image.new("RGBA", (SUPERSAMPLED_WIDTH_LOCAL, SUPERSAMPLED_HEIGHT_LOCAL), (*colors["background"], 255))
    draw = ImageDraw.Draw(img)
    
    # Scale all coordinates and sizes for supersampled canvas
    scale = supersample_factor
    
    # Scale coordinate arrays
    all_coords = all_coords * scale
    if shelf0_coords is not None:
        shelf0_coords = shelf0_coords * scale
    if shelf0_coords_progressive is not None:
        shelf0_coords_progressive = shelf0_coords_progressive * scale
    if centroid_coord is not None:
        centroid_coord = (centroid_coord[0] * scale, centroid_coord[1] * scale) if isinstance(centroid_coord, (tuple, list)) else centroid_coord * scale
    
    # Scale lines_to_draw coordinates (they're in the same coordinate space as all_coords_2d)
    if lines_to_draw is not None:
        scaled_lines_to_draw = []
        for line_data in lines_to_draw:
            if len(line_data) == 4:  # (start, end, progress, is_closest)
                start, end, progress, is_closest = line_data
                # Scale start and end coordinates - convert to tuples for consistency
                if isinstance(start, np.ndarray):
                    start = (float(start[0] * scale), float(start[1] * scale))
                elif isinstance(start, (tuple, list)) and len(start) >= 2:
                    start = (float(start[0] * scale), float(start[1] * scale))
                if isinstance(end, np.ndarray):
                    end = (float(end[0] * scale), float(end[1] * scale))
                elif isinstance(end, (tuple, list)) and len(end) >= 2:
                    end = (float(end[0] * scale), float(end[1] * scale))
                scaled_lines_to_draw.append((start, end, progress, is_closest))
            else:  # (start, end, progress)
                start, end, progress = line_data
                # Scale start and end coordinates - convert to tuples for consistency
                if isinstance(start, np.ndarray):
                    start = (float(start[0] * scale), float(start[1] * scale))
                elif isinstance(start, (tuple, list)) and len(start) >= 2:
                    start = (float(start[0] * scale), float(start[1] * scale))
                if isinstance(end, np.ndarray):
                    end = (float(end[0] * scale), float(end[1] * scale))
                elif isinstance(end, (tuple, list)) and len(end) >= 2:
                    end = (float(end[0] * scale), float(end[1] * scale))
                scaled_lines_to_draw.append((start, end, progress))
        lines_to_draw = scaled_lines_to_draw
    
    # Scale size constants
    POINT_SIZE_SCALED = POINT_SIZE * scale
    POINT_SIZE_LARGE_SCALED = POINT_SIZE_LARGE * scale
    POINT_SIZE_REPRESENTATIVE_SCALED = POINT_SIZE_REPRESENTATIVE * scale
    LINE_WIDTH_SCALED = LINE_WIDTH * scale
    RULER_DOT_SIZE_SCALED = 16 * scale if step == "ruler" else None
    
    # Scale dimension constants
    MAP_WIDTH_SCALED = MAP_WIDTH * scale
    PANEL_WIDTH_SCALED = PANEL_WIDTH * scale
    MAP_HEIGHT_SCALED = MAP_HEIGHT * scale
    PANEL_HEIGHT_SCALED = PANEL_HEIGHT * scale
    CANVAS_WIDTH_SCALED = SUPERSAMPLED_WIDTH_LOCAL
    CANVAS_HEIGHT_SCALED = SUPERSAMPLED_HEIGHT_LOCAL
    
    # Load fonts with appropriate weights - scale font sizes for supersampled canvas
    font_title = get_font(int(FONT_SIZE_TITLE * scale), "medium")  # Medium for titles to stand out
    font_label = get_font(int(FONT_SIZE_LABEL * scale), "thin")  # Thin for labels
    font_info = get_font(int(FONT_SIZE_INFO * scale), "thin", mono=True)  # Monofont for info text
    font_small = get_font(int(FONT_SIZE_SMALL * scale), "thin", mono=True)  # Monofont for small text
    
    # Draw step label (subtitle) at top
    step_labels = {
        "all": "All Embeddings",
        "highlight_slow": "Identify Regal items",
        "highlight": f"Highlighting Regal {target_shelf}",
        "centroid": f"Highlighting Regal {target_shelf}",
        "distances": f"Highlighting Regal {target_shelf}",
        "representative": "Finding Representatives" if search_mode == "representative" else "Representative Found",
        "ruler": "Measuring Distances",
        "representatives": "Top 10 Representatives",
        "zoom": "Selected Representative",
        "outlier": "Outlier Found",
        "top10": "Top 10 Representatives and Outliers"
    }
    step_text = step_labels.get(step, step)
    
    # Add distance display for top10 step
    if step == "top10":
        # Calculate current distance being displayed based on what's shown
        current_display_distance = None
        if top10_reps_shown is not None and top10_reps_shown > 0 and top_representatives is not None:
            # Show distance of the currently appearing representative
            if top10_reps_shown <= len(top_representatives):
                current_display_distance = top_representatives[top10_reps_shown - 1][1]
        elif top10_outliers_shown is not None and top10_outliers_shown > 0 and top_outliers is not None:
            # Show distance of the currently appearing outlier
            if top10_outliers_shown <= len(top_outliers):
                current_display_distance = top_outliers[top10_outliers_shown - 1][1]
        
        if current_display_distance is not None:
            step_text += f" | Distance: {current_display_distance:.4f}"
    
    # Add search mode and distance to subtitle for representative step
    if step == "representative":
        if search_mode is not None:
            mode_text = "Representative" if search_mode == "representative" else "Outlier"
            step_text = f"Finding {mode_text}s"
            if current_distance is not None:
                step_text += f" | Distance: {current_distance:.4f}"
        else:
            # Determine if it's representative or outlier based on highlighted artwork
            # Check if highlighted artwork is in top_representatives or top_outliers
            is_representative = False
            is_outlier = False
            if highlighted_artwork_idx is not None and top_representatives is not None:
                for rep_idx, _ in top_representatives[:5]:
                    if rep_idx == highlighted_artwork_idx:
                        is_representative = True
                        break
            if highlighted_artwork_idx is not None and top_outliers is not None:
                for outlier_idx, _ in top_outliers[:5]:
                    if outlier_idx == highlighted_artwork_idx:
                        is_outlier = True
                        break
            
            if is_representative:
                step_text = "Representative Found | Closest to Centroid"
            elif is_outlier:
                step_text = "Outlier Found | Farthest from Centroid"
            else:
                # Check distance to determine - closer items are likely representatives
                if current_distance is not None:
                    # Use median distance as threshold (approximate)
                    if current_distance < 0.3:  # Closer items are representatives
                        step_text = "Representative Found | Closest to Centroid"
                    else:
                        step_text = "Outlier Found | Farthest from Centroid"
                else:
                    step_text = "Representative Found | Closest to Centroid"  # Default fallback
    
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
    
    # Subtitle should be white (not grey)
    draw.text((50 * scale, 60 * scale), step_text, fill=colors["text"], font=font_label)
    
    # Draw title at left side, higher up
    title_text = f"Regal {target_shelf}"
    title_x = 50 * scale  # Left side
    title_y = CANVAS_HEIGHT_SCALED - 100 * scale  # Higher up from bottom
    draw.text((title_x, title_y), title_text, fill=colors["text"], font=font_title)
    
    # Draw calculation info panel FIRST (so it's behind the divider line)
    # This ensures the panel background is drawn before other elements
    if highlighted_artwork_idx is not None and df is not None:
        # Draw background for panel (consistent with main canvas)
        draw.rectangle([MAP_WIDTH_SCALED, 0, CANVAS_WIDTH_SCALED, CANVAS_HEIGHT_SCALED], fill=colors["background"])
    
    # Draw divider line - draw AFTER panel background
    draw.line([(MAP_WIDTH_SCALED, 0), (MAP_WIDTH_SCALED, CANVAS_HEIGHT_SCALED)], fill=colors["text"], width=int(LINE_WIDTH_SCALED * 2))
    
    # Draw all points (fill, no stroke for non-Regal items) with lower opacity
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
            shelf0_indices = np.where(shelf0_mask)[0]
            shelf0_indices_list = list(shelf0_indices)
            if i in shelf0_indices_list:
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
    
    # Draw lines BEFORE green circles (so they appear behind)
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
                # Determine line color with opacity
                base_color = colors["connection"]
                if connection_lines_opacity is not None:
                    # Apply opacity by blending with background
                    opacity = max(0.0, min(1.0, connection_lines_opacity))
                    r = int(base_color[0] * opacity + colors["background"][0] * (1 - opacity))
                    g = int(base_color[1] * opacity + colors["background"][1] * (1 - opacity))
                    b = int(base_color[2] * opacity + colors["background"][2] * (1 - opacity))
                    line_color = (r, g, b)
                else:
                    line_color = base_color
                
                for i in range(min(num_to_show, len(coords_to_use))):
                    for j in range(i + 1, min(num_to_show, len(coords_to_use))):
                        x1, y1 = coords_to_use[i]
                        x2, y2 = coords_to_use[j]
                        draw.line([(x1, y1), (x2, y2)], fill=line_color, width=int(LINE_WIDTH_SCALED))
    
    # Extract centroid coordinates (needed for lines)
    cx, cy = None, None
    if centroid_coord is not None:
        try:
            if isinstance(centroid_coord, np.ndarray):
                cx, cy = float(centroid_coord[0]), float(centroid_coord[1])
            elif isinstance(centroid_coord, (tuple, list)) and len(centroid_coord) >= 2:
                cx, cy = float(centroid_coord[0]), float(centroid_coord[1])
            if cx is not None and cy is not None:
                cx = max(50 * scale, min(cx, MAP_WIDTH_SCALED - 50 * scale))
                cy = max(100 * scale, min(cy, MAP_HEIGHT_SCALED - 50 * scale))
        except (ValueError, TypeError, IndexError, AttributeError):
            cx, cy = None, None
    
    # Draw lines from centroid to Regal 0 points (if centroid is available and not in slow step)
    # Draw BEFORE green circles
    # In "representative" step, only draw green lines (from lines_to_draw), no grey lines
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
                                font_small_num = get_font(int(12 * scale), "thin", mono=True)  # Monofont for numbers
                                num_text = f"{current_distance:.3f}"
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
            # Draw lines from centroid to Regal 0 points (gray on black - less obvious)
            # Only draw grey lines in highlight/centroid/distances steps, not after
            if step in ["highlight", "centroid", "distances"]:
                # Use progressive coords if available
                # Apply opacity if connection_lines_opacity is provided (for fade out animation)
                coords_to_use = shelf0_coords_progressive if (shelf0_coords_progressive is not None and num_shelf0_shown is not None) else shelf0_coords
                if coords_to_use is not None:
                    num_to_show = num_shelf0_shown if num_shelf0_shown is not None else len(coords_to_use)
                    # Determine line color with opacity
                    base_color = colors["connection"]
                    if connection_lines_opacity is not None:
                        # Apply opacity by blending with background
                        opacity = max(0.0, min(1.0, connection_lines_opacity))
                        r = int(base_color[0] * opacity + colors["background"][0] * (1 - opacity))
                        g = int(base_color[1] * opacity + colors["background"][1] * (1 - opacity))
                        b = int(base_color[2] * opacity + colors["background"][2] * (1 - opacity))
                        line_color = (r, g, b)
                    else:
                        line_color = base_color
                    
                    for i in range(min(num_to_show, len(coords_to_use))):
                        x, y = coords_to_use[i]
                        draw.line([(cx, cy), (x, y)], fill=line_color, width=int(LINE_WIDTH_SCALED))
    
    # Draw expanding gray circle for highlight_slow and highlight steps (BEFORE green circle)
    if step in ["highlight_slow", "highlight"] and highlighted_artwork_idx is not None and shelf0_coords is not None and circle_expand_progress is not None:
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
        shelf0_indices_list = [int(x) for x in shelf0_indices.tolist()]
        if highlight_idx_check in shelf0_indices_list:
            highlight_idx_in_shelf0 = shelf0_indices_list.index(highlight_idx_check)
            if highlight_idx_in_shelf0 < len(shelf0_coords):
                hx, hy = shelf0_coords[highlight_idx_in_shelf0]
                # Expanding circle animation: start from small (5px) to full size (20px radius)
                min_radius = 5
                max_radius = 20
                # Use ease-in-out curve
                eased = circle_expand_progress * circle_expand_progress * (3 - 2 * circle_expand_progress)
                current_radius = min_radius + (max_radius - min_radius) * eased
                # Draw gray expanding circle BEHIND green circle
                draw.ellipse([hx - current_radius, hy - current_radius, hx + current_radius, hy + current_radius],
                            fill=None, outline=colors["circle_gray"], width=2)
    
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
            
            draw.ellipse([x - POINT_SIZE_LARGE_SCALED, y - POINT_SIZE_LARGE_SCALED, 
                        x + POINT_SIZE_LARGE_SCALED, y + POINT_SIZE_LARGE_SCALED],
                       fill=current_color, outline=None, width=0)
        
        # Draw title text for Regal items
        # In highlight_slow step, show text for all identified items (they stay visible once identified)
        # This includes the currently appearing item - text appears when item turns green
        # In highlight step, show text for all items (all identified items)
        # In other steps, show text for all items
        if shelf0_coords is not None and df is not None:
            shelf0_indices_list = list(np.where(shelf0_mask)[0])
            # Determine how many items to show text for
            if step == "highlight_slow" and num_shelf0_shown is not None:
                # In identify step, show text for all items that have been identified (0 to num_shelf0_shown - 1)
                # This includes the currently appearing item (at index num_shelf0_shown - 1)
                num_text_to_show = num_shelf0_shown
                text_indices_to_show = list(range(num_text_to_show))
            elif step == "highlight":
                # In highlight step, show text for all items (all identified items)
                num_text_to_show = len(shelf0_coords)
                text_indices_to_show = list(range(num_text_to_show))
            else:
                # In other steps, show text for all items
                num_text_to_show = len(shelf0_coords)
                text_indices_to_show = list(range(num_text_to_show))
            
            for i, (x, y) in enumerate(shelf0_coords):
                if step == "highlight_slow":
                    # Show text for all identified items
                    if i >= num_text_to_show:
                        continue
                elif i >= num_text_to_show:
                    continue  # Skip items not yet shown
                try:
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
                                text_x = x + POINT_SIZE_LARGE_SCALED + 5 * scale
                                text_y = y - 10 * scale
                                font_map_text = get_font(int(FONT_SIZE_MAP_TEXT * scale), "thin")
                                draw.text((int(text_x), int(text_y)), title, fill=colors["text"], font=font_map_text)
                except Exception:
                    pass
    elif step != "all" and shelf0_coords is not None:
        # Normal mode: show all Regal 0 points (but not in "all" step)
        for idx, (x, y) in enumerate(shelf0_coords):
            shelf0_indices = np.where(shelf0_mask)[0]
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
            
            # Draw title text for all Regal items
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
                                text_x = x + POINT_SIZE_LARGE_SCALED + 5 * scale
                                text_y = y - 10 * scale
                                font_map_text = get_font(int(FONT_SIZE_MAP_TEXT * scale), "thin")
                                draw.text((int(text_x), int(text_y)), title, fill=colors["text"], font=font_map_text)
                except Exception:
                    pass
    
    # Highlight currently displayed artwork (if different from representative)
    # Draw gray circle AFTER green circle (so it appears on top but behind other elements)
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
                # Draw gray circle around highlighted artwork (not black) - properly scaled
                circle_radius = int(20 * scale)
                draw.ellipse([int(hx - circle_radius), int(hy - circle_radius), 
                            int(hx + circle_radius), int(hy + circle_radius)],
                            fill=None, outline=colors["circle_gray"], width=int(2 * scale))
    
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
                    # Draw larger bright green circle around representative - properly scaled
                    circle_radius = int(15 * scale)
                    draw.ellipse([int(rx - circle_radius), int(ry - circle_radius), 
                                int(rx + circle_radius), int(ry + circle_radius)],
                                fill=None, outline=colors["lime"], width=int(3 * scale))
    
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
                    # Draw bright green circle around representative - properly scaled
                    circle_radius = int(12 * scale)
                    draw.ellipse([int(x - circle_radius), int(y - circle_radius), 
                                int(x + circle_radius), int(y + circle_radius)],
                                fill=None, outline=colors["lime"], width=int(2 * scale))
    
    # Draw calculation info panel content (text and images) - draw LAST so it's on top
    if highlighted_artwork_idx is not None and df is not None:
        # Find artwork in dataframe
        try:
            artwork_id = all_artwork_ids[highlighted_artwork_idx]
            artwork = None
            
            # Use pre-computed lookup if available (much faster)
            if artwork_lookup is not None:
                artwork_id_str = str(artwork_id).strip()
                artwork_id_normalized = artwork_id_str.replace('.0', '').strip()
                artwork_id_float = None
                try:
                    artwork_id_float = float(artwork_id)
                except (ValueError, TypeError):
                    pass
                
                # Try multiple lookup keys
                lookup_idx = None
                if artwork_id_str in artwork_lookup:
                    lookup_idx = artwork_lookup[artwork_id_str]
                elif artwork_id_normalized in artwork_lookup:
                    lookup_idx = artwork_lookup[artwork_id_normalized]
                elif artwork_id_float is not None and artwork_id_float in artwork_lookup:
                    lookup_idx = artwork_lookup[artwork_id_float]
                
                if lookup_idx is not None:
                    artwork = df.iloc[lookup_idx]
            else:
                # Fallback to original lookup method
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
                
                if not artwork_row.empty:
                    artwork = artwork_row.iloc[0]
        except (IndexError, KeyError, TypeError) as e:
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
                
                # No border - paste image directly
                img_x = MAP_WIDTH_SCALED + (PANEL_WIDTH_SCALED - image.width) // 2
                # Paste image - need to use alpha composite if image has alpha, otherwise direct paste
                if image.mode == 'RGBA':
                    img.paste(image, (int(img_x), int(y_pos)), image)
                else:
                    img.paste(image, (int(img_x), int(y_pos)))
                y_pos += image.height + 20 * scale  # Standardized spacing
            else:
                y_pos += 20 * scale
            
            # Draw divider line after image - standardized spacing
            draw.line([(panel_x, y_pos), (CANVAS_WIDTH_SCALED - 30 * scale, y_pos)], 
                     fill=colors["text"], width=int(LINE_WIDTH_SCALED))
            y_pos += 20 * scale  # Standardized spacing
            
            # Standardized font sizes
            font_section = get_font(int(FONT_SIZE_TABLE * scale), "thin")
            
            # Title - standardized size
            font_title_large = get_font(int(20 * scale), "medium")  # Standardized: 20pt Medium
            title = str(artwork.get("title", "Unknown"))
            max_title_width = PANEL_WIDTH_SCALED - 60 * scale
            y_pos = draw_artwork_title(draw, title, panel_x, y_pos, max_title_width, colors, font_title_large, scale)
            
            # Artist and Year - standardized sizes
            artist = str(artwork.get("artist", "Unknown"))
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
            # Year with standardized font (use monofont for content)
            font_year = get_font(int(FONT_SIZE_TABLE * scale), "thin", mono=True)  # Monofont for year
            draw.text((panel_x, y_pos), year, fill=colors["text"], font=font_year)
            y_pos += 25 * scale  # Standardized spacing
            
            # Divider before two-column fields - standardized layout
            draw.line([(panel_x, y_pos), (CANVAS_WIDTH_SCALED - 30 * scale, y_pos)], 
                     fill=colors["text"], width=int(LINE_WIDTH_SCALED))
            y_pos += 20 * scale  # Standardized spacing
            
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
                draw.line([(panel_x, y_pos), (CANVAS_WIDTH_SCALED - 30 * scale, y_pos)], 
                         fill=colors["text"], width=int(LINE_WIDTH_SCALED))
                y_pos += 20 * scale  # Standardized spacing
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
                            # Use lookup if available
                            if artwork_lookup is not None:
                                artwork_id_str = str(artwork_id).strip()
                                artwork_id_normalized = artwork_id_str.replace('.0', '').strip()
                                lookup_idx = None
                                try:
                                    artwork_id_float = float(artwork_id)
                                    if artwork_id_float in artwork_lookup:
                                        lookup_idx = artwork_lookup[artwork_id_float]
                                except (ValueError, TypeError):
                                    pass
                                if lookup_idx is None and artwork_id_str in artwork_lookup:
                                    lookup_idx = artwork_lookup[artwork_id_str]
                                if lookup_idx is None and artwork_id_normalized in artwork_lookup:
                                    lookup_idx = artwork_lookup[artwork_id_normalized]
                                if lookup_idx is not None:
                                    artwork = df.iloc[lookup_idx]
                                else:
                                    artwork = None
                            else:
                                artwork_row = df[df["id"].astype(float) == float(artwork_id)]
                                if artwork_row.empty:
                                    artwork_row = df[df["id"].astype(str).str.strip() == str(artwork_id).strip()]
                                if not artwork_row.empty:
                                    artwork = artwork_row.iloc[0]
                                else:
                                    artwork = None
                            
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
                                                       fill=None, outline=outline_color, width=3)
                        except Exception:
                            pass
    
    
    # Draw ruler lines from centroid to representative and outlier (ruler step)
    if step == "ruler" and cx is not None and cy is not None and ruler_progress is not None and shelf0_coords is not None:
        # Find representative and outlier coordinates
        rep_coord = None
        outlier_coord = None
        rep_distance = None
        outlier_distance = None
        
        shelf0_indices = np.where(shelf0_mask)[0]
        shelf0_indices_list = [int(x) for x in shelf0_indices.tolist()]
        
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
        
        # Draw ruler line(s)
        # When ruler_to_rep is True: only draw to representative
        # When ruler_to_rep is False: draw to both representative (already complete) and outlier (animating)
        targets_to_draw = []
        if ruler_to_rep:
            # Drawing to representative only
            if rep_coord is not None and rep_distance is not None:
                targets_to_draw.append((rep_coord, rep_distance, True))  # (coord, distance, is_rep)
        else:
            # Drawing to outlier, but also show completed representative line
            if rep_coord is not None and rep_distance is not None:
                targets_to_draw.append((rep_coord, rep_distance, True))  # Show completed rep line
            if outlier_coord is not None and outlier_distance is not None:
                targets_to_draw.append((outlier_coord, outlier_distance, False))  # Animate outlier line
        
        for target_coord, target_distance, is_rep in targets_to_draw:
            # Use full progress for representative when drawing outlier (it's already complete)
            line_progress = 1.0 if (not ruler_to_rep and is_rep) else ruler_progress
            
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
                    label_x = cx + (tx - cx) * (line_progress * 0.5)
                    label_y = cy + (ty - cy) * (line_progress * 0.5)
                    
                    # Offset label perpendicular to line
                    label_offset = 35 * scale
                    label_x += perp_x * label_offset
                    label_y += perp_y * label_offset
                    
                    # Draw distance text in green with better styling
                    distance_text = f"{target_distance:.4f}"
                    font_ruler = get_font(int(FONT_SIZE_LABEL * scale), "medium")  # Larger, bolder font
                    
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
                    bbox = draw.textbbox((int(label_x), int(label_y)), distance_text, font=font_ruler)
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
                    
                    # Draw text on top with fade
                    text_r = int(colors["lime"][0] * fade_progress + colors["background"][0] * (1 - fade_progress))
                    text_g = int(colors["lime"][1] * fade_progress + colors["background"][1] * (1 - fade_progress))
                    text_b = int(colors["lime"][2] * fade_progress + colors["background"][2] * (1 - fade_progress))
                    draw.text((int(label_x), int(label_y)), distance_text, 
                             fill=(text_r, text_g, text_b), font=font_ruler)
    
    # Make representative and outlier dots and text more visible during ruler step
    if step == "ruler" and shelf0_coords is not None and df is not None:
        shelf0_indices = np.where(shelf0_mask)[0]
        shelf0_indices_list = [int(x) for x in shelf0_indices.tolist()]
        
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
        for idx_in_shelf0, coord in enumerate(shelf0_coords):
            if idx_in_shelf0 == rep_idx_in_shelf0 or idx_in_shelf0 == outlier_idx_in_shelf0:
                x, y = coord
                # Draw larger bright green dot
                draw.ellipse([x - RULER_DOT_SIZE_SCALED, y - RULER_DOT_SIZE_SCALED, 
                            x + RULER_DOT_SIZE_SCALED, y + RULER_DOT_SIZE_SCALED],
                           fill=colors["lime"], outline=None, width=0)
                # Draw bright green circle around it
                draw.ellipse([x - RULER_DOT_SIZE_SCALED - 5 * scale, y - RULER_DOT_SIZE_SCALED - 5 * scale, 
                            x + RULER_DOT_SIZE_SCALED + 5 * scale, y + RULER_DOT_SIZE_SCALED + 5 * scale],
                           fill=None, outline=colors["lime"], width=int(3 * scale))
        
        # Draw larger, more visible text labels for representative and outlier
        if rep_idx_in_shelf0 is not None and rep_idx_in_shelf0 < len(shelf0_coords):
            x, y = shelf0_coords[rep_idx_in_shelf0]
            try:
                artwork_idx_for_text = shelf0_indices_list[rep_idx_in_shelf0]
                artwork_id = all_artwork_ids[artwork_idx_for_text]
                artwork_row = df[df["id"].astype(float) == float(artwork_id)]
                if artwork_row.empty:
                    artwork_row = df[df["id"].astype(str).str.strip() == str(artwork_id).strip()]
                if not artwork_row.empty:
                    title = str(artwork_row.iloc[0].get("title", ""))
                    if title and title != "nan":
                        # Draw title text with larger font and background
                        text_x = x + RULER_DOT_SIZE_SCALED + 10 * scale
                        text_y = y - 15 * scale
                        font_ruler_text = get_font(int(24 * scale), "medium")  # Larger font for ruler step
                        # Draw background for text
                        bbox = draw.textbbox((text_x, text_y), title, font=font_ruler_text)
                        padding = int(6 * scale)
                        bg_rect = [bbox[0] - padding, bbox[1] - padding, 
                                  bbox[2] + padding, bbox[3] + padding]
                        draw.rectangle(bg_rect, fill=colors["background"], outline=colors["lime"], width=2)
                        draw.text((text_x, text_y), title, fill=colors["lime"], font=font_ruler_text)
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
                        text_x = x + RULER_DOT_SIZE_SCALED + 10 * scale
                        text_y = y - 15 * scale
                        font_ruler_text = get_font(int(24 * scale), "medium")  # Larger font for ruler step
                        # Draw background for text
                        bbox = draw.textbbox((text_x, text_y), title, font=font_ruler_text)
                        padding = int(6 * scale)
                        bg_rect = [bbox[0] - padding, bbox[1] - padding, 
                                  bbox[2] + padding, bbox[3] + padding]
                        draw.rectangle(bg_rect, fill=colors["background"], outline=colors["lime"], width=2)
                        draw.text((text_x, text_y), title, fill=colors["lime"], font=font_ruler_text)
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
            draw.text((panel_x, y_pos_rep), "Representative", fill=colors["lime"], font=font_section_label)
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
            # Year with standardized font (use monofont for content)
            font_year = get_font(int(FONT_SIZE_TABLE * scale), "thin", mono=True)  # Monofont for year
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
            draw.text((panel_x, y_pos), "Outlier", fill=colors["lime"], font=font_section_label)
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
            # Year with standardized font (use monofont for content)
            font_year = get_font(int(FONT_SIZE_TABLE * scale), "thin", mono=True)  # Monofont for year
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
                    draw.line([(x1, y1), (x2, y2)], fill=colors["lime"], width=int(LINE_WIDTH_SCALED))
    
    # Draw top 10 table (5 representatives + 5 outliers) on right side
    if step == "top10" and top_representatives is not None and top_outliers is not None and df is not None:
        panel_x = MAP_WIDTH_SCALED + 30 * scale
        y_pos = 60 * scale  # Higher up
        
        # Title - larger, more prominent with medium weight (changed from "Top 10")
        font_section_title = get_font(int(32 * scale), "medium")  # Larger, medium weight for prominence
        draw.text((panel_x, y_pos), "Closest to Centroid", fill=colors["lime"], font=font_section_title)
        y_pos += int(50 * scale)  # Less spacing
        
        # Representatives section header (use monofont for labels)
        font_table_label = get_font(int(FONT_SIZE_TABLE * scale), "thin")  # Labels
        font_table_content = get_font(int(FONT_SIZE_TABLE * scale), "thin", mono=True)  # Content (monofont)
        draw.text((panel_x + int(80 * scale), y_pos), "Rank", fill=colors["text"], font=font_table_label)
        draw.text((panel_x + int(140 * scale), y_pos), "Title", fill=colors["text"], font=font_table_label)
        draw.text((panel_x + int(350 * scale), y_pos), "Artist", fill=colors["text"], font=font_table_label)
        draw.text((panel_x + int(520 * scale), y_pos), "Distance", fill=colors["text"], font=font_table_label)
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
                    is_selected = (aesthetic_representative_id is not None and 
                                  (float(artwork_id) == float(aesthetic_representative_id) or 
                                   str(artwork_id) == str(aesthetic_representative_id)))
                    
                    # Only highlight selected items in green (not currently appearing items)
                    # Currently appearing items should stay in normal text color
                    if is_selected:
                        row_color = colors["lime"]
                    else:
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
                    
                    # Title with text wrapping (constraint box: from 140 to 350)
                    title = str(artwork.get("title", "Unknown"))
                    title_x = panel_x + int(140 * scale)
                    title_max_width = int(350 * scale) - title_x - int(10 * scale)  # Leave 10px margin
                    title_words = title.split()
                    title_lines = []
                    current_line = ""
                    for word in title_words:
                        test_line = current_line + (" " if current_line else "") + word
                        bbox = draw.textbbox((0, 0), test_line, font=font_table_content)
                        if bbox[2] - bbox[0] <= title_max_width:
                            current_line = test_line
                        else:
                            if current_line:
                                title_lines.append(current_line)
                            current_line = word
                    if current_line:
                        title_lines.append(current_line)
                    # Draw title (max 2 lines)
                    title_y = y_pos
                    for line in title_lines[:2]:
                        draw.text((title_x, title_y), line, fill=row_color, font=font_table_content)
                        title_y += int(18 * scale)
                    
                    # Artist with text wrapping (constraint box: from 350 to 520)
                    artist = str(artwork.get("artist", "Unknown"))
                    artist_x = panel_x + int(350 * scale)
                    artist_max_width = int(520 * scale) - artist_x - int(10 * scale)  # Leave 10px margin
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
                    draw.text((panel_x + int(520 * scale), y_pos), f"{distance:.4f}", fill=row_color, font=font_table_content)
                    
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
        draw.text((panel_x, y_pos), "Farthest from Centroid", fill=colors["lime"], font=font_section_title)
        y_pos += int(50 * scale)  # Less spacing
        
        # Outliers section header (use monofont for labels)
        font_table_label = get_font(int(FONT_SIZE_TABLE * scale), "thin")  # Labels
        font_table_content = get_font(int(FONT_SIZE_TABLE * scale), "thin", mono=True)  # Content (monofont)
        draw.text((panel_x + int(80 * scale), y_pos), "Rank", fill=colors["text"], font=font_table_label)
        draw.text((panel_x + int(140 * scale), y_pos), "Title", fill=colors["text"], font=font_table_label)
        draw.text((panel_x + int(350 * scale), y_pos), "Artist", fill=colors["text"], font=font_table_label)
        draw.text((panel_x + int(520 * scale), y_pos), "Distance", fill=colors["text"], font=font_table_label)
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
                    is_selected = (aesthetic_outlier_id is not None and 
                                  (float(artwork_id) == float(aesthetic_outlier_id) or 
                                   str(artwork_id) == str(aesthetic_outlier_id)))
                    
                    # Only highlight selected items in green (not currently appearing items)
                    # Currently appearing items should stay in normal text color
                    if is_selected:
                        row_color = colors["lime"]
                    else:
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
                    
                    # Title with text wrapping (constraint box: from 140 to 350)
                    title = str(artwork.get("title", "Unknown"))
                    title_x = panel_x + int(140 * scale)
                    title_max_width = int(350 * scale) - title_x - int(10 * scale)  # Leave 10px margin
                    title_words = title.split()
                    title_lines = []
                    current_line = ""
                    for word in title_words:
                        test_line = current_line + (" " if current_line else "") + word
                        bbox = draw.textbbox((0, 0), test_line, font=font_table_content)
                        if bbox[2] - bbox[0] <= title_max_width:
                            current_line = test_line
                        else:
                            if current_line:
                                title_lines.append(current_line)
                            current_line = word
                    if current_line:
                        title_lines.append(current_line)
                    # Draw title (max 2 lines)
                    title_y = y_pos
                    for line in title_lines[:2]:
                        draw.text((title_x, title_y), line, fill=row_color, font=font_table_content)
                        title_y += int(18 * scale)
                    
                    # Artist with text wrapping (constraint box: from 350 to 520)
                    artist = str(artwork.get("artist", "Unknown"))
                    artist_x = panel_x + int(350 * scale)
                    artist_max_width = int(520 * scale) - artist_x - int(10 * scale)  # Leave 10px margin
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
                    draw.text((panel_x + int(520 * scale), y_pos), f"{distance:.4f}", fill=row_color, font=font_table_content)
                    
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
                info_x = left_x + 300  # Image takes ~300px, info starts after
                
                # Image on the left - bigger size
                rep_image = load_image(rep_artwork.get("thumbnail", ""))
                if rep_image:
                    img_w, img_h = rep_image.size
                    fixed_img_height = min(item_height - 40, 300)  # Bigger image
                    ratio = fixed_img_height / img_h
                    new_w, new_h = int(img_w * ratio), fixed_img_height
                    if new_w > 280:  # Limit image width
                        ratio = 280 / img_w
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
                draw.text((info_x, info_y), "Representative", fill=colors["lime"], font=font_label)
                info_y += 35
                # Wrap title if needed - constrain to MAP_WIDTH_SCALED
                max_title_width = MAP_WIDTH_SCALED - info_x - 20
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
                    info_y += 25
                info_y += 10
                
                # Artist - truncate if too long (use smaller font)
                artist = str(rep_artwork.get("artist", "Unknown"))
                draw.text((info_x, info_y), "Artist", fill=colors["text"], font=font_small)
                info_y += 18
                max_artist_width = MAP_WIDTH_SCALED - info_x - 20
                artist_bbox = draw.textbbox((0, 0), artist, font=font_side)
                if artist_bbox[2] - artist_bbox[0] > max_artist_width:
                    # Truncate artist name
                    while artist and draw.textbbox((0, 0), artist + "...", font=font_side)[2] > max_artist_width:
                        artist = artist[:-1]
                    artist = artist + "..." if artist else "..."
                draw.text((info_x, info_y), artist, fill=colors["text"], font=font_side)
                info_y += 25
                
                # Year
                year = str(rep_artwork.get("year", "N/A"))
                draw.text((info_x, info_y), "Year", fill=colors["text"], font=font_small)
                info_y += 18
                draw.text((info_x, info_y), year, fill=colors["text"], font=font_side)
                info_y += 25
                
                # Size - truncate if too long
                size = str(rep_artwork.get("size", "N/A"))
                if size and size != "N/A" and size != "nan":
                    draw.text((info_x, info_y), "Size", fill=colors["text"], font=font_small)
                    info_y += 18
                    max_size_width = MAP_WIDTH_SCALED - info_x - 20
                    size_bbox = draw.textbbox((0, 0), size, font=font_side)
                    if size_bbox[2] - size_bbox[0] > max_size_width:
                        # Truncate size
                        while size and draw.textbbox((0, 0), size + "...", font=font_side)[2] > max_size_width:
                            size = size[:-1]
                        size = size + "..." if size else "..."
                    draw.text((info_x, info_y), size, fill=colors["text"], font=font_side)
                    info_y += 25
                
                # Distance to centroid
                if distances is not None and rep_idx is not None:
                    shelf0_indices = np.where(shelf0_mask)[0]
                    if rep_idx in shelf0_indices:
                        shelf0_idx = list(shelf0_indices).index(rep_idx)
                        if shelf0_idx < len(distances):
                            distance = distances[shelf0_idx]
                            draw.text((info_x, info_y), "Distance to Centroid", fill=colors["text"], font=font_small)
                            info_y += 18
                            draw.text((info_x, info_y), f"{distance:.4f}", fill=colors["lime"], font=font_side)
                            info_y += 25
                
                # Description (wrapped text)
                description = str(rep_artwork.get("description", ""))
                if description and description != "nan" and description.strip():
                    draw.line([(info_x, info_y), (MAP_WIDTH_SCALED - 50, info_y)], fill=colors["text"], width=1)
                    info_y += 15
                    draw.text((info_x, info_y), "Description", fill=colors["text"], font=font_small)
                    info_y += 18
                    max_width = MAP_WIDTH_SCALED - info_x - 20
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
                        info_y += 16
            
            # Draw divider between representative and outlier on left (aligned to exact center)
            # Calculate exact center of left side area
            divider_y = MAP_HEIGHT_SCALED // 2  # Exact center of canvas height
            divider_start_x = left_x
            divider_end_x = MAP_WIDTH_SCALED - 50 * scale
            draw.line([(divider_start_x, divider_y), (divider_end_x, divider_y)], fill=colors["text"], width=1)
            
            # Outlier (bottom half of left side) - horizontal layout
            if outlier_artwork is not None:
                outlier_y = divider_y + 20
                img_x = left_x
                info_x = left_x + 300  # Image takes ~300px, info starts after
                
                # Image on the left - bigger size
                outlier_image = load_image(outlier_artwork.get("thumbnail", ""))
                if outlier_image:
                    img_w, img_h = outlier_image.size
                    fixed_img_height = min(item_height - 40, 300)  # Bigger image
                    ratio = fixed_img_height / img_h
                    new_w, new_h = int(img_w * ratio), fixed_img_height
                    if new_w > 280:  # Limit image width
                        ratio = 280 / img_w
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
                draw.text((info_x, info_y), "Outlier", fill=colors["lime"], font=font_label)
                info_y += 35
                # Wrap title if needed - constrain to MAP_WIDTH_SCALED
                max_title_width = MAP_WIDTH_SCALED - info_x - 20
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
                    info_y += 25
                info_y += 10
                
                # Artist - truncate if too long (use smaller font)
                artist = str(outlier_artwork.get("artist", "Unknown"))
                draw.text((info_x, info_y), "Artist", fill=colors["text"], font=font_small)
                info_y += 18
                max_artist_width = MAP_WIDTH_SCALED - info_x - 20
                artist_bbox = draw.textbbox((0, 0), artist, font=font_side)
                if artist_bbox[2] - artist_bbox[0] > max_artist_width:
                    # Truncate artist name
                    while artist and draw.textbbox((0, 0), artist + "...", font=font_side)[2] > max_artist_width:
                        artist = artist[:-1]
                    artist = artist + "..." if artist else "..."
                draw.text((info_x, info_y), artist, fill=colors["text"], font=font_side)
                info_y += 25
                
                # Year
                year = str(outlier_artwork.get("year", "N/A"))
                draw.text((info_x, info_y), "Year", fill=colors["text"], font=font_small)
                info_y += 18
                draw.text((info_x, info_y), year, fill=colors["text"], font=font_side)
                info_y += 25
                
                # Size - truncate if too long
                size = str(outlier_artwork.get("size", "N/A"))
                if size and size != "N/A" and size != "nan":
                    draw.text((info_x, info_y), "Size", fill=colors["text"], font=font_small)
                    info_y += 18
                    max_size_width = MAP_WIDTH_SCALED - info_x - 20
                    size_bbox = draw.textbbox((0, 0), size, font=font_side)
                    if size_bbox[2] - size_bbox[0] > max_size_width:
                        # Truncate size
                        while size and draw.textbbox((0, 0), size + "...", font=font_side)[2] > max_size_width:
                            size = size[:-1]
                        size = size + "..." if size else "..."
                    draw.text((info_x, info_y), size, fill=colors["text"], font=font_side)
                    info_y += 25
                
                # Distance to centroid
                if distances is not None and outlier_idx is not None:
                    shelf0_indices = np.where(shelf0_mask)[0]
                    if outlier_idx in shelf0_indices:
                        shelf0_idx = list(shelf0_indices).index(outlier_idx)
                        if shelf0_idx < len(distances):
                            distance = distances[shelf0_idx]
                            draw.text((info_x, info_y), "Distance to Centroid", fill=colors["text"], font=font_small)
                            info_y += 18
                            draw.text((info_x, info_y), f"{distance:.4f}", fill=colors["lime"], font=font_side)
                            info_y += 25
                
                # Description (wrapped text)
                description = str(outlier_artwork.get("description", ""))
                if description and description != "nan" and description.strip():
                    draw.line([(info_x, info_y), (MAP_WIDTH_SCALED - 50, info_y)], fill=colors["text"], width=1)
                    info_y += 15
                    draw.text((info_x, info_y), "Description", fill=colors["text"], font=font_small)
                    info_y += 18
                    max_width = MAP_WIDTH_SCALED - info_x - 20
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
                        info_y += 16
            
            # Draw rank table on right side (match top10 structure with thumbnails)
            panel_x = MAP_WIDTH_SCALED + 30
            y_pos = 100
            
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
            
            y_pos += 20
            draw.line([(panel_x, y_pos), (CANVAS_WIDTH_SCALED - 30, y_pos)], fill=colors["text"], width=1)
            y_pos += 30
            
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
    
    # Downscale from supersampled resolution to final resolution for smooth anti-aliasing
    # Use LANCZOS resampling for highest quality downscaling
    img = img.resize((CANVAS_WIDTH, CANVAS_HEIGHT), Image.Resampling.LANCZOS)
    
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


def main(target_shelf: str = "0", mode: str = "both", white_background: bool = False, supersample_factor: float = 2.0):
    """Main function to generate visualization frames and video.
    
    Args:
        target_shelf: The Regal number to visualize (as string, e.g., "0", "5")
        mode: "representative", "outlier", or "both" - which type to visualize (default: "both")
        white_background: If True, use white background with inverted colors (default: False)
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
    frames_dir = SCRIPT_DIR / "frames" / f"shelf{target_shelf}_both"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    
    # Step 1: Show all embeddings
    print(f"   Generating step 1: All embeddings...")
    for i in tqdm(range(FRAMES_PER_STEP), desc="Step 1 frames"):
        img = create_frame("all", all_coords_2d, all_artwork_ids, shelf0_mask, all_embeddings, 
                          target_shelf=target_shelf, top_representatives=top_items,
                          aesthetic_representative_id=aesthetic_representative_id,
                          white_background=white_background, supersample_factor=supersample_factor,
                          artwork_lookup=artwork_lookup, df=df)
        # Save with high quality PNG settings
        img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG", compress_level=1, optimize=False)
        frame_count += 1
    
    # Step 2: Show Regal items one by one slowly (NO LINES, just identification) with color easing
    print(f"   Generating step 2: Identifying Regal {target_shelf} items (slow, no lines)...")
    shelf0_indices_list = np.where(shelf0_mask)[0].tolist()
    num_shelf0 = len(shelf0_indices_list)
    
    # Show each artwork one by one, more slowly with color easing
    # IMPORTANT: Only show ONE item at a time - each item should appear individually
    FRAMES_PER_ARTWORK_SLOW = 30  # More frames per artwork for slower pace
    EASE_IN_FRAMES = 15  # Frames for color easing animation
    for artwork_num in tqdm(range(1, num_shelf0 + 1), desc="Step 2: Identifying items"):
        # Get the artwork index being added
        current_artwork_idx_in_all = shelf0_indices_list[artwork_num - 1]
        
        # Generate frames for this artwork - NO lines, NO centroid, just showing the item
        CIRCLE_EXPAND_FRAMES = 20  # Frames for circle expansion animation
        for i in range(FRAMES_PER_ARTWORK_SLOW):
            # Calculate color ease progress for the currently appearing item
            if i < EASE_IN_FRAMES:
                color_ease = (i + 1) / EASE_IN_FRAMES
            else:
                color_ease = None  # Fully appeared, no easing needed
            
            # Calculate circle expansion progress (only for the currently appearing item)
            circle_expand = None
            if i < CIRCLE_EXPAND_FRAMES:
                circle_expand = (i + 1) / CIRCLE_EXPAND_FRAMES
            
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
                              circle_expand_progress=circle_expand)
            # Save with high quality PNG settings
            img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG", compress_level=1, optimize=False)
            frame_count += 1
    
    # Hold final state for a bit - show all Regal items
    if num_shelf0 > 0:
        last_artwork_idx = shelf0_indices_list[-1]
        for i in tqdm(range(FRAMES_PER_STEP // 2), desc="Step 2: Hold final"):
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
            img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG", compress_level=1, optimize=False)
            frame_count += 1
    
    # Step 3: Highlight Regal with centroid and distances (combined step) with easing
    print(f"   Generating step 3: Highlighting Regal {target_shelf} with centroid and distances (with easing)...")
    
    # IMPORTANT: All green dots from "identify regal items" step should already be shown
    # We show all items (num_shelf0_shown=num_shelf0) but animate lines/centroid progressively
    # Animate adding each artwork one by one with lines, centroid, and distances
    EASE_IN_FRAMES_STEP3 = 10  # Frames for easing in each item
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
        CIRCLE_EXPAND_FRAMES_STEP3 = 15  # Frames for circle expansion animation in highlight step
        for i in range(FRAMES_PER_ADDITION):
            # Calculate color ease progress for the currently appearing item
            if i < EASE_IN_FRAMES_STEP3:
                color_ease = (i + 1) / EASE_IN_FRAMES_STEP3
            else:
                color_ease = None  # Fully appeared
            
            # Calculate circle expansion progress (only for the currently appearing item)
            circle_expand = None
            if i < CIRCLE_EXPAND_FRAMES_STEP3:
                circle_expand = (i + 1) / CIRCLE_EXPAND_FRAMES_STEP3
            
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
                              circle_expand_progress=circle_expand)
            # Save with high quality PNG settings
            img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG", compress_level=1, optimize=False)
            frame_count += 1
    
    # Hold final state for a bit - show all items with final centroid and distances
    # Increase hold frames for centroid highlighting (more still frames)
    if num_shelf0 > 0:
        last_artwork_idx = shelf0_indices_list[-1]
        for i in tqdm(range(FRAMES_PER_STEP * 2), desc="Step 3: Hold final"):  # Double the hold frames
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
            img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG", compress_level=1, optimize=False)
            frame_count += 1
    
    # Step 3.5: Animate grey connection lines disappearing
    print("   Generating step 3.5: Fading out connection lines...")
    FRAMES_FADE_OUT = 30  # Frames to fade out lines
    for fade_frame in tqdm(range(FRAMES_FADE_OUT), desc="Step 3.5: Fade out"):
        fade_progress = 1.0 - (fade_frame + 1) / FRAMES_FADE_OUT  # 1.0 to 0.0
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
                          connection_lines_opacity=fade_progress)
        # Save with high quality PNG settings
        img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG", compress_level=1, optimize=False)
        frame_count += 1
    
    # Step 4: Cycle through each Regal artwork showing calculations
    # Do representatives first, then outliers
    print(f"   Generating step 4: Cycling through Regal {target_shelf} artworks (representatives first, then outliers)...")
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
    for artwork_idx, dist, shelf0_idx in tqdm(representatives, desc="Representatives"):
        # Draw line from centroid to this artwork (animated)
        artwork_coord = shelf0_coords_2d[shelf0_idx]
        start_coord = centroid_coord_2d
        
        # Animate line drawing for this artwork
        FRAMES_PER_REP_LINE = 25  # Frames to draw line for each representative (slower)
        for frame_in_line in range(FRAMES_PER_REP_LINE):
            line_progress = (frame_in_line + 1) / FRAMES_PER_REP_LINE
            lines_to_draw = [(start_coord, artwork_coord, line_progress, False)]
            
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
                              search_mode="representative")
            # Save with high quality PNG settings
            img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG", compress_level=1, optimize=False)
            frame_count += 1
        
        # Hold the line and show artwork info
        for i in range(FRAMES_PER_ARTWORK - FRAMES_PER_REP_LINE):
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
                              search_mode="representative")
            # Save with high quality PNG settings
            img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG", compress_level=1, optimize=False)
            frame_count += 1
    
    # Then show all outliers
    for artwork_idx, dist, shelf0_idx in tqdm(outliers, desc="Outliers"):
        # Draw line from centroid to this artwork (animated)
        artwork_coord = shelf0_coords_2d[shelf0_idx]
        start_coord = centroid_coord_2d
        
        # Animate line drawing for this artwork
        FRAMES_PER_REP_LINE = 25  # Frames to draw line for each representative (slower)
        for frame_in_line in range(FRAMES_PER_REP_LINE):
            line_progress = (frame_in_line + 1) / FRAMES_PER_REP_LINE
            lines_to_draw = [(start_coord, artwork_coord, line_progress, False)]
            
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
                              search_mode="outlier")
            # Save with high quality PNG settings
            img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG", compress_level=1, optimize=False)
            frame_count += 1
        
        # Hold the line and show artwork info
        for i in range(FRAMES_PER_ARTWORK - FRAMES_PER_REP_LINE):
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
                              search_mode="outlier")
            # Save with high quality PNG settings
            img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG", compress_level=1, optimize=False)
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
            img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG", compress_level=1, optimize=False)
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
                img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG", compress_level=1, optimize=False)
                frame_count += 1
    
    # Step 6: Show top 10 (5 representatives + 5 outliers) appearing one by one with simultaneous line drawing
    print(f"   Generating step 6: Top 10 (5 Representatives + 5 Outliers) appearing one by one...")
    
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
                              top10_reps_shown=rep_num,
                              top10_outliers_shown=0,  # Not showing outliers yet
                              top10_item_ease_progress=ease_progress)
            # Save with high quality PNG settings
            img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG", compress_level=1, optimize=False)
            frame_count += 1
    
    # Hold all representatives shown for a moment
    for i in tqdm(range(FRAMES_PER_STEP // 2), desc="Step 6: Hold reps"):
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
                          top10_outliers_shown=0)
        # Save with high quality PNG settings
        img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG", compress_level=1, optimize=False)
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
            img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG", compress_level=1, optimize=False)
            frame_count += 1
    
    # Hold final state with all items shown
    for i in tqdm(range(FRAMES_PER_STEP), desc="Step 6: Hold final"):
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
        # Save with high quality PNG settings
        img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG", compress_level=1, optimize=False)
        frame_count += 1
    
    # Step 7: Draw ruler lines from centroid to representative and outlier
    print("   Generating step 7: Drawing ruler lines to representative and outlier...")
    FRAMES_PER_RULER = 50  # Frames to draw each ruler line (slower)
    
    # First draw to representative
    for frame_num in tqdm(range(FRAMES_PER_RULER), desc="Step 7: Representative ruler"):
        progress = (frame_num + 1) / FRAMES_PER_RULER
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
                          ruler_to_rep=True)
        # Save with high quality PNG settings
        img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG", compress_level=1, optimize=False)
        frame_count += 1
    
    # Hold representative ruler for a moment
    for i in tqdm(range(FRAMES_PER_STEP // 2), desc="Step 7: Hold rep ruler"):
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
                          ruler_to_rep=True)
        # Save with high quality PNG settings
        img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG", compress_level=1, optimize=False)
        frame_count += 1
    
    # Then draw to outlier (while keeping representative line visible)
    for frame_num in tqdm(range(FRAMES_PER_RULER), desc="Step 7: Outlier ruler"):
        progress = (frame_num + 1) / FRAMES_PER_RULER
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
                          ruler_to_rep=False)
        # Save with high quality PNG settings
        img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG", compress_level=1, optimize=False)
        frame_count += 1
    
    # Hold both rulers visible with info on right side for a couple seconds
    for i in tqdm(range(FRAMES_PER_STEP * 2), desc="Step 7: Hold final"):  # Hold for 4 seconds (120 frames at 30fps)
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
        # Save with high quality PNG settings
        img.save(frames_dir / f"frame_{frame_count:05d}.png", "PNG", compress_level=1, optimize=False)
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
        help="Supersampling factor for anti-aliasing (1.0 = no supersampling for fast test renders, 2.0 = 2x resolution for production quality, default: 2.0)"
    )
    
    args = parser.parse_args()
    main(target_shelf=args.shelf, mode=args.mode, white_background=args.white_background, supersample_factor=args.scale)

