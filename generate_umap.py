# pip install torch transformers faiss-cpu pandas pillow tqdm umap-learn
"""
Generate embeddings for all artworks and save to cache.
This script should be run first before using find_cluster_representatives.py
"""
import os
import pickle
import math
import re
from pathlib import Path
from typing import Dict, Optional
from PIL import Image, ImageDraw

import numpy as np
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import umap  # for UMAP dimensionality reduction

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

MODEL_NAME = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and processor
print(f"Loading CLIP model: {MODEL_NAME}")
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()


def extract_filename_from_asset(asset_value) -> Optional[str]:
    """Extract filename from Sanity asset reference."""
    if not asset_value or pd.isna(asset_value):
        return None
    
    asset_str = str(asset_value).strip()
    
    # Handle Sanity asset format: "image@file://./images/filename.jpg"
    if "@file://" in asset_str:
        if "./images/" in asset_str:
            filename = asset_str.split("./images/")[-1]
        elif "./files/" in asset_str:
            filename = asset_str.split("./files/")[-1]
        elif "/" in asset_str:
            parts = asset_str.split("/")
            filename = parts[-1]
        else:
            return None
        # Remove query parameters if present
        if "?" in filename:
            filename = filename.split("?")[0]
        return filename
    
    return None


def find_actual_image_file(filename: str, base_dir: str = "production-export-2025-11-04t14-27-00-000z") -> Optional[str]:
    """Find the actual image file in the images directory, handling variations in filename."""
    if not filename:
        return None
    
    images_dir = Path(base_dir) / "images"
    if not images_dir.exists():
        return None
    
    # Try exact match first
    exact_path = images_dir / filename
    if exact_path.exists():
        return filename
    
    # Try to find by hash (first part before dimensions)
    # Filenames are like: "b44fd62fdf7c85cf0e0fbd5e49c30cf137e118-7890x5320.jpg"
    # Extract hash part (before first dash or dimension pattern)
    hash_part = filename.split("-")[0] if "-" in filename else filename.split(".")[0]
    
    # Search for files starting with this hash
    for img_file in images_dir.iterdir():
        if img_file.is_file() and img_file.name.startswith(hash_part):
            return img_file.name
    
    # Try case-insensitive search
    filename_lower = filename.lower()
    for img_file in images_dir.iterdir():
        if img_file.is_file() and img_file.name.lower() == filename_lower:
            return img_file.name
    
    return None


def load_image(image_path: str, base_dir: str = "production-export-2025-11-04t14-27-00-000z") -> Optional[Image.Image]:
    """Load image from thumbnail path. Handles relative paths from CSV."""
    if pd.isna(image_path) or not image_path:
        return None
    
    # Paths from CSV are relative like: production-export-2025-11-04t14-27-00-000z/images/filename.jpg
    # Try using the path directly first
    full_path = Path(image_path)
    if full_path.exists() and full_path.is_file():
        try:
            return Image.open(full_path).convert("RGB")
        except Exception:
            pass
    
    # If direct path doesn't work, try extracting filename and looking in base_dir/images
    # Handle paths like "production-export-2025-11-04t14-27-00-000z/images/filename.jpg"
    if "images/" in image_path:
        filename = image_path.split("images/")[-1]
    elif image_path.startswith("/thumbnails/"):
        filename = image_path.replace("/thumbnails/", "")
    else:
        filename = image_path
    
    # Try in base_dir/images
    images_dir = Path(base_dir) / "images"
    if images_dir.exists():
        full_path = images_dir / filename
        if full_path.exists():
            try:
                return Image.open(full_path).convert("RGB")
            except Exception:
                pass
        
        # Try to find by hash or partial match
        actual_file = find_actual_image_file(filename, base_dir)
        if actual_file:
            full_path = images_dir / actual_file
            if full_path.exists():
                try:
                    return Image.open(full_path).convert("RGB")
                except Exception:
                    pass
    
    return None


@torch.inference_mode()
def embed_text(text: str, normalize: bool = True) -> np.ndarray:
    """Embed a single text string."""
    if not text or pd.isna(text):
        return np.zeros(512)  # CLIP text embedding dimension
    
    inputs = processor(text=[str(text)], return_tensors="pt", padding=True, truncation=True).to(device)
    text_features = model.get_text_features(**inputs)
    if normalize:
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)
    return text_features.cpu().numpy()[0]


@torch.inference_mode()
def embed_image(image: Image.Image, normalize: bool = True) -> np.ndarray:
    """Embed a single image."""
    if image is None:
        return np.zeros(512)  # CLIP image embedding dimension
    
    inputs = processor(images=[image], return_tensors="pt").to(device)
    image_features = model.get_image_features(**inputs)
    if normalize:
        image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
    return image_features.cpu().numpy()[0]


@torch.inference_mode()
def embed_artwork(row: pd.Series, base_dir: str = "production-export-2025-11-04t14-27-00-000z") -> np.ndarray:
    """
    Create a combined embedding for an artwork.
    Combines text fields (artist, title, year, description, size, handling_status, internalNote) and thumbnail image.
    Title has stronger weight in the embedding.
    Uses weighted combination: 60% text, 40% image (or 100% text if image unavailable).
    """
    # Collect text fields
    text_parts = []
    
    # Artist name
    artist = row.get("artist", "")
    if pd.notna(artist) and str(artist).strip():
        text_parts.append(f"artist: {str(artist).strip()}")
    
    # Title (will be embedded separately with higher weight)
    title = row.get("title", "")
    title_text = ""
    if pd.notna(title) and str(title).strip():
        title_text = f"title: {str(title).strip()}"
        text_parts.append(title_text)
    
    # Year
    year = row.get("year", "")
    if pd.notna(year) and str(year).strip():
        text_parts.append(f"year: {str(year).strip()}")
    
    # Description
    description = row.get("description", "")
    if pd.notna(description) and str(description).strip():
        text_parts.append(f"description: {str(description).strip()}")
    
    # Size
    size = row.get("size", "")
    if pd.notna(size) and str(size).strip():
        text_parts.append(f"size: {str(size).strip()}")
    
    # Handling status
    handling_status = row.get("handling_status", "")
    if pd.notna(handling_status) and str(handling_status).strip():
        text_parts.append(f"handling_status: {str(handling_status).strip()}")
    
    # Internal note
    internal_note = row.get("internalNote", "")
    if pd.notna(internal_note) and str(internal_note).strip():
        text_parts.append(f"internalNote: {str(internal_note).strip()}")
    
    # Combine all text fields (except title which will be weighted separately)
    other_text = ". ".join([p for p in text_parts if not p.startswith("title:")]) if text_parts else ""
    
    # Embed title separately with higher weight
    title_emb = embed_text(title_text) if title_text else np.zeros(512)
    
    # Embed other fields
    other_emb = embed_text(other_text) if other_text else np.zeros(512)
    
    # Combine text embeddings with title having stronger weight (40% title, 60% other)
    text_emb = 0.4 * title_emb + 0.6 * other_emb
    
    # Load and embed thumbnail image
    # Read thumbnail path directly from CSV (already set by read_data_ting.py)
    thumbnail_path = row.get("thumbnail", "")
    image = load_image(thumbnail_path, base_dir) if thumbnail_path and pd.notna(thumbnail_path) else None
    image_emb = embed_image(image) if image is not None else None
    
    # Combine text and image embeddings
    if image_emb is not None and np.any(image_emb):  # Image available and non-zero
        # Weighted combination: 60% text, 40% image
        combined_emb = 0.6 * text_emb + 0.4 * image_emb
    else:
        # Use only text embedding
        combined_emb = text_emb
    
    # Normalize the combined embedding
    norm = np.linalg.norm(combined_emb)
    if norm > 0:
        combined_emb = combined_emb / norm
    
    return combined_emb


def get_cache_dir() -> Path:
    """Get the cache directory for embeddings."""
    cache_dir = Path("embeddings_cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def get_embeddings_cache_file() -> Path:
    """Get the path to the global embeddings cache file."""
    cache_dir = get_cache_dir()
    return cache_dir / "all_embeddings.pkl"


def save_all_embeddings(embeddings_dict: Dict, metadata: Optional[Dict] = None):
    """
    Save all artwork embeddings as a tensor indexed by artwork ID.
    
    Args:
        embeddings_dict: Dictionary mapping artwork ID (int or str) to embedding vector
        metadata: Optional metadata dictionary (e.g., CSV hash, model name, etc.)
    """
    cache_file = get_embeddings_cache_file()
    
    cache_data = {
        "embeddings": embeddings_dict,  # Dict[str, np.ndarray] - ID -> embedding
        "metadata": metadata or {},
        "num_embeddings": len(embeddings_dict),
        "embedding_dim": next(iter(embeddings_dict.values())).shape[0] if embeddings_dict else 512
    }
    
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f)
    
    print(f"✓ Saved {len(embeddings_dict)} embeddings to cache")


def generate_embeddings(
    csv_path: str,
    base_dir: str = "production-export-2025-11-04t14-27-00-000z",
    force_regenerate: bool = False
) -> Dict:
    """
    Generate embeddings for all artworks in the CSV and save to cache.
    
    Args:
        csv_path: Path to CSV file with artwork data
        base_dir: Base directory for images
        force_regenerate: If True, regenerate all embeddings even if cache exists
    
    Returns:
        Dictionary mapping artwork ID to embedding vector
    """
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} artworks")
    
    # Check if cache exists and load it
    cache_file = get_embeddings_cache_file()
    cached_embeddings = {}
    
    if cache_file.exists() and not force_regenerate:
        try:
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)
            cached_embeddings = cache_data.get("embeddings", {})
            print(f"✓ Found {len(cached_embeddings)} cached embeddings")
        except Exception as e:
            print(f"⚠ Error loading cache: {e}, will regenerate")
            cached_embeddings = {}
    
    # Generate embeddings for all artworks
    print("Generating embeddings...")
    embeddings_dict = {}  # ID -> embedding
    new_count = 0
    cached_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing artworks"):
        # Get artwork ID - try to keep as int if possible
        raw_id = row.get("id", idx)
        # Try to convert to int, fallback to original value if not numeric
        try:
            artwork_id = int(float(raw_id)) if pd.notna(raw_id) else idx
        except (ValueError, TypeError):
            artwork_id = raw_id if pd.notna(raw_id) else idx
        
        # Check if we already have this embedding in cache
        emb = None
        if cached_embeddings and not force_regenerate:
            if artwork_id in cached_embeddings:
                emb = cached_embeddings[artwork_id]
            elif str(artwork_id) in cached_embeddings:
                emb = cached_embeddings[str(artwork_id)]
        
        if emb is not None:
            embeddings_dict[artwork_id] = emb
            cached_count += 1
        else:
            # Compute new embedding
            try:
                emb = embed_artwork(row, base_dir)
                embeddings_dict[artwork_id] = emb
                new_count += 1
            except Exception as e:
                print(f"Error embedding artwork at index {idx} (ID: {artwork_id}): {e}")
                continue
    
    # Save all embeddings to cache
    print(f"\n✓ Generated {new_count} new embeddings")
    print(f"✓ Used {cached_count} cached embeddings")
    print(f"✓ Total embeddings: {len(embeddings_dict)}")
    
    if new_count > 0 or force_regenerate:
        print("Saving embeddings to cache...")
        metadata = {
            "csv_path": csv_path,
            "model_name": MODEL_NAME,
            "num_artworks": len(df)
        }
        save_all_embeddings(embeddings_dict, metadata)
    
    return embeddings_dict


# ---------- size parsing (NEW) ----------

def parse_size_str(s):
    if not isinstance(s, str) or not s.strip():
        return np.nan, np.nan, np.nan
    s = s.lower().replace('×', 'x').strip()
    h_match = re.search(r'h\s*:\s*(-?\d+(?:\.\d+)?)', s)
    w_match = re.search(r'w\s*:\s*(-?\d+(?:\.\d+)?)', s)
    d_match = re.search(r'd\s*:\s*(-?\d+(?:\.\d+)?)', s)
    h = float(h_match.group(1)) if h_match else np.nan
    w = float(w_match.group(1)) if w_match else np.nan
    d = float(d_match.group(1)) if d_match else np.nan
    return h, w, d


# ---------- UMAP + CSV utilities (NEW) ----------

def compute_umap_2d(embeddings_dict: Dict) -> tuple[np.ndarray, list]:
    """
    Compute a 2D UMAP embedding (x, y) for all embeddings.

    Args:
        embeddings_dict: Dict[artwork_id, np.ndarray] of shape (embedding_dim,)

    Returns:
        reduced: np.ndarray of shape (num_items, 2)
        ids: list of artwork_ids in the same order as reduced
    """
    if not embeddings_dict:
        raise ValueError("No embeddings provided for UMAP.")

    # Stack embeddings into a matrix respecting dict insertion order
    ids = list(embeddings_dict.keys())
    emb_matrix = np.stack([embeddings_dict[k] for k in ids], axis=0)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=18,   # Increased for more global structure
        min_dist=0.0,     # Lower value to pack points closer together
        spread=12,        # Lower spread for tighter clusters
        metric="cosine",
        random_state=21,
    )
    reduced = reducer.fit_transform(emb_matrix)
    return reduced, ids


def generate_size_cube_images(
    df: pd.DataFrame,
    width_col: str = "width",
    height_col: str = "height",
    depth_col: str = "depth",
    output_dir: Path | str = "size_cubes",
    image_size: int = 1024,
):
    """
    Generate isometric cube PNGs for each artwork based on width/height/depth.

    - Black lines on transparent background.
    - All cubes scaled so the largest artwork roughly fills the canvas.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect global max dimensions
    dims = []

    for _, row in df.iterrows():
        w = row.get(width_col, np.nan)
        h = row.get(height_col, np.nan)
        d = row.get(depth_col, np.nan)
        for v in (w, h, d):
            if pd.notna(v) and v > 0:
                dims.append(v)

    if not dims:
        print("⚠ No valid size data to generate cubes.")
        return

    # Global max per dimension
    w_max = df[width_col].replace({np.nan: 0}).max()
    h_max = df[height_col].replace({np.nan: 0}).max()
    d_max = df[depth_col].replace({np.nan: 0}).max()

    # Isometric projection parameters
    angle = math.radians(30)
    cos30 = math.cos(angle)
    sin30 = math.sin(angle)

    # Estimate max projected width/height for the biggest artwork
    proj_width = (w_max + d_max) * cos30
    proj_height = (w_max + d_max) * sin30 + h_max

    if proj_width <= 0 or proj_height <= 0:
        print("⚠ Invalid projected dimensions, skipping cube generation.")
        return

    target_extent = image_size * 0.8  # leave a margin
    scale = min(target_extent / proj_width, target_extent / proj_height)

    def iso_project(x, y, z):
        px = (x - y) * cos30
        py = (x + y) * sin30 - z
        return px, py

    for idx, row in df.iterrows():
        w = row.get(width_col, np.nan)
        h = row.get(height_col, np.nan)
        d = row.get(depth_col, np.nan)

        if not (pd.notna(w) and pd.notna(h) and pd.notna(d)):
            continue
        if w <= 0 or h <= 0 or d <= 0:
            continue

        # Scaled dimensions
        ws = w * scale
        hs = h * scale
        ds = d * scale

        # 3D vertices of the box
        vertices = [
            (0, 0, 0),
            (ws, 0, 0),
            (ws, ds, 0),
            (0, ds, 0),
            (0, 0, hs),
            (ws, 0, hs),
            (ws, ds, hs),
            (0, ds, hs),
        ]

        # Project to 2D
        proj = [iso_project(x, y, z) for (x, y, z) in vertices]
        xs = [p[0] for p in proj]
        ys = [p[1] for p in proj]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width_span = max_x - min_x
        height_span = max_y - min_y

        # Center in image
        offset_x = (image_size - width_span) / 2 - min_x
        offset_y = (image_size - height_span) / 2 - min_y

        proj_centered = [(x + offset_x, y + offset_y) for (x, y) in proj]

        # Create image
        img = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # edges of cube (by vertex indices)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # top
            (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
        ]

        for a, b in edges:
            x1, y1 = proj_centered[a]
            x2, y2 = proj_centered[b]
            draw.line((x1, y1, x2, y2), fill=(0, 0, 0, 255), width=3)

        # Determine filename based on artwork id (same logic as embeddings)
        raw_id = row.get("id", idx)
        try:
            artwork_id = int(float(raw_id)) if pd.notna(raw_id) else idx
        except (ValueError, TypeError):
            artwork_id = raw_id if pd.notna(raw_id) else idx

        filename = f"{artwork_id}.png"
        img.save(output_dir / filename, "PNG")

    print(f"✓ Saved size cubes to {output_dir}")


def save_embeddings_and_umap_to_csv(
    embeddings_dict: Dict,
    original_csv_path: str,
    output_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    Save a CSV that contains:
      - all columns from the original CSV
      - parsed size columns: 'height', 'width', 'depth'
      - 'umap_x', 'umap_y'
      - 'emb_0' ... 'emb_(d-1)'

    Embeddings are matched to rows via the same artwork_id logic
    used in generate_embeddings (based on 'id' column or row index).

    Also generates isometric cube PNGs for each artwork in a
    folder '<output_csv_stem>_size_cubes'.
    """
    df = pd.read_csv(original_csv_path)

    # Parse size -> height, width, depth
    if "size" in df.columns:
        parsed = df["size"].apply(lambda x: parse_size_str(x))
        df["height"], df["width"], df["depth"] = zip(*parsed)
    else:
        df["height"] = np.nan
        df["width"] = np.nan
        df["depth"] = np.nan

    if output_csv is None:
        stem, ext = os.path.splitext(original_csv_path)
        output_csv = stem + "_with_embeddings_umap" + (ext or ".csv")

    # Build an ordered embeddings dict consistent with the row order of df
    ordered_embeddings: Dict = {}
    id_to_row_index: Dict = {}

    for idx, row in df.iterrows():
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
            # keep in dataframe row order
            ordered_embeddings[artwork_id] = emb
            id_to_row_index[artwork_id] = idx

    if not ordered_embeddings:
        raise ValueError("No matching embeddings found for rows in original CSV.")

    # Compute UMAP on these (in consistent order)
    reduced, ordered_ids = compute_umap_2d(ordered_embeddings)
    emb_matrix = np.stack([ordered_embeddings[k] for k in ordered_ids], axis=0)
    num_items, emb_dim = emb_matrix.shape

    # Initialize new columns with NaNs
    df["umap_x"] = np.nan
    df["umap_y"] = np.nan
    for dim in range(emb_dim):
        df[f"emb_{dim}"] = np.nan

    # Map UMAP + embedding values back into df rows
    row_indices = [id_to_row_index[art_id] for art_id in ordered_ids]

    # UMAP columns
    df.loc[row_indices, "umap_x"] = reduced[:, 0]
    df.loc[row_indices, "umap_y"] = reduced[:, 1]

    # Embedding columns
    for dim in range(emb_dim):
        col_values = np.full(len(df), np.nan, dtype=float)
        col_values[row_indices] = emb_matrix[:, dim]
        df[f"emb_{dim}"] = col_values

    df.to_csv(output_csv, index=False)
    print(f"✓ Saved full CSV (original + size + embeddings + UMAP) for {num_items} artworks to {output_csv}")

    # Generate cubes into a folder derived from the CSV name
    cube_dir = Path(output_csv).parent / (Path(output_csv).stem + "_size_cubes")
    generate_size_cube_images(df, width_col="width", height_col="height", depth_col="depth", output_dir=cube_dir)

    return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for all artworks and save to cache.")
    parser.add_argument("--csv", default="artworks_with_thumbnails_ting.csv", help="Path to CSV file")
    parser.add_argument("--base-dir", default="production-export-2025-11-04t14-27-00-000z", help="Base directory for images")
    parser.add_argument("--force", action="store_true", help="Force regeneration of all embeddings (ignore cache)")
    # This now saves a CSV with original columns + embeddings + UMAP + size cols
    parser.add_argument(
        "--save-umap-csv",
        default=None,
        help="Optional path to save full CSV (original columns + size + embeddings + 2D UMAP). "
             "If omitted but flag is present, uses '<input>_with_embeddings_umap.csv'.",
    )
    args = parser.parse_args()
    
    embeddings_dict = generate_embeddings(args.csv, args.base_dir, force_regenerate=args.force)

    # If requested, save full merged CSV
    if args.save_umap_csv is not None:
        # If user passed an empty string by accident, treat as None
        out_path = args.save_umap_csv if args.save_umap_csv.strip() else None
        save_embeddings_and_umap_to_csv(embeddings_dict, args.csv, out_path)

    print("\n✓ Embedding generation complete!")


if __name__ == "__main__":
    main()
