# pip install torch transformers faiss-cpu pandas pillow tqdm
"""
Generate embeddings for all artworks and save to cache.
This script should be run first before using find_cluster_representatives.py
"""
import os
import pickle
from pathlib import Path
from typing import Dict, Optional
from PIL import Image

import numpy as np
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

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
        except Exception as e:
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
            except Exception as e:
                pass
        
        # Try to find by hash or partial match
        actual_file = find_actual_image_file(filename, base_dir)
        if actual_file:
            full_path = images_dir / actual_file
            if full_path.exists():
                try:
                    return Image.open(full_path).convert("RGB")
                except Exception as e:
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


def get_embeddings_cache_file(cache_name: Optional[str] = None) -> Path:
    """Get the path to the global embeddings cache file."""
    cache_dir = get_cache_dir()
    if cache_name:
        return cache_dir / f"{cache_name}.pkl"
    return cache_dir / "all_embeddings.pkl"


def save_all_embeddings(embeddings_dict: Dict, metadata: Optional[Dict] = None, cache_name: Optional[str] = None):
    """
    Save all artwork embeddings as a tensor indexed by artwork ID.
    
    Args:
        embeddings_dict: Dictionary mapping artwork ID (int or str) to embedding vector
        metadata: Optional metadata dictionary (e.g., CSV hash, model name, etc.)
        cache_name: Optional custom name for the cache file (without .pkl extension)
    """
    cache_file = get_embeddings_cache_file(cache_name)
    
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
    force_regenerate: bool = False,
    cache_name: Optional[str] = None
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
    cache_file = get_embeddings_cache_file(cache_name)
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
        save_all_embeddings(embeddings_dict, metadata, cache_name)
    
    return embeddings_dict


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for all artworks and save to cache.")
    parser.add_argument("--csv", default="artworks_with_thumbnails_ting.csv", help="Path to CSV file")
    parser.add_argument("--base-dir", default="production-export-2025-11-04t14-27-00-000z", help="Base directory for images")
    parser.add_argument("--force", action="store_true", help="Force regeneration of all embeddings (ignore cache)")
    parser.add_argument("--cache-name", default=None, help="Custom name for the embeddings cache file (without .pkl extension)")
    args = parser.parse_args()
    
    generate_embeddings(args.csv, args.base_dir, force_regenerate=args.force, cache_name=args.cache_name)
    print("\n✓ Embedding generation complete!")


if __name__ == "__main__":
    main()

