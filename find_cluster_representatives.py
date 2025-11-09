# pip install torch transformers faiss-cpu pandas pillow tqdm
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

try:
    faiss.omp_set_num_threads(1)
except Exception:
    pass

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


def make_thumbnail_path(row: pd.Series, base_dir: str = "production-export-2025-11-04t14-27-00-000z") -> str:
    """Construct a thumbnail path for each artwork, verifying the file actually exists."""
    BASE_THUMB_PATH = "/thumbnails"
    images_dir = Path(base_dir) / "images"
    
    # Try to extract filename from thumbnail._sanityAsset
    # Access the column with dot in name - use direct indexing which works for Series
    thumb_asset = None
    if "thumbnail._sanityAsset" in row.index:
        thumb_asset = row["thumbnail._sanityAsset"]
    else:
        # Fallback: try get method or convert to dict
        thumb_asset = row.get("thumbnail._sanityAsset", None)
        if thumb_asset is None:
            row_dict = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
            thumb_asset = row_dict.get("thumbnail._sanityAsset")
    
    if thumb_asset and pd.notna(thumb_asset):
        filename = extract_filename_from_asset(thumb_asset)
        if filename:
            # Verify the file actually exists
            actual_file = find_actual_image_file(filename, base_dir)
            if actual_file:
                return f"{BASE_THUMB_PATH}/{actual_file}"
    
    # Fallback: search images directory for files matching hash patterns in the row
    if images_dir.exists():
        row_dict = row.to_dict() if hasattr(row, 'to_dict') else {k: row[k] for k in row.index}
        for key, value in row_dict.items():
            if value and isinstance(value, str) and len(value) > 20:
                # Look for hash-like patterns
                for img_file in images_dir.iterdir():
                    if img_file.is_file():
                        # Check if hash part matches
                        hash_part = value[:40] if len(value) >= 40 else value
                        if hash_part in img_file.name or img_file.name.startswith(hash_part):
                            return f"{BASE_THUMB_PATH}/{img_file.name}"
    
    # Fallback to slug-based naming (but won't verify existence)
    slug = row.get("slug", "")
    if slug and pd.notna(slug):
        slug_str = str(slug).strip().replace(" ", "-").replace("_", "-")
        slug_str = slug_str.replace("/", "-").replace("\\", "-")
        while "--" in slug_str:
            slug_str = slug_str.replace("--", "-")
        slug_str = slug_str.strip("-")
        if slug_str:
            # Check if slug-based file exists
            slug_file = find_actual_image_file(f"{slug_str}.jpg", base_dir)
            if slug_file:
                return f"{BASE_THUMB_PATH}/{slug_file}"
            return f"{BASE_THUMB_PATH}/{slug_str}.jpg"
    
    # Last resort: use _id
    artwork_id = row.get("_id", "")
    if artwork_id and pd.notna(artwork_id):
        artwork_id_clean = str(artwork_id).replace("drafts.", "").replace(".", "-")
        id_file = find_actual_image_file(f"{artwork_id_clean}.jpg", base_dir)
        if id_file:
            return f"{BASE_THUMB_PATH}/{id_file}"
        return f"{BASE_THUMB_PATH}/{artwork_id_clean}.jpg"
    
    return f"{BASE_THUMB_PATH}/unknown.jpg"


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


def find_cluster_representatives(
    csv_path: str,
    base_dir: str = "production-export-2025-11-04t14-27-00-000z",
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Find the top 5 most representative artworks for each cluster (closest to centroid).
    Returns a DataFrame with cluster, index, id, title, artist, artwork_no, 
    shelf_no, rank, num_artworks_in_cluster, and distance_to_centroid.
    """
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} artworks")
    
    # Filter to artworks with cluster information
    if "cluster" not in df.columns:
        print("ERROR: 'cluster' column not found in CSV")
        return pd.DataFrame()
    
    df = df[df["cluster"].notna()].copy()
    print(f"Artworks with cluster info: {len(df)}")
    
    # Create embeddings for all artworks
    print("Creating embeddings...")
    embeddings = []
    valid_indices = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding artworks"):
        try:
            emb = embed_artwork(row, base_dir)
            embeddings.append(emb)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error embedding artwork at index {idx}: {e}")
            continue
    
    if not embeddings:
        print("ERROR: No embeddings created")
        return pd.DataFrame()
    
    embeddings = np.array(embeddings, dtype=np.float32)
    df_valid = df.loc[valid_indices].copy()
    print(f"Created {len(embeddings)} embeddings")
    
    # Group by cluster and find representative for each
    print("Finding cluster representatives...")
    results = []
    
    unique_clusters = sorted(df_valid["cluster"].unique())
    for cluster_id in tqdm(unique_clusters, desc="Processing clusters"):
        cluster_mask = df_valid["cluster"] == cluster_id
        cluster_df = df_valid[cluster_mask].copy()
        cluster_embeddings = embeddings[cluster_mask]
        
        if len(cluster_embeddings) == 0:
            continue
        
        # Calculate centroid (average embedding) of the cluster
        centroid = np.mean(cluster_embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)  # Normalize
        
        # Find distances to centroid for all artworks in cluster
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        
        # Get top 5 closest artworks (sorted by distance)
        top_k = min(5, len(distances))
        top_indices = np.argsort(distances)[:top_k]
        
        # Add top 5 artworks to results
        for rank, idx in enumerate(top_indices, 1):
            artwork = cluster_df.iloc[idx]
            
            # Get the id column value from CSV (not row index)
            csv_id = artwork.get("id", "N/A")
            
            title = artwork.get("title", "Unknown")
            artist = artwork.get("artist", "Unknown")
            artwork_no = artwork.get("artworkNo", artwork.get("id", "N/A"))
            shelf_no = artwork.get("shelfNo", "N/A")
            artwork_id = artwork.get("_id", artwork.get("id", "N/A"))
            # Read thumbnail path directly from CSV (already set by read_data_ting.py)
            thumbnail = artwork.get("thumbnail", "N/A")
            
            results.append({
                "cluster": cluster_id,
                "index": csv_id,
                "id": artwork_id,
                "title": title,
                "artist": artist,
                "artwork_no": artwork_no,
                "shelf_no": shelf_no,
                "thumbnail": thumbnail,
                "rank": rank,
                "num_artworks_in_cluster": len(cluster_df),
                "distance_to_centroid": float(distances[idx])
            })
        
    return pd.DataFrame(results)


def find_cluster_outliers(
    csv_path: str,
    base_dir: str = "production-export-2025-11-04t14-27-00-000z",
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Find the top 5 outlier artworks for each cluster (furthest from centroid).
    Returns a DataFrame with cluster, index, id, title, artist, artwork_no, 
    shelf_no, rank, num_artworks_in_cluster, and distance_to_centroid.
    """
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} artworks")
    
    # Filter to artworks with cluster information
    if "cluster" not in df.columns:
        print("ERROR: 'cluster' column not found in CSV")
        return pd.DataFrame()
    
    df = df[df["cluster"].notna()].copy()
    print(f"Artworks with cluster info: {len(df)}")
    
    # Create embeddings for all artworks
    print("Creating embeddings...")
    embeddings = []
    valid_indices = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding artworks"):
        try:
            emb = embed_artwork(row, base_dir)
            embeddings.append(emb)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error embedding artwork at index {idx}: {e}")
            continue
    
    if not embeddings:
        print("ERROR: No embeddings created")
        return pd.DataFrame()
    
    embeddings = np.array(embeddings, dtype=np.float32)
    df_valid = df.loc[valid_indices].copy()
    print(f"Created {len(embeddings)} embeddings")
    
    # Group by cluster and find outliers for each
    print("Finding cluster outliers...")
    results = []
    
    unique_clusters = sorted(df_valid["cluster"].unique())
    for cluster_id in tqdm(unique_clusters, desc="Processing clusters"):
        cluster_mask = df_valid["cluster"] == cluster_id
        cluster_df = df_valid[cluster_mask].copy()
        cluster_embeddings = embeddings[cluster_mask]
        
        if len(cluster_embeddings) == 0:
            continue
        
        # Calculate centroid (average embedding) of the cluster
        centroid = np.mean(cluster_embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)  # Normalize
        
        # Find distances to centroid for all artworks in cluster
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        
        # Get top 5 furthest artworks (sorted by distance, descending)
        top_k = min(5, len(distances))
        top_indices = np.argsort(distances)[-top_k:][::-1]  # Get largest distances first
        
        # Add top 5 outliers to results
        for rank, idx in enumerate(top_indices, 1):
            artwork = cluster_df.iloc[idx]
            
            # Get the id column value from CSV (not row index)
            csv_id = artwork.get("id", "N/A")
            
            title = artwork.get("title", "Unknown")
            artist = artwork.get("artist", "Unknown")
            artwork_no = artwork.get("artworkNo", artwork.get("id", "N/A"))
            shelf_no = artwork.get("shelfNo", "N/A")
            artwork_id = artwork.get("_id", artwork.get("id", "N/A"))
            # Read thumbnail path directly from CSV (already set by read_data_ting.py)
            thumbnail = artwork.get("thumbnail", "N/A")
            
            results.append({
                "cluster": cluster_id,
                "index": csv_id,
                "id": artwork_id,
                "title": title,
                "artist": artist,
                "artwork_no": artwork_no,
                "shelf_no": shelf_no,
                "thumbnail": thumbnail,
                "rank": rank,
                "num_artworks_in_cluster": len(cluster_df),
                "distance_to_centroid": float(distances[idx])
            })
        
    return pd.DataFrame(results)


def find_shelf_representatives(
    csv_path: str,
    base_dir: str = "production-export-2025-11-04t14-27-00-000z",
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Find the top 5 most representative artworks for each shelf (closest to centroid).
    Returns a DataFrame with shelf_no, index, id, title, artist, artwork_no, 
    rank, num_artworks_in_shelf, and distance_to_centroid.
    """
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} artworks")
    
    # Filter to artworks with shelf information
    if "shelfNo" not in df.columns:
        print("ERROR: 'shelfNo' column not found in CSV")
        return pd.DataFrame()
    
    df = df[df["shelfNo"].notna()].copy()
    df = df[df["shelfNo"] != ""].copy()  # Remove empty shelf numbers
    print(f"Artworks with shelf info: {len(df)}")
    
    # Handle multiple shelf numbers per artwork (split by semicolon or comma)
    # This duplicates the logic from read_data_ting.py to ensure proper splitting
    def parse_shelf_numbers(shelf_value):
        """Parse shelfNo value and return list of individual shelf numbers."""
        if pd.isna(shelf_value) or shelf_value is None:
            return []
        shelf_str = str(shelf_value).strip()
        if not shelf_str:
            return []
        # Split by semicolon or comma
        shelf_numbers = []
        for part in shelf_str.replace(",", ";").split(";"):
            part = part.strip()
            if part:
                try:
                    normalized = str(int(part))  # Remove leading zeros
                    shelf_numbers.append(normalized)
                except ValueError:
                    shelf_numbers.append(part)
        return shelf_numbers
    
    # Parse shelf numbers and explode to create one row per shelf
    df["shelf_list"] = df["shelfNo"].apply(parse_shelf_numbers)
    df = df[df["shelf_list"].apply(len) > 0].copy()  # Remove rows with no shelf numbers
    df = df.explode("shelf_list", ignore_index=True)
    df["shelfNo"] = df["shelf_list"].astype(str)
    df = df.drop(columns=["shelf_list"])
    print(f"After splitting multiple shelf numbers: {len(df)} artworks")
    
    # Create embeddings for all artworks
    print("Creating embeddings...")
    embeddings = []
    valid_indices = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding artworks"):
        try:
            emb = embed_artwork(row, base_dir)
            embeddings.append(emb)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error embedding artwork at index {idx}: {e}")
            continue
    
    if not embeddings:
        print("ERROR: No embeddings created")
        return pd.DataFrame()
    
    embeddings = np.array(embeddings, dtype=np.float32)
    df_valid = df.loc[valid_indices].copy()
    print(f"Created {len(embeddings)} embeddings")
    
    # Group by shelf and find representative for each
    print("Finding shelf representatives...")
    results = []
    
    unique_shelves = sorted(df_valid["shelfNo"].unique(), key=lambda x: str(x))
    for shelf_no in tqdm(unique_shelves, desc="Processing shelves"):
        shelf_mask = df_valid["shelfNo"] == shelf_no
        shelf_df = df_valid[shelf_mask].copy()
        shelf_embeddings = embeddings[shelf_mask]
        
        if len(shelf_embeddings) == 0:
            continue
        
        # Calculate centroid (average embedding) of the shelf
        centroid = np.mean(shelf_embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)  # Normalize
        
        # Find distances to centroid for all artworks on this shelf
        distances = np.linalg.norm(shelf_embeddings - centroid, axis=1)
        
        # Get top 5 closest artworks (sorted by distance)
        top_k = min(5, len(distances))
        top_indices = np.argsort(distances)[:top_k]
        
        # Add top 5 artworks to results
        for rank, idx in enumerate(top_indices, 1):
            artwork = shelf_df.iloc[idx]
            
            # Get the id column value from CSV (not row index)
            csv_id = artwork.get("id", "N/A")
            
            title = artwork.get("title", "Unknown")
            artist = artwork.get("artist", "Unknown")
            artwork_no = artwork.get("artworkNo", artwork.get("id", "N/A"))
            artwork_id = artwork.get("_id", artwork.get("id", "N/A"))
            cluster = artwork.get("cluster", "N/A")
            # Read thumbnail path directly from CSV (already set by read_data_ting.py)
            thumbnail = artwork.get("thumbnail", "N/A")
            
            results.append({
                "shelf_no": shelf_no,
                "index": csv_id,
                "id": artwork_id,
                "title": title,
                "artist": artist,
                "artwork_no": artwork_no,
                "cluster": cluster,
                "thumbnail": thumbnail,
                "rank": rank,
                "num_artworks_in_shelf": len(shelf_df),
                "distance_to_centroid": float(distances[idx])
            })
        
    return pd.DataFrame(results)


def find_shelf_outliers(
    csv_path: str,
    base_dir: str = "production-export-2025-11-04t14-27-00-000z",
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Find the top 5 outlier artworks for each shelf (furthest from centroid).
    Returns a DataFrame with shelf_no, index, id, title, artist, artwork_no, 
    cluster, rank, num_artworks_in_shelf, and distance_to_centroid.
    """
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} artworks")
    
    # Filter to artworks with shelf information
    if "shelfNo" not in df.columns:
        print("ERROR: 'shelfNo' column not found in CSV")
        return pd.DataFrame()
    
    df = df[df["shelfNo"].notna()].copy()
    print(f"Artworks with shelf info: {len(df)}")
    
    # Handle multiple shelf numbers (split by semicolon or comma)
    # Create a list column for shelf numbers
    if "shelfNo" in df.columns:
        def parse_shelf_list(shelf_value):
            if pd.isna(shelf_value):
                return []
            shelf_str = str(shelf_value).strip()
            if not shelf_str:
                return []
            # Replace commas with semicolons for consistent splitting
            shelf_str = shelf_str.replace(",", ";")
            return [s.strip() for s in shelf_str.split(";") if s.strip()]
        
        df["shelf_list"] = df["shelfNo"].apply(parse_shelf_list)
        
        # Explode to create separate rows for each shelf number
        artworks_no_shelf = df[df["shelf_list"].apply(len) == 0].copy()
        artworks_with_shelf = df[df["shelf_list"].apply(len) > 0].copy()
        
        if len(artworks_with_shelf) > 0:
            artworks_exploded = artworks_with_shelf.explode("shelf_list")
            artworks_exploded["shelfNo"] = artworks_exploded["shelf_list"]
            df = pd.concat([artworks_no_shelf, artworks_exploded], ignore_index=True)
            df = df.drop(columns=["shelf_list"])
    
    print(f"After splitting multiple shelf numbers: {len(df)} artworks")
    
    # Create embeddings for all artworks
    print("Creating embeddings...")
    embeddings = []
    valid_indices = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding artworks"):
        try:
            emb = embed_artwork(row, base_dir)
            embeddings.append(emb)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error embedding artwork at index {idx}: {e}")
            continue
    
    if not embeddings:
        print("ERROR: No embeddings created")
        return pd.DataFrame()
    
    embeddings = np.array(embeddings, dtype=np.float32)
    df_valid = df.loc[valid_indices].copy()
    print(f"Created {len(embeddings)} embeddings")
    
    # Group by shelf and find outliers for each
    print("Finding shelf outliers...")
    results = []
    
    unique_shelves = sorted(df_valid["shelfNo"].unique(), key=lambda x: str(x))
    for shelf_no in tqdm(unique_shelves, desc="Processing shelves"):
        shelf_mask = df_valid["shelfNo"] == shelf_no
        shelf_df = df_valid[shelf_mask].copy()
        shelf_embeddings = embeddings[shelf_mask]
        
        if len(shelf_embeddings) == 0:
            continue
        
        # Calculate centroid (average embedding) of the shelf
        centroid = np.mean(shelf_embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)  # Normalize
        
        # Find distances to centroid for all artworks on this shelf
        distances = np.linalg.norm(shelf_embeddings - centroid, axis=1)
        
        # Select diverse outliers (ensure artist diversity)
        unique_artists = shelf_df["artist"].unique()
        selected_indices = []
        selected_artists = set()
        
        # First pass: select furthest from centroid from each artist
        for artist in unique_artists:
            if len(selected_indices) >= 5:
                break
            artist_mask = shelf_df["artist"] == artist
            artist_indices = np.where(artist_mask)[0]
            if len(artist_indices) > 0:
                artist_distances = distances[artist_indices]
                furthest_idx = artist_indices[np.argmax(artist_distances)]
                selected_indices.append(furthest_idx)
                selected_artists.add(artist)
        
        # Second pass: fill remaining slots with furthest from centroid
        remaining = min(5, len(distances)) - len(selected_indices)
        if remaining > 0:
            all_indices = set(range(len(shelf_embeddings)))
            remaining_candidates = sorted(all_indices - set(selected_indices),
                                        key=lambda i: distances[i], reverse=True)[:remaining]
            selected_indices.extend(remaining_candidates)
        
        # Sort selected indices by distance to centroid (descending) for ranking
        selected_indices = sorted(selected_indices, key=lambda i: distances[i], reverse=True)[:5]
        top_indices = selected_indices
        
        # Add top 5 outliers to results
        for rank, idx in enumerate(top_indices, 1):
            artwork = shelf_df.iloc[idx]
            
            # Get the id column value from CSV (not row index)
            csv_id = artwork.get("id", "N/A")
            
            title = artwork.get("title", "Unknown")
            artist = artwork.get("artist", "Unknown")
            artwork_no = artwork.get("artworkNo", artwork.get("id", "N/A"))
            artwork_id = artwork.get("_id", artwork.get("id", "N/A"))
            cluster = artwork.get("cluster", "N/A")
            # Read thumbnail path directly from CSV (already set by read_data_ting.py)
            thumbnail = artwork.get("thumbnail", "N/A")
            
            results.append({
                "shelf_no": shelf_no,
                "index": csv_id,
                "id": artwork_id,
                "title": title,
                "artist": artist,
                "artwork_no": artwork_no,
                "cluster": cluster,
                "thumbnail": thumbnail,
                "rank": rank,
                "num_artworks_in_shelf": len(shelf_df),
                "distance_to_centroid": float(distances[idx])
            })
        
    return pd.DataFrame(results)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Find representative artworks for clusters or shelves.")
    parser.add_argument("--csv", default="artworks_with_thumbnails_ting.csv", help="Path to CSV file")
    parser.add_argument("--base-dir", default="production-export-2025-11-04t14-27-00-000z", help="Base directory for images")
    parser.add_argument("--output", default="cluster_representatives.csv", help="Output CSV file")
    parser.add_argument("--mode", default="cluster", choices=["cluster", "shelf", "cluster-outliers", "shelf-outliers"], help="Group by cluster or shelf, or find outliers")
    args = parser.parse_args()
    
    if args.mode == "cluster":
        results_df = find_cluster_representatives(args.csv, args.base_dir)
        
        if not results_df.empty:
            # Save to CSV
            results_df.to_csv(args.output, index=False)
            num_clusters = len(results_df["cluster"].unique())
            num_results = len(results_df)
            print(f"\n✓ Processed {num_clusters} clusters")
            print(f"✓ Found {num_results} representative artworks (top 5 per cluster)")
            print(f"✓ Results saved to {args.output}")
        else:
            print("No results to display")
    
    elif args.mode == "shelf":
        results_df = find_shelf_representatives(args.csv, args.base_dir)
        
        if not results_df.empty:
            # Save to CSV
            results_df.to_csv(args.output, index=False)
            num_shelves = len(results_df["shelf_no"].unique())
            num_results = len(results_df)
            print(f"\n✓ Processed {num_shelves} shelves")
            print(f"✓ Found {num_results} representative artworks (top 5 per shelf)")
            print(f"✓ Results saved to {args.output}")
        else:
            print("No results to display")
    
    elif args.mode == "cluster-outliers":
        results_df = find_cluster_outliers(args.csv, args.base_dir)
        
        if not results_df.empty:
            # Save to CSV
            results_df.to_csv(args.output, index=False)
            num_clusters = len(results_df["cluster"].unique())
            num_results = len(results_df)
            print(f"\n✓ Processed {num_clusters} clusters")
            print(f"✓ Found {num_results} outlier artworks (top 5 furthest per cluster)")
            print(f"✓ Results saved to {args.output}")
        else:
            print("No results to display")
    
    elif args.mode == "shelf-outliers":
        results_df = find_shelf_outliers(args.csv, args.base_dir)
        
        if not results_df.empty:
            # Save to CSV
            results_df.to_csv(args.output, index=False)
            num_shelves = len(results_df["shelf_no"].unique())
            num_results = len(results_df)
            print(f"\n✓ Processed {num_shelves} shelves")
            print(f"✓ Found {num_results} outlier artworks (top 5 furthest per shelf)")
            print(f"✓ Results saved to {args.output}")
        else:
            print("No results to display")


if __name__ == "__main__":
    main()

