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


def load_image(image_path: str, base_dir: str = "production-export-2025-11-04t14-27-00-000z") -> Optional[Image.Image]:
    """Load image from thumbnail path."""
    if pd.isna(image_path) or not image_path:
        return None
    
    # Handle thumbnail paths like "/thumbnails/filename.jpg"
    if image_path.startswith("/thumbnails/"):
        filename = image_path.replace("/thumbnails/", "")
        full_path = Path(base_dir) / "images" / filename
    else:
        # Try direct path
        full_path = Path(image_path)
    
    if not full_path.exists():
        return None
    
    try:
        return Image.open(full_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {full_path}: {e}")
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
    Combines text fields (artist, title, year, description, size, handling_status) and thumbnail image.
    Uses weighted combination: 60% text, 40% image (or 100% text if image unavailable).
    """
    # Collect text fields
    text_parts = []
    
    # Artist name
    artist = row.get("artist", "")
    if pd.notna(artist) and str(artist).strip():
        text_parts.append(f"artist: {str(artist).strip()}")
    
    # Title
    title = row.get("title", "")
    if pd.notna(title) and str(title).strip():
        text_parts.append(f"title: {str(title).strip()}")
    
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
    
    # Combine all text fields
    combined_text = ". ".join(text_parts) if text_parts else ""
    
    # Embed text
    text_emb = embed_text(combined_text)
    
    # Load and embed thumbnail image
    thumbnail_path = row.get("thumbnail", "")
    image = load_image(thumbnail_path, base_dir) if thumbnail_path else None
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
            
            results.append({
                "cluster": cluster_id,
                "index": csv_id,
                "id": artwork_id,
                "title": title,
                "artist": artist,
                "artwork_no": artwork_no,
                "shelf_no": shelf_no,
                "rank": rank,
                "num_artworks_in_cluster": len(cluster_df),
                "distance_to_centroid": float(distances[idx])
            })
        
        # Print summary for this cluster
        top_artwork = cluster_df.iloc[top_indices[0]]
        print(f"Cluster {cluster_id}: Top artwork - {top_artwork.get('title', 'Unknown')} by {top_artwork.get('artist', 'Unknown')} (No. {top_artwork.get('artworkNo', top_artwork.get('id', 'N/A'))}, Shelf: {top_artwork.get('shelfNo', 'N/A')}) - {len(cluster_df)} artworks total")
    
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
            
            results.append({
                "shelf_no": shelf_no,
                "index": csv_id,
                "id": artwork_id,
                "title": title,
                "artist": artist,
                "artwork_no": artwork_no,
                "cluster": cluster,
                "rank": rank,
                "num_artworks_in_shelf": len(shelf_df),
                "distance_to_centroid": float(distances[idx])
            })
        
        # Print summary for this shelf
        top_artwork = shelf_df.iloc[top_indices[0]]
        print(f"Shelf {shelf_no}: Top artwork - {top_artwork.get('title', 'Unknown')} by {top_artwork.get('artist', 'Unknown')} (No. {top_artwork.get('artworkNo', top_artwork.get('id', 'N/A'))}) - {len(shelf_df)} artworks total")
    
    return pd.DataFrame(results)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Find representative artworks for clusters or shelves.")
    parser.add_argument("--csv", default="artworks_with_thumbnails_ting.csv", help="Path to CSV file")
    parser.add_argument("--base-dir", default="production-export-2025-11-04t14-27-00-000z", help="Base directory for images")
    parser.add_argument("--output", default="cluster_representatives.csv", help="Output CSV file")
    parser.add_argument("--mode", default="cluster", choices=["cluster", "shelf"], help="Group by cluster or shelf")
    args = parser.parse_args()
    
    if args.mode == "cluster":
        results_df = find_cluster_representatives(args.csv, args.base_dir)
        
        if not results_df.empty:
            # Display results grouped by cluster
            print("\n" + "="*60)
            print("CLUSTER REPRESENTATIVES (Top 5 per cluster)")
            print("="*60)
            
            for cluster_id in sorted(results_df["cluster"].unique()):
                cluster_results = results_df[results_df["cluster"] == cluster_id].sort_values("rank")
                print(f"\nCluster {cluster_id} ({cluster_results.iloc[0]['num_artworks_in_cluster']} artworks total):")
                print("-" * 60)
                
                for _, row in cluster_results.iterrows():
                    print(f"  Rank {row['rank']}: {row['title']} by {row['artist']} (No. {row['artwork_no']})")
                    print(f"    - Index: {row['index']}")
                    print(f"    - ID: {row['id']}")
                    print(f"    - Shelf No: {row['shelf_no']}")
                    print(f"    - Distance to centroid: {row['distance_to_centroid']:.4f}")
                    print()
            
            # Save to CSV
            results_df.to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")
        else:
            print("No results to display")
    
    elif args.mode == "shelf":
        results_df = find_shelf_representatives(args.csv, args.base_dir)
        
        if not results_df.empty:
            # Display results grouped by shelf
            print("\n" + "="*60)
            print("SHELF REPRESENTATIVES (Top 5 per shelf)")
            print("="*60)
            
            for shelf_no in sorted(results_df["shelf_no"].unique(), key=lambda x: str(x)):
                shelf_results = results_df[results_df["shelf_no"] == shelf_no].sort_values("rank")
                print(f"\nShelf {shelf_no} ({shelf_results.iloc[0]['num_artworks_in_shelf']} artworks total):")
                print("-" * 60)
                
                for _, row in shelf_results.iterrows():
                    print(f"  Rank {row['rank']}: {row['title']} by {row['artist']} (No. {row['artwork_no']})")
                    print(f"    - Index: {row['index']}")
                    print(f"    - ID: {row['id']}")
                    print(f"    - Cluster: {row['cluster']}")
                    print(f"    - Distance to centroid: {row['distance_to_centroid']:.4f}")
                    print()
            
            # Save to CSV
            results_df.to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")
        else:
            print("No results to display")


if __name__ == "__main__":
    main()

