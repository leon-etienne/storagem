# pip install torch transformers faiss-cpu pandas pillow tqdm
import os
import json
import base64
from io import BytesIO
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


def image_to_base64(image: Image.Image, max_size: int = 300) -> str:
    """Convert PIL Image to base64 string, with optional resizing."""
    if image is None:
        return ""
    
    try:
        # Resize if too large (keep aspect ratio)
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Save to bytes
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        img_bytes = buffer.getvalue()
        
        # Encode to base64
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{img_base64}"
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return ""


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


def find_shelf_outliers(
    csv_path: str,
    base_dir: str = "production-export-2025-11-04t14-27-00-000z",
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Find the top 10 outlier artworks for each shelf (furthest from centroid).
    Returns a DataFrame with reordered columns: shelf_no, rank, index, title, artist, cluster, 
    num_artworks_in_shelf, distance_to_centroid, thumbnail (base64 encoded).
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
        
        # Get top 10 furthest artworks (sorted by distance, descending)
        top_k = min(10, len(distances))
        top_indices = np.argsort(distances)[-top_k:][::-1]  # Get largest distances first
        
        # Add top 10 outliers to results with reordered columns and embedded images
        for rank, idx in enumerate(top_indices, 1):
            artwork = shelf_df.iloc[idx]
            
            # Get the id column value from CSV (not row index)
            csv_id = artwork.get("id", "N/A")
            
            title = artwork.get("title", "Unknown")
            artist = artwork.get("artist", "Unknown")
            cluster = artwork.get("cluster", "N/A")
            # Read thumbnail path directly from CSV (already set by read_data_ting.py)
            thumbnail_path = artwork.get("thumbnail", "N/A")
            
            # Load and encode image as base64
            image = load_image(thumbnail_path, base_dir) if thumbnail_path and pd.notna(thumbnail_path) and thumbnail_path != "N/A" else None
            thumbnail_base64 = image_to_base64(image) if image else ""
            
            results.append({
                "shelf_no": shelf_no,
                "rank": rank,
                "index": csv_id,
                "title": title,
                "artist": artist,
                "cluster": cluster,
                "num_artworks_in_shelf": len(shelf_df),
                "distance_to_centroid": float(distances[idx]),
                "thumbnail": thumbnail_base64
            })
        
    # Create DataFrame and reorder columns
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        # Reorder columns as specified
        column_order = ["shelf_no", "rank", "index", "title", "artist", "cluster", 
                       "num_artworks_in_shelf", "distance_to_centroid", "thumbnail"]
        # Only include columns that exist
        available_columns = [col for col in column_order if col in results_df.columns]
        results_df = results_df[available_columns]
    
    return results_df


def find_shelf_representatives(
    csv_path: str,
    base_dir: str = "production-export-2025-11-04t14-27-00-000z",
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Find the top 10 representative artworks for each shelf (closest to centroid).
    Returns a DataFrame with reordered columns: shelf_no, rank, index, title, artist, cluster, 
    num_artworks_in_shelf, distance_to_centroid, thumbnail (base64 encoded).
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
    
    # Group by shelf and find representatives for each
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
        
        # Get top 10 closest artworks (sorted by distance, ascending)
        top_k = min(10, len(distances))
        top_indices = np.argsort(distances)[:top_k]  # Get smallest distances first
        
        # Add top 10 representatives to results with reordered columns and embedded images
        for rank, idx in enumerate(top_indices, 1):
            artwork = shelf_df.iloc[idx]
            
            # Get the id column value from CSV (not row index)
            csv_id = artwork.get("id", "N/A")
            
            title = artwork.get("title", "Unknown")
            artist = artwork.get("artist", "Unknown")
            cluster = artwork.get("cluster", "N/A")
            # Read thumbnail path directly from CSV (already set by read_data_ting.py)
            thumbnail_path = artwork.get("thumbnail", "N/A")
            
            # Load and encode image as base64
            image = load_image(thumbnail_path, base_dir) if thumbnail_path and pd.notna(thumbnail_path) and thumbnail_path != "N/A" else None
            thumbnail_base64 = image_to_base64(image) if image else ""
            
            results.append({
                "shelf_no": shelf_no,
                "rank": rank,
                "index": csv_id,
                "title": title,
                "artist": artist,
                "cluster": cluster,
                "num_artworks_in_shelf": len(shelf_df),
                "distance_to_centroid": float(distances[idx]),
                "thumbnail": thumbnail_base64
            })
        
    # Create DataFrame and reorder columns
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        # Reorder columns as specified
        column_order = ["shelf_no", "rank", "index", "title", "artist", "cluster", 
                       "num_artworks_in_shelf", "distance_to_centroid", "thumbnail"]
        # Only include columns that exist
        available_columns = [col for col in column_order if col in results_df.columns]
        results_df = results_df[available_columns]
    
    return results_df


def create_html_view(results_df: pd.DataFrame, output_file: str, mode: str = "outlier"):
    """Create an HTML file with embedded images for better viewing."""
    mode_title = "Shelf Representatives" if mode == "representative" else "Shelf Outliers"
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{mode_title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            padding: 12px;
            text-align: left;
            border: 1px solid #ddd;
        }}
        td {{
            padding: 10px;
            border: 1px solid #ddd;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .thumbnail {{
            max-width: 150px;
            max-height: 150px;
            object-fit: contain;
        }}
        .shelf-header {{
            background-color: #2196F3;
            color: white;
            font-weight: bold;
            font-size: 1.2em;
            padding: 15px;
            margin-top: 20px;
            page-break-after: avoid;
            orphans: 3;
            widows: 3;
        }}
        .shelf-section {{
            page-break-after: always;
            page-break-inside: avoid;
            orphans: 3;
            widows: 3;
        }}
        @media print {{
            .shelf-section {{
                page-break-after: always;
                page-break-inside: avoid;
                orphans: 3;
                widows: 3;
            }}
            .shelf-header {{
                page-break-after: avoid;
                page-break-before: avoid;
                orphans: 3;
                widows: 3;
            }}
            .shelf-section:first-of-type .shelf-header {{
                page-break-before: auto;
            }}
            table {{
                page-break-inside: auto;
            }}
            tr {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <h1>{mode_title} - Top 10 per Shelf</h1>
"""
    
    # Group by shelf_no
    for shelf_no in sorted(results_df["shelf_no"].unique(), key=lambda x: str(x)):
        shelf_data = results_df[results_df["shelf_no"] == shelf_no].sort_values("rank")
        
        mode_label = "representatives" if mode == "representative" else "outliers"
        html_content += f'    <div class="shelf-section">\n'
        html_content += f'    <div class="shelf-header">Shelf {shelf_no} ({len(shelf_data)} {mode_label})</div>\n'
        html_content += '    <table>\n'
        html_content += '        <tr>\n'
        html_content += '            <th>Rank</th>\n'
        html_content += '            <th>Index</th>\n'
        html_content += '            <th>Title</th>\n'
        html_content += '            <th>Artist</th>\n'
        html_content += '            <th>Cluster</th>\n'
        html_content += '            <th>Distance</th>\n'
        html_content += '            <th>Thumbnail</th>\n'
        html_content += '        </tr>\n'
        
        for _, row in shelf_data.iterrows():
            thumbnail_img = f'<img src="{row["thumbnail"]}" class="thumbnail" alt="Thumbnail" />' if row["thumbnail"] else "No image"
            
            html_content += '        <tr>\n'
            html_content += f'            <td>{row["rank"]}</td>\n'
            html_content += f'            <td>{row["index"]}</td>\n'
            html_content += f'            <td>{row["title"]}</td>\n'
            html_content += f'            <td>{row["artist"]}</td>\n'
            html_content += f'            <td>{row["cluster"]}</td>\n'
            html_content += f'            <td>{row["distance_to_centroid"]:.4f}</td>\n'
            html_content += f'            <td>{thumbnail_img}</td>\n'
            html_content += '        </tr>\n'
        
        html_content += '    </table>\n'
        html_content += '    </div>\n'
    
    html_content += """
</body>
</html>
"""
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Find shelf outlier or representative artworks.")
    parser.add_argument("--csv", default="artworks_with_thumbnails_ting.csv", help="Path to CSV file")
    parser.add_argument("--base-dir", default="production-export-2025-11-04t14-27-00-000z", help="Base directory for images")
    parser.add_argument("--output", default="shelf_outliers_print.csv", help="Output CSV file")
    parser.add_argument("--mode", default="outliers", choices=["outliers", "representatives"], help="Find outliers (furthest) or representatives (closest)")
    args = parser.parse_args()
    
    if args.mode == "outliers":
        results_df = find_shelf_outliers(args.csv, args.base_dir)
        mode_name = "outlier"
    else:  # representatives
        results_df = find_shelf_representatives(args.csv, args.base_dir)
        mode_name = "representative"
    
    if not results_df.empty:
        # Save to CSV with embedded images
        results_df.to_csv(args.output, index=False)
        
        # Also create HTML file for better viewing
        html_output = args.output.replace(".csv", ".html")
        create_html_view(results_df, html_output, mode_name)
        
        num_shelves = len(results_df["shelf_no"].unique())
        num_results = len(results_df)
        print(f"\n✓ Processed {num_shelves} shelves")
        print(f"✓ Found {num_results} {mode_name} artworks (top 10 per shelf)")
        print(f"✓ Results saved to {args.output}")
        print(f"✓ HTML view saved to {html_output}")
    else:
        print("No results to display")


if __name__ == "__main__":
    main()
