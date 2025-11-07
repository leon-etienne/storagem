# pip install torch transformers pandas pillow tqdm plotly umap-learn scikit-learn
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image

import numpy as np
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available, will use t-SNE instead")

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

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


def reduce_dimensions(embeddings: np.ndarray, method: str = "umap", n_components: int = 2) -> np.ndarray:
    """
    Reduce embedding dimensions to 2D or 3D for visualization.
    
    Args:
        embeddings: High-dimensional embeddings (n_samples, n_features)
        method: "umap" or "tsne"
        n_components: 2 or 3
    
    Returns:
        Reduced embeddings (n_samples, n_components)
    """
    print(f"Reducing dimensions using {method.upper()} from {embeddings.shape[1]}D to {n_components}D...")
    
    if method.lower() == "umap":
        if not UMAP_AVAILABLE:
            print("UMAP not available, falling back to t-SNE")
            method = "tsne"
        else:
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine',
                random_state=42
            )
            reduced = reducer.fit_transform(embeddings)
            print(f"UMAP reduction complete: {reduced.shape}")
            return reduced
    
    if method.lower() == "tsne":
        # Note: t-SNE doesn't support cosine metric directly, but works with euclidean on normalized embeddings
        reducer = TSNE(
            n_components=n_components,
            perplexity=30,
            random_state=42,
            max_iter=1000
        )
        reduced = reducer.fit_transform(embeddings)
        print(f"t-SNE reduction complete: {reduced.shape}")
        return reduced
    
    raise ValueError(f"Unknown method: {method}")


def create_interactive_visualization(
    csv_path: str,
    base_dir: str = "production-export-2025-11-04t14-27-00-000z",
    method: str = "umap",
    output_file: str = "embeddings_interactive.html"
):
    """
    Create an interactive visualization of artwork embeddings.
    
    Args:
        csv_path: Path to CSV file with artworks
        base_dir: Base directory for images
        method: Dimensionality reduction method ("umap" or "tsne")
        output_file: Output HTML file for interactive visualization
    """
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} artworks")
    
    # Filter to artworks with cluster information if available
    if "cluster" in df.columns:
        df = df[df["cluster"].notna()].copy()
        print(f"Artworks with cluster info: {len(df)}")
    
    # Create embeddings for all artworks
    print("Creating embeddings...")
    embeddings = []
    artwork_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding artworks"):
        try:
            emb = embed_artwork(row, base_dir)
            embeddings.append(emb)
            
            # Store artwork metadata
            artwork_data.append({
                "index": row.get("id", idx),
                "title": row.get("title", "Unknown"),
                "artist": row.get("artist", "Unknown"),
                "year": row.get("year", "N/A"),
                "cluster": str(row.get("cluster", "0")) if "cluster" in row else "0",
                "shelf_no": row.get("shelfNo", "N/A"),
                "description": str(row.get("description", ""))[:200] if pd.notna(row.get("description")) else "",
                "thumbnail": row.get("thumbnail", ""),
            })
        except Exception as e:
            print(f"Error embedding artwork at index {idx}: {e}")
            continue
    
    if not embeddings:
        print("ERROR: No embeddings created")
        return
    
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Created {len(embeddings)} embeddings")
    
    # Reduce dimensions
    reduced_embeddings = reduce_dimensions(embeddings, method=method, n_components=2)
    
    # Prepare data for visualization
    viz_df = pd.DataFrame({
        "x": reduced_embeddings[:, 0],
        "y": reduced_embeddings[:, 1],
        "title": [d["title"] for d in artwork_data],
        "artist": [d["artist"] for d in artwork_data],
        "year": [d["year"] for d in artwork_data],
        "cluster": [d["cluster"] for d in artwork_data],
        "shelf_no": [d["shelf_no"] for d in artwork_data],
        "index": [d["index"] for d in artwork_data],
        "description": [d["description"] for d in artwork_data],
    })
    
    # Create hover text
    hover_text = []
    for _, row in viz_df.iterrows():
        hover = f"<b>{row['title']}</b><br>"
        hover += f"Artist: {row['artist']}<br>"
        hover += f"Year: {row['year']}<br>"
        hover += f"Cluster: {row['cluster']}<br>"
        hover += f"Shelf: {row['shelf_no']}<br>"
        hover += f"ID: {row['index']}"
        if row['description']:
            hover += f"<br><br>{row['description'][:150]}..."
        hover_text.append(hover)
    
    viz_df["hover_text"] = hover_text
    
    # Create interactive plot
    print("Creating interactive visualization...")
    
    # Get unique clusters for coloring
    unique_clusters = sorted(viz_df["cluster"].unique())
    colors = px.colors.qualitative.Set3[:len(unique_clusters)]
    cluster_color_map = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}
    viz_df["color"] = viz_df["cluster"].map(cluster_color_map)
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot for each cluster (for better legend)
    for cluster in unique_clusters:
        cluster_data = viz_df[viz_df["cluster"] == cluster]
        fig.add_trace(go.Scatter(
            x=cluster_data["x"],
            y=cluster_data["y"],
            mode='markers',
            name=f'Cluster {cluster}',
            marker=dict(
                size=8,
                color=cluster_color_map[cluster],
                line=dict(width=0.5, color='white'),
                opacity=0.7
            ),
            text=cluster_data["hover_text"],
            hoverinfo='text',
            customdata=cluster_data[["title", "artist", "year", "cluster", "shelf_no", "index"]].values,
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'Artwork Embeddings Visualization ({method.upper()})<br><sub>{len(viz_df)} artworks</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title=f'{method.upper()} Dimension 1',
        yaxis_title=f'{method.upper()} Dimension 2',
        hovermode='closest',
        width=1200,
        height=800,
        template='plotly_white',
        legend=dict(
            title="Clusters",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        )
    )
    
    # Add interactive features
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Save as HTML
    fig.write_html(output_file)
    print(f"Interactive visualization saved to {output_file}")
    print(f"Open {output_file} in a web browser to explore the embeddings!")
    
    # Also save the reduced embeddings data
    data_file = output_file.replace(".html", "_data.csv")
    viz_df.to_csv(data_file, index=False)
    print(f"Embedding coordinates saved to {data_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create interactive visualization of artwork embeddings.")
    parser.add_argument("--csv", default="artworks_with_thumbnails_ting.csv", help="Path to CSV file")
    parser.add_argument("--base-dir", default="production-export-2025-11-04t14-27-00-000z", help="Base directory for images")
    parser.add_argument("--method", default="umap", choices=["umap", "tsne"], help="Dimensionality reduction method")
    parser.add_argument("--output", default="embeddings_interactive.html", help="Output HTML file")
    args = parser.parse_args()
    
    create_interactive_visualization(
        args.csv,
        args.base_dir,
        method=args.method,
        output_file=args.output
    )


if __name__ == "__main__":
    main()

