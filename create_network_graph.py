# pip install torch transformers pandas pillow tqdm networkx matplotlib numpy scipy
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
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

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


def calculate_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Calculate pairwise cosine similarity matrix."""
    # Since embeddings are normalized, cosine similarity is just dot product
    similarity_matrix = np.dot(embeddings, embeddings.T)
    return similarity_matrix


def create_network_graph(
    csv_path: str,
    base_dir: str = "production-export-2025-11-04t14-27-00-000z",
    similarity_threshold: float = 0.7,
    max_edges_per_node: int = 10,
    output_file: str = "network_graph.png"
) -> nx.Graph:
    """
    Create a network graph from artwork embeddings.
    
    Args:
        csv_path: Path to CSV file with artworks
        base_dir: Base directory for images
        similarity_threshold: Minimum similarity to create an edge
        max_edges_per_node: Maximum number of edges per node (top-k similar)
        output_file: Output file for the graph visualization
    
    Returns:
        NetworkX graph object
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
    valid_indices = []
    artwork_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding artworks"):
        try:
            emb = embed_artwork(row, base_dir)
            embeddings.append(emb)
            valid_indices.append(idx)
            
            # Store artwork metadata
            artwork_data.append({
                "index": row.get("id", idx),
                "title": row.get("title", "Unknown"),
                "artist": row.get("artist", "Unknown"),
                "cluster": row.get("cluster", "0") if "cluster" in row else "0",
                "shelf_no": row.get("shelfNo", "N/A"),
            })
        except Exception as e:
            print(f"Error embedding artwork at index {idx}: {e}")
            continue
    
    if not embeddings:
        print("ERROR: No embeddings created")
        return nx.Graph()
    
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Created {len(embeddings)} embeddings")
    
    # Calculate similarity matrix
    print("Calculating similarity matrix...")
    similarity_matrix = calculate_cosine_similarity(embeddings)
    
    # Create network graph
    print("Creating network graph...")
    G = nx.Graph()
    
    # Add nodes with attributes
    for i, data in enumerate(artwork_data):
        G.add_node(i, **data)
    
    # Add edges based on similarity
    print(f"Adding edges (threshold: {similarity_threshold}, max per node: {max_edges_per_node})...")
    edges_added = 0
    
    for i in tqdm(range(len(embeddings)), desc="Adding edges"):
        # Get similarities for this node
        similarities = similarity_matrix[i]
        
        # Get top-k most similar (excluding self)
        top_indices = np.argsort(similarities)[::-1][1:max_edges_per_node+1]
        
        for j in top_indices:
            similarity = similarities[j]
            if similarity >= similarity_threshold:
                # Add edge with similarity as weight
                if not G.has_edge(i, j):
                    G.add_edge(i, j, weight=float(similarity), similarity=float(similarity))
                    edges_added += 1
    
    print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Visualize the graph
    print("Visualizing graph...")
    visualize_graph(G, artwork_data, output_file)
    
    # Save graph data
    graph_data_file = output_file.replace(".png", "_data.json")
    save_graph_data(G, artwork_data, graph_data_file)
    
    return G


def visualize_graph(G: nx.Graph, artwork_data: List[Dict], output_file: str):
    """Visualize the network graph."""
    if G.number_of_nodes() == 0:
        print("No nodes to visualize")
        return
    
    # Create figure
    plt.figure(figsize=(20, 20))
    
    # Use spring layout for positioning
    print("Calculating layout...")
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Get cluster information for coloring
    clusters = [data.get("cluster", "0") for data in artwork_data[:G.number_of_nodes()]]
    unique_clusters = sorted(set(clusters))
    
    # Create color map
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    cluster_color_map = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}
    node_colors = [cluster_color_map.get(cluster, "gray") for cluster in clusters]
    
    # Draw edges
    edges = G.edges()
    edge_weights = [G[u][v].get("weight", 1.0) for u, v in edges]
    edge_alphas = [(w - min(edge_weights)) / (max(edge_weights) - min(edge_weights) + 1e-6) 
                   for w in edge_weights]
    edge_alphas = [max(0.1, min(0.5, a)) for a in edge_alphas]  # Clamp between 0.1 and 0.5
    
    nx.draw_networkx_edges(
        G, pos,
        alpha=edge_alphas,
        width=[w * 0.5 for w in edge_weights],
        edge_color="gray"
    )
    
    # Draw nodes
    # Node size is based on degree (number of connections)
    # Base size: 300, each connection adds 50
    # So nodes with more connections (more similar artworks) appear larger
    node_sizes = [300 + G.degree(n) * 50 for n in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8
    )
    
    # Draw labels (only for nodes with high degree or important nodes)
    important_nodes = [n for n in G.nodes() if G.degree(n) >= np.percentile([G.degree(n) for n in G.nodes()], 75)]
    labels = {n: f"{artwork_data[n]['title'][:20]}..." if len(artwork_data[n]['title']) > 20 
              else artwork_data[n]['title'] 
              for n in important_nodes}
    
    nx.draw_networkx_labels(G, pos, labels, font_size=6, font_weight="bold")
    
    # Create legend
    legend_elements = [mpatches.Patch(facecolor=cluster_color_map[cluster], label=f"Cluster {cluster}")
                       for cluster in unique_clusters]
    plt.legend(handles=legend_elements, loc="upper left", fontsize=8)
    
    plt.title(f"Artwork Network Graph\n{G.number_of_nodes()} nodes, {G.number_of_edges()} edges", 
              fontsize=16, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Graph saved to {output_file}")
    plt.close()


def save_graph_data(G: nx.Graph, artwork_data: List[Dict], output_file: str):
    """Save graph data to JSON file."""
    graph_data = {
        "nodes": [],
        "edges": []
    }
    
    # Add nodes
    for node_id in G.nodes():
        node_attrs = G.nodes[node_id]
        graph_data["nodes"].append({
            "id": node_id,
            **node_attrs
        })
    
    # Add edges
    for u, v, data in G.edges(data=True):
        graph_data["edges"].append({
            "source": int(u),
            "target": int(v),
            "weight": data.get("weight", 1.0),
            "similarity": data.get("similarity", 1.0)
        })
    
    with open(output_file, "w") as f:
        json.dump(graph_data, f, indent=2)
    
    print(f"Graph data saved to {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create a network graph from artwork embeddings.")
    parser.add_argument("--csv", default="artworks_with_thumbnails_ting.csv", help="Path to CSV file")
    parser.add_argument("--base-dir", default="production-export-2025-11-04t14-27-00-000z", help="Base directory for images")
    parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold for edges (0-1)")
    parser.add_argument("--max-edges", type=int, default=10, help="Maximum edges per node")
    parser.add_argument("--output", default="network_graph.png", help="Output image file")
    args = parser.parse_args()
    
    G = create_network_graph(
        args.csv,
        args.base_dir,
        similarity_threshold=args.threshold,
        max_edges_per_node=args.max_edges,
        output_file=args.output
    )
    
    if G.number_of_nodes() > 0:
        print(f"\nGraph Statistics:")
        print(f"  Nodes: {G.number_of_nodes()}")
        print(f"  Edges: {G.number_of_edges()}")
        print(f"  Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
        print(f"  Number of connected components: {nx.number_connected_components(G)}")
        
        if nx.number_connected_components(G) > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            print(f"  Largest component size: {len(largest_cc)} nodes")


if __name__ == "__main__":
    main()

