#!/usr/bin/env python
import os
from typing import List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import faiss
from sklearn.manifold import TSNE

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ----------------------------
# Config
# ----------------------------
MODEL_NAME = "openai/clip-vit-base-patch32"
CSV_PATH = "artworks_with_thumbnails_final_with_thumbs.csv"   # CSV that contains full + thumb paths

FULL_IMAGE_COLUMN = "thumbnail"        # full-size image paths
THUMB_COLUMN = "thumb_small"           # small thumbnails for the map

ID_COLUMN = "id"
TITLE_COLUMN = "title"
ARTIST_COLUMN = "artist"

TEXT_WEIGHT = 0.7            # balance image vs text in embedding
TOP_K_DEFAULT = 15

# KNN graph settings (for /api/graph if you ever want it)
K_GRAPH = 8                  # neighbors per node for the graph


# ----------------------------
# Helpers
# ----------------------------
def normalize_id(x):
    """Normalize IDs like 142, '142', 142.0, '142.0' → '142'."""
    try:
        f = float(str(x).strip())
        if f.is_integer():
            return str(int(f))
        return ("%.6f" % f).rstrip("0").rstrip(".")
    except Exception:
        return str(x).strip()


def build_metadata_text(row: pd.Series) -> str:
    """
    Text that will be embedded for each artwork.
    Currently: only title (you can change this if you like).
    """
    title = row.get(TITLE_COLUMN, "")
    if pd.isna(title) or not str(title).strip():
        return "Untitled artwork"
    return str(title)


# ----------------------------
# Load CLIP model
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP model {MODEL_NAME} on {device}...")
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()


# ----------------------------
# Load CSV + images metadata
# ----------------------------
print(f"Loading CSV from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# Normalized IDs
if ID_COLUMN in df.columns:
    df["id_norm"] = df[ID_COLUMN].apply(normalize_id)
else:
    df["id_norm"] = [str(i) for i in range(len(df))]

# Keep rows that have at least a thumbnail
df = df[df[THUMB_COLUMN].notna()].copy()

# Ensure columns exist
if FULL_IMAGE_COLUMN not in df.columns:
    raise ValueError(f"CSV is missing FULL_IMAGE_COLUMN='{FULL_IMAGE_COLUMN}'")
if THUMB_COLUMN not in df.columns:
    raise ValueError(f"CSV is missing THUMB_COLUMN='{THUMB_COLUMN}'")

df[FULL_IMAGE_COLUMN] = df[FULL_IMAGE_COLUMN].astype(str)
df[THUMB_COLUMN] = df[THUMB_COLUMN].astype(str)

ids = df["id_norm"].tolist()
titles = df[TITLE_COLUMN].fillna("").astype(str).tolist()

# Artists (if missing, fill with "")
if ARTIST_COLUMN in df.columns:
    artists = df[ARTIST_COLUMN].fillna("").astype(str).tolist()
else:
    artists = [""] * len(df)

full_image_paths = df[FULL_IMAGE_COLUMN].tolist()   # for embeddings + right panel
thumb_image_paths = df[THUMB_COLUMN].tolist()       # for map thumbnails

metadata_texts = [build_metadata_text(row) for _, row in df.iterrows()]

num_items = len(df)
print(f"Using {num_items} rows with images.")


# ----------------------------
# Build embeddings (image + text)
# ----------------------------
print("Computing CLIP embeddings (image + text)...")
emb_dim = model.config.projection_dim
all_embs = np.zeros((num_items, emb_dim), dtype=np.float32)

BATCH_SIZE = 32

with torch.no_grad():
    for start in range(0, num_items, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_items)

        batch_images = []
        for p in full_image_paths[start:end]:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                # fallback if image missing/broken
                img = Image.new("RGB", (224, 224), color=(0, 0, 0))
            batch_images.append(img)

        batch_texts = metadata_texts[start:end]

        img_inputs = processor(images=batch_images, return_tensors="pt")
        img_features = model.get_image_features(
            pixel_values=img_inputs["pixel_values"].to(device)
        )

        text_inputs = processor(
            text=batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        text_features = model.get_text_features(
            input_ids=text_inputs["input_ids"].to(device),
            attention_mask=text_inputs["attention_mask"].to(device),
        )

        img_features = F.normalize(img_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        combined = img_features + TEXT_WEIGHT * text_features
        combined = F.normalize(combined, p=2, dim=1)

        all_embs[start:end] = combined.cpu().numpy().astype(np.float32)

print(f"Embedding matrix shape: {all_embs.shape}")


# ----------------------------
# Build FAISS index (cosine via inner product)
# ----------------------------
# all_embs is normalized, so inner product == cosine similarity
faiss.normalize_L2(all_embs)
index = faiss.IndexFlatIP(emb_dim)
index.add(all_embs)
print(f"FAISS index built with {index.ntotal} vectors.")


# ----------------------------
# 2D projection (for D3) using t-SNE
# ----------------------------
print("Computing 2D projection (t-SNE)... this may take a bit.")

tsne = TSNE(
    n_components=2,
    perplexity=40,              # tweak 20–50 depending on dataset size
    learning_rate=200,
    n_iter=1500,
    n_iter_without_progress=300,
    early_exaggeration=16.0,
    metric="cosine",            # match CLIP's cosine space
    init="random",
    verbose=1,
    random_state=21,
)

coords_2d = tsne.fit_transform(all_embs)  # (N, 2)

# normalize to [0,1] for plotting
min_xy = coords_2d.min(axis=0)
max_xy = coords_2d.max(axis=0)
scale = max_xy - min_xy
scale[scale == 0] = 1.0
coords_2d_norm = (coords_2d - min_xy) / scale


# ----------------------------
# Build KNN graph from embeddings (for /api/graph)
# ----------------------------
print(f"Building KNN graph with K={K_GRAPH} from embeddings...")
distances_graph, indices_graph = index.search(all_embs, K_GRAPH + 1)  # self + neighbors

graph_edges = []
seen_pairs = set()

for i in range(num_items):
    for j_idx, sim in zip(indices_graph[i, 1:], distances_graph[i, 1:]):  # skip self
        if j_idx < 0:
            continue
        a = i
        b = int(j_idx)
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        graph_edges.append(
            {
                "source": ids[a],
                "target": ids[b],
                "weight": float(sim),  # cosine similarity
            }
        )

print(f"KNN graph edges: {len(graph_edges)}")


# ----------------------------
# API models
# ----------------------------
class Point(BaseModel):
    id: str
    title: str
    artist: str
    thumb: str   # small thumbnail for map
    x: float
    y: float


class SearchResult(BaseModel):
    id: str
    title: str
    artist: str
    image: str   # full-size image
    thumb: str   # small thumbnail (if needed)
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]


class GraphNode(BaseModel):
    id: str
    title: str
    artist: str
    thumb: str
    x: float
    y: float


class GraphEdge(BaseModel):
    source: str
    target: str
    weight: float


class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI()

# Allow web frontend (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g. ["http://localhost:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/points", response_model=List[Point])
def get_points():
    """Return all points with 2D coordinates for visualization (thumbnails only)."""
    points_out: List[Point] = []
    for i in range(num_items):
        points_out.append(
            Point(
                id=ids[i],
                title=titles[i],
                artist=artists[i],
                thumb=thumb_image_paths[i],
                x=float(coords_2d_norm[i, 0]),
                y=float(coords_2d_norm[i, 1]),
            )
        )
    return points_out


@app.get("/api/search", response_model=SearchResponse)
def search(
    q: str = Query(..., description="Query text"),
    k: int = Query(TOP_K_DEFAULT, description="Top K"),
):
    """Search for nearest artworks given a text query."""
    q = q.strip()
    if not q:
        return SearchResponse(results=[])

    with torch.no_grad():
        text_inputs = processor(
            text=[q],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        q_features = model.get_text_features(
            input_ids=text_inputs["input_ids"].to(device),
            attention_mask=text_inputs["attention_mask"].to(device),
        )
        q_features = F.normalize(q_features, p=2, dim=1)

    q_np = q_features.cpu().numpy().astype(np.float32)
    faiss.normalize_L2(q_np)

    distances, indices = index.search(q_np, k)
    distances = distances[0]
    indices = indices[0]

    results: List[SearchResult] = []
    for score, idx in zip(distances, indices):
        if idx < 0:
            continue
        results.append(
            SearchResult(
                id=ids[idx],
                title=titles[idx],
                artist=artists[idx],
                image=full_image_paths[idx],   # full-size
                thumb=thumb_image_paths[idx],  # small
                score=float(score),
            )
        )

    return SearchResponse(results=results)


@app.get("/api/graph", response_model=GraphResponse)
def get_graph():
    """Return nodes + embedding-based edges for force-directed layout (optional)."""
    nodes_out: List[GraphNode] = []
    for i in range(num_items):
        nodes_out.append(
            GraphNode(
                id=ids[i],
                title=titles[i],
                artist=artists[i],
                thumb=thumb_image_paths[i],
                x=float(coords_2d_norm[i, 0]),
                y=float(coords_2d_norm[i, 1]),
            )
        )
    edges_out = [GraphEdge(**e) for e in graph_edges]
    return GraphResponse(nodes=nodes_out, edges=edges_out)
# uvicorn backend:app --reload --port 8000