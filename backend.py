#!/usr/bin/env python
import os
from typing import List, Optional

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
CSV_PATH = "artworks_with_thumbnails_ting.csv"

ID_COLUMN = "id"
TITLE_COLUMN = "title"
ARTIST_COLUMN = "artist"          # change if your CSV uses another name
IMAGE_COLUMN = "image"            # full-size image for the right panel
THUMB_COLUMN = "thumbnail"        # small thumb for the scatter map
SHELF_COLUMN = "shelfNo"          # <-- your shelf column

TEXT_WEIGHT = 0.7                 # balance image vs text in embedding
TOP_K_DEFAULT = 15

TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000


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
    """Text that will be embedded for each artwork (currently: title only)."""
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
# Load CSV + metadata
# ----------------------------
print(f"Loading CSV from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# Normalized ID
if ID_COLUMN in df.columns:
    df["id_norm"] = df[ID_COLUMN].apply(normalize_id)
else:
    df["id_norm"] = [str(i) for i in range(len(df))]

# Only keep rows with a thumbnail (and optionally an image)
if THUMB_COLUMN in df.columns:
    df = df[df[THUMB_COLUMN].notna()].copy()
else:
    raise ValueError(f"CSV must contain a '{THUMB_COLUMN}' column for thumbnails.")

df[THUMB_COLUMN] = df[THUMB_COLUMN].astype(str)

# If IMAGE_COLUMN not present, fall back to thumbnail for full image as well
if IMAGE_COLUMN in df.columns:
    df[IMAGE_COLUMN] = df[IMAGE_COLUMN].astype(str)
else:
    df[IMAGE_COLUMN] = df[THUMB_COLUMN]

# Artist column optional
if ARTIST_COLUMN in df.columns:
    df[ARTIST_COLUMN] = df[ARTIST_COLUMN].fillna("").astype(str)
else:
    df[ARTIST_COLUMN] = ""

# Shelf column: expected to be numeric-ish
if SHELF_COLUMN in df.columns:
    # If there are NaNs, set them to a default shelf (e.g. 0)
    df[SHELF_COLUMN] = df[SHELF_COLUMN].fillna(0)
    # cast to int if possible
    try:
        df["shelf_id"] = df[SHELF_COLUMN].astype(int)
    except Exception:
        # if that fails, cast to string and later the frontend can handle mapping differently
        df["shelf_id"] = df[SHELF_COLUMN].astype(str)
else:
    # default shelf id 0 if column missing
    df["shelf_id"] = 0

ids = df["id_norm"].tolist()
titles = df[TITLE_COLUMN].fillna("").astype(str).tolist()
artists = df[ARTIST_COLUMN].tolist()
image_paths = df[IMAGE_COLUMN].tolist()
thumb_paths = df[THUMB_COLUMN].tolist()
shelf_ids = df["shelf_id"].tolist()
metadata_texts = [build_metadata_text(row) for _, row in df.iterrows()]

num_items = len(df)
print(f"Using {num_items} rows with thumbnails.")

# Optionally make paths absolute or prepend a root directory here
# IMAGE_ROOT = "/absolute/path/to/images"
# image_paths = [os.path.join(IMAGE_ROOT, p) for p in image_paths]
# thumb_paths = [os.path.join(IMAGE_ROOT, p) for p in thumb_paths]


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
        for p in image_paths[start:end]:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
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
faiss.normalize_L2(all_embs)
index = faiss.IndexFlatIP(emb_dim)
index.add(all_embs)
print(f"FAISS index built with {index.ntotal} vectors.")


# ----------------------------
# 2D projection with t-SNE
# ----------------------------
print("Computing 2D projection (t-SNE)...")
tsne = TSNE(
    n_components=2,
    perplexity=TSNE_PERPLEXITY,
    n_iter=TSNE_N_ITER,
    init="random",
    learning_rate="auto",
    verbose=1,
)
coords_2d = tsne.fit_transform(all_embs)  # (N, 2)

# Normalize to [0, 1] for plotting
min_xy = coords_2d.min(axis=0)
max_xy = coords_2d.max(axis=0)
scale = max_xy - min_xy
scale[scale == 0] = 1.0
coords_2d_norm = (coords_2d - min_xy) / scale


# ----------------------------
# API models
# ----------------------------
class Point(BaseModel):
    id: str
    title: str
    artist: Optional[str]
    image: str
    thumb: str
    x: float
    y: float
    shelf_id: Optional[int]  # or str if your CSV was non-numeric


class SearchResult(BaseModel):
    id: str
    title: str
    artist: Optional[str]
    image: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]


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
    """
    Return all points with 2D coordinates for visualization.
    Fields used by the frontend:
      - id, title, artist, thumb, x, y, shelf_id
      - image is also included but used mainly in search results panel.
    """
    points_out: List[Point] = []
    for i in range(num_items):
        shelf_raw = shelf_ids[i]
        # if shelf_id ended up as string, you can keep it; frontend only expects a number 0..9.
        # here we try to cast to int, else keep as is.
        try:
            shelf_val = int(shelf_raw)
        except Exception:
            shelf_val = None

        points_out.append(
            Point(
                id=ids[i],
                title=titles[i],
                artist=artists[i],
                image=image_paths[i],
                thumb=thumb_paths[i],
                x=float(coords_2d_norm[i, 0]),
                y=float(coords_2d_norm[i, 1]),
                shelf_id=shelf_val,
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
                image=image_paths[idx],  # full-size for right panel
                score=float(score),
            )
        )

    return SearchResponse(results=results)
# uvicorn backend:app --reload --port 8000