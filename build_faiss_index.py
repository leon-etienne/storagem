# pip install torch transformers faiss-cpu pandas
import torch
import faiss
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# -----------------------------
# 0) Model
# -----------------------------
MODEL_NAME = "intfloat/e5-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

@torch.inference_mode()
def embed_texts(texts, batch_size=64, normalize=True, add_instruction=True):
    if add_instruction:
        texts = [f"passage: {t}" for t in texts]
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i+batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        if normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        out.append(emb.cpu())
    return torch.cat(out, 0).numpy()

@torch.inference_mode()
def embed_query(q, normalize=True, add_instruction=True):
    if add_instruction:
        q = f"query: {q}"
    enc = tokenizer(q, return_tensors="pt", truncation=True, max_length=512).to(device)
    h = model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1)
    emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
    if normalize:
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()[0]

# -----------------------------
# 1) Load + duplicate rows per shelf
# -----------------------------
df_raw = pd.read_csv("artworks_with_thumbnails.csv")

# keep only artworks if such a column exists
if "_type" in df_raw.columns:
    df_aw = df_raw[df_raw["_type"] == "artwork"].copy()
else:
    df_aw = df_raw.copy()

# parse multi-shelf cells like "3; 5" / "3,5" → [3,5]
def parse_shelf_list(x):
    if pd.isna(x):
        return [np.nan]
    vals = []
    for part in str(x).replace(",", ";").split(";"):
        part = part.strip()
        if part.isdigit():
            vals.append(int(part))
    return vals or [np.nan]

df_aw["shelf_list"] = df_aw.get("shelfNo", np.nan).apply(parse_shelf_list)

# remember original row id before exploding
df_aw["orig_id"] = df_aw.index.astype(int)

# explode so each shelf becomes its own row (a copy per shelf)
df_exp = df_aw.explode("shelf_list", ignore_index=False).copy()
# optional: if you want to drop rows with no shelf, comment/uncomment the next line
# df_exp = df_exp[~df_exp["shelf_list"].isna()]

# make a single-shelf scalar column for this copy
df_exp["shelf_single"] = df_exp["shelf_list"].apply(lambda s: ("NaN" if pd.isna(s) else int(s)))

# reindex to get contiguous indices for embeddings/FAISS
df = df_exp.reset_index(drop=True)

# -----------------------------
# 2) Text building (per copy)
# -----------------------------
cols_to_index = [c for c in ["title", "description", "artist", "year"] if c in df.columns]

def row_to_text(row):
    parts = []
    for c in cols_to_index:
        v = row.get(c, None)
        if pd.notna(v):
            parts.append(f"{c}: {v}")
    parts.append(f"shelf: {row['shelf_single']}")
    parts.append(f"orig_id: {row['orig_id']}")
    return ". ".join(parts)

docs = [row_to_text(r) for _, r in df.iterrows()]

# -----------------------------
# 3) Embeddings + FAISS index
# -----------------------------
X = embed_texts(docs, add_instruction=True).astype(np.float32)
dim = X.shape[1]

index = faiss.IndexHNSWFlat(dim, 32)
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 64
index.add(X)

# maps/metadata
id_map = {i: i for i in range(len(df))}                 # faiss id -> df row (copy)
metadata = df.to_dict(orient="index")                   # each row is a per-shelf copy

# quick helpers
def cosine(a, b):  # X must be L2-normalized already
    return float(np.dot(a, b))

def search(q, top_k=5, shelf=None, dedup_by_orig=False):
    """
    shelf: filter results to a specific shelf (e.g., 3 or 'NaN')
    dedup_by_orig: if True, keep only the best result per original artwork
    """
    qv = embed_query(q, add_instruction=True).astype(np.float32).reshape(1, -1)
    D, I = index.search(qv, top_k * 10)

    rows = []
    seen_orig = set()
    for rank, idx_ in enumerate(I[0]):
        if idx_ < 0:
            continue
        rid = id_map[idx_]
        row = metadata[rid]
        if (shelf is not None) and (str(row["shelf_single"]) != str(shelf)):
            continue
        if dedup_by_orig:
            oid = row["orig_id"]
            if oid in seen_orig:
                continue
            seen_orig.add(oid)
        rows.append({"rank": rank, "score": float(D[0][rank]), **row})
        if len(rows) >= top_k:
            break
    return pd.DataFrame(rows)

print(f"Built FAISS over {len(df)} per-shelf copies "
      f"from {df['orig_id'].nunique()} original artworks, "
      f"{df['shelf_single'].nunique()} shelves.")

results = search("old", top_k=5)
print(results[["rank", "score"] + cols_to_index])