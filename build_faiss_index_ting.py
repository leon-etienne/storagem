# pip install torch transformers faiss-cpu pandas
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

try:
    faiss.omp_set_num_threads(1)
except Exception:
    pass

MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_COLS = ["title", "description", "artist", "year"]
ARTIFACT_INDEX = "faiss.index"
ARTIFACT_METADATA = "metadata.pkl"
ARTIFACT_CONFIG = "search_config.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()


@torch.inference_mode()
def embed_texts(texts: List[str], batch_size: int = 64, normalize: bool = True, add_instruction: bool = False) -> np.ndarray:
    if add_instruction:
        texts = [f"passage: {t}" for t in texts]
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i : i + batch_size],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        emb = model.get_text_features(**enc)
        if normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        out.append(emb.cpu())
    return torch.cat(out, 0).numpy()


@torch.inference_mode()
def embed_query(q: str, normalize: bool = True, add_instruction: bool = False) -> np.ndarray:
    if add_instruction:
        q = f"query: {q}"
    enc = tokenizer(q, return_tensors="pt", truncation=True, max_length=77).to(device)
    emb = model.get_text_features(**enc)
    if normalize:
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()[0]


def _parse_shelf_list(value) -> List[float]:
    if pd.isna(value):
        return [np.nan]
    vals = []
    for part in str(value).replace(",", ";").split(";"):
        part = part.strip()
        if part.isdigit():
            vals.append(int(part))
    return vals or [np.nan]


def prepare_dataframe(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    if "_type" in df_raw.columns:
        df_aw = df_raw[df_raw["_type"] == "artwork"].copy()
    else:
        df_aw = df_raw.copy()

    df_aw["shelf_list"] = df_aw.get("shelfNo", np.nan).apply(_parse_shelf_list)
    df_aw["orig_id"] = df_aw.index.astype(int)

    df_exp = df_aw.explode("shelf_list", ignore_index=False).copy()
    df_exp["shelf_single"] = df_exp["shelf_list"].apply(lambda s: ("NaN" if pd.isna(s) else int(s)))

    df = df_exp.reset_index(drop=True)
    cols_to_index = [c for c in DEFAULT_COLS if c in df.columns]
    return df, cols_to_index


def dataframe_to_documents(df: pd.DataFrame, cols_to_index: List[str]) -> List[str]:
    docs = []
    for _, row in df.iterrows():
        parts = []
        for col in cols_to_index:
            value = row.get(col)
            if pd.notna(value):
                parts.append(f"{col}: {value}")
        parts.append(f"shelf: {row['shelf_single']}")
        parts.append(f"orig_id: {row['orig_id']}")
        docs.append(". ".join(parts))
    return docs


def build_faiss_index(embeddings: np.ndarray, m: int = 32, ef_construction: int = 200, ef_search: int = 64) -> faiss.Index:
    embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, m)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(embeddings)
    return index


def save_artifacts(
    index: faiss.Index,
    df: pd.DataFrame,
    cols_to_index: List[str],
    output_dir: Path,
    *,
    ef_search: int,
    m: int,
    model_name: str,
    normalize_embeddings: bool,
    add_instruction: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = output_dir / ARTIFACT_INDEX
    metadata_path = output_dir / ARTIFACT_METADATA
    config_path = output_dir / ARTIFACT_CONFIG

    faiss.write_index(index, str(faiss_path))
    df.reset_index(drop=True).to_pickle(metadata_path)

    config = {
        "cols_to_index": cols_to_index,
        "model_name": model_name,
        "normalize_embeddings": normalize_embeddings,
        "add_instruction": add_instruction,
        "ef_search": ef_search,
        "hnsw_m": m,
        "metadata": {
            "num_rows": int(df.shape[0]),
            "num_original": int(df["orig_id"].nunique()),
            "num_shelves": int(df["shelf_single"].nunique()),
        },
    }
    config_path.write_text(json.dumps(config, indent=2))


def build_artifacts(
    csv_path: Path,
    output_dir: Path,
    *,
    batch_size: int = 64,
    normalize_embeddings: bool = True,
    add_instruction: bool = False,
    hnsw_m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
) -> Dict[str, str]:
    df_raw = pd.read_csv(csv_path)
    df, cols_to_index = prepare_dataframe(df_raw)
    documents = dataframe_to_documents(df, cols_to_index)
    embeddings = embed_texts(documents, batch_size=batch_size, normalize=normalize_embeddings, add_instruction=add_instruction)
    index = build_faiss_index(embeddings, m=hnsw_m, ef_construction=ef_construction, ef_search=ef_search)
    save_artifacts(
        index,
        df,
        cols_to_index,
        output_dir,
        ef_search=ef_search,
        m=hnsw_m,
        model_name=MODEL_NAME,
        normalize_embeddings=normalize_embeddings,
        add_instruction=add_instruction,
    )

    return {
        "index_path": str(output_dir / ARTIFACT_INDEX),
        "metadata_path": str(output_dir / ARTIFACT_METADATA),
        "config_path": str(output_dir / ARTIFACT_CONFIG),
        "num_rows": int(df.shape[0]),
        "num_original": int(df["orig_id"].nunique()),
        "num_shelves": int(df["shelf_single"].nunique()),
    }


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index artifacts for artwork search.")
    parser.add_argument("--csv", default="artworks_with_thumbnails.csv", help="Path to the CSV file with artworks data.")
    parser.add_argument("--out", default="artifacts", help="Directory where the FAISS index and metadata will be stored.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding generation.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing artifacts if present.")
    args = parser.parse_args()

    output_dir = Path(args.out)
    faiss_path = output_dir / ARTIFACT_INDEX
    metadata_path = output_dir / ARTIFACT_METADATA
    config_path = output_dir / ARTIFACT_CONFIG

    if not args.force and faiss_path.exists() and metadata_path.exists() and config_path.exists():
        print(f"Artifacts already exist in {output_dir}. Use --force to rebuild.")
        return

    summary = build_artifacts(
        Path(args.csv),
        output_dir,
        batch_size=args.batch_size,
    )

    print(
        "Built FAISS index at {index_path} with {num_rows} per-shelf copies, "
        "{num_original} original artworks across {num_shelves} shelves.".format(**summary)
    )


if __name__ == "__main__":
    main()