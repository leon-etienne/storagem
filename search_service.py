import json
import os
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import faiss
import numpy as np
import pandas as pd

from build_faiss_index_ting import (
    ARTIFACT_CONFIG,
    ARTIFACT_INDEX,
    ARTIFACT_METADATA,
    embed_query,
)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

try:
    faiss.omp_set_num_threads(1)
except Exception:
    pass


class SearchNotReadyError(RuntimeError):
    pass


class MissingArtifactsError(FileNotFoundError):
    def __init__(self, artifact_dir: Path):
        super().__init__(
            f"FAISS artifacts not found in '{artifact_dir}'. "
            "Run build_faiss_index_ting.py to generate them."
        )


class _SearchState:
    def __init__(self) -> None:
        self.index: Optional[faiss.Index] = None
        self.metadata: Optional[pd.DataFrame] = None
        self.cols_to_index: List[str] = []
        self.config: Dict = {}
        self.artifact_dir: Optional[Path] = None
        self.lock = Lock()


_STATE = _SearchState()
DEFAULT_ARTIFACT_DIR = Path("artifacts")


def _artifact_paths(artifact_dir: Path) -> Dict[str, Path]:
    return {
        "index": artifact_dir / ARTIFACT_INDEX,
        "metadata": artifact_dir / ARTIFACT_METADATA,
        "config": artifact_dir / ARTIFACT_CONFIG,
    }


def load(artifact_dir: Optional[Path] = None, *, force: bool = False) -> None:
    target_dir = Path(artifact_dir) if artifact_dir else DEFAULT_ARTIFACT_DIR

    with _STATE.lock:
        if not force and _STATE.index is not None and _STATE.artifact_dir == target_dir:
            return

        paths = _artifact_paths(target_dir)
        if not all(p.exists() for p in paths.values()):
            raise MissingArtifactsError(target_dir)

        config = json.loads(paths["config"].read_text())
        metadata = pd.read_pickle(paths["metadata"])
        index = faiss.read_index(str(paths["index"]))

        ef_search = config.get("ef_search")
        if ef_search and hasattr(index, "hnsw"):
            index.hnsw.efSearch = ef_search

        _STATE.index = index
        _STATE.metadata = metadata.reset_index(drop=True)
        _STATE.cols_to_index = list(config.get("cols_to_index", []))
        _STATE.config = config
        _STATE.artifact_dir = target_dir


def is_ready() -> bool:
    return _STATE.index is not None and _STATE.metadata is not None


def ensure_ready() -> None:
    if not is_ready():
        raise SearchNotReadyError(
            "Search artifacts have not been loaded. Call search_service.load() first."
        )


def _iter_results(
    distances: np.ndarray,
    indices: np.ndarray,
    *,
    top_k: int,
    shelf: Optional[str],
    dedup_by_orig: bool,
):
    metadata = _STATE.metadata
    if metadata is None:
        return

    seen_orig = set()
    result_count = 0

    for rank, idx in enumerate(indices):
        if idx < 0:
            continue

        row = metadata.iloc[idx]
        if shelf is not None and str(row["shelf_single"]) != str(shelf):
            continue

        if dedup_by_orig:
            orig_id = row["orig_id"]
            if orig_id in seen_orig:
                continue
            seen_orig.add(orig_id)

        payload = {
            "rank": rank,
            "score": float(distances[rank]),
            **row.to_dict(),
        }
        yield payload

        result_count += 1
        if result_count >= top_k:
            break


def search(
    query: str,
    top_k: int = 5,
    *,
    shelf: Optional[str] = None,
    dedup_by_orig: bool = False,
) -> pd.DataFrame:
    ensure_ready()

    config = _STATE.config
    query_vector = embed_query(
        query,
        normalize=config.get("normalize_embeddings", True),
        add_instruction=config.get("add_instruction", False),
    ).astype(np.float32).reshape(1, -1)

    index = _STATE.index
    assert index is not None

    distance_matrix, index_matrix = index.search(query_vector, top_k * 10)
    distances = distance_matrix[0]
    indices = index_matrix[0]

    rows = list(
        _iter_results(
            distances,
            indices,
            top_k=top_k,
            shelf=shelf,
            dedup_by_orig=dedup_by_orig,
        )
    )

    return pd.DataFrame(rows)


def get_metadata() -> pd.DataFrame:
    ensure_ready()
    assert _STATE.metadata is not None
    return _STATE.metadata


def get_cols_to_index() -> List[str]:
    ensure_ready()
    return _STATE.cols_to_index


def get_config() -> Dict:
    ensure_ready()
    return dict(_STATE.config)

