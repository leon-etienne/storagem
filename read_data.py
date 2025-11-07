# --- NDJSON Parser + Preview (all-in-one notebook) ---

import json
from typing import Any, Dict, Iterable, Iterator, List, Optional, TextIO, Union
import pandas as pd  # optional but useful for visualization

JSONLike = Dict[str, Any]

# ---------- PARSER FUNCTIONS ----------

def _open_maybe(path_or_file: Union[str, TextIO]) -> Iterator[TextIO]:
    """Yield a file-like object from path or an already-open file handle."""
    if hasattr(path_or_file, "read"):
        yield path_or_file  # type: ignore
        return
    close_me: Optional[TextIO] = None
    try:
        close_me = open(path_or_file, "r", encoding="utf-8")
        yield close_me
    finally:
        if close_me is not None:
            close_me.close()


def iter_ndjson(path_or_file: Union[str, TextIO]) -> Iterator[JSONLike]:
    """Stream-parse NDJSON, yielding one JSON object per non-empty line."""
    for fh in _open_maybe(path_or_file):
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                ctx = f"Invalid JSON on line {lineno}: {e.msg} (pos {e.pos})"
                raise ValueError(ctx) from e
            yield obj


def read_ndjson(path_or_file: Union[str, TextIO]) -> List[JSONLike]:
    """Read all NDJSON objects into a list."""
    return list(iter_ndjson(path_or_file))


def to_dataframe(path_or_file: Union[str, TextIO], *, record_path: Optional[str] = None):
    """
    Load NDJSON into a pandas DataFrame using json_normalize.
    - record_path: if objects contain a repeated field you'd like exploded, pass its key.
    """
    records = list(iter_ndjson(path_or_file))
    if not records:
        return pd.DataFrame()
    from pandas import json_normalize
    if record_path:
        return json_normalize(
            records,
            record_path=record_path,
            meta=[k for k in records[0].keys() if k != record_path],
        )
    return json_normalize(records)


# ---------- PREVIEW UTILITY ----------

def preview_ndjson(path: str, n: Optional[int] = None) -> pd.DataFrame:
    """
    Quick preview of first n NDJSON rows.
    If n is None, loads *all* rows in the file.
    """
    rows = []
    for i, obj in enumerate(iter_ndjson(path)):
        rows.append(obj)
        if n is not None and i + 1 >= n:
            break

    df = pd.json_normalize(rows)
    print(df)
    return df


# ---------- EXAMPLE USAGE ----------

path = "production-export-2025-11-04t14-27-00-000z/data.ndjson"
df = preview_ndjson(path)  # loads ALL if n not given

# Filter artworks only
artworks = df[df["_type"] == "artwork"].copy()

# ---------- ADD THUMBNAIL PATH FIELD ----------

BASE_THUMB_PATH = "/thumbnails"  # adjust to your real path or CDN base

def make_thumbnail_path(row):
    """Construct a simple static thumbnail path for each artwork."""
    slug = row.get("slug.current") or row.get("slug") or row["_id"]
    # strip spaces or invalid characters if needed
    slug_str = str(slug).strip().replace(" ", "-")
    return f"{BASE_THUMB_PATH}/{slug_str}.jpg"

artworks["thumbnail"] = artworks.apply(make_thumbnail_path, axis=1)

# Preview a few artworks with the new thumbnail field
print(artworks[["_id", "title", "thumbnail"]].head())
artworks.to_csv("artworks_with_thumbnails.csv", index=False)