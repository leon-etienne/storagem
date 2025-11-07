# --- NDJSON Parser + Preview (all-in-one notebook) ---

import ast
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

# Path to the NDJSON data file (now inside the storagem folder)
path = "production-export-2025-11-04t14-27-00-000z/data.ndjson"
df = preview_ndjson(path)  # loads ALL if n not given

# ---------- RESOLVE ARTIST REFERENCES ----------
# Create a mapping from artist ID to artist name
artists_df = df[df["_type"] == "artist"].copy()
artist_map = {}

for _, artist_row in artists_df.iterrows():
    artist_id = artist_row.get("_id", "")
    if artist_id:
        # Handle both "drafts.xxx" and regular IDs
        artist_id_clean = artist_id.replace("drafts.", "")
        
        # Get artist name from name.fName and name.lName
        fname = artist_row.get("name.fName", "")
        lname = artist_row.get("name.lName", "")
        
        # Combine first and last name
        if fname and lname:
            full_name = f"{fname} {lname}".strip()
        elif fname:
            full_name = fname
        elif lname:
            full_name = lname
        else:
            full_name = None
        
        if full_name:
            artist_map[artist_id] = full_name
            artist_map[artist_id_clean] = full_name  # Also map without "drafts." prefix

print(f"Loaded {len(artist_map)} artist names")

# Filter artworks only
artworks = df[df["_type"] == "artwork"].copy()

# Resolve artist references to names
def resolve_artist_name(row):
    """Resolve artist._ref to actual artist name."""
    artist_ref = row.get("artist._ref", "")
    
    # Handle NaN, None, or empty values
    if pd.isna(artist_ref) or artist_ref is None or artist_ref == "":
        return None
    
    # Convert to string if it's not already
    artist_ref = str(artist_ref).strip()
    if not artist_ref:
        return None
    
    # Try direct lookup
    if artist_ref in artist_map:
        return artist_map[artist_ref]
    
    # Try with "drafts." prefix
    if not artist_ref.startswith("drafts."):
        drafts_ref = f"drafts.{artist_ref}"
        if drafts_ref in artist_map:
            return artist_map[drafts_ref]
    
    # Try without "drafts." prefix
    clean_ref = artist_ref.replace("drafts.", "")
    if clean_ref and clean_ref in artist_map:
        return artist_map[clean_ref]
    
    return None

# Add resolved artist names
artworks["artist"] = artworks.apply(resolve_artist_name, axis=1)
print(f"Resolved artist names for {artworks['artist'].notna().sum()} out of {len(artworks)} artworks")

# ---------- CLEAN UP COMPLEX FIELDS ----------
# Many fields are stored as string representations of Python lists/dicts
# We need to extract actual text values from them

def extract_text_from_complex(value):
    """Extract text from complex structures (lists, dicts, or their string representations)."""
    # Handle NaN/None - check for array-like first
    try:
        # Check if it's a numpy array or pandas Series
        if hasattr(value, '__array__') or (hasattr(value, '__len__') and not isinstance(value, (str, dict, list, tuple))):
            # Convert array-like to list for processing
            try:
                value = list(value) if len(value) > 0 else None
            except (TypeError, ValueError):
                pass
    except (TypeError, ValueError, AttributeError):
        pass
    
    # Now check for NaN/None properly
    if value is None:
        return None
    
    # Check for NaN - handle scalar values only (not arrays/lists)
    try:
        # Only use pd.isna for scalar values
        # Strings are iterable but we want to check them, so handle separately
        if isinstance(value, str):
            # Strings are fine, continue
            pass
        elif isinstance(value, (list, tuple, dict)):
            # Lists/tuples/dicts are fine, continue
            pass
        else:
            # For other types, check if it's NaN
            if pd.isna(value):
                return None
    except (TypeError, ValueError):
        # If pd.isna fails, continue processing
        pass
    
    # If it's already a simple string (not a string representation of a list/dict), return it
    if isinstance(value, str):
        # Check if it's a string representation of a list or dict
        value_stripped = value.strip()
        if (value_stripped.startswith('[') and value_stripped.endswith(']')) or \
           (value_stripped.startswith('{') and value_stripped.endswith('}')):
            # Try to parse it
            try:
                parsed = ast.literal_eval(value)
                value = parsed
            except (ValueError, SyntaxError):
                # If parsing fails, return as is
                return value
    
    # If it's a list/array, extract text from items
    if isinstance(value, (list, tuple)):
        texts = []
        for item in value:
            if isinstance(item, dict):
                # Handle block/rich text structures
                if "children" in item:
                    # Extract text from children blocks
                    for child in item.get("children", []):
                        if isinstance(child, dict) and "text" in child:
                            texts.append(str(child["text"]))
                elif "value" in item:
                    texts.append(str(item["value"]))
                elif "text" in item:
                    texts.append(str(item["text"]))
            elif isinstance(item, str):
                texts.append(item)
        return " ".join(texts).strip() if texts else None
    
    # If it's a dict, try to extract text
    if isinstance(value, dict):
        if "children" in value:
            texts = []
            for child in value.get("children", []):
                if isinstance(child, dict) and "text" in child:
                    texts.append(str(child["text"]))
            return " ".join(texts).strip() if texts else None
        elif "value" in value:
            return str(value["value"])
        elif "text" in value:
            return str(value["text"])
    
    # If it's a simple value, return as string
    try:
        return str(value) if value else None
    except (TypeError, ValueError):
        return None

def clean_list_field(value):
    """Convert list fields to readable strings."""
    # Handle NaN/None - check for array-like first
    try:
        # Check if it's a numpy array or pandas Series
        if hasattr(value, '__array__') or (hasattr(value, '__len__') and not isinstance(value, (str, dict, list, tuple))):
            # Convert array-like to list for processing
            try:
                value = list(value) if len(value) > 0 else None
            except (TypeError, ValueError):
                pass
    except (TypeError, ValueError, AttributeError):
        pass
    
    # Now check for NaN/None properly
    if value is None:
        return None
    
    # Check for NaN - handle scalar values only (not arrays/lists)
    try:
        # Only use pd.isna for scalar values
        if isinstance(value, str):
            # Strings are fine, continue
            pass
        elif isinstance(value, (list, tuple, dict)):
            # Lists/tuples/dicts are fine, continue
            pass
        else:
            # For other types, check if it's NaN
            if pd.isna(value):
                return None
    except (TypeError, ValueError):
        # If pd.isna fails, continue processing
        pass
    
    # If it's a string representation of a list, parse it
    if isinstance(value, str):
        value_stripped = value.strip()
        if (value_stripped.startswith('[') and value_stripped.endswith(']')):
            try:
                parsed = ast.literal_eval(value)
                value = parsed
            except (ValueError, SyntaxError):
                return value
    
    # If it's a list, join with commas
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    
    # Return as string if it's a simple value
    try:
        return str(value) if value else None
    except (TypeError, ValueError):
        return None

# Clean up description field (required for indexing)
if "description" in artworks.columns:
    artworks["description"] = artworks["description"].apply(extract_text_from_complex)
    print(f"Cleaned description field: {artworks['description'].notna().sum()} artworks have descriptions")

# Clean up other text fields that might have complex structures
text_fields = ["addition", "internalNote", "shortBio"]
cleaned_text_fields = []
for field in text_fields:
    if field in artworks.columns:
        artworks[field] = artworks[field].apply(extract_text_from_complex)
        cleaned_text_fields.append(field)
if cleaned_text_fields:
    print(f"Cleaned text fields: {', '.join(cleaned_text_fields)}")

# Clean up list fields (convert to readable comma-separated strings)
list_fields = ["rental", "handling.status", "status", "media"]
cleaned_list_fields = []
for field in list_fields:
    if field in artworks.columns:
        artworks[field] = artworks[field].apply(clean_list_field)
        cleaned_list_fields.append(field)
if cleaned_list_fields:
    print(f"Cleaned list fields: {', '.join(cleaned_list_fields)}")

# ---------- ADD THUMBNAIL PATH FIELD ----------

import os
from pathlib import Path
import re

# Base path for thumbnails in the web app (Flask will serve from actual image directory)
BASE_THUMB_PATH = "/thumbnails"

# Path to the images directory
IMAGES_DIR = Path("production-export-2025-11-04t14-27-00-000z/images")

# Cache of image files to avoid repeated directory scans
_image_files_cache = None

def get_image_files():
    """Get list of all image files in the images directory (cached)."""
    global _image_files_cache
    if _image_files_cache is not None:
        return _image_files_cache
    
    if not IMAGES_DIR.exists():
        _image_files_cache = []
        return []
    
    _image_files_cache = [f.name for f in IMAGES_DIR.iterdir() if f.is_file()]
    return _image_files_cache

def find_image_file(artwork_id, slug=None, row_data=None):
    """
    Try to find the actual image file in the images directory.
    Returns the filename if found, None otherwise.
    
    Checks multiple strategies:
    1. Look for hash in thumbnail._sanityAsset or other asset fields
    2. Search by artwork ID patterns
    3. Search by slug patterns
    """
    if not IMAGES_DIR.exists():
        return None
    
    image_files = get_image_files()
    if not image_files:
        return None
    
    artwork_id_str = str(artwork_id) if artwork_id else ""
    artwork_id_clean = artwork_id_str.replace("drafts.", "").replace(".", "").replace("-", "")
    
    # Strategy 1: Check if row_data has any hash-like values that match image filenames
    if row_data is not None:
        # Look for hash patterns in various fields
        for key, value in row_data.items():
            if value and isinstance(value, str) and len(value) > 20:
                # Check if this value matches any image filename (hash part)
                for img_file in image_files:
                    # Extract hash part (before first dash or dimension pattern)
                    img_hash = img_file.split("-")[0] if "-" in img_file else img_file.split(".")[0]
                    if img_hash in value or value in img_file:
                        return img_file
    
    # Strategy 2: Search by artwork ID patterns
    if artwork_id_clean:
        for img_file in image_files:
            if artwork_id_clean in img_file:
                return img_file
    
    # Strategy 3: Search by slug (less reliable)
    if slug:
        slug_clean = str(slug).replace(" ", "-").replace("_", "-")
        for img_file in image_files:
            if slug_clean.lower() in img_file.lower():
                return img_file
    
    return None

def extract_filename_from_asset(asset_value):
    """
    Extract filename from Sanity asset reference.
    Asset references are in format: "image@file://./images/filename.jpg"
    or "file@file://./files/filename.pdf"
    """
    if not asset_value or not isinstance(asset_value, str):
        return None
    
    asset_str = str(asset_value).strip()
    
    # Handle Sanity asset format: "image@file://./images/filename.jpg"
    if "@file://" in asset_str:
        # Extract the path part after file://
        if "./images/" in asset_str:
            filename = asset_str.split("./images/")[-1]
        elif "./files/" in asset_str:
            filename = asset_str.split("./files/")[-1]
        elif "/" in asset_str:
            # Generic path extraction
            parts = asset_str.split("/")
            filename = parts[-1]
        else:
            return None
        # Remove query parameters if present
        if "?" in filename:
            filename = filename.split("?")[0]
        return filename
    
    # If it looks like a URL, extract the filename
    if "/" in asset_str:
        parts = asset_str.split("/")
        filename = parts[-1]
        # Remove query parameters if present
        if "?" in filename:
            filename = filename.split("?")[0]
        return filename
    
    # If it's a hash (long alphanumeric string), it might be the filename
    if len(asset_str) > 20 and re.match(r'^[a-f0-9-]+', asset_str, re.IGNORECASE):
        return asset_str
    
    return None

def make_thumbnail_path(row):
    """
    Construct a thumbnail path for each artwork pointing to actual image files.
    
    Priority order:
    1. Extract filename from thumbnail._sanityAsset if it contains a hash/filename
    2. Search images directory for matching files by artwork ID or slug
    3. Use slug-based naming as fallback (may not match actual files)
    4. Use _id as last resort
    
    The path will be like "/thumbnails/2a1607fc4327119b81585e0e309446cf0eced630-1366x969.png"
    which the Flask app will serve from production-export-2025-11-04t14-27-00-000z/images/
    """
    artwork_id = row.get("_id", "")
    slug = row.get("slug.current") or row.get("slug", "")
    
    # First, try to extract filename from thumbnail._sanityAsset
    # This is the most reliable source - it contains the actual image filename
    # Format: "image@file://./images/54720d2d6a042b48ec14902d6958ec8022e38017-1920x2880.jpg"
    thumb_asset = row.get("thumbnail._sanityAsset")
    if thumb_asset:
        filename = extract_filename_from_asset(thumb_asset)
        if filename:
            # Use the filename directly - it's the actual image file name
            return f"{BASE_THUMB_PATH}/{filename}"
    
    # Try to find the actual image file in the images directory
    # Pass the entire row to check for hash patterns in other fields
    # Convert pandas Series to dict
    try:
        row_dict = row.to_dict()
    except (AttributeError, TypeError):
        row_dict = {k: row[k] for k in row.index}
    found_filename = find_image_file(artwork_id, slug, row_data=row_dict)
    if found_filename:
        return f"{BASE_THUMB_PATH}/{found_filename}"
    
    # Fallback to slug-based naming (may not match actual files)
    if slug:
        slug_str = str(slug).strip()
        # Clean up the slug: replace spaces, remove invalid chars
        slug_str = slug_str.replace(" ", "-").replace("_", "-")
        # Remove any path separators
        slug_str = slug_str.replace("/", "-").replace("\\", "-")
        # Remove multiple consecutive dashes
        while "--" in slug_str:
            slug_str = slug_str.replace("--", "-")
        # Remove leading/trailing dashes
        slug_str = slug_str.strip("-")
        
        if slug_str:
            return f"{BASE_THUMB_PATH}/{slug_str}.jpg"
    
    # Last resort: use _id
    if artwork_id:
        artwork_id_clean = str(artwork_id).replace("drafts.", "").replace(".", "-")
        return f"{BASE_THUMB_PATH}/{artwork_id_clean}.jpg"
    
    return f"{BASE_THUMB_PATH}/unknown.jpg"

artworks["thumbnail"] = artworks.apply(make_thumbnail_path, axis=1)

# Verify required fields for build_faiss_index_ting.py
# Required: title, description, artist, year
required_fields = ["title", "description", "artist", "year"]
missing_fields = [f for f in required_fields if f not in artworks.columns]
if missing_fields:
    print(f"\nWARNING: Missing fields in CSV: {missing_fields}")
    print("Available fields:", [c for c in artworks.columns if any(rf in c.lower() for rf in required_fields)])
else:
    print(f"\n✓ All required fields present: {required_fields}")

# Preview a few artworks with key fields
print("\nSample artworks:")
display_cols = ["title", "artist", "year", "thumbnail"]
available_display = [c for c in display_cols if c in artworks.columns]
print(artworks[available_display].head())
print(f"\nTotal artworks: {len(artworks)}")
print(f"Artworks with artist names: {artworks['artist'].notna().sum() if 'artist' in artworks.columns else 0}")
print(f"Artworks with thumbnails: {artworks['thumbnail'].notna().sum()}")

# Save to CSV - this file will be used by build_faiss_index_ting.py
artworks.to_csv("artworks_with_thumbnails.csv", index=False)
print("\n✓ Saved to artworks_with_thumbnails.csv")