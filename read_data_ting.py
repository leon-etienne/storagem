# --- NDJSON Parser with Improved Nested Content Handling ---

import json
from typing import Any, Dict, Iterator, List, Optional, TextIO, Union
import pandas as pd
from pathlib import Path

JSONLike = Dict[str, Any]

# ---------- PARSER FUNCTIONS ----------

def _open_maybe(path_or_file: Union[str, TextIO]) -> Iterator[TextIO]:
    """Yield a file-like object from path or an already-open file handle."""
    if hasattr(path_or_file, "read"):
        yield path_or_file
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


# ---------- NESTED CONTENT HANDLING FUNCTIONS ----------

def extract_internationalized_value(value: Any, preferred_lang: str = "de") -> Optional[str]:
    """
    Extract value from internationalized array structure.
    Structure: [{"_key": "de", "_type": "...", "value": "..."}, ...]
    Returns the value for preferred_lang, or first available value.
    """
    if not value or not isinstance(value, list):
        return None
    
    # Try to find preferred language first
    for item in value:
        if isinstance(item, dict):
            if item.get("_key") == preferred_lang and "value" in item:
                val = item.get("value")
                if val:
                    return str(val).strip()
    
    # Fall back to first available value
    for item in value:
        if isinstance(item, dict) and "value" in item:
            val = item.get("value")
            if val:
                return str(val).strip()
    
    return None


def extract_rich_text(value: Any) -> Optional[str]:
    """
    Extract text from rich text structure (block content).
    Structure: [{"_key": "...", "_type": "block", "children": [{"_type": "span", "text": "..."}], ...}]
    Or: [{"_key": "de", "_type": "...", "value": [blocks...]}]
    """
    if not value:
        return None
    
    texts = []
    
    # If it's a list of internationalized values with nested value field
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                # Check if it's an internationalized array with value containing blocks
                if "value" in item:
                    val = item.get("value")
                    # If value is a list of blocks
                    if isinstance(val, list):
                        for block in val:
                            if isinstance(block, dict) and "children" in block:
                                for child in block.get("children", []):
                                    if isinstance(child, dict) and "text" in child:
                                        text = child.get("text", "").strip()
                                        if text:
                                            texts.append(text)
                    # If value is a simple string
                    elif val:
                        texts.append(str(val).strip())
                # Direct block structure (no value wrapper)
                elif "children" in item:
                    for child in item.get("children", []):
                        if isinstance(child, dict) and "text" in child:
                            text = child.get("text", "").strip()
                            if text:
                                texts.append(text)
                # Direct text in block
                elif "text" in item:
                    text_val = item.get("text", "")
                    if isinstance(text_val, str):
                        text = text_val.strip()
                        if text:
                            texts.append(text)
                    elif isinstance(text_val, list):
                        # Handle list of text values
                        for t in text_val:
                            if isinstance(t, str) and t.strip():
                                texts.append(t.strip())
    
    # Direct block structure (dict, not list)
    elif isinstance(value, dict):
        if "children" in value:
            for child in value.get("children", []):
                if isinstance(child, dict) and "text" in child:
                    text = child.get("text", "").strip()
                    if text:
                        texts.append(text)
        elif "text" in value:
            text_val = value.get("text", "")
            if isinstance(text_val, str):
                text = text_val.strip()
                if text:
                    texts.append(text)
            elif isinstance(text_val, list):
                # Handle list of text values
                for t in text_val:
                    if isinstance(t, str) and t.strip():
                        texts.append(t.strip())
    
    return " ".join(texts).strip() if texts else None


def combine_name(obj: Dict[str, Any]) -> Optional[str]:
    """Combine fName and lName from nested name object."""
    name_obj = obj.get("name")
    if isinstance(name_obj, dict):
        fname = name_obj.get("fName", "").strip()
        lname = name_obj.get("lName", "").strip()
        if fname and lname:
            return f"{fname} {lname}"
        elif fname:
            return fname
        elif lname:
            return lname
    return None


def combine_size(obj: Dict[str, Any]) -> Optional[str]:
    """Combine height, width, depth from nested size object."""
    size_obj = obj.get("size")
    if isinstance(size_obj, dict):
        parts = []
        if "height" in size_obj:
            parts.append(f"h:{size_obj['height']}")
        if "width" in size_obj:
            parts.append(f"w:{size_obj['width']}")
        if "depth" in size_obj:
            parts.append(f"d:{size_obj['depth']}")
        return " x ".join(parts) if parts else None
    return None


def combine_address(obj: Dict[str, Any]) -> Optional[str]:
    """Combine street, city, zip from nested address object."""
    addr_obj = obj.get("address")
    if isinstance(addr_obj, dict):
        parts = []
        if "street" in addr_obj:
            parts.append(addr_obj["street"].strip())
        if "city" in addr_obj:
            parts.append(addr_obj["city"].strip())
        if "zip" in addr_obj:
            parts.append(addr_obj["zip"].strip())
        return ", ".join(parts) if parts else None
    return None


def extract_slug(obj: Dict[str, Any]) -> Optional[str]:
    """Extract current value from nested slug object."""
    slug_obj = obj.get("slug")
    if isinstance(slug_obj, dict):
        return slug_obj.get("current", "").strip() or None
    return None


def extract_handling_status(obj: Dict[str, Any]) -> Optional[str]:
    """Extract and combine handling status array."""
    handling_obj = obj.get("handling")
    if isinstance(handling_obj, dict):
        status = handling_obj.get("status")
        if isinstance(status, list):
            return ", ".join(str(s) for s in status if s)
    return None


def flatten_nested_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested objects in the JSON structure.
    Handles:
    - name.fName + name.lName → name
    - size.height, width, depth → size
    - address.street, city, zip → address
    - slug.current → slug
    - Internationalized arrays → extract value
    - Rich text structures → extract text
    """
    flattened = {}
    
    # Copy all top-level non-nested fields
    for key, value in obj.items():
        # Skip nested objects we'll handle specially
        if key in ["name", "size", "address", "slug", "handling"]:
            continue
        
        # Handle internationalized arrays (title, subtitle, description, etc.)
        if isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], dict) and "_key" in value[0]:
                # For description and other rich text fields, try rich text first
                if key in ["description", "addition", "internalNote", "shortBio"]:
                    extracted = extract_rich_text(value)
                    if extracted:
                        flattened[key] = extracted
                        continue
                    # If extraction failed and it's an empty structure, set to None
                    else:
                        flattened[key] = None
                        continue
                # For simple internationalized values (title, subtitle), try simple extraction first
                else:
                    extracted = extract_internationalized_value(value)
                    if extracted:
                        flattened[key] = extracted
                        continue
                    # Fallback to rich text for other fields
                    extracted = extract_rich_text(value)
                    if extracted:
                        flattened[key] = extracted
                        continue
                    # If extraction failed, set to None instead of keeping raw structure
                    flattened[key] = None
                    continue
        
        # Handle rich text structures (non-internationalized)
        if isinstance(value, list):
            extracted = extract_rich_text(value)
            if extracted:
                flattened[key] = extracted
                continue
        
        # Copy other values as-is
        flattened[key] = value
    
    # Handle nested name object
    name = combine_name(obj)
    if name:
        flattened["name"] = name
    
    # Handle nested size object
    size = combine_size(obj)
    if size:
        flattened["size"] = size
    
    # Handle nested address object
    address = combine_address(obj)
    if address:
        flattened["address"] = address
    
    # Handle nested slug object
    slug = extract_slug(obj)
    if slug:
        flattened["slug"] = slug
    
    # Handle handling.status
    handling_status = extract_handling_status(obj)
    if handling_status:
        flattened["handling_status"] = handling_status
    
    return flattened


# ---------- MAIN PROCESSING ----------

def process_ndjson_file(path: str) -> pd.DataFrame:
    """
    Read and process NDJSON file with proper nested content handling.
    """
    print(f"Reading NDJSON file: {path}")
    records = list(iter_ndjson(path))
    print(f"Loaded {len(records)} records")
    
    # Flatten nested structures
    print("Flattening nested structures...")
    flattened_records = [flatten_nested_object(record) for record in records]
    
    # Convert to DataFrame
    print("Converting to DataFrame...")
    df = pd.json_normalize(flattened_records)
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df


# ---------- ARTIST RESOLUTION ----------

def resolve_artist_references(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve artist references in artworks to actual artist names.
    """
    # Create artist mapping
    artists_df = df[df["_type"] == "artist"].copy()
    artist_map = {}
    
    for _, artist_row in artists_df.iterrows():
        artist_id = artist_row.get("_id", "")
        if artist_id:
            # Handle both "drafts.xxx" and regular IDs
            artist_id_clean = artist_id.replace("drafts.", "")
            
            # Get artist name (already flattened)
            artist_name = artist_row.get("name")
            if artist_name:
                artist_map[artist_id] = artist_name
                artist_map[artist_id_clean] = artist_name
    
    print(f"Loaded {len(artist_map)} artist names")
    
    # Filter artworks
    artworks = df[df["_type"] == "artwork"].copy()
    
    # Resolve artist references
    def resolve_artist_name(row):
        artist_ref = row.get("artist._ref", "")
        
        if pd.isna(artist_ref) or artist_ref is None or artist_ref == "":
            return None
        
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
    
    artworks["artist"] = artworks.apply(resolve_artist_name, axis=1)
    print(f"Resolved artist names for {artworks['artist'].notna().sum()} out of {len(artworks)} artworks")
    
    return artworks


# ---------- THUMBNAIL PATH HANDLING ----------

def extract_filename_from_asset(asset_value: Any) -> Optional[str]:
    """Extract filename from Sanity asset reference."""
    if not asset_value or not isinstance(asset_value, str):
        return None
    
    asset_str = str(asset_value).strip()
    
    # Handle Sanity asset format: "image@file://./images/filename.jpg"
    if "@file://" in asset_str:
        if "./images/" in asset_str:
            filename = asset_str.split("./images/")[-1]
        elif "./files/" in asset_str:
            filename = asset_str.split("./files/")[-1]
        elif "/" in asset_str:
            parts = asset_str.split("/")
            filename = parts[-1]
        else:
            return None
        # Remove query parameters if present
        if "?" in filename:
            filename = filename.split("?")[0]
        return filename
    
    return None


def make_thumbnail_path(row: pd.Series) -> str:
    """Construct a thumbnail path for each artwork."""
    BASE_THUMB_PATH = "/thumbnails"
    
    # Try to extract filename from thumbnail._sanityAsset
    thumb_asset = row.get("thumbnail._sanityAsset")
    if thumb_asset:
        filename = extract_filename_from_asset(thumb_asset)
        if filename:
            return f"{BASE_THUMB_PATH}/{filename}"
    
    # Fallback to slug-based naming
    slug = row.get("slug", "")
    if slug:
        slug_str = str(slug).strip().replace(" ", "-").replace("_", "-")
        slug_str = slug_str.replace("/", "-").replace("\\", "-")
        while "--" in slug_str:
            slug_str = slug_str.replace("--", "-")
        slug_str = slug_str.strip("-")
        if slug_str:
            return f"{BASE_THUMB_PATH}/{slug_str}.jpg"
    
    # Last resort: use _id
    artwork_id = row.get("_id", "")
    if artwork_id:
        artwork_id_clean = str(artwork_id).replace("drafts.", "").replace(".", "-")
        return f"{BASE_THUMB_PATH}/{artwork_id_clean}.jpg"
    
    return f"{BASE_THUMB_PATH}/unknown.jpg"


# ---------- SHELF NUMBER HANDLING ----------

def parse_shelf_numbers(shelf_value: Any) -> List[str]:
    """
    Parse shelfNo value and return list of individual shelf numbers.
    Handles formats like: "3", "3;4", "07", "3;4;5", etc.
    Normalizes shelf numbers by removing leading zeros (e.g., "07" -> "7").
    """
    if pd.isna(shelf_value) or shelf_value is None:
        return []
    
    shelf_str = str(shelf_value).strip()
    if not shelf_str:
        return []
    
    # Split by semicolon and clean each part
    shelf_numbers = []
    for part in shelf_str.split(";"):
        part = part.strip()
        if part:
            # Normalize by converting to int (removes leading zeros) then back to string
            try:
                # Try to convert to int to remove leading zeros
                normalized = str(int(part))
                shelf_numbers.append(normalized)
            except ValueError:
                # If it's not a number, keep as-is (e.g., "A", "B1", etc.)
                shelf_numbers.append(part)
    
    return shelf_numbers


def duplicate_artworks_by_shelf(artworks: pd.DataFrame) -> pd.DataFrame:
    """
    Duplicate artwork entries when shelfNo contains multiple shelf numbers (e.g., "3;4").
    Each duplicate will have a single shelfNo value.
    """
    if "shelfNo" not in artworks.columns:
        return artworks
    
    # Parse shelf numbers for each row
    artworks["shelf_list"] = artworks["shelfNo"].apply(parse_shelf_numbers)
    
    # Count how many rows will be created
    shelf_lengths = artworks["shelf_list"].apply(len)
    total_duplicates = shelf_lengths.sum()
    single_shelf_count = (shelf_lengths == 1).sum()
    multi_shelf_count = (shelf_lengths > 1).sum()
    no_shelf_count = (shelf_lengths == 0).sum()
    
    print(f"\nShelf number processing:")
    print(f"  - Artworks with single shelf: {single_shelf_count}")
    print(f"  - Artworks with multiple shelves: {multi_shelf_count}")
    print(f"  - Artworks with no shelf: {no_shelf_count}")
    print(f"  - Total rows before: {len(artworks)}")
    print(f"  - Total rows after: {total_duplicates + no_shelf_count}")
    
    # Explode the shelf_list to create duplicate rows
    # First, handle rows with no shelf numbers (set to empty string or keep original)
    artworks_no_shelf = artworks[artworks["shelf_list"].apply(len) == 0].copy()
    if len(artworks_no_shelf) > 0:
        artworks_no_shelf["shelfNo"] = ""
    
    # Handle rows with shelf numbers
    artworks_with_shelf = artworks[artworks["shelf_list"].apply(len) > 0].copy()
    
    if len(artworks_with_shelf) > 0:
        # Explode to create one row per shelf number
        artworks_exploded = artworks_with_shelf.explode("shelf_list", ignore_index=False)
        artworks_exploded["shelfNo"] = artworks_exploded["shelf_list"]
        
        # Drop the temporary shelf_list column
        artworks_exploded = artworks_exploded.drop(columns=["shelf_list"])
        
        # Combine with artworks that have no shelf numbers
        if len(artworks_no_shelf) > 0:
            artworks_no_shelf = artworks_no_shelf.drop(columns=["shelf_list"])
            result = pd.concat([artworks_exploded, artworks_no_shelf], ignore_index=True)
        else:
            result = artworks_exploded.reset_index(drop=True)
    else:
        # No artworks with shelf numbers
        if len(artworks_no_shelf) > 0:
            artworks_no_shelf = artworks_no_shelf.drop(columns=["shelf_list"])
            result = artworks_no_shelf.reset_index(drop=True)
        else:
            result = artworks.reset_index(drop=True)
            result = result.drop(columns=["shelf_list"])
    
    return result


# ---------- MAIN EXECUTION ----------

if __name__ == "__main__":
    # Path to the NDJSON data file
    path = "production-export-2025-11-04t14-27-00-000z/data.ndjson"
    
    # Process the file
    df = process_ndjson_file(path)
    
    # Resolve artist references for artworks
    artworks = resolve_artist_references(df)
    
    # Add thumbnail paths
    artworks["thumbnail"] = artworks.apply(make_thumbnail_path, axis=1)
    
    # Sanitize and duplicate artworks by shelf number
    artworks = duplicate_artworks_by_shelf(artworks)
    
    # Convert shelfNo to string format (remove .0 from floats, keep as clean numbers)
    if "shelfNo" in artworks.columns:
        def format_shelf_number(value):
            if pd.isna(value) or value == "":
                return ""
            # Convert to int then string to remove leading zeros and .0
            try:
                return str(int(float(value)))
            except (ValueError, TypeError):
                return str(value) if value else ""
        artworks["shelfNo"] = artworks["shelfNo"].apply(format_shelf_number)
    
    # Add cluster column based on shelf number
    def assign_cluster(shelf_value):
        """
        Assign cluster based on shelf number:
        Cluster 1: 5, 9, 1
        Cluster 2: 3
        Cluster 3: 4
        Cluster 4: 2, 6, 7
        Cluster 5: 8, 0
        Cluster 0: all other shelf numbers and empty values
        Returns just the number (0, 1, 2, 3, 4, 5).
        """
        if pd.isna(shelf_value) or shelf_value == "":
            return "0"
        
        try:
            # Convert to int for comparison
            shelf_num = int(float(str(shelf_value)))
            
            # Assign cluster based on shelf number
            if shelf_num in [5, 9, 1]:
                return "1"
            elif shelf_num == 3:
                return "2"
            elif shelf_num == 4:
                return "3"
            elif shelf_num in [2, 6, 7]:
                return "4"
            elif shelf_num in [8, 0]:
                return "5"
            else:
                # Shelf number not in any defined cluster, assign to cluster 0
                return "0"
        except (ValueError, TypeError):
            # Non-numeric shelf value, assign to cluster 0
            return "0"
    
    artworks["cluster"] = artworks["shelfNo"].apply(assign_cluster)
    
    # Clean up list fields (convert arrays to comma-separated strings)
    list_fields = ["rental", "status", "media"]
    for field in list_fields:
        if field in artworks.columns:
            def clean_list(value):
                if isinstance(value, list):
                    return ", ".join(str(v) for v in value if v)
                return value
            artworks[field] = artworks[field].apply(clean_list)
    
    # Verify required fields
    required_fields = ["title", "description", "artist", "year"]
    missing_fields = [f for f in required_fields if f not in artworks.columns]
    if missing_fields:
        print(f"\nWARNING: Missing fields in CSV: {missing_fields}")
        print("Available fields:", [c for c in artworks.columns if any(rf in c.lower() for rf in required_fields)])
    else:
        print(f"\n✓ All required fields present: {required_fields}")
    
    # Preview sample artworks
    print("\nSample artworks:")
    display_cols = ["title", "artist", "year", "thumbnail", "name", "size", "address"]
    available_display = [c for c in display_cols if c in artworks.columns]
    print(artworks[available_display].head())
    print(f"\nTotal artworks: {len(artworks)}")
    print(f"Artworks with artist names: {artworks['artist'].notna().sum() if 'artist' in artworks.columns else 0}")
    print(f"Artworks with thumbnails: {artworks['thumbnail'].notna().sum()}")
    
    # Save to CSV
    output_file = "artworks_with_thumbnails_ting.csv"
    artworks.to_csv(output_file, index=False)
    print(f"\n✓ Saved to {output_file}")

