#!/usr/bin/env python3
"""
Find the image that's causing the DecompressionBombWarning.
The error says: Image size (101756928 pixels) exceeds limit of 89478485 pixels.
"""
import pandas as pd
from pathlib import Path
from PIL import Image
from PIL.Image import DecompressionBombWarning
import warnings

# Get project root
PROJECT_ROOT = Path(__file__).parent
CSV_PATH = PROJECT_ROOT / "artworks_with_thumbnails_ting.csv"
BASE_DIR = PROJECT_ROOT / "production-export-2025-11-04t14-27-00-000z"
FALLBACK_IMAGE = PROJECT_ROOT / "video_making/font/no_image.png"

# PIL's default limit is 89478485 pixels
PIL_LIMIT = 89478485
TARGET_PIXELS = 101756928  # The exact size from the error

def load_image_safe(image_path: str) -> tuple:
    """Load image and return its dimensions, or None if error.
    Temporarily increases PIL's image size limit to check large images."""
    if pd.isna(image_path) or not image_path or image_path == "N/A":
        return None, None, None
    
    # Temporarily increase PIL's image size limit
    Image.MAX_IMAGE_PIXELS = 500_000_000  # 500 million pixels
    
    # Try direct path (absolute or relative to project root)
    full_path = Path(image_path)
    if not full_path.is_absolute():
        full_path = PROJECT_ROOT / image_path
    
    if full_path.exists() and full_path.is_file():
        try:
            with Image.open(full_path) as img:
                width, height = img.size
                pixels = width * height
                return width, height, pixels
        except Exception as e:
            return None, None, str(e)
    
    # Try extracting filename
    if "images/" in image_path:
        filename = image_path.split("images/")[-1]
    else:
        filename = image_path
    
    images_dir = BASE_DIR / "images"
    if images_dir.exists():
        full_path = images_dir / filename
        if full_path.exists():
            try:
                with Image.open(full_path) as img:
                    width, height = img.size
                    pixels = width * height
                    return width, height, pixels
            except Exception as e:
                return None, None, str(e)
    
    return None, None, None


def main():
    print("=" * 60)
    print("Finding large images that exceed PIL limits")
    print("=" * 60)
    
    # Load CSV
    print(f"\n1. Loading CSV: {CSV_PATH}")
    df = pd.read_csv(str(CSV_PATH))
    print(f"   Loaded {len(df)} artworks")
    
    # Parse shelf numbers to find shelf 1 (group 1)
    def parse_shelf_numbers_flexible(shelf_value):
        """Parse shelfNo value handling both float and string formats."""
        if pd.isna(shelf_value) or shelf_value is None:
            return []
        if isinstance(shelf_value, (int, float)):
            shelf_str = str(int(shelf_value))
        else:
            shelf_str = str(shelf_value).strip()
        if not shelf_str:
            return []
        shelf_numbers = []
        for part in shelf_str.replace(",", ";").split(";"):
            part = part.strip()
            if part:
                try:
                    normalized = str(int(float(part)))
                    shelf_numbers.append(normalized)
                except (ValueError, TypeError):
                    shelf_numbers.append(part)
        return shelf_numbers
    
    df["shelf_list"] = df["shelfNo"].apply(parse_shelf_numbers_flexible)
    df = df[df["shelf_list"].apply(len) > 0].copy()
    df = df.explode("shelf_list", ignore_index=True)
    df["shelfNo"] = df["shelf_list"].astype(str)
    
    # Filter for shelf 1 (group 1)
    df_shelf1 = df[df["shelfNo"] == "1"].copy()
    print(f"\n2. Found {len(df_shelf1)} artworks on shelf 1 (group 1)")
    
    # Check all images, but prioritize shelf 1
    print("\n3. Checking images...")
    print("   Checking shelf 1 images first...")
    
    large_images = []
    
    # Check shelf 1 images
    for idx, row in df_shelf1.iterrows():
        artwork_id = row.get("id", idx)
        title = row.get("title", "Unknown")
        artist = row.get("artist", "Unknown")
        thumbnail_path = row.get("thumbnail", "")
        
        width, height, pixels = load_image_safe(thumbnail_path)
        
        if pixels is not None:
            if isinstance(pixels, int):
                if pixels >= PIL_LIMIT:
                    large_images.append({
                        "artwork_id": artwork_id,
                        "title": title,
                        "artist": artist,
                        "thumbnail_path": thumbnail_path,
                        "width": width,
                        "height": height,
                        "pixels": pixels,
                        "shelf": "1"
                    })
                    print(f"   FOUND LARGE IMAGE: ID={artwork_id}, Title={title}, Size={width}x{height} ({pixels:,} pixels)")
                    if pixels == TARGET_PIXELS:
                        print(f"   *** THIS IS THE EXACT SIZE FROM THE ERROR! ***")
    
    # If not found in shelf 1, check all images
    if not large_images:
        print("\n   Not found in shelf 1, checking all images...")
        for idx, row in df.iterrows():
            artwork_id = row.get("id", idx)
            title = row.get("title", "Unknown")
            artist = row.get("artist", "Unknown")
            thumbnail_path = row.get("thumbnail", "")
            shelf = row.get("shelfNo", "N/A")
            
            width, height, pixels = load_image_safe(thumbnail_path)
            
            if pixels is not None:
                if isinstance(pixels, int):
                    if pixels >= PIL_LIMIT:
                        large_images.append({
                            "artwork_id": artwork_id,
                            "title": title,
                            "artist": artist,
                            "thumbnail_path": thumbnail_path,
                            "width": width,
                            "height": height,
                            "pixels": pixels,
                            "shelf": shelf
                        })
                        print(f"   FOUND LARGE IMAGE: ID={artwork_id}, Title={title}, Shelf={shelf}, Size={width}x{height} ({pixels:,} pixels)")
                        if pixels == TARGET_PIXELS:
                            print(f"   *** THIS IS THE EXACT SIZE FROM THE ERROR! ***")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if large_images:
        print(f"\nFound {len(large_images)} image(s) exceeding PIL limit ({PIL_LIMIT:,} pixels):")
        for img in large_images:
            print(f"\n  Artwork ID: {img['artwork_id']}")
            print(f"  Title: {img['title']}")
            print(f"  Artist: {img['artist']}")
            print(f"  Shelf: {img['shelf']}")
            print(f"  Thumbnail Path: {img['thumbnail_path']}")
            print(f"  Dimensions: {img['width']} x {img['height']}")
            print(f"  Total Pixels: {img['pixels']:,}")
            if img['pixels'] == TARGET_PIXELS:
                print(f"  *** MATCHES ERROR SIZE EXACTLY! ***")
        
        # Find the exact match
        exact_matches = [img for img in large_images if img['pixels'] == TARGET_PIXELS]
        if exact_matches:
            print(f"\n*** EXACT MATCH FOUND ***")
            for img in exact_matches:
                print(f"  This is the problematic image:")
                print(f"  ID: {img['artwork_id']}")
                print(f"  Title: {img['title']}")
                print(f"  Path: {img['thumbnail_path']}")
    else:
        print("\nNo images found exceeding the PIL limit.")
        print("The image might be loaded from a different path or the error")
        print("might be coming from a different source.")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Suppress warnings temporarily to see our own output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DecompressionBombWarning)
        main()

