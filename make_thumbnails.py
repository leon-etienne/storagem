#!/usr/bin/env python
import os
from pathlib import Path

import pandas as pd
from PIL import Image

CSV_PATH = "artworks_with_thumbnails_final.csv"
IMAGE_COLUMN = "thumbnail"       # current column with image path
THUMB_COLUMN = "thumb_small"     # new column to write
IMAGE_ROOT = "."                 # or wherever your images live
THUMB_ROOT = "thumbs_256"        # output folder for generated thumbs
MAX_SIZE = (256, 256)            # max width, height

def main():
    df = pd.read_csv(CSV_PATH)

    out_dir = Path(THUMB_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    thumb_paths = []

    for idx, row in df.iterrows():
        src_rel = str(row[IMAGE_COLUMN])
        src_path = Path(IMAGE_ROOT) / src_rel
        if not src_path.exists():
            print("Missing:", src_path)
            thumb_paths.append("")  # or keep original
            continue

        # create deterministic name (id + original ext)
        stem = f"{row.get('id', idx)}"
        ext = src_path.suffix.lower() or ".jpg"
        thumb_name = f"{stem}{ext}"
        dst_path = out_dir / thumb_name

        if not dst_path.exists():
            try:
                im = Image.open(src_path).convert("RGB")
                im.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)
                # You can force jpeg/webp here if you want
                if ext in [".jpg", ".jpeg"]:
                    im.save(dst_path, quality=80, optimize=True)
                elif ext == ".webp":
                    im.save(dst_path, format="WEBP", quality=75)
                else:
                    im.save(dst_path, "JPEG", quality=80, optimize=True)
                print("Saved thumb:", dst_path)
            except Exception as e:
                print("Error on", src_path, "->", e)
                thumb_paths.append("")
                continue

        thumb_paths.append(str(dst_path))

    df[THUMB_COLUMN] = thumb_paths
    df.to_csv(CSV_PATH.replace(".csv", "_with_thumbs.csv"), index=False)
    print("Wrote updated CSV with thumbs column:", CSV_PATH.replace(".csv", "_with_thumbs.csv"))

if __name__ == "__main__":
    main()
