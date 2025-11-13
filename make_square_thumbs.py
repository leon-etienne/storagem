import sys
import os
from pathlib import Path
from PIL import Image

def make_square_thumbnail(in_path, out_path, size):
    # Open image and ensure it has an alpha channel (RGBA)
    img = Image.open(in_path).convert("RGBA")

    # Resize while keeping aspect ratio – largest side == size
    img.thumbnail((size, size), Image.LANCZOS)

    # Create a transparent square background
    square = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    # Center the image on the square
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    square.paste(img, (x, y), img)

    # Always save as PNG to keep transparency
    out_path = out_path.with_suffix(".png")
    square.save(out_path, format="PNG")


def main():
    if len(sys.argv) < 3:
        print("Usage: python make_square_thumbs.py <input_folder> <output_folder> [size]")
        print("Example: python make_square_thumbs.py ./images ./thumbs 512")
        sys.exit(1)

    input_folder = Path(sys.argv[1])
    output_folder = Path(sys.argv[2])
    size = int(sys.argv[3]) if len(sys.argv) >= 4 else 512  # default size 512x512

    if not input_folder.is_dir():
        print(f"Input folder does not exist or is not a directory: {input_folder}")
        sys.exit(1)

    output_folder.mkdir(parents=True, exist_ok=True)

    # Common image extensions – extend if you like
    exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

    for in_file in input_folder.iterdir():
        if not in_file.is_file():
            continue
        if in_file.suffix.lower() not in exts:
            continue

        out_file = output_folder / in_file.stem  # suffix set in make_square_thumbnail
        try:
            print(f"Processing {in_file} -> {out_file.with_suffix('.png')}")
            make_square_thumbnail(in_file, out_file, size)
        except Exception as e:
            print(f"Failed to process {in_file}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
