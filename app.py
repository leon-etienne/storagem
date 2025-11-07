import os
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory

import search_service
from search_service import MissingArtifactsError

app = Flask(__name__)

ARTIFACT_DIR = Path(os.getenv("SEARCH_ARTIFACT_DIR", "artifacts"))
# Try to find thumbnails directory - check multiple possible locations
# Now checking the production-export directory inside the storagem folder
THUMBNAIL_DIRS = [
    Path("thumbnails"),
    Path("production-export-2025-11-04t14-27-00-000z/images"),  # Updated: now relative to storagem folder
    Path("/Users/tcliu/Downloads/production-export-2025-11-04t14-27-00-000z/images"),  # Keep old path as fallback
    Path(os.getenv("THUMBNAIL_DIR", "")),
]
THUMBNAIL_DIR = next((d for d in THUMBNAIL_DIRS if d and d.exists()), None)

try:
    search_service.load(ARTIFACT_DIR)
except MissingArtifactsError as exc:
    print(f"WARNING: {exc}")

if THUMBNAIL_DIR:
    print(f"Thumbnails will be served from: {THUMBNAIL_DIR}")
else:
    print("WARNING: Thumbnail directory not found. Thumbnail images may not display.")


@app.route("/thumbnails/<path:filename>")
def serve_thumbnail(filename):
    """
    Serve thumbnail images from the configured directory.
    
    IMPORTANT PATH MISMATCH ISSUE:
    The CSV contains slug-based filenames (e.g., "wunsch-2-4.jpg"), but the actual
    image files use hash-based names (e.g., "005daa237f1c296f109abb9eb43e44cf57be8cdf-1667x2500.jpg").
    
    This means most requests will return 404 unless you:
    1. Create a mapping file (slug -> actual filename) and update this function
    2. Update read_data.py to extract actual image filenames from the NDJSON data
    3. Rename image files to match the slug-based names
    
    For now, this function only serves files that match exactly.
    """
    if THUMBNAIL_DIR and THUMBNAIL_DIR.exists():
        file_path = THUMBNAIL_DIR / filename
        if file_path.exists() and file_path.is_file():
            return send_from_directory(str(THUMBNAIL_DIR), filename)
    
    return "Thumbnail not found", 404


@app.route("/", methods=["GET", "POST"])
def index():
    q = request.form.get("q", "") if request.method == "POST" else ""
    results = None
    error = None
    cols_to_index = []

    if search_service.is_ready():
        cols_to_index = search_service.get_cols_to_index()
    else:
        error = (
            "Search index not loaded. Run build_faiss_index_ting.py to generate the FAISS artifacts."
        )

    if q and search_service.is_ready():
        df = search_service.search(q, top_k=5)
        if not df.empty:
            # Include all relevant fields for display
            display_cols = ["rank", "score", "title", "artist", "year", "thumbnail", "description"]
            # Only include columns that exist in the dataframe
            available_cols = [col for col in display_cols if col in df.columns]
            results = df[available_cols].to_dict(orient="records")
            # Normalize thumbnail paths to work with our /thumbnails/ route
            for result in results:
                if result.get("thumbnail"):
                    thumb_path = result["thumbnail"]
                    # Extract filename from paths like "/thumbnails/filename.jpg"
                    if thumb_path.startswith("/thumbnails/"):
                        filename = thumb_path.replace("/thumbnails/", "")
                        result["thumbnail"] = f"/thumbnails/{filename}"
                    elif not thumb_path.startswith("/"):
                        result["thumbnail"] = f"/thumbnails/{thumb_path}"
                    # If it's already a full path, keep it as is
        else:
            results = []

    return render_template(
        "index.html",
        q=q,
        results=results,
        cols_to_index=cols_to_index,
        error=error,
    )

if __name__ == "__main__":
    # Run on localhost
    app.run(host="127.0.0.1", port=5000, debug=True)
