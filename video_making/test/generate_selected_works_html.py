#!/usr/bin/env python3
"""
Generate an HTML file displaying selected representative and outlier works for each shelf.
"""
import pandas as pd
from pathlib import Path

# Get script directory and project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Constants
CSV_PATH = PROJECT_ROOT / "artworks_with_thumbnails_ting.csv"
BASE_DIR = PROJECT_ROOT / "production-export-2025-11-13t13-42-48-005z"
OUTPUT_HTML = SCRIPT_DIR / "selected_works.html"

# Selected works (representative and outlier for each shelf)
# Matches aesthetic.md curation (video_making/font/aesthetic.md lines 150-152)
# Also matches visualize_shelf0_representative.py REPRESENTATIVES and OUTLIERS dictionaries
REPRESENTATIVES = {
    0: 464, 1: 152, 2: 161, 3: 119, 4: 454,
    5: 360, 6: 468, 7: 107, 8: 385, 9: 185
}
OUTLIERS = {
    0: 479, 1: 386, 2: 326, 3: 82, 4: 424,
    5: 310, 6: 93, 7: 96, 8: 343, 9: 441
}

def load_image_path(thumbnail_path: str) -> str:
    """Get the image path, handling relative paths from HTML file location."""
    if pd.isna(thumbnail_path) or not thumbnail_path or thumbnail_path == "N/A":
        return ""
    
    # HTML file is in video_making/test/, so we need to go up 2 levels (../../) 
    # to reach the project root, then use the thumbnail path
    
    # If thumbnail_path is already a relative path from project root, prepend ../../
    if not Path(thumbnail_path).is_absolute():
        # Check if it's already a full path from project root
        if "production-export" in thumbnail_path or "images/" in thumbnail_path:
            # Path is like "production-export-2025-11-13t13-42-48-005z/images/..."
            # Make it relative to HTML file location
            return f"../../{thumbnail_path}"
        
        # Try to find the file
        full_path = PROJECT_ROOT / thumbnail_path
        if full_path.exists() and full_path.is_file():
            return f"../../{thumbnail_path}"
        
        # Try extracting filename and looking in BASE_DIR/images
        if "images/" in thumbnail_path:
            filename = thumbnail_path.split("images/")[-1]
        else:
            filename = thumbnail_path
        
        images_dir = BASE_DIR / "images"
        if images_dir.exists():
            full_path = images_dir / filename
            if full_path.exists():
                # Return relative path from HTML location
                rel_path = full_path.relative_to(PROJECT_ROOT)
                return f"../../{rel_path}"
    
    # If absolute path, try to make it relative
    full_path = Path(thumbnail_path)
    if full_path.is_absolute() and full_path.exists():
        try:
            rel_path = full_path.relative_to(PROJECT_ROOT)
            return f"../../{rel_path}"
        except ValueError:
            pass
    
    # Fallback: return as-is (might be a URL or already correct)
    return str(thumbnail_path)

def find_artwork(df: pd.DataFrame, artwork_id: int) -> pd.Series:
    """Find artwork by ID in the dataframe."""
    # Try multiple matching methods
    artwork_row = pd.Series(dtype=object)
    
    try:
        # Try exact match with float
        artwork_row = df[df["id"].astype(float) == float(artwork_id)]
    except (ValueError, TypeError):
        pass
    
    if artwork_row.empty:
        # Try string match
        try:
            artwork_row = df[df["id"].astype(str).str.strip() == str(artwork_id).strip()]
        except (ValueError, TypeError):
            pass
    
    if artwork_row.empty:
        # Try removing .0 from float strings
        try:
            artwork_row = df[df["id"].astype(str).str.replace('.0', '', regex=False).str.strip() == str(artwork_id).strip()]
        except (ValueError, TypeError):
            pass
    
    if not artwork_row.empty:
        return artwork_row.iloc[0]
    return pd.Series(dtype=object)

def generate_html():
    """Generate HTML file with selected works."""
    print("Loading CSV...")
    df = pd.read_csv(str(CSV_PATH))
    print(f"Loaded {len(df)} artworks")
    
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Selected Works by Shelf</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        html, body {
            height: 100vh;
            overflow: auto;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Helvetica, Arial, sans-serif;
            background: rgb(0, 0, 0);
            color: rgb(255, 255, 255);
            padding: 20px;
            line-height: 1.4;
            font-size: 14px;
        }
        h1 {
            font-size: 32px;
            font-weight: 500;
            margin-bottom: 20px;
            text-align: center;
            color: rgb(255, 255, 255);
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 12px;
            width: 100%;
        }
        .shelf-item {
            border: 1px solid rgb(80, 80, 80);
            padding: 12px;
            display: flex;
            flex-direction: column;
            background: rgb(0, 0, 0);
        }
        .shelf-header {
            font-weight: 500;
            text-align: center;
            font-size: 18px;
            margin-bottom: 12px;
            border-bottom: 1px solid rgb(80, 80, 80);
            padding-bottom: 8px;
            color: rgb(0, 255, 0);
        }
        .works-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        .work {
            display: flex;
            flex-direction: column;
            font-size: 12px;
        }
        .work-label {
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 8px;
            text-transform: uppercase;
            color: rgb(0, 255, 0);
            letter-spacing: 0.5px;
        }
        .work-image {
            width: 100%;
            max-width: 100%;
            height: auto;
            margin-bottom: 8px;
            border: 1px solid rgb(80, 80, 80);
            display: block;
            background: rgb(0, 0, 0);
        }
        .work-info {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .work-id {
            font-family: 'Courier New', monospace;
            font-size: 11px;
            color: rgb(200, 200, 200);
            font-weight: 300;
        }
        .work-title {
            font-weight: 500;
            font-size: 16px;
            line-height: 1.3;
            color: rgb(255, 255, 255);
            margin-top: 4px;
        }
        .work-artist {
            font-size: 14px;
            color: rgb(200, 200, 200);
            font-weight: 300;
        }
        @media print {
            html, body {
                height: auto;
                overflow: visible;
            }
            @page {
                size: A4;
                margin: 0.5cm;
            }
            body {
                padding: 15px;
                font-size: 10px;
                background: white;
                color: black;
            }
            h1 {
                font-size: 24px;
                margin-bottom: 15px;
                color: black;
            }
            .grid {
                grid-template-columns: repeat(3, 1fr);
                gap: 8px;
            }
            .shelf-item {
                padding: 8px;
                border: 1px solid black;
                background: white;
            }
            .shelf-header {
                font-size: 14px;
                margin-bottom: 8px;
                border-bottom: 1px solid black;
                color: black;
            }
            .works-container {
                gap: 8px;
            }
            .work-label {
                font-size: 11px;
                color: black;
            }
            .work-image {
                margin-bottom: 6px;
                border: 1px solid black;
            }
            .work-title {
                font-size: 12px;
                color: black;
            }
            .work-id, .work-artist {
                font-size: 9px;
                color: rgb(80, 80, 80);
            }
        }
    </style>
</head>
<body>
    <h1>Selected Works by Shelf</h1>
    <div class="grid">
""")
    
    # Generate HTML grid items for each shelf
    for shelf_num in sorted(REPRESENTATIVES.keys()):
        rep_id = REPRESENTATIVES[shelf_num]
        outlier_id = OUTLIERS[shelf_num]
        
        rep_artwork = find_artwork(df, rep_id)
        outlier_artwork = find_artwork(df, outlier_id)
        
        html_parts.append(f'        <div class="shelf-item">\n')
        html_parts.append(f'            <div class="shelf-header">Shelf {shelf_num}</div>\n')
        html_parts.append(f'            <div class="works-container">\n')
        
        # Representative work
        html_parts.append(f'                <div class="work">\n')
        html_parts.append(f'                    <div class="work-label">Representative</div>\n')
        if not rep_artwork.empty:
            rep_title = str(rep_artwork.get("title", "Unknown"))
            rep_artist = str(rep_artwork.get("artist", "Unknown"))
            rep_id_display = str(rep_artwork.get("id", rep_id))
            rep_thumbnail = load_image_path(rep_artwork.get("thumbnail", ""))
            
            if rep_thumbnail:
                html_parts.append(f'                    <img src="{rep_thumbnail}" alt="{rep_title}" class="work-image" onerror="this.style.display=\'none\'">\n')
            html_parts.append(f'                    <div class="work-info">\n')
            html_parts.append(f'                        <div class="work-id">{rep_id_display}</div>\n')
            html_parts.append(f'                        <div class="work-title">{rep_title}</div>\n')
            html_parts.append(f'                        <div class="work-artist">{rep_artist}</div>\n')
            html_parts.append(f'                    </div>\n')
        else:
            html_parts.append(f'                    <div class="work-info">Not found (ID: {rep_id})</div>\n')
        html_parts.append(f'                </div>\n')
        
        # Outlier work
        html_parts.append(f'                <div class="work">\n')
        html_parts.append(f'                    <div class="work-label">Outlier</div>\n')
        if not outlier_artwork.empty:
            outlier_title = str(outlier_artwork.get("title", "Unknown"))
            outlier_artist = str(outlier_artwork.get("artist", "Unknown"))
            outlier_id_display = str(outlier_artwork.get("id", outlier_id))
            outlier_thumbnail = load_image_path(outlier_artwork.get("thumbnail", ""))
            
            if outlier_thumbnail:
                html_parts.append(f'                    <img src="{outlier_thumbnail}" alt="{outlier_title}" class="work-image" onerror="this.style.display=\'none\'">\n')
            html_parts.append(f'                    <div class="work-info">\n')
            html_parts.append(f'                        <div class="work-id">{outlier_id_display}</div>\n')
            html_parts.append(f'                        <div class="work-title">{outlier_title}</div>\n')
            html_parts.append(f'                        <div class="work-artist">{outlier_artist}</div>\n')
            html_parts.append(f'                    </div>\n')
        else:
            html_parts.append(f'                    <div class="work-info">Not found (ID: {outlier_id})</div>\n')
        html_parts.append(f'                </div>\n')
        
        html_parts.append(f'            </div>\n')
        html_parts.append(f'        </div>\n')
    
    html_parts.append("""    </div>
</body>
</html>""")
    
    # Write HTML file
    html_content = "".join(html_parts)
    OUTPUT_HTML.write_text(html_content, encoding='utf-8')
    print(f"\nHTML file generated: {OUTPUT_HTML}")
    print(f"Open it in a browser to view the selected works.")

if __name__ == "__main__":
    generate_html()

