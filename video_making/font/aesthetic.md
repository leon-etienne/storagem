# Visual Identity Reference

## Design Principles
- Clean, Simple, black & white
- No Emoji
- No computer science data visualization aesthetic
- Modern Designer aesthetic

## Typography
- **Font**: `video_making/font/NeueHaasUnicaW1G-Regular.ttf`

## Color Palette
- **Primary Color**: Pure green (RGB: 0, 255, 0) / HTML: "lime"
- Colors should be simple and follow a pattern related to lime green
- Keep color usage minimal

## Technical Requirements
- Image generation should be smooth (for ffmpeg video conversion)
- All visualizations should follow these guidelines
- **Fallback Image**: `video_making/font/no_image.png` (use when artwork image cannot be found)

---

## Data Sources
- **CSV Data**: `artworks_with_thumbnails_ting.csv`
- **Embeddings**: `embeddings_cache/all_embeddings.pkl`
- **Artwork ID Field**: Use `id` column from CSV (e.g., "464", "152", "161")

---

## Curation Data

### Groups (Regal Clustering)
- **Group 1**: Regals 1, 5, 9
- **Group 2**: Regals 3, 4
- **Group 3**: Regals 2, 6, 7
- **Group 4**: Regals 0, 8

### Representatives (by Regal)
| Regal | Artwork ID | Notes |
|-------|------------|-------|
| 0     | 464        |       |
| 1     | 152        |       |
| 2     | 161        |       |
| 3     | 376        |       |
| 4     | 454        |       |
| 5     | 360        |       |
| 6     | 468        |       |
| 7     | 107        |       |
| 8     | 389        |       |
| 9     | 185        |       |

### Outliers (by Regal)
| Regal | Artwork ID |
|-------|------------|
| 0     | 479        |
| 1     | 386        |
| 2     | 326        |
| 3     | 82         |
| 4     | 424        |
| 5     | 310        |
| 6     | 93         |
| 7     | 96         |
| 8     | 343        |
| 9     | 441        |

### Quick Reference (Python Dict Format)
```python
GROUPS = {
    1: [1, 5, 9],
    2: [3, 4],
    3: [2, 6, 7],
    4: [0, 8]
}

REPRESENTATIVES = {
    0: 464, 1: 152, 2: 161, 3: 376, 4: 454,
    5: 360, 6: 468, 7: 107, 8: 389, 9: 185
}

OUTLIERS = {
    0: 479, 1: 386, 2: 326, 3: 82, 4: 424,
    5: 310, 6: 93, 7: 96, 8: 343, 9: 441
}
```

---

## Visualization Requirements

### File Naming System
- **Frames Directory**: `frames/regal{regal_number}_{mode}/`
  - Example: `frames/regal0_both/`, `frames/regal0_representative/`, `frames/regal0_outlier/`
- **Video Output**: `frames/regal{regal_number}_{mode}.mp4`
  - Example: `regal0_both.mp4`, `regal0_representative.mp4`, `regal0_outlier.mp4`
- **Mode Options**: `representative`, `outlier`, or `both` (default: both)

### Visual Elements

#### Points and Circles
- **Non-Regal points**: White fill, no outline, lower opacity (gray)
- **Regal points**: 
  - Highlighted: Bright green fill, no border
  - Non-highlighted: Lower opacity green fill, no border
- **Top 10 items**: Circular thumbnail images (40px diameter), no border
- **Centroid**: Pure white circle (18px radius), white fill and outline
- **Representatives/Outliers**: Marked with bright green circles

#### Text
- **Title text**: Displayed next to ALL Regal items (not just highlighted)
- **Text color**: White on black background
- **Font weights**: Thin for labels/info, Medium for titles

#### Lines and Connections
- **Connection lines between Regal items**: Gray, thin, less obvious
- **Lines from centroid**: Gray (except closest/farthest in green)
- **Lines between top 10 items**: Bright green lines connecting all pairs of top 10 items (5 representatives + 5 outliers)

#### Top 10 Display (5 Representatives + 5 Outliers)
- **Images**: Circular thumbnails on map, no borders
- **Aesthetic item**: Highlighted with bright green circle around image
- **Table on right side**: Shows Rank, Title, Distance for both Representatives and Outliers sections
- **Aesthetic item in table**: Highlighted in green
- **Green lines**: Bright green lines connect all pairs of top 10 items simultaneously
- **Title**: "Representatives" and "Outliers" as separate section titles (not "Top 10")

#### Zoom Effect
- **Duration**: 180 frames (6 seconds at 30fps)
- **Animation**: Smooth ease-in-out easing function
- **Image quality**: High-resolution images with LANCZOS resampling
- **Aspect ratio**: Maintained during zoom
- **Background**: Hides all points and connections during zoom

### Calculation Logic

#### Selection Process
- Finds **5 closest** artworks to centroid (representatives)
- Finds **5 farthest** artworks from centroid (outliers)
- **Aesthetic IDs are guaranteed**: The aesthetic representative and outlier IDs from `REPRESENTATIVES` and `OUTLIERS` dicts are always included in the top 5 lists, even if they don't rank in the top 5 by distance
- Sorts by distance (closest first for representatives, farthest first for outliers)
- Uses `REPRESENTATIVES` dict for aesthetic representative item
- Uses `OUTLIERS` dict for aesthetic outlier item

### Video Generation
- **FPS**: 60 frames per second
- **Codec**: H.264 (libx264)
- **Quality**: CRF 23
- **Format**: MP4 (yuv420p)

### Visualization Steps

1. **All Embeddings**: Show all artworks as points
2. **Identify Regal Items**: Highlight Regal items one by one (slow, no lines)
3. **Centroid & Distances**: Show Regal items with centroid and distance calculations
4. **Cycle Through Artworks**: Display each artwork with its calculation details
5. **Draw Lines from Centroid**: Animate lines from centroid to all Regal items
6. **Top 10 (5 Representatives + 5 Outliers)**: Show top 5 representatives and top 5 outliers appearing one by one with green lines connecting them simultaneously
7. **Side-by-Side View**: Show selected representative and outlier on left side with images and info, rank table on right side

### Last Step Layout
- **Left Side**: Both representative and outlier stacked vertically, each with:
  - Image (centered)
  - "Representative" or "Outlier" label in green
  - Title (white, wrapped if needed)
  - Artist (white)
  - Year (white)
- **Right Side**: Rank table showing:
  - "Representatives" section with top 5 (Rank, Title, Distance)
  - "Outliers" section with top 5 (Rank, Title, Distance)
  - Aesthetic items highlighted in green

### Usage
```bash
# Generate both representative and outlier video for Regal 0 (default)
python visualize_shelf0_representative.py --shelf 0 --mode both

# Generate representative video for Regal 0
python visualize_shelf0_representative.py --shelf 0 --mode representative

# Generate outlier video for Regal 0
python visualize_shelf0_representative.py --shelf 0 --mode outlier

# Generate for all Regals (both modes)
for regal in {0..9}; do
    python visualize_shelf0_representative.py --shelf $regal --mode both
done
```
