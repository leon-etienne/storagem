# Visual Identity & Design System

## Core Principles

- **Clean & Minimal**: Black & white with lime green accent
- **Modern Designer Aesthetic**: Focus on typography, spacing, and visual hierarchy
- **No Emoji**: Keep it professional

---

## Typography

**Font**: Neue Haas Unica W1G (`video_making/font/`)
- Regular, Light, Medium, Thin weights available

**Font Sizes**:
- **Title**: 48pt Medium
- **Subtitle/Stage**: 28pt Thin (position: 60px from top)
- **Section Titles** (Top 10, Ruler): 32pt Medium, lime green
- **Artwork Title**: 24pt Medium
- **Artist**: 20pt Medium
- **Info Text**: 18pt Thin
- **Small Text**: 14pt Thin
- **Table Text**: 16pt Thin
- **Ruler Step Labels**: 18pt Medium (text in selected boxes)

---

## Colors

**Primary**:
- Lime Green: RGB(0, 255, 0) - Accent color
- Black: RGB(0, 0, 0) - Background
- White: RGB(255, 255, 255) - Text

**Secondary**:
- Basic Dots: RGB(90, 90, 90) / RGB(80, 80, 80)
- Text Gray: RGB(200, 200, 200) / RGB(150, 150, 150)
- Connection Lines: RGB(100, 100, 100)

**Usage**: Lime green only for highlights, markers, measurements, and section labels.

---

## Layout

**Canvas**: 
- **Full HD**: 1920×1080px (default)
- **4K**: 3840×2160px (use `--scale 4.0`)
- **Map Area**: 1200px (Full HD) / 2400px (4K) (left)
- **Panel Area**: 720px (Full HD) / 1440px (4K) (right)

**Positions**:
- Floor plan: Top left (50px from left, 60px from top), rotated 90° counter-clockwise, max 300px
- Subtitle: Right side of title, vertically centered
- Panel content: 60px from top
- Title: 50px from left, 100px from bottom

---

## Visual Elements

### Points
- **Basic Dots**: 4px, dark grey (RGB 80-90)
- **Regal Points**: 8px, lime green
- **Representative Points**: 40px circular thumbnails
- **Centroid**: Crosshair (12px circle, 20px lines)

### Lines
- **Connection Lines**: 1px grey, fade out after highlight stage
- **Finding Representatives**: 3px green with distance numbers (12pt)
- **Ruler Lines**: 3px green with tick marks, arrowhead, distance label (32pt)
- **Top 10 Connections**: 1px green between all pairs

---

## Stages

### 1. All Embeddings
- 60 frames + 2 seconds hold (180 frames total)
- All points grey
- Floor plan visible at top left

### 2. Identify Regal Items
- 30 frames per item
- Items appear one by one with easing
- No lines or centroid
- + 2 seconds hold at end

### 3. Highlight with Centroid
- 20 frames per item
- **All green dots from Stage 2 already visible**
- Lines and centroid animate progressively
- + 2 seconds hold at end

### 3.5. Fade Out Lines
- 30 frames
- Grey connection lines fade out
- + 2 seconds hold at end

### 4. Cycle Through Artworks
- 25 frames to draw line + hold per artwork
- Green lines (3px) with distance numbers (shown on lines)
- Subtitle: "Finding Representatives | Distance: X.XXXX" or "Representative Found | Closest to Centroid" / "Outlier Found | Farthest from Centroid"
- + 2 seconds hold at end

### 5. Draw Lines from Centroid
- 30 frames per line
- Lines drawn one by one
- + 2 seconds hold at end

### 6. Top 10 Items
- 40 frames per item
- Section titles: "Closest to Centroid" / "Farthest from Centroid" (32pt Medium, lime green)
- 5 representatives, then 5 outliers appear one by one
- Selected items use same color as others (no green highlighting)
- + 2 seconds hold at end

### 6.5. Blink Effect
- 2 seconds (120 frames)
- Selected items blink between gray and green
- Smooth sine wave animation
- After blink, items stay green and transition to measuring stage

### 7. Measuring Distances
- 50 frames per ruler line
- **Gradual animations**:
  - Circle starts at regular point size, expands to full size over 30 frames
  - Text fades in over 40 frames (instead of popping out)
- + 2 seconds hold at end
- **Layout** (right panel):
  - Image at top (no border)
  - Section label: 28pt Medium, lime green
  - **Distance prominently first**: 32pt Medium, lime green
  - Title: 24pt Medium
  - Artist: 20pt Medium
  - Year: 16pt Thin
  - Divider
  - Two-column details (ID, Size, Handling Status, Raw Embedding / Delivery Date, Weight)

---

## Info Panel Layout

**Representative Cycling & Ruler Steps**:
1. Image (200px height, no border)
2. Divider
3. Section label (lime green)
4. Distance (32pt Medium, lime green) - **Ruler step only**
5. Title (24pt Medium)
6. Artist (20pt Medium)
7. Year (16pt Thin)
8. Divider
9. Two-column fields

**Top 10 Table**:
- Section titles: 32pt Medium, lime green
- Thumbnails: 60px × 60px
- Font: 16pt Thin (monofont for content)
- Selected items: Same color as others (no green highlighting during display)
- Blink effect: Selected items blink gray→green for 2 seconds after all items shown

---

## Data

**Files**:
- CSV: `artworks_with_thumbnails_final.csv`
- Embeddings: `embeddings_cache/all_embeddings.pkl`
- Images: `production-export-2025-11-14t12-58-07-436z/images/`

**Curation**:
```python
REPRESENTATIVES = {0: 464, 1: 152, 2: 161, 3: 119, 4: 454, 5: 360, 6: 468, 7: 107, 8: 385, 9: 185}
OUTLIERS = {0: 479, 1: 386, 2: 326, 3: 82, 4: 424, 5: 310, 6: 93, 7: 96, 8: 343, 9: 441}
```

---

## Technical

**Video**: 60fps, H.264, CRF 18 (high quality)

**Output Locations**:
- **Default**: `video_making/test/frames/shelf{regal}_both/` (frames) and `shelf{regal}_both.mp4` (video)
- **White Background**: `video_making/test/frames/shelf{regal}_both_white/` (frames) and `shelf{regal}_both_white.mp4` (video)
- **External Drive** (if mounted): `/Volumes/NO NAME/storageMuseum/frames/shelf{regal}_both/` (falls back to default if not found)

**Rendering**:
- **Full HD** (default): Renders at 2x (3840×2160) and downscales to 1920×1080 for anti-aliasing
- **4K**: Renders directly at 3840×2160 (no downscaling)
- Use `--scale 2.0` for Full HD, `--scale 4.0` for 4K

**UMAP Parameters**:
- `min_dist=0.1` (slightly increased for better spacing)
- `spread=18` (increased for more global structure)
- `n_neighbors=18`
- `metric='cosine'`
- `random_state=21`

**Usage**:
```bash
# Full HD (default, black background)
python visualize_shelf0_representative.py --shelf 0 --mode both

# Full HD with white background
python visualize_shelf0_representative.py --shelf 0 --mode both --white-background

# 4K output
python visualize_shelf0_representative.py --shelf 0 --mode both --scale 4.0

# To run for all shelves (0-9), running two processes at a time:
for shelf in {0..9}; do
    python visualize_shelf0_representative.py --shelf $shelf --mode both &
    if (( (shelf + 1) % 2 == 0 )); then
        wait
    fi
done
wait
