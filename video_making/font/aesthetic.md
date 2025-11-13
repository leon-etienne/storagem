# Visual Identity & Design System

## Core Design Principles

- **Clean & Minimal**: Black & white with lime green accent
- **No Emoji**: Keep it professional
- **No Data Visualization Aesthetic**: Avoid typical computer science chart styling
- **Modern Designer Aesthetic**: Focus on typography, spacing, and deliberate visual hierarchy

---

## Typography

### Font Family
- **Primary Font**: Neue Haas Unica W1G
- **Font Files Location**: `video_making/font/`
  - `Neue Haas Unica W1G Regular.ttf`
  - `Neue Haas Unica W1G Light.ttf`
  - `Neue Haas Unica W1G Medium.ttf`
  - `Neue Haas Unica W1G Thin.ttf`

### Font Usage
- **Titles**: Medium weight, 48pt
- **Labels/Subtitle**: Thin weight, 28pt (slightly bigger)
- **Info Text**: Thin weight, 18pt
- **Small Text**: Thin weight, 14pt
- **Map Text**: Thin weight, 18pt
- **Table Text**: Thin weight, 16pt (for top10 and side-by-side views)

---

## Color Palette

### Primary Colors
- **Lime Green**: RGB(0, 255, 0) - Primary accent color
- **Black**: RGB(0, 0, 0) - Background
- **White**: RGB(255, 255, 255) - Primary text

### Secondary Colors
- **Light Gray**: RGB(200, 200, 200) - Secondary text on black
- **Medium Gray**: RGB(150, 150, 150) - Tertiary text
- **Point Gray**: RGB(140, 140, 140) - Regular points
- **Low Opacity Gray**: RGB(120, 120, 120) - Non-Regal points
- **Circle Gray**: RGB(100, 100, 100) - Selection circles
- **Connection Gray**: RGB(100, 100, 100) - Connection lines

### Color Usage Rules
- Keep color usage minimal
- Use lime green only for:
  - Regal item highlights
  - Representative/outlier markers
  - Ruler lines and measurements
  - Selected items in tables
  - Section labels ("Representative", "Outlier")

---

## Layout & Composition

### Canvas Dimensions
- **Width**: 1920px
- **Height**: 1080px
- **Map Area**: 1200px width (left side)
- **Panel Area**: 720px width (right side)

### Text Positioning
- **Title**: Left side bottom (50px from left, 100px from bottom) - higher up
- **Subtitle/Stage Name**: Top left (50px, 30px), 28pt font
- **Centroid Position**: Appended to subtitle in highlight steps
- **Distance & Search Mode**: Shown in subtitle during representative cycling (e.g., "Finding Representatives | Distance: 0.1234")
- **Panel Content**: Starts at 60px from top (higher position)
- **Section Titles**: 28pt font (smaller), positioned at 60px from top

---

## Visual Elements

### Points & Circles

#### Regular Points (Non-Regal)
- **Size**: 4px diameter
- **Color**: Gray (low opacity)
- **Style**: Fill only, no outline

#### Regal Points
- **Size**: 8px diameter (larger)
- **Highlighted**: Bright lime green fill
- **Non-highlighted**: Lower opacity lime green fill
- **No outline**

#### Representative Points (Top 10)
- **Size**: 40px diameter
- **Style**: Circular thumbnail images
- **Aesthetic Selection**: Bright green circle outline (3px width)

#### Centroid Marker
- **Style**: Crosshair design
- **Size**: 12px center circle, 20px line length
- **Color**: White
- **Width**: 2px

### Lines & Connections

#### Connection Lines (Between Regal Items)
- **Color**: Gray
- **Width**: 1px
- **Style**: Subtle, less obvious
- **Position**: Always behind points
- **Visibility**: Only shown in highlight/centroid/distances steps
- **Fade Out**: After highlight stage, lines fade out over 30 frames (opacity 1.0 to 0.0)
- **Removed**: No grey lines after finding centroid (not shown in representative, ruler, top10 steps)

#### Lines from Centroid
- **Color**: Gray (in highlight/centroid/distances steps) or Green (in representative step)
- **Width**: 1px
- **Style**: Connect centroid to Regal items
- **Position**: Always behind points
- **In Representative Step**: Only green lines shown (from lines_to_draw), no grey lines
- **Visibility**: Grey lines only in highlight/centroid/distances steps

#### Top 10 Connection Lines
- **Color**: Bright lime green
- **Width**: 1px
- **Style**: Connect all pairs of top 10 items simultaneously
- **Position**: Always behind points

#### Ruler Lines (Distance Measurement)
- **Main Line**: 
  - Color: Bright lime green
  - Width: 3px
  - Style: Prominent measurement line
- **Tick Marks**:
  - Total: 20 ticks
  - Major ticks: Every 5th tick, 12px length, 2px width
  - Minor ticks: 6px length, 1px width
  - Color: Lime green
- **Arrowhead**:
  - Appears when line is 80%+ complete
  - Size: 12px
  - Style: Filled triangle pointing to target
  - Color: Lime green
- **Distance Label**:
  - Font: Medium weight, 24pt
  - Background: Black rectangle with green border (2px)
  - Padding: 8px
  - Offset: 35px perpendicular to line
  - Color: Lime green text

---

## Data Sources

### Files
- **CSV Data**: `artworks_with_thumbnails_ting.csv`
- **Embeddings**: `embeddings_cache/all_embeddings.pkl`
- **Production Export**: `production-export-2025-11-13t13-42-48-005z/`
- **Fallback Image**: `video_making/font/no_image.png`

### Data Fields
- **Artwork ID**: Use `id` column from CSV (numeric, e.g., 464, 152, 161)
- **Thumbnail Path**: Relative path in CSV, fallback to `production-export-*/images/`

---

## Curation Data

### Regal Groups (Clustering)
```python
GROUPS = {
    1: [1, 5, 9],
    2: [3, 4],
    3: [2, 6, 7],
    4: [0, 8]
}
```

### Representatives (by Regal)
```python
REPRESENTATIVES = {
    0: 464, 1: 152, 2: 161, 3: 376, 4: 454,
    5: 360, 6: 468, 7: 107, 8: 389, 9: 185
}
```

| Regal | Artwork ID |
|-------|------------|
| 0     | 464        |
| 1     | 152        |
| 2     | 161        |
| 3     | 376        |
| 4     | 454        |
| 5     | 360        |
| 6     | 468        |
| 7     | 107        |
| 8     | 389        |
| 9     | 185        |

### Outliers (by Regal)
```python
OUTLIERS = {
    0: 479, 1: 386, 2: 326, 3: 82, 4: 424,
    5: 310, 6: 93, 7: 96, 8: 343, 9: 441
}
```

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

---

## Visualization Stages

### Stage 1: All Embeddings
- **Duration**: 60 frames (2 seconds at 30fps)
- **Visual**: All points shown as gray
- **Title**: "All Embeddings" (subtitle)
- **Regal Title**: "Regal X" (bottom, centered)

### Stage 2: Identify Regal Items
- **Duration**: 30 frames per item + hold
- **Visual**: Items appear one by one with color easing
- **Animation**: Expanding gray circle, then green point
- **Text**: Title text appears one by one (only for items that have been shown)
- **No lines or centroid shown**
- **Title**: "Identify Regal items" (subtitle)

### Stage 3: Highlight with Centroid & Distances
- **Duration**: 10 frames per item + hold
- **Visual**: Items added progressively with lines and centroid
- **Animation**: Expanding gray circle effect (15 frames) before green point appears
- **Centroid**: Updates dynamically as items are added
- **Distances**: Calculated and displayed for each item
- **Title**: "Highlighting Regal X | Centroid: (x, y)" (subtitle)
- **Note**: Previously identified items stay green
- **Title Text**: All Regal item titles persist from identify stage

### Stage 3.5: Fade Out Connection Lines
- **Duration**: 30 frames
- **Visual**: Grey connection lines fade out (opacity 1.0 to 0.0)
- **Animation**: Smooth fade by blending with background color
- **Purpose**: Clean transition before cycling through artworks

### Stage 4: Cycle Through Artworks
- **Duration**: 15 frames to draw line + 5 frames hold per artwork (20 total)
- **Visual**: Each Regal artwork displayed with details
- **Line Drawing**: Animated green line from centroid to each artwork point (15 frames)
- **No Grey Lines**: Only green lines shown during this stage
- **Sorting**: By distance to centroid (closest first)
- **Panel**: Shows artwork info on right side (starts at 60px, uses 16pt font)
- **Subtitle**: "Finding Representatives | Distance: X.XXXX" (updates in realtime)
- **Search Mode**: Subtitle indicates whether finding representatives or outliers

### Stage 5: Draw Lines from Centroid
- **Duration**: 30 frames per line + pause after closest
- **Visual**: Lines drawn one by one from centroid
- **Sorting**: By distance (closest first)
- **Highlight**: Closest item (representative) paused and highlighted

### Stage 6: Top 10 Items
- **Duration**: 20 frames per item
- **Visual**: 
  - First: 5 representatives appear one by one
  - Hold: All representatives shown
  - Then: 5 outliers appear one by one
  - Hold: All items shown with green connection lines
- **Map**: Circular thumbnails (40px) on map positions
- **Table**: Right side shows rank table with thumbnails (60px), 16pt font
- **Section Titles**: "Representatives" and "Outliers" use 28pt font, positioned at 60px from top
- **Highlighting**: Currently appearing item highlighted in green in the table
- **Lines**: Green lines connect all pairs of top 10 items
- **Title**: "Top 10 Representatives" (subtitle)

### Stage 7: Ruler Lines (Measuring Distances)
- **Duration**: 30 frames per ruler line + 120 frames hold (4 seconds)
- **Visual**: 
  - Left Side: Ruler lines drawn to representative and outlier
  - Right Side: Two selected works info (representative and outlier)
  - First: Draw ruler to representative
  - Hold: Show representative ruler
  - Then: Draw ruler to outlier (keeping rep line visible)
  - Hold: Show both rulers with info on right side for 4 seconds
- **Right Side Layout**:
  - Representative section (top): Image (120px max), ID, Title, Artist, Year, Size, Delivery Date, Handling Status, Weight, Raw Embedding, Distance to Centroid
  - Divider line
  - Outlier section (bottom): Same fields as representative
  - Font: 16pt for all content, 14pt for labels
  - Position: Starts at 60px from top
- **Design**: Prominent ruler with tick marks, arrowhead, and distance label
- **Title**: "Measuring Distances" (subtitle)
- **Note**: Comes after Top 10 stage, replaces side-by-side view

### Stage 8: Removed
- **Note**: Side-by-side view has been removed. The measuring distance stage (Stage 7) now shows the two selected works info on the right side.

---

## Layout Specifications

### Right Side Panel (All Stages)

#### Panel Positioning
- **Start Position**: 60px from top (higher up)
- **X Position**: MAP_WIDTH + 30px
- **Font**: 16pt (table font) for all content, 14pt for labels
- **Section Titles**: 28pt font, positioned at 60px from top

#### Information Fields (Representative Cycling & Ruler Steps)
- **ID**: Displayed as integer (converted from float/string)
- **Title**: Wrapped to max 2 lines, 16pt font
- **Artist**: Truncated with "..." if too long, 16pt font
- **Year**: 16pt font
- **Size**: Truncated with "..." if too long, 16pt font
- **Delivery Date**: From `deliveryDate` field, 16pt font
- **Handling Status**: From `handling_status` field, 16pt font
- **Weight**: From `weight` field, 16pt font
- **Raw Embedding**: First 8-10 values shown with "... (N dims)" if truncated, wrapped to max 2-3 lines
- **Distance to Centroid**: Displayed in green, 16pt font

#### Removed Fields
- **Embedding Calculation Section**: Removed (Dimension, Norm, Mean no longer shown)

#### Top 10 Table Layout
- **Section Titles**: "Representatives" and "Outliers" - 28pt font, positioned at 60px from top
- **Thumbnails**: 60px × 60px, positioned at panel_x
- **Column Positions**:
  - Rank: panel_x + 80px
  - Title: panel_x + 140px (max 30 chars)
  - Artist: panel_x + 350px (max 25 chars)
  - Distance: panel_x + 520px
- **Row Height**: 70px
- **Font**: 16pt (table font) for all table content
- **Highlighting**: Currently appearing item highlighted in green
- **Transparency**: Thumbnails preserve RGBA mode if transparent

---

## Calculation Logic

### Selection Process
1. Calculate centroid as mean of all Regal embeddings
2. Calculate distances from centroid to all Regal items
3. Find **5 closest** (representatives)
4. Find **5 farthest** (outliers)
5. **Guarantee aesthetic IDs**: If aesthetic representative/outlier not in top 5, replace 5th item
6. Sort by distance (closest first for reps, farthest first for outliers)

### Distance Calculation
- **Method**: Euclidean distance in embedding space
- **Display**: 4 decimal places
- **Format**: `f"{distance:.4f}"`

---

## Video Generation

### Technical Specifications
- **FPS**: 60 frames per second
- **Codec**: H.264 (libx264)
- **Quality**: CRF 23
- **Pixel Format**: yuv420p
- **Output Format**: MP4

### File Naming
- **Frames Directory**: `frames/shelf{regal_number}_both/`
- **Video Output**: `frames/shelf{regal_number}_both.mp4`
- **Mode Options**: `representative`, `outlier`, or `both` (default: both)

---

## Usage

### Command Line
```bash
# Generate both representative and outlier video for Regal 0 (default)
python visualize_shelf0_representative.py --shelf 0 --mode both

# Generate representative video only
python visualize_shelf0_representative.py --shelf 0 --mode representative

# Generate outlier video only
python visualize_shelf0_representative.py --shelf 0 --mode outlier

# White background mode
python visualize_shelf0_representative.py --shelf 0 --mode both --white-background

# Generate for all Regals
for regal in {0..9}; do
    python visualize_shelf0_representative.py --shelf $regal --mode both
done
```

### Arguments
- `--shelf` / `-s`: Regal number (0-9), default: "0"
- `--mode` / `-m`: Visualization mode ("representative", "outlier", or "both"), default: "both"
- `--white-background` / `-w`: Use white background with inverted colors

---

## Implementation Notes

### Text Overflow Prevention
- All text fields have width constraints
- Titles: Wrapped to max 2 lines
- Artist/Size: Truncated with "..." if too long
- Description: Wrapped to max 3 lines
- Constraint formula: `MAP_WIDTH - info_x - 20`

### Color Preservation
- In Stage 3, previously identified items (from Stage 2) remain green
- Only new items ease in with color animation

### Title Text Persistence
- **In Identify Stage**: Title text appears one by one (only for items that have been shown)
- **In Other Stages**: Title text for all Regal items is drawn for ALL items
- This ensures titles persist from identify stage through highlight stage
- Text is drawn in a separate loop after point drawing

### Line Drawing Order
- Lines are always drawn before points/dots
- This ensures lines appear behind visual elements
- **Grey Lines**: Only shown in highlight/centroid/distances steps, fade out after highlight stage (Stage 3.5)
- **Green Lines**: Shown in representative step (from lines_to_draw), no grey lines in this stage
- Lines from centroid are drawn during representative cycling (Stage 4) in green

### Expanding Circle Animation
- **Identify Stage**: Expanding gray circle (5px to 20px radius) before green point appears
- **Highlight Stage**: Same expanding circle effect (15 frames) before green point appears
- **Easing**: Smooth ease-in-out curve for expansion

### Image Loading
- Primary: Use path from CSV
- Fallback: `production-export-2025-11-13t13-42-48-005z/images/`
- Final fallback: `video_making/font/no_image.png` (transparent PNG - preserves RGBA mode)
- **Transparency Support**: All images maintain RGBA mode if transparent, otherwise convert to RGB

---

## Quality Standards

### Image Quality
- **Resampling**: LANCZOS for all image operations
- **Aspect Ratio**: Always maintained
- **Thumbnail Sizes**: 
  - Map circles: 40px diameter
  - Table thumbnails: 60px × 60px
  - Side-by-side images: Max 300px height, 280px width

### Animation Quality
- **Easing**: Smooth ease-in-out curves
- **Frame Consistency**: All animations use consistent timing
- **Smooth Transitions**: No abrupt changes

### Visual Consistency
- **Spacing**: Consistent padding and margins throughout
- **Alignment**: Text aligned to consistent grid
- **Hierarchy**: Clear visual hierarchy through size and weight
