# Visual Identity

## Design Principles

- **Clean & Minimal**: Black & white with lime green accent
- **Modern Typography**: Focus on spacing and hierarchy
- **No Emoji**: Keep it professional

## Typography

**Font**: Neue Haas Unica W1G
- Regular, Light, Medium, Thin weights

**Sizes**:
- Title: 36pt Medium
- Subtitle: 28pt Thin
- Section Titles: 32pt Medium (lime green)
- Artwork Title: 24pt Medium
- Artist: 20pt Medium
- Info Text: 18pt Thin
- Small Text: 16pt Thin

## Colors

- **Lime Green**: RGB(0, 255, 0) - Accent only
- **Black**: RGB(0, 0, 0) - Background
- **White**: RGB(255, 255, 255) - Text
- **Grey**: RGB(90, 90, 90) - Dots and lines

## Layout

**Canvas**: 1920×1080px (Full HD) or 3840×2160px (4K)
- Map: 1200px left (Full HD) / 2400px (4K)
- Panel: 720px right (Full HD) / 1440px (4K)

**Progress Bar**: Top right, extends from map to panel border

## Visual Elements

- **Points**: 4px grey (basic), 8px lime (Regal), 40px thumbnails (representatives)
- **Lines**: 1px grey (connections), 3px lime (measurements)
- **Centroid**: Crosshair (12px circle, 20px lines)

## Stages

1. All Embeddings
2. Identify Regal Items
3. Highlight with Centroid
4. Cycle Through Artworks
5. Draw Lines from Centroid
6. Top 10 Items
7. Measuring Distances

## Usage

```bash
# Full HD
python visualize_shelf0_representative.py --shelf 0 --mode both

# 4K
python visualize_shelf0_representative.py --shelf 0 --scale 4.0

# White background
python visualize_shelf0_representative.py --shelf 0 --white-background

# Generate all shelves (0-9) in 4K, white background only for group 2 (Cluster 2: shelves 3, 4)
for shelf in {0..9}; do
    if [[ "$shelf" == "3" ]] || [[ "$shelf" == "4" ]]; then
        python visualize_shelf0_representative.py --shelf $shelf --mode both --scale 4.0 --white-background
    else
        python visualize_shelf0_representative.py --shelf $shelf --mode both --scale 4.0
    fi
done
```
