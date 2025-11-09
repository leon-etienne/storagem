# Storage Museum - Artwork Analysis and Search

This repository contains tools for processing, analyzing, and searching artwork collections using CLIP embeddings and FAISS indexing.

## Overview

The codebase processes artwork data from Sanity CMS exports, creates embeddings using CLIP (Contrastive Language-Image Pre-training), and provides functionality to:
- Process raw NDJSON data into structured CSV files
- Find representative and outlier artworks by shelf/cluster
- Build searchable FAISS indices
- Generate HTML reports with embedded images

## Installation

### Requirements

```bash
pip install torch transformers faiss-cpu pandas pillow tqdm
```

### Dependencies

- **torch**: PyTorch for deep learning models
- **transformers**: Hugging Face transformers library for CLIP models
- **faiss-cpu**: Facebook AI Similarity Search for efficient similarity search
- **pandas**: Data manipulation and analysis
- **pillow**: Image processing
- **tqdm**: Progress bars

## Project Structure

```
storagem/
├── read_data.py                          # Process NDJSON to CSV
├── find_cluster_representatives.py       # Main analysis script
├── build_faiss_index.py                 # Build FAISS index (E5 model)
├── build_faiss_index_clip.py            # Build FAISS index (CLIP model)
├── artworks_with_thumbnails.csv          # Processed CSV output
├── production-export-2025-11-04t14-27-00-000z/  # Source data directory
│   ├── data.ndjson                       # Raw artwork data
│   └── images/                          # Artwork images
└── artifacts/                           # Generated indices and metadata
```

## Main Scripts

### 1. `read_data.py` - Data Processing

Processes raw NDJSON data from Sanity CMS exports and creates a structured CSV file with artwork metadata.

**Usage:**
```bash
python read_data.py
```

**What it does:**
- Reads NDJSON data from `production-export-2025-11-04t14-27-00-000z/data.ndjson`
- Flattens nested JSON structures
- Resolves artist references
- Generates thumbnail paths (relative paths to images)
- Assigns clusters based on shelf numbers
- Outputs `artworks_with_thumbnails.csv` (or `artworks_with_thumbnails_ting.csv`)

**Key Features:**
- Handles multiple shelf numbers (splits by comma or semicolon)
- Verifies image file existence
- Provides statistics on artworks (total, with artist names, with images, in stock, etc.)
- Assigns clusters based on shelf number mapping:
  - Cluster 1: Shelves 5, 9, 1
  - Cluster 2: Shelf 3
  - Cluster 3: Shelf 4
  - Cluster 4: Shelves 2, 6, 7
  - Cluster 5: Shelves 8, 0

### 2. `find_cluster_representatives.py` - Artwork Analysis

Main script for finding representative and outlier artworks using CLIP embeddings. This script contains multiple functions for different analysis tasks.

**Usage:**
```bash
# Find outliers (furthest from centroid)
python find_cluster_representatives.py --mode outliers --output shelf_outliers_print.csv

# Find representatives (closest to centroid)
python find_cluster_representatives.py --mode representatives --output shelf_representatives_print.csv
```

**Command-line Arguments:**
- `--csv`: Path to CSV file (default: `artworks_with_thumbnails_ting.csv`)
- `--base-dir`: Base directory for images (default: `production-export-2025-11-04t14-27-00-000z`)
- `--output`: Output CSV file (default: `shelf_outliers_print.csv`)
- `--mode`: Analysis mode - `outliers` or `representatives` (default: `outliers`)

**Note:** The default CSV filename is `artworks_with_thumbnails_ting.csv`. If you're using `artworks_with_thumbnails.csv`, specify it with `--csv`.

**Output:**
- CSV file with base64-encoded images
- HTML file for better visualization (one page per shelf)

## Functions in `find_cluster_representatives.py`

### Core Embedding Functions

#### `embed_text(text: str, normalize: bool = True) -> np.ndarray`
Embeds a single text string using CLIP's text encoder.

**Parameters:**
- `text`: Text string to embed
- `normalize`: Whether to normalize the embedding vector (default: True)

**Returns:**
- Normalized 512-dimensional embedding vector

#### `embed_image(image: Image.Image, normalize: bool = True) -> np.ndarray`
Embeds a single image using CLIP's image encoder.

**Parameters:**
- `image`: PIL Image object
- `normalize`: Whether to normalize the embedding vector (default: True)

**Returns:**
- Normalized 512-dimensional embedding vector

#### `embed_artwork(row: pd.Series, base_dir: str = "production-export-2025-11-04t14-27-00-000z") -> np.ndarray`
Creates a combined embedding for an artwork by combining text and image embeddings.

**Parameters:**
- `row`: Pandas Series containing artwork data
- `base_dir`: Base directory for image paths

**Embedding Strategy:**
1. **Text Fields**: Combines artist, title, year, description, size, handling_status, and internalNote
2. **Title Weighting**: Title is embedded separately and given 40% weight (other text fields get 60%)
3. **Image Weighting**: If image is available, final embedding is 60% text + 40% image. Otherwise, uses 100% text
4. **Normalization**: Final embedding is normalized to unit length

**Text Fields Used:**
- `artist`: Artist name
- `title`: Artwork title (weighted more heavily)
- `year`: Year of creation
- `description`: Artwork description
- `size`: Physical dimensions
- `handling_status`: Handling instructions
- `internalNote`: Internal notes

**Returns:**
- Normalized 512-dimensional combined embedding vector

### Analysis Functions

#### `find_shelf_outliers(csv_path: str, base_dir: str = "production-export-2025-11-04t14-27-00-000z", batch_size: int = 32) -> pd.DataFrame`
Finds the top 10 outlier artworks for each shelf (furthest from the shelf's centroid).

**Parameters:**
- `csv_path`: Path to CSV file with artwork data
- `base_dir`: Base directory for images
- `batch_size`: Batch size for processing (currently not used, kept for compatibility)

**Process:**
1. Loads CSV and filters artworks with shelf information
2. Handles multiple shelf numbers (splits and creates separate rows)
3. Creates embeddings for all artworks
4. For each shelf:
   - Calculates centroid (average embedding of all artworks on that shelf)
   - Computes distances from each artwork to the centroid
   - Selects top 10 artworks with largest distances (furthest from centroid)

**Returns:**
- DataFrame with columns:
  - `shelf_no`: Shelf number
  - `rank`: Rank (1-10, where 1 is furthest from centroid)
  - `index`: Artwork ID from CSV
  - `title`: Artwork title
  - `artist`: Artist name
  - `cluster`: Cluster assignment
  - `num_artworks_in_shelf`: Total number of artworks on this shelf
  - `distance_to_centroid`: Euclidean distance to shelf centroid
  - `thumbnail`: Base64-encoded image (for CSV embedding)

#### `find_shelf_representatives(csv_path: str, base_dir: str = "production-export-2025-11-04t14-27-00-000z", batch_size: int = 32) -> pd.DataFrame`
Finds the top 10 representative artworks for each shelf (closest to the shelf's centroid).

**Parameters:**
- Same as `find_shelf_outliers`

**Process:**
- Same as `find_shelf_outliers`, but selects top 10 artworks with **smallest** distances (closest to centroid)

**Returns:**
- Same DataFrame structure as `find_shelf_outliers`, but with artworks closest to centroid

**Use Cases:**
- **Outliers**: Find artworks that are unusual or different from the typical shelf content
- **Representatives**: Find artworks that best represent the typical content of a shelf

### Utility Functions

#### `load_image(image_path: str, base_dir: str = "production-export-2025-11-04t14-27-00-000z") -> Optional[Image.Image]`
Loads an image from a thumbnail path, handling relative paths from CSV.

**Parameters:**
- `image_path`: Path to image (can be relative or absolute)
- `base_dir`: Base directory for resolving relative paths

**Returns:**
- PIL Image object or None if image not found

#### `image_to_base64(image: Image.Image, max_size: int = 300) -> str`
Converts a PIL Image to a base64-encoded string for embedding in CSV/HTML.

**Parameters:**
- `image`: PIL Image object
- `max_size`: Maximum dimension for resizing (keeps aspect ratio)

**Returns:**
- Base64-encoded data URL string (format: `data:image/jpeg;base64,...`)

#### `create_html_view(results_df: pd.DataFrame, output_file: str, mode: str = "outlier")`
Creates an HTML file with embedded images for better visualization.

**Parameters:**
- `results_df`: DataFrame with results (from `find_shelf_outliers` or `find_shelf_representatives`)
- `output_file`: Path to output HTML file
- `mode`: Mode string ("outlier" or "representative") for title customization

**Features:**
- One page per shelf (with CSS page breaks)
- Embedded images displayed inline
- Print-friendly styling
- Responsive table layout

### Main Function

#### `main()`
Command-line interface for running the analysis.

**Example Usage:**
```bash
# Find outliers
python find_cluster_representatives.py --mode outliers --csv artworks_with_thumbnails.csv --output outliers.csv

# Find representatives
python find_cluster_representatives.py --mode representatives --csv artworks_with_thumbnails.csv --output representatives.csv
```

## Workflow Example

### Step 1: Process Raw Data
```bash
python read_data.py
```
This creates `artworks_with_thumbnails.csv` with all artwork metadata and thumbnail paths.

### Step 2: Find Shelf Outliers
```bash
python find_cluster_representatives.py \
    --mode outliers \
    --csv artworks_with_thumbnails.csv \
    --output shelf_outliers_print.csv
```
This generates:
- `shelf_outliers_print.csv`: CSV with top 10 outliers per shelf
- `shelf_outliers_print.html`: HTML visualization

### Step 3: Find Shelf Representatives
```bash
python find_cluster_representatives.py \
    --mode representatives \
    --csv artworks_with_thumbnails.csv \
    --output shelf_representatives_print.csv
```
This generates:
- `shelf_representatives_print.csv`: CSV with top 10 representatives per shelf
- `shelf_representatives_print.html`: HTML visualization

## Understanding the Results

### Distance to Centroid

The `distance_to_centroid` column shows the Euclidean distance from an artwork's embedding to the average embedding (centroid) of all artworks on that shelf.

- **For Outliers**: Higher values = more unusual/different from shelf average
- **For Representatives**: Lower values = more typical/representative of shelf average

### Rank

The `rank` column indicates the position within the top 10:
- Rank 1 = most extreme (furthest for outliers, closest for representatives)
- Rank 10 = least extreme within the top 10

## Technical Details

### CLIP Model
- **Model**: `openai/clip-vit-base-patch32`
- **Embedding Dimension**: 512
- **Text Encoder**: Transformer-based
- **Image Encoder**: Vision Transformer (ViT)

### Embedding Combination
The final artwork embedding combines:
- **Text (60%)**: Weighted combination of all text fields
  - Title: 40% of text weight
  - Other fields: 60% of text weight
- **Image (40%)**: If available, otherwise 0%

All embeddings are normalized to unit length for cosine similarity calculations.

### Shelf Number Handling
- Multiple shelf numbers are split by comma or semicolon
- Each artwork with multiple shelf numbers is duplicated (one row per shelf)
- This allows analysis per individual shelf

## Troubleshooting

### Missing Images
If thumbnails are not found:
- Check that `production-export-2025-11-04t14-27-00-000z/images/` directory exists
- Verify image paths in CSV are correct (relative paths)
- Run `read_data.py` again to regenerate paths

### Memory Issues
If you run out of memory:
- Process smaller batches by filtering CSV first
- Use CPU instead of GPU (slower but less memory)
- Reduce image size in `image_to_base64` function

### CUDA/GPU Issues
The script automatically falls back to CPU if CUDA is not available. To force CPU:
```python
device = "cpu"  # In the script
```

## Additional Scripts

### `build_faiss_index.py`
Builds a FAISS index using the E5 text embedding model for text-based search.

### `build_faiss_index_clip.py`
Builds a FAISS index using CLIP for multimodal (text + image) search.

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

