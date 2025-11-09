# Storage Museum - Artwork Analysis

Tools for finding representative and outlier artworks using CLIP embeddings.

## Installation

```bash
pip install torch transformers faiss-cpu pandas pillow tqdm
```

## Usage

### Step 1: Generate Embeddings

Create embeddings for all artworks (run once):

```bash
python generate_embeddings.py --csv artworks_with_thumbnails_ting.csv
```

This saves embeddings to `embeddings_cache/all_embeddings.pkl`.

### Step 2: Find Representatives or Outliers

**Find representatives** (most typical artworks per shelf):

```bash
python find_representatives.py --output shelf_representatives.csv
```

**Find outliers** (most unusual artworks per shelf):

```bash
python find_representatives.py --outliers --output shelf_outliers.csv
```

**With HTML report** (includes embedded images):

```bash
python find_representatives.py --print-mode --output shelf_representatives_print.csv
python find_representatives.py --outliers --print-mode --output shelf_outliers_print.csv
```

## Options

- `--csv` - Input CSV file (default: `artworks_with_thumbnails_ting.csv`)
- `--output` - Output CSV file name
- `--outliers` - Find outliers instead of representatives
- `--top-k` - Number of results per shelf (default: 5, automatically 15 in print mode)
- `--print-mode` - Generate HTML report with embedded images

## Other Scripts

**Inspect embeddings cache:**
```bash
python unpack_embeddings.py
python unpack_embeddings.py --artwork-id 123
```

**Process raw data** (if needed):
```bash
python read_data.py
```
