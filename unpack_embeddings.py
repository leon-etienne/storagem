#!/usr/bin/env python3
"""
Simple script to inspect the embeddings cache file.
"""

import pickle
import numpy as np
from pathlib import Path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect embeddings cache file")
    parser.add_argument("--cache-file", default="embeddings_cache/all_embeddings.pkl", 
                       help="Path to embeddings cache file")
    parser.add_argument("--artwork-id", type=str, help="Show embedding for specific artwork ID")
    args = parser.parse_args()
    
    # Load cache
    cache_path = Path(args.cache_file)
    if not cache_path.exists():
        print(f"Error: Cache file not found: {args.cache_file}")
        return
    
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)
    
    embeddings = cache_data.get("embeddings", {})
    metadata = cache_data.get("metadata", {})
    
    # Print basic info
    print(f"Total embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {cache_data.get('embedding_dim', 512)}")
    
    if metadata:
        print(f"Metadata: {metadata}")
    
    # Show sample IDs
    if embeddings:
        sample_ids = list(embeddings.keys())[:5]
        print(f"\nSample artwork IDs: {sample_ids}")
        
        sample_emb = next(iter(embeddings.values()))
        print(f"Embedding shape: {sample_emb.shape}, dtype: {sample_emb.dtype}")
        print(f"Embedding norm: {np.linalg.norm(sample_emb):.6f}")
    
    # Show specific artwork if requested
    if args.artwork_id:
        artwork_id = args.artwork_id
        emb = None
        
        # Try different ID formats
        if artwork_id in embeddings:
            emb = embeddings[artwork_id]
        else:
            try:
                int_id = int(float(artwork_id))
                if int_id in embeddings:
                    emb = embeddings[int_id]
            except (ValueError, TypeError):
                pass
        
        if emb is not None:
            print(f"\nEmbedding for artwork ID '{artwork_id}':")
            print(f"  Shape: {emb.shape}")
            print(f"  First 10 values: {emb[:10]}")
            print(f"  Norm: {np.linalg.norm(emb):.6f}")
        else:
            print(f"\nArtwork ID '{artwork_id}' not found in cache")


if __name__ == "__main__":
    main()

