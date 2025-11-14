#!/usr/bin/env python3
"""
Simple script to swap embeddings for IDs 385 and 389 in all_embeddings.pkl
"""

import pickle
from pathlib import Path


def main():
    # Load the original embeddings file
    input_file = Path("embeddings_cache/all_embeddings.pkl")
    output_file = Path("embeddings_cache/final_embeddings.pkl")
    
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        return
    
    print(f"Loading {input_file}...")
    with open(input_file, "rb") as f:
        cache_data = pickle.load(f)
    
    embeddings = cache_data.get("embeddings", {})
    
    # Try to find IDs 385 and 389 (they might be strings or integers)
    id_385 = None
    id_389 = None
    
    # Check both string and int formats
    for key in embeddings.keys():
        if str(key) == "385":
            id_385 = key
        if str(key) == "389":
            id_389 = key
    
    if id_385 is None:
        print("Error: ID 385 not found in embeddings")
        return
    
    if id_389 is None:
        print("Error: ID 389 not found in embeddings")
        return
    
    print(f"Found ID 385: {id_385}")
    print(f"Found ID 389: {id_389}")
    
    # Swap the embeddings
    temp = embeddings[id_385].copy()
    embeddings[id_385] = embeddings[id_389].copy()
    embeddings[id_389] = temp
    
    print("Swapped embeddings for IDs 385 and 389")
    
    # Update num_embeddings if it exists
    if "num_embeddings" in cache_data:
        cache_data["num_embeddings"] = len(embeddings)
    
    # Save to new file
    print(f"Saving to {output_file}...")
    with open(output_file, "wb") as f:
        pickle.dump(cache_data, f)
    
    print("Done!")


if __name__ == "__main__":
    main()

