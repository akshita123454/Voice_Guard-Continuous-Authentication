# src/data_prep/cache_new_xtts_dsp.py
import os
import glob
import numpy as np
from tqdm import tqdm
from src.features.acoustic_extractor import extract_dsp_features

def append_to_dsp_cache():
    cache_path = "data/dsp_feature_cache.npz"
    xtts_files = sorted(glob.glob("data/synthetic_voice/diverse_clones/*.wav"))
    print(f"Found {len(xtts_files)} XTTS files to extract.")

    # Load existing cache
    print("Loading existing DSP cache...")
    old_cache = np.load(cache_path, allow_pickle=True)
    existing_paths = set(old_cache["paths"])
    print(f"Existing cache has {len(existing_paths):,} entries.")

    new_files = [f for f in xtts_files if f not in existing_paths]
    print(f"Files needing extraction: {len(new_files):,}")

    if not new_files:
        print("✅ All XTTS files already in cache!")
        return

    new_paths = []
    new_features = []

    for path in tqdm(new_files, desc="Extracting 45-dim DSP for XTTS"):
        try:
            feats = extract_dsp_features(path)
            new_paths.append(path)
            new_features.append(feats)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    # Merge
    combined_paths = np.array(list(old_cache["paths"]) + new_paths)
    combined_features = np.vstack([old_cache["features"], np.array(new_features, dtype=np.float32)])

    # Backup old cache & save new
    os.rename(cache_path, "data/dsp_feature_cache.npz.bak")
    np.savez_compressed(cache_path, paths=combined_paths, features=combined_features)
    print(f"✅ Cache updated: {len(combined_paths):,} total files saved to {cache_path}")

if __name__ == "__main__":
    append_to_dsp_cache()