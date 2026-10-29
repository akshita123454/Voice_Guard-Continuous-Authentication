# src/data_prep/cache_dsp_features.py
"""
Pre-extracts all 45 DSP acoustic features (MFCCs, Pitch, Jitter, Shimmer, HNR)
for every audio file in the manifest and saves them as a single .npz cache file.

This runs ONCE (~20 minutes on CPU). After this, the training loop loads
features instantly from the cache instead of recomputing librosa on every epoch.
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.features.acoustic_extractor import AcousticFeatureExtractor

MANIFEST_PATH = "data/voice_auth_manifest.csv"
CACHE_PATH = "data/dsp_feature_cache.npz"

def main():
    if os.path.exists(CACHE_PATH):
        print(f"Cache already exists at {CACHE_PATH}")
        print("Delete it manually if you want to re-extract.")
        data = np.load(CACHE_PATH, allow_pickle=True)
        print(f"Cached entries: {len(data['features'])}")
        return

    df = pd.read_csv(MANIFEST_PATH)
    extractor = AcousticFeatureExtractor(sample_rate=16000)

    all_features = []
    all_paths = []
    failed_files = []

    print(f"Extracting DSP features for {len(df)} files...")
    print("This will take approximately 20-30 minutes.\n")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        audio_path = row["audio_path"]
        try:
            feat = extractor.extract_features(audio_path)
            all_features.append(feat)
            all_paths.append(audio_path)
        except Exception as e:
            # Store zeros for failed files (will be filtered during training)
            all_features.append(np.zeros(45, dtype=np.float32))
            all_paths.append(audio_path)
            failed_files.append((audio_path, str(e)))

    features_array = np.array(all_features, dtype=np.float32)

    np.savez_compressed(
        CACHE_PATH,
        features=features_array,
        paths=np.array(all_paths)
    )

    print(f"\n{'=' * 60}")
    print(f"DSP Feature Cache Saved: {CACHE_PATH}")
    print(f"{'=' * 60}")
    print(f"Total files processed: {len(all_features)}")
    print(f"Feature shape per file: {all_features[0].shape} (45 dimensions)")
    print(f"Cache file size: {os.path.getsize(CACHE_PATH) / 1024 / 1024:.1f} MB")
    print(f"Failed extractions: {len(failed_files)}")

    if failed_files:
        print(f"\nFirst 5 failures:")
        for path, err in failed_files[:5]:
            print(f"  {path}: {err}")

if __name__ == "__main__":
    main()