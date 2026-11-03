# src/models/calibrate_forensic_thresholds.py
"""
VoiceGuard: Forensic Threshold Calibration Engine
Extracts empirical distributions of physical acoustic features from REAL banking speech.

Why this matters:
  A bank's Fraud Operations unit requires Explainable AI (XAI) for SAR (Suspicious Activity Report)
  filings under regulatory compliance (e.g. FinCEN, OCC, RBI).
  Instead of hardcoded arbitrary thresholds, this script calculates the true 5th and 95th
  percentiles of genuine human voice across 6 critical vocal-tract indicators.
"""
import numpy as np
import pandas as pd


def calibrate():
    print("=" * 70)
    print("VOICEGUARD FORENSIC THRESHOLD CALIBRATION")
    print("=" * 70)

    # 1. Load manifest and DSP cache
    manifest_path = "data/voice_auth_manifest.csv"
    cache_path = "data/dsp_feature_cache.npz"

    df = pd.read_csv(manifest_path)
    cache = np.load(cache_path, allow_pickle=True)
    cache_paths = list(cache["paths"])
    cache_features = cache["features"]

    path_to_idx = {str(p): i for i, p in enumerate(cache_paths)}

    # 2. Extract Training Set Real vs Fake Features
    train_df = df[df["split"] == "train"]
    real_df = train_df[train_df["label"] == "real"]
    fake_df = train_df[train_df["label"] == "fake"]

    real_feats = np.array([
        cache_features[path_to_idx[str(row["audio_path"])]]
        for _, row in real_df.iterrows()
        if str(row["audio_path"]) in path_to_idx
    ])
    fake_feats = np.array([
        cache_features[path_to_idx[str(row["audio_path"])]]
        for _, row in fake_df.iterrows()
        if str(row["audio_path"]) in path_to_idx
    ])

    print(f"Training Samples Analyzed: {len(real_feats):,} Real | {len(fake_feats):,} Fake")

    # 3. 45-dim DSP Layout indices
    forensic_features = {
        39: "Pitch (Hz)",
        40: "Jitter (Relative)",
        41: "Shimmer (Relative)",
        42: "HNR (Harmonics/Noise dB)",
        43: "Spectral Centroid",
        44: "Spectral Flatness",
    }

    header = (
        f"{'Feature Name':<26} "
        f"{'Real P5':>10} {'Real P50':>10} {'Real P95':>10} │ "
        f"{'Fake P50':>10} │ {'Δ Median':>10}"
    )
    print(f"\n{header}")
    print("─" * len(header))

    ranges = {}
    for idx, name in forensic_features.items():
        r = real_feats[:, idx]
        f = fake_feats[:, idx]
        r_p5, r_p50, r_p95 = np.percentile(r, [5, 50, 95])
        f_p50 = np.percentile(f, 50)
        delta = abs(r_p50 - f_p50)
        ranges[name] = [round(float(r_p5), 4), round(float(r_p95), 4)]
        print(
            f"{name:<26} "
            f"{r_p5:>10.4f} {r_p50:>10.4f} {r_p95:>10.4f} │ "
            f"{f_p50:>10.4f} │ {delta:>10.4f}"
        )

    print("─" * len(header))
    print("\n" + "=" * 70)
    print("CALIBRATED FORENSIC_RANGES (Paste into voice_authenticator.py):")
    print("=" * 70)
    print("FORENSIC_RANGES = {")
    for name, bounds in ranges.items():
        print(f'    "{name}": {bounds},')
    print("}")
    print("=" * 70)


if __name__ == "__main__":
    calibrate()