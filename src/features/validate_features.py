# src/features/validate_features.py
import os
import glob
import numpy as np
import pandas as pd
from src.features.acoustic_extractor import AcousticFeatureExtractor

def compute_group_metrics(file_list, label_name):
    extractor = AcousticFeatureExtractor(sample_rate=16000)
    records = []
    
    print(f"Extracting features from {len(file_list)} {label_name} files...")
    for f in file_list:
        try:
            feat = extractor.extract_features(f)
            # feat index mapping from acoustic_extractor.py:
            # 0-19: MFCC mean
            # 20-39: MFCC std
            # 40: f0_mean, 41: f0_std, 42: jitter, 43: shimmer, 44: hnr
            records.append({
                "f0_mean": feat[40],
                "f0_std": feat[41],
                "jitter": feat[42],
                "shimmer": feat[43],
                "hnr": feat[44],
                "label": label_name
            })
        except Exception as e:
            pass
            
    return pd.DataFrame(records)

def main():
    # 1. Grab 20 real files from LibriSpeech
    real_files = glob.glob("data/librispeech/**/*.flac", recursive=True)[:20]
    
    # 2. Grab 20 normalized synthetic files
    synth_files = glob.glob("data/synthetic_voice_16k/**/*.wav", recursive=True)[:20]
    
    if len(real_files) == 0 or len(synth_files) == 0:
        print("❌ Error: Missing files. Ensure both LibriSpeech and synthetic_voice_16k are present.")
        return

    df_real = compute_group_metrics(real_files, "Real (LibriSpeech)")
    df_synth = compute_group_metrics(synth_files, "Synthetic (TTS)")
    
    df_all = pd.concat([df_real, df_synth], ignore_index=True)
    
    # 3. Print statistical summary
    summary = df_all.groupby("label")[["jitter", "shimmer", "hnr", "f0_mean", "f0_std"]].mean()
    
    print("\n" + "=" * 65)
    print("      ACOUSTIC FEATURE COMPARISON (AVERAGE VALUES)")
    print("=" * 65)
    print(summary.to_string())
    print("=" * 65)
    
    # 4. Physical interpretation check
    diff_jitter = abs(summary.loc["Real (LibriSpeech)", "jitter"] - summary.loc["Synthetic (TTS)", "jitter"])
    diff_hnr = abs(summary.loc["Real (LibriSpeech)", "hnr"] - summary.loc["Synthetic (TTS)", "hnr"])
    
    print("\n[Physical Analysis]")
    print(f"• Jitter Separation Delta:  {diff_jitter:.5f}")
    print(f"• HNR Separation Delta:     {diff_hnr:.2f} dB")
    
    if diff_jitter > 1e-4 or diff_hnr > 1.0:
        print("✅ SUCCESS: Strong physical separation detected between Real and Synthetic voices.")
        print("   The DSP features are capturing meaningful acoustic discrepancies.")
    else:
        print("⚠️ WARNING: Separation is subtle; relying heavily on Stream 2 (Spectrogram CNN) will be essential.")

if __name__ == "__main__":
    main()