# src/data_prep/audit_datasets.py
import os
import glob
import soundfile as sf
import numpy as np

def audit_directory(dir_path, name, ext="*.flac"):
    files = glob.glob(os.path.join(dir_path, "**", ext), recursive=True)
    if not files:
        # Try .wav if .flac had no matches
        files = glob.glob(os.path.join(dir_path, "**", "*.wav"), recursive=True)
        
    print(f"\n{'='*50}")
    print(f"AUDITING: {name}")
    print(f"Directory: {dir_path}")
    print(f"Total files found: {len(files)}")
    print(f"{'='*50}")

    if len(files) == 0:
        print("❌ WARNING: No audio files found in this directory!")
        return

    sample_rates = {}
    durations = []
    silent_files = 0
    corrupt_files = 0

    # Inspect first 1000 files (or all if fewer) to avoid waiting too long
    check_limit = min(len(files), 1000)
    for i in range(check_limit):
        f = files[i]
        try:
            info = sf.info(f)
            sr = info.samplerate
            sample_rates[sr] = sample_rates.get(sr, 0) + 1
            durations.append(info.duration)

            # Check for silence/zeros in a small sample
            data, _ = sf.read(f, frames=min(info.frames, 16000))
            if np.max(np.abs(data)) < 1e-4:
                silent_files += 1

        except Exception as e:
            corrupt_files += 1

    print(f"Inspected Sample Size: {check_limit} files")
    print(f"Sample Rates Detected: {sample_rates}")
    print(f"Average Duration: {np.mean(durations):.2f} seconds (Min: {np.min(durations):.2f}s, Max: {np.max(durations):.2f}s)")
    print(f"Silent / Zero-Amplitude Files: {silent_files}")
    print(f"Corrupt / Unreadable Files: {corrupt_files}")

    if len(sample_rates) == 1 and 16000 in sample_rates and corrupt_files == 0:
        print("✅ STATUS: Pristine. Ready for model training.")
    else:
        print("⚠️ STATUS: Requires normalization/resampling before training.")

if __name__ == "__main__":
    audit_directory("data/librispeech", "LibriSpeech Clean Corpus")
    audit_directory("data/synthetic_voice", "Synthetic Voice Samples")