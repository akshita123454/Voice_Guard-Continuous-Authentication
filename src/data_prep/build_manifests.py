# src/data_prep/build_voice_auth_manifest.py
"""
Builds a single unified manifest.csv for voice authenticity training.
Combines:
  1. ASVspoof 2019 LA (bonafide + spoof) — the scientific benchmark
  2. Self-generated synthetic banking audio — the domain-specific threat
  
Each row: audio_path, label (real/fake), attack_type, split, source_dataset
"""
import os
import glob
import csv
import soundfile as sf
from collections import Counter

BASE_DIR = "/data/user3/voice_guard"
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "voice_auth_manifest.csv")

def parse_asvspoof_protocol(protocol_path, audio_dir, split_name):
    """Parse an ASVspoof protocol file and return list of record dicts."""
    records = []
    if not os.path.exists(protocol_path):
        print(f"  ⚠️  Protocol not found: {protocol_path}")
        return records

    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            speaker_id = parts[0]
            audio_id = parts[1]
            attack_type = parts[3]  # '-' for bonafide, 'A01'-'A19' for spoof
            label_raw = parts[4]    # 'bonafide' or 'spoof'

            audio_path = os.path.join(audio_dir, f"{audio_id}.flac")
            if not os.path.exists(audio_path):
                continue

            records.append({
                "audio_path": audio_path,
                "label": "real" if label_raw == "bonafide" else "fake",
                "attack_type": attack_type if label_raw == "spoof" else "none",
                "split": split_name,
                "source": "asvspoof2019"
            })
    return records

def collect_asvspoof():
    """Collect all ASVspoof 2019 LA records across train/dev/eval splits."""
    la_root = os.path.join(BASE_DIR, "data", "asvspoof2019", "LA")
    proto_dir = os.path.join(la_root, "ASVspoof2019_LA_cm_protocols")
    
    all_records = []

    # Training split
    proto_file = os.path.join(proto_dir, "ASVspoof2019.LA.cm.train.trn.txt")
    audio_dir = os.path.join(la_root, "ASVspoof2019_LA_train", "flac")
    records = parse_asvspoof_protocol(proto_file, audio_dir, "train")
    print(f"  ASVspoof train: {len(records)} files")
    all_records.extend(records)

    # Dev split
    proto_file = os.path.join(proto_dir, "ASVspoof2019.LA.cm.dev.trl.txt")
    audio_dir = os.path.join(la_root, "ASVspoof2019_LA_dev", "flac")
    records = parse_asvspoof_protocol(proto_file, audio_dir, "dev")
    print(f"  ASVspoof dev:   {len(records)} files")
    all_records.extend(records)

    # Eval split
    proto_file = os.path.join(proto_dir, "ASVspoof2019.LA.cm.eval.trl.txt")
    audio_dir = os.path.join(la_root, "ASVspoof2019_LA_eval", "flac")
    records = parse_asvspoof_protocol(proto_file, audio_dir, "eval")
    print(f"  ASVspoof eval:  {len(records)} files")
    all_records.extend(records)

    return all_records

def collect_synthetic():
    """Collect our self-generated normalized synthetic banking audio."""
    synth_dir = os.path.join(BASE_DIR, "data", "synthetic_voice_16k")
    records = []

    wav_files = glob.glob(os.path.join(synth_dir, "**", "*.wav"), recursive=True)
    for f in wav_files:
        records.append({
            "audio_path": f,
            "label": "fake",
            "attack_type": "self_generated_tts",
            "split": "train",
            "source": "synthetic_banking"
        })

    print(f"  Synthetic banking: {len(records)} files")
    return records

def main():
    print("=" * 60)
    print("Building Unified Voice Auth Training Manifest")
    print("=" * 60)

    print("\n[1/2] Collecting ASVspoof 2019 LA...")
    asvspoof_records = collect_asvspoof()

    print("\n[2/2] Collecting Synthetic Banking Audio...")
    synth_records = collect_synthetic()

    all_records = asvspoof_records + synth_records

    # Write CSV
    fieldnames = ["audio_path", "label", "attack_type", "split", "source"]
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    # Print summary statistics
    label_counts = Counter(r["label"] for r in all_records)
    split_counts = Counter(r["split"] for r in all_records)
    source_counts = Counter(r["source"] for r in all_records)

    print(f"\n{'=' * 60}")
    print(f"MANIFEST WRITTEN: {OUTPUT_FILE}")
    print(f"{'=' * 60}")
    print(f"Total records:     {len(all_records)}")
    print(f"By label:          {dict(label_counts)}")
    print(f"By split:          {dict(split_counts)}")
    print(f"By source:         {dict(source_counts)}")
    
    # Verify class balance for training split
    train_records = [r for r in all_records if r["split"] == "train"]
    train_real = sum(1 for r in train_records if r["label"] == "real")
    train_fake = sum(1 for r in train_records if r["label"] == "fake")
    ratio = train_fake / max(train_real, 1)
    
    print(f"\n[Training Split Balance]")
    print(f"  Real:  {train_real}")
    print(f"  Fake:  {train_fake}")
    print(f"  Ratio (Fake/Real): {ratio:.2f}")
    
    if 0.5 <= ratio <= 2.0:
        print("  ✅ Class balance is acceptable for training.")
    else:
        print("  ⚠️  Significant class imbalance. Consider weighted loss or oversampling.")

if __name__ == "__main__":
    main()