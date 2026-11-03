# src/data_prep/index_noise_rir.py
import json
from pathlib import Path

def index_corpora():
    musan_dir = Path("data/musan")
    rir_dir = Path("data/RIRS_NOISES")
    if not rir_dir.exists():
        rir_dir = Path("data/rirs_noises")

    musan_files = [str(p) for p in musan_dir.rglob("*.wav")]
    rir_files = [str(p) for p in rir_dir.rglob("*.wav")]

    print(f"Indexed MUSAN audio: {len(musan_files):,} files")
    print(f"Indexed RIR profiles: {len(rir_files):,} files")

    catalog = {
        "musan": musan_files,
        "rir": rir_files
    }

    out_path = Path("data/noise_rir_catalog.json")
    with open(out_path, "w") as f:
        json.dump(catalog, f)
    print(f"✅ Catalog saved to {out_path}")

if __name__ == "__main__":
    index_corpora()