# src/data_prep/normalize_audio.py
import os
import glob
import torchaudio
import torchaudio.transforms as T
import torch
from tqdm import tqdm

TARGET_SAMPLE_RATE = 16000

def normalize_file(input_path, output_path):
    """
    Standardizes any audio file:
    1. Loads audio (handles any format supported by soundfile/sox)
    2. Converts Stereo to Mono
    3. Resamples to exact 16,000 Hz using high-quality sinc interpolation
    4. Applies Peak Loudness Normalization (-1.0 to 1.0)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    waveform, sr = torchaudio.load(input_path)
    
    # 1. Convert to Mono if multi-channel
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # 2. Resample to 16,000 Hz if necessary
    if sr != TARGET_SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
        
    # 3. Peak Normalization (prevent quiet files from confusing the model)
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
        
    # 4. Save as standardized 16kHz 16-bit PCM WAV
    torchaudio.save(output_path, waveform, TARGET_SAMPLE_RATE, encoding="PCM_S", bits_per_sample=16)

def process_directory(source_dir, dest_dir, ext="*.wav"):
    files = glob.glob(os.path.join(source_dir, "**", ext), recursive=True)
    print(f"\nNormalizing {len(files)} files from '{source_dir}' -> '{dest_dir}'...")
    
    for f in tqdm(files):
        # Maintain relative directory structure
        rel_path = os.path.relpath(f, source_dir)
        # Ensure target file extension is .wav
        out_rel_path = os.path.splitext(rel_path)[0] + ".wav"
        out_full_path = os.path.join(dest_dir, out_rel_path)
        
        try:
            normalize_file(f, out_full_path)
        except Exception as e:
            print(f"Error processing {f}: {e}")

if __name__ == "__main__":
    # Standardize synthetic audio to a clean normalized folder
    process_directory(
        source_dir="data/synthetic_voice",
        dest_dir="data/synthetic_voice_16k",
        ext="*.wav"
    )
    print("\nNormalization complete!")