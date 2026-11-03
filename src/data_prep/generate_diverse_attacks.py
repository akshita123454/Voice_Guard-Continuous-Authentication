#!/usr/bin/env python3
"""
VoiceGuard: Diverse Neural Voice Clone Generator
Hardware: NVIDIA H100 80GB HBM3
Engine: Coqui XTTS-v2 (Cloning via conditioning latents)

Generates thousands of realistic synthetic speech samples to eliminate
the Zero-Day vulnerability (unseen TTS algorithms A07-A19).
"""
import os
import glob
import random
import argparse
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from pathlib import Path
# Patch PyTorch 2.6 default weights_only=True for Coqui TTS compatibility
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load
# Fix potential PyTorch / CUDA thread contention
torch.set_num_threads(4)

# Diverse banking, social engineering, security, and conversational utterances
FRAUD_AND_BANKING_PROMPTS = [
    # Wire & Transfer scenarios
    "Please authorize an immediate wire transfer of forty-five thousand dollars to our Singapore supplier.",
    "I am calling to verify the international remittance requested from my savings account this morning.",
    "Can you increase the daily withdrawal limit on corporate account four zero nine eight?",
    "Confirming payment authorization for invoice number seven eight one nine to Alpha Logistics.",
    "Kindly release the escrow hold on transaction ID eight two six five four immediately.",
    "I need to transfer twelve thousand five hundred euros to account number three three two nine.",
    
    # Security & Verification
    "My one time passcode is four eight two nine one zero, please confirm my identity.",
    "I have lost my security token and need to reset my online banking multi-factor authentication.",
    "Yes, I authorize this phone transaction using my voice biometric verification phrase.",
    "Please send the verification code to my secondary registered mobile number.",
    "I am confirming that I initiated the password reset request on my mobile banking app.",
    
    # Conversational & Impersonation
    "Good morning, this is David Miller calling regarding our company's line of credit renewal.",
    "Hello, I am calling on behalf of the Chief Financial Officer to clear an urgent vendor draft.",
    "Can you check if the RTGS batch for this afternoon has cleared the settlement queue?",
    "I am currently traveling abroad and need emergency funds authorized on my corporate card.",
    "Please update our registered billing email address to accounting at nexus group dot com.",
    "Could you please review the pending transactions on my business checking account?",
    "I noticed an unauthorized charge of eight hundred dollars from an online retailer.",
    "Please hold the outbound transaction until I can confirm the beneficiary IBAN with our legal team.",
    "I would like to confirm that our quarterly interest dividend has been credited to our portfolio.",
    "Can you please read back the account balance and the last three outgoing payments?"
]

def load_librispeech_speakers(librispeech_dir="data/librispeech", min_duration_sec=3.0, max_speakers=100):
    """
    Discovers diverse speaker references from LibriSpeech train-clean-100.
    Returns a dict: {speaker_id: [list of candidate audio paths]}
    """
    print(f"Scanning {librispeech_dir} for speaker references...")
    wav_files = glob.glob(f"{librispeech_dir}/**/*.flac", recursive=True)
    if not wav_files:
        wav_files = glob.glob(f"{librispeech_dir}/**/*.wav", recursive=True)

    speaker_map = {}
    for path in wav_files:
        # LibriSpeech structure: .../speaker_id/chapter_id/speaker_id-chapter_id-utterance.flac
        parts = Path(path).parts
        if len(parts) >= 3:
            spk_id = parts[-3]
        else:
            spk_id = "default"
        
        if spk_id not in speaker_map:
            speaker_map[spk_id] = []
        speaker_map[spk_id].append(path)

    print(f"Found {len(speaker_map)} distinct speaker IDs in LibriSpeech.")
    
    # Filter speakers that have at least 3 audio samples
    valid_speakers = {k: v for k, v in speaker_map.items() if len(v) >= 3}
    selected_speakers = dict(list(valid_speakers.items())[:max_speakers])
    print(f"Selected {len(selected_speakers)} diverse speakers for voice cloning.")
    return selected_speakers

def main():
    parser = argparse.ArgumentParser(description="VoiceGuard Diverse Neural Attack Generator")
    parser.add_argument("--num_samples", type=int, default=2500, help="Number of synthetic samples to generate")
    parser.add_argument("--output_dir", type=str, default="data/synthetic_voice/diverse_clones", help="Output path")
    parser.add_argument("--librispeech_dir", type=str, default="data/librispeech", help="LibriSpeech path")
    parser.add_argument("--batch_save", type=int, default=50, help="Interval to save manifest checkpoint")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "manifest.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print("VOICEGUARD DIVERSE ATTACK GENERATOR (XTTS-v2)")
    print(f"Target count: {args.num_samples:,} samples")
    print(f"Output folder: {args.output_dir}")
    print(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 65)

    # 1. Resume from existing manifest if present
    existing_records = []
    completed_files = set()
    if os.path.exists(manifest_path):
        existing_df = pd.read_csv(manifest_path)
        existing_records = existing_df.to_dict("records")
        completed_files = set(existing_df["audio_path"].astype(str).values)
        print(f"Found existing manifest with {len(existing_records)} samples. Resuming...")

    needed_samples = args.num_samples - len(completed_files)
    if needed_samples <= 0:
        print(f"✅ Target count of {args.num_samples} already met!")
        return

    # 2. Load TTS Engine
    print("Loading Coqui XTTS-v2 onto H100...")
    from TTS.api import TTS
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=(device == "cuda"))

    # 3. Reference Speakers
    speaker_map = load_librispeech_speakers(args.librispeech_dir, max_speakers=150)
    if not speaker_map:
        raise RuntimeError("No LibriSpeech speaker audio found. Check data/librispeech path.")

    speaker_ids = list(speaker_map.keys())

    # Resampler: XTTS outputs at 24000 Hz. ASVspoof / LibriSpeech / VoiceGuard standard is 16000 Hz.
    resampler = T.Resample(orig_freq=24000, new_freq=16000)

    # 4. Generation Loop
    records = list(existing_records)
    current_idx = len(records)

    pbar = tqdm(total=args.num_samples, initial=len(records), desc="Synthesizing Clones")

    try:
        while len(records) < args.num_samples:
            current_idx += 1
            spk_id = random.choice(speaker_ids)
            ref_audio = random.choice(speaker_map[spk_id])
            text = random.choice(FRAUD_AND_BANKING_PROMPTS)

            # Add subtle natural variations (numbers, amounts, speed)
            if random.random() < 0.4:
                rand_num = random.randint(1000, 99999)
                text = text.replace("forty-five thousand", f"{rand_num}")

            filename = f"xtts_clone_{current_idx:06d}_spk{spk_id}.wav"
            out_path = os.path.join(args.output_dir, filename)

            if out_path in completed_files:
                continue

            try:
                # Generate via XTTS conditioning
                # wav output is numpy array float32 at 24kHz
                wav = tts.tts(text=text, speaker_wav=ref_audio, language="en")
                
                # Convert to tensor and strictly resample to 16,000 Hz mono
                wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
                wav_16k = resampler(wav_tensor)

                # Normalize amplitude to prevent clipping (-1 dB headroom)
                max_val = wav_16k.abs().max()
                if max_val > 0.95:
                    wav_16k = wav_16k / max_val * 0.95

                # Save 16kHz WAV
                torchaudio.save(out_path, wav_16k, 16000, encoding="PCM_S", bits_per_sample=16)

                duration_sec = wav_16k.shape[-1] / 16000.0

                record = {
                    "audio_path": out_path,
                    "label": "fake",
                    "attack_type": "XTTS_v2_neural_clone",
                    "target_speaker_id": spk_id,
                    "duration_sec": round(duration_sec, 2),
                    "text": text,
                    "sample_rate": 16000
                }
                records.append(record)
                completed_files.add(out_path)
                pbar.update(1)

                # Periodic save
                if len(records) % args.batch_save == 0:
                    pd.DataFrame(records).to_csv(manifest_path, index=False)

            except Exception as e:
                print(f"\n⚠️ Skipped sample {current_idx} due to error: {e}")
                continue

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving progress...")

    finally:
        pbar.close()
        pd.DataFrame(records).to_csv(manifest_path, index=False)
        print(f"\nSaved {len(records)} synthesized samples to manifest: {manifest_path}")
        print("=" * 65)
        print("SYNTHESIS COMPLETE")
        print("=" * 65)

if __name__ == "__main__":
    main()
