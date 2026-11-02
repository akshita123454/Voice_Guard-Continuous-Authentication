# src/models/voice_auth_dataset.py
"""
VoiceGuard v3 Dataset: Physics-First Dual Stream

Key changes from v2:
  ✓ FIXED: DSP Z-score normalization uses TRAINING split stats for ALL splits
  ✓ Augment BOTH classes equally at 50% (prevents augmentation fingerprint)
  ✓ SNR ≥ 15 dB for noise augmentation (preserves DSP features)
  ✗ Removed SpecAugment (hurt zero-day generalization in v2)
  ✗ Removed separate augment_prob_real / augment_prob_fake
"""
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T


class VoiceAuthDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str = "data/voice_auth_manifest.csv",
        dsp_cache_path: str = "data/dsp_feature_cache.npz",
        split: str = "train",
        target_seconds: float = 3.0,
        sample_rate: int = 16000,
        augment: bool = False,
        augment_prob: float = 0.5,   # v3: SAME probability for both classes
    ):
        full_df = pd.read_csv(manifest_csv)
        self.df = full_df[full_df["split"] == split].reset_index(drop=True)

        # Load DSP feature cache
        cache_data = np.load(dsp_cache_path, allow_pickle=True)
        cache_paths = list(cache_data["paths"])
        cache_features = cache_data["features"]
        self.dsp_lookup = {
            str(path): cache_features[i] for i, path in enumerate(cache_paths)
        }

        self.sr = sample_rate
        self.target_samples = int(sample_rate * target_seconds)

        # Mel spectrogram transform (same as v1/v2)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            power=2.0,
        )

        self.augment = augment
        self.augment_prob = augment_prob

        # Telephone codec simulation (pure PyTorch — no sox dependency)
        self.downsample = T.Resample(sample_rate, 8000)
        self.upsample = T.Resample(8000, sample_rate)
        self.mu_encode = T.MuLawEncoding(quantization_channels=256)
        self.mu_decode = T.MuLawDecoding(quantization_channels=256)

        # Pre-built pitch shift transforms (±0.5 and ±1.0 semitones)
        self.pitch_shifts = [
            T.PitchShift(sample_rate, n_steps=s)
            for s in [-1.0, -0.5, 0.5, 1.0]
        ]

        # ==============================================================
        # [v3 CRITICAL FIX] Z-score normalization from TRAINING split
        # stats — used for ALL splits (train, dev, eval).
        #
        # v2 BUG: used `self.df` (current split) instead of `train_df`
        # which caused each split to normalize by its own statistics
        # = data leakage for dev/eval.
        # ==============================================================
        train_df = full_df[full_df["split"] == "train"]
        train_paths = set(train_df["audio_path"].astype(str).values)
        train_feats = np.array([
            cache_features[i]
            for i, p in enumerate(cache_paths)
            if str(p) in train_paths
        ])
        self.dsp_mean = train_feats.mean(axis=0).astype(np.float32)
        self.dsp_std = (train_feats.std(axis=0) + 1e-8).astype(np.float32)

    def __len__(self):
        return len(self.df)

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------
    def _pad_or_crop(self, waveform):
        """Pad with zeros or random-crop to exactly target_samples."""
        n = waveform.shape[-1]
        if n >= self.target_samples:
            start = random.randint(0, n - self.target_samples)
            return waveform[..., start : start + self.target_samples]
        else:
            pad = self.target_samples - n
            return torch.nn.functional.pad(waveform, (0, pad))

    # ------------------------------------------------------------------
    # Augmentation pipeline (applied to waveform BEFORE mel computation)
    # ------------------------------------------------------------------
    def _augment_waveform(self, waveform):
        """Apply ONE random augmentation.

        Types:
          telephone — 8 kHz downsample + µ-law encode/decode + upsample
          noise     — additive Gaussian, SNR ∈ [15, 30] dB (DSP-safe)
          pitch     — ±0.5 or ±1.0 semitones
          volume    — ±3 dB gain
        """
        aug_type = random.choice(["telephone", "noise", "pitch", "volume"])

        if aug_type == "telephone":
            down = self.downsample(waveform)
            encoded = self.mu_encode(down)
            decoded = self.mu_decode(encoded)
            waveform = self.upsample(decoded)

        elif aug_type == "noise":
            snr_db = random.uniform(15.0, 30.0)
            sig_power = waveform.pow(2).mean()
            noise_power = sig_power / (10 ** (snr_db / 10))
            noise = torch.randn_like(waveform) * noise_power.sqrt()
            waveform = waveform + noise

        elif aug_type == "pitch":
            ps = random.choice(self.pitch_shifts)
            waveform = ps(waveform)

        elif aug_type == "volume":
            gain_db = random.uniform(-3.0, 3.0)
            gain = 10 ** (gain_db / 20)
            waveform = waveform * gain

        return waveform

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------
    @torch.no_grad()
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row["audio_path"]
        label = 0 if row["label"] == "real" else 1   # 0=real, 1=fake

        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sr:
            waveform = T.Resample(sr, self.sr)(waveform)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad / crop to target length
        waveform = self._pad_or_crop(waveform)

        # ==========================================================
        # [v3] Augment BOTH classes with the SAME probability
        # ==========================================================
        if self.augment and random.random() < self.augment_prob:
            waveform = self._augment_waveform(waveform)

        # Mel spectrogram (log scale)
        mel = self.mel_transform(waveform)          # (1, 80, T)
        log_mel = torch.log(mel + 1e-9)

        # DSP features (Z-score normalized from training stats)
        dsp = self.dsp_lookup.get(str(audio_path))
        if dsp is not None:
            dsp = (dsp - self.dsp_mean) / self.dsp_std
            dsp = torch.tensor(dsp, dtype=torch.float32)
        else:
            dsp = torch.zeros(45, dtype=torch.float32)

        return log_mel.detach(), dsp.detach(), label
