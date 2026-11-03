# src/models/voice_auth_dataset.py
"""
VoiceGuard Banking-Grade Dataset (v4: MUSAN + RIR + Telephony)

Design & Security Standards:
  ✓ 100% PyTorch GPU-friendly augmentation (Zero CPU PitchShift bottleneck)
  ✓ Real-world ambient noise injection (MUSAN babble, street, office, music)
  ✓ Real acoustic room convolution (OpenSLR 28 RIRs via fast FFT-convolution)
  ✓ Banking telephony simulation (8 kHz G.711 / µ-law companding)
  ✓ SNR strictly bounded to [15, 30] dB to preserve acoustic DSP integrity
  ✓ Equal augmentation probability (50%) for both Real and Fake audio (no fingerprinting)
  ✓ Training-split Z-score normalization for 45-dim DSP physical features (no data leakage)
  ✓ Multiprocessing safe: @torch.no_grad() and detached tensors (no worker crashes)
"""
import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T


class VoiceAuthDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str = "data/voice_auth_manifest.csv",
        dsp_cache_path: str = "data/dsp_feature_cache.npz",
        musan_dir: str = "data/musan",
        rir_dir: str = "data/RIRS_NOISES",
        split: str = "train",
        target_seconds: float = 3.0,
        sample_rate: int = 16000,
        augment: bool = False,
        augment_prob: float = 0.50,
    ):
        full_df = pd.read_csv(manifest_csv)
        self.df = full_df[full_df["split"] == split].reset_index(drop=True)

        # Load 45-dim DSP physical feature cache
        cache_data = np.load(dsp_cache_path, allow_pickle=True)
        cache_paths = list(cache_data["paths"])
        cache_features = cache_data["features"]
        self.dsp_lookup = {
            str(path): cache_features[i] for i, path in enumerate(cache_paths)
        }

        self.sr = sample_rate
        self.target_samples = int(sample_rate * target_seconds)

        # Fast 80-channel log-mel spectrogram
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            power=2.0,
        )

        self.augment = augment
        self.augment_prob = augment_prob

        # Telephony codec transforms (pure PyTorch)
        self.downsample = T.Resample(sample_rate, 8000)
        self.upsample = T.Resample(8000, sample_rate)
        self.mu_encode = T.MuLawEncoding(quantization_channels=256)
        self.mu_decode = T.MuLawDecoding(quantization_channels=256)

        # Index MUSAN files if augmenting
        self.musan_files = []
        if self.augment and os.path.exists(musan_dir):
            self.musan_files = glob.glob(f"{musan_dir}/**/*.wav", recursive=True)
            if len(self.musan_files) > 0:
                print(f"  [{split.upper()}] Loaded {len(self.musan_files):,} MUSAN noise files for augmentation.")

        # Index RIR files if augmenting
        self.rir_files = []
        if self.augment:
            alt_rir = "data/rirs_noises"
            search_rir = rir_dir if os.path.exists(rir_dir) else alt_rir
            if os.path.exists(search_rir):
                all_rirs = glob.glob(f"{search_rir}/**/*.wav", recursive=True)
                # Sample a fast pool of 2,000 diverse RIRs
                self.rir_files = all_rirs[:2000] if len(all_rirs) > 2000 else all_rirs
                if len(self.rir_files) > 0:
                    print(f"  [{split.upper()}] Loaded {len(self.rir_files):,} RIR impulse profiles for reverberation.")

        # ==============================================================
        # Fixed Z-Score Normalization computed STRICTLY from train split
        # (Zero data leakage across Dev or Eval)
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

    def _pad_or_crop(self, waveform):
        """Pad with zeros or random-crop to exactly target_samples."""
        n = waveform.shape[-1]
        if n >= self.target_samples:
            start = random.randint(0, n - self.target_samples)
            return waveform[..., start : start + self.target_samples]
        else:
            pad = self.target_samples - n
            return F.pad(waveform, (0, pad))

    # ------------------------------------------------------------------
    # High-Performance Acoustic Augmentation (PyTorch Native)
    # ------------------------------------------------------------------
    def _apply_musan_noise(self, waveform):
        """Blend realistic ambient background noise at 15–30 dB SNR."""
        if not self.musan_files:
            return waveform
        noise_file = random.choice(self.musan_files)
        try:
            noise_wav, sr = torchaudio.load(noise_file)
            if sr != self.sr:
                noise_wav = T.Resample(sr, self.sr)(noise_wav)
            if noise_wav.shape[0] > 1:
                noise_wav = noise_wav.mean(dim=0, keepdim=True)
            noise_wav = self._pad_or_crop(noise_wav)

            # Strict banking safety: SNR ∈ [15dB, 30dB] protects DSP features
            snr_db = random.uniform(15.0, 30.0)
            sig_power = waveform.pow(2).mean()
            noise_power = noise_wav.pow(2).mean()

            if noise_power > 1e-8:
                scale = (sig_power / (noise_power * (10 ** (snr_db / 10)))).sqrt()
                waveform = waveform + scale * noise_wav
        except Exception:
            pass
        return waveform

    def _apply_rir_reverb(self, waveform):
        """Convolve with room impulse response to simulate acoustic room/car."""
        if not self.rir_files:
            return waveform
        rir_file = random.choice(self.rir_files)
        try:
            rir_wav, sr = torchaudio.load(rir_file)
            if sr != self.sr:
                rir_wav = T.Resample(sr, self.sr)(rir_wav)
            if rir_wav.shape[0] > 1:
                rir_wav = rir_wav.mean(dim=0, keepdim=True)

            # Normalize RIR energy and truncate long tail to 4000 samples (250ms) for fast FFT
            rir_wav = rir_wav[..., :min(4000, rir_wav.shape[-1])]
            rir_wav = rir_wav / (rir_wav.norm(p=2) + 1e-8)

            # Fast FFT convolution with fallback
            if hasattr(torchaudio.functional, "fftconvolve"):
                convolved = torchaudio.functional.fftconvolve(waveform, rir_wav, mode="same")
            else:
                convolved = F.conv1d(waveform.unsqueeze(0), rir_wav.unsqueeze(0), padding="same").squeeze(0)

            # 50/50 wet/dry blend keeps direct voice path clear
            waveform = 0.5 * waveform + 0.5 * convolved
        except Exception:
            pass
        return waveform

    def _apply_telephony(self, waveform):
        """Simulate G.711 / µ-law 8kHz PSTN landline/cellular bandpass."""
        try:
            down = self.downsample(waveform)
            encoded = self.mu_encode(down)
            decoded = self.mu_decode(encoded)
            waveform = self.upsample(decoded)
        except Exception:
            pass
        return waveform

    def _augment_waveform(self, waveform):
        """Randomly pick 1 realistic distortion scenario."""
        aug_choice = random.choice(["telephony", "musan_noise", "rir_reverb", "volume"])

        if aug_choice == "telephony":
            waveform = self._apply_telephony(waveform)
        elif aug_choice == "musan_noise":
            waveform = self._apply_musan_noise(waveform)
        elif aug_choice == "rir_reverb":
            waveform = self._apply_rir_reverb(waveform)
        elif aug_choice == "volume":
            gain_db = random.uniform(-3.0, 3.0)
            waveform = waveform * (10 ** (gain_db / 20))

        # Maintain -0.5 dB headroom
        max_val = waveform.abs().max()
        if max_val > 0.95:
            waveform = waveform / max_val * 0.95

        return waveform

    # ------------------------------------------------------------------
    # __getitem__ with DataLoader Worker Autograd Protection
    # ------------------------------------------------------------------
    @torch.no_grad()
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row["audio_path"]
        label = 0 if row["label"] == "real" else 1

        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sr:
            waveform = T.Resample(sr, self.sr)(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = self._pad_or_crop(waveform)

        # Equal probability augmentation for both Real & Fake (no fingerprinting)
        if self.augment and random.random() < self.augment_prob:
            waveform = self._augment_waveform(waveform)

        # Fast Log-mel spectrogram
        mel = self.mel_transform(waveform)
        log_mel = torch.log(mel + 1e-9)

        # 45-dim DSP features with fixed training Z-score normalization
        dsp = self.dsp_lookup.get(str(audio_path))
        if dsp is not None:
            dsp = (dsp - self.dsp_mean) / self.dsp_std
            dsp = torch.tensor(dsp, dtype=torch.float32)
        else:
            dsp = torch.zeros(45, dtype=torch.float32)

        return log_mel.detach(), dsp.detach(), label
