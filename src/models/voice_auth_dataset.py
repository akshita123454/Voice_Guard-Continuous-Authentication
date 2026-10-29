# src/models/voice_auth_dataset.py
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
        augment_prob_real: float = 0.70,
        augment_prob_fake: float = 0.30,
    ):
        full_df = pd.read_csv(manifest_csv)
        self.df = full_df[full_df["split"] == split].reset_index(drop=True)

        cache_data = np.load(dsp_cache_path, allow_pickle=True)
        cache_paths = list(cache_data["paths"])
        cache_features = cache_data["features"]

        self.dsp_lookup = {str(path): cache_features[i] for i, path in enumerate(cache_paths)}

        self.sr = sample_rate
        self.target_samples = int(sample_rate * target_seconds)

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            power=2.0
        )

        self.augment = augment
        self.aug_prob_real = augment_prob_real
        self.aug_prob_fake = augment_prob_fake

        # Pure PyTorch telephone simulation
        self.downsample = T.Resample(sample_rate, 8000)
        self.upsample = T.Resample(8000, sample_rate)
        self.mu_encode = T.MuLawEncoding(quantization_channels=256)
        self.mu_decode = T.MuLawDecoding(quantization_channels=256)

        counts = self.df["label"].value_counts().to_dict()
        print(f"  [{split.upper()}] Loaded {len(self.df)} samples: {counts}")

    def __len__(self):
        return len(self.df)

    def _load_and_prepare_audio(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        current_len = waveform.shape[1]
        if current_len > self.target_samples:
            max_start = current_len - self.target_samples
            start = random.randint(0, max_start) if self.augment else 0
            waveform = waveform[:, start : start + self.target_samples]
        elif current_len < self.target_samples:
            pad_amount = self.target_samples - current_len
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        return waveform

    def _apply_augmentation(self, waveform):
        # 1. Bounded noise (SNR 15 to 30 dB - preserves DSP features)
        if random.random() < 0.5:
            snr_db = random.uniform(15.0, 30.0)
            noise = torch.randn_like(waveform)
            sig_power = torch.mean(waveform ** 2)
            noise_power = torch.mean(noise ** 2)
            scale = torch.sqrt(sig_power / (10 ** (snr_db / 10) * noise_power + 1e-8))
            waveform = waveform + scale * noise

        # 2. Volume jitter (+/- 4 dB)
        if random.random() < 0.5:
            gain_db = random.uniform(-4.0, 4.0)
            waveform = waveform * (10 ** (gain_db / 20))

        # 3. Telephony codec artifact simulation
        if random.random() < 0.3:
            w8 = self.downsample(waveform)
            w_mu = self.mu_decode(self.mu_encode(w8))
            waveform = self.upsample(w_mu)

        return torch.clamp(waveform, -1.0, 1.0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = str(row["audio_path"])
        label = 0 if row["label"] == "real" else 1

        dsp_feat = self.dsp_lookup.get(audio_path, np.zeros(45, dtype=np.float32))
        dsp_tensor = torch.tensor(dsp_feat, dtype=torch.float32)

        waveform = self._load_and_prepare_audio(audio_path)

        # Apply augmentation to both classes (different probability)
        if self.augment:
            prob = self.aug_prob_real if label == 0 else self.aug_prob_fake
            if random.random() < prob:
                waveform = self._apply_augmentation(waveform)

        mel_spec = self.mel_transform(waveform)
        log_mel = torch.log(mel_spec + 1e-6).squeeze(0)  # Shape: [80, Time]

        return dsp_tensor, log_mel, torch.tensor(label, dtype=torch.long)