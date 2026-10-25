# src/features/acoustic_extractor.py
import numpy as np
import librosa
import soundfile as sf
import torch

class AcousticFeatureExtractor:
    def __init__(self, sample_rate=16000, n_mfcc=20):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        Extracts pitch (F0), jitter, shimmer, HNR, and MFCCs from a 16kHz WAV file.
        """
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        if len(y) == 0:
            raise ValueError(f"Empty audio file: {audio_path}")

        # 1. MFCC Features (Mean & Std Dev across frames)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # 2. Pitch (Fundamental Frequency F0) using Yin algorithm
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        f0_clean = f0[~np.isnan(f0)]
        f0_mean = np.mean(f0_clean) if len(f0_clean) > 0 else 0.0
        f0_std = np.std(f0_clean) if len(f0_clean) > 0 else 0.0

        # 3. Jitter (Cycle-to-cycle pitch period variation)
        if len(f0_clean) > 1:
            periods = 1.0 / (f0_clean + 1e-6)
            jitter = np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-6)
        else:
            jitter = 0.0

        # 4. Shimmer (Cycle-to-cycle amplitude variation)
        hop_length = 512
        frame_energy = np.array([
            np.sqrt(np.mean(y[i:i + hop_length]**2))
            for i in range(0, len(y) - hop_length, hop_length)
        ])
        if len(frame_energy) > 1:
            shimmer = np.mean(np.abs(np.diff(frame_energy))) / (np.mean(frame_energy) + 1e-6)
        else:
            shimmer = 0.0

        # 5. Harmonics-to-Noise Ratio (HNR)
        harmonic, percussive = librosa.effects.hpss(y)
        noise_energy = np.mean(percussive**2) + 1e-8
        hnr = 10 * np.log10((np.mean(harmonic**2) + 1e-8) / noise_energy)

        # Concatenate into a single feature vector
        feature_vector = np.concatenate([
            mfcc_mean,       # 20 features
            mfcc_std,        # 20 features
            [f0_mean, f0_std, jitter, shimmer, hnr]  # 5 features
        ])
        return feature_vector.astype(np.float32)

if __name__ == "__main__":
    extractor = AcousticFeatureExtractor()
    print("AcousticFeatureExtractor initialized successfully. Ready to process audio.")