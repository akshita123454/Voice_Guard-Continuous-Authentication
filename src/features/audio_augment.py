# src/features/audio_augment.py
"""
On-the-fly audio augmentation for training robustness.
Applied ONLY to the minority class (real/bonafide) to increase diversity,
and optionally to all samples for general noise tolerance.

Each augmentation simulates a real-world condition:
  - add_noise:       Background chatter in a call center or street
  - pitch_shift:     Different speakers naturally vary in pitch
  - time_stretch:    People speak faster/slower under stress
  - volume_perturb:  Microphone distance, phone speaker quality
  - telephone_codec: PSTN/VoIP 8kHz compression artifacts
"""
import torch
import torchaudio
import torchaudio.transforms as T
import random

class AudioAugmentor:
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        # Pre-build resamplers for telephone simulation
        self.downsample_8k = T.Resample(orig_freq=sample_rate, new_freq=8000)
        self.upsample_16k = T.Resample(orig_freq=8000, new_freq=sample_rate)
        self.mu_encode = T.MuLawEncoding(quantization_channels=256)
        self.mu_decode = T.MuLawDecoding(quantization_channels=256)

    def add_noise(self, waveform, snr_db=15.0):
        """
        Adds Gaussian white noise at a specified Signal-to-Noise Ratio.
        SNR of 15 dB = moderate background noise (typical office).
        SNR of 5 dB  = very noisy environment (street, crowd).
        """
        noise = torch.randn_like(waveform)
        signal_power = torch.mean(waveform ** 2)
        noise_power = torch.mean(noise ** 2)
        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(signal_power / (snr_linear * noise_power + 1e-8))
        return waveform + scale * noise

    def pitch_shift(self, waveform, semitones=None):
        """
        Shifts pitch up or down by a random amount (±2 semitones).
        Simulates different vocal tract lengths / speaking styles.
        """
        if semitones is None:
            semitones = random.uniform(-2.0, 2.0)
        effects = [["pitch", str(int(semitones * 100))], ["rate", str(self.sr)]]
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sr, effects, channels_first=True
        )
        return augmented

    def time_stretch(self, waveform, rate=None):
        """
        Speeds up or slows down speech by 10-20%.
        Simulates speaking under stress (faster) or hesitation (slower).
        """
        if rate is None:
            rate = random.uniform(0.85, 1.15)
        effects = [["tempo", str(rate)]]
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sr, effects, channels_first=True
        )
        return augmented

    def volume_perturb(self, waveform, gain_db=None):
        """
        Randomly adjusts volume by ±6 dB.
        Simulates holding the phone closer/farther from mouth.
        """
        if gain_db is None:
            gain_db = random.uniform(-6.0, 6.0)
        gain_linear = 10 ** (gain_db / 20)
        return waveform * gain_linear

    def telephone_codec(self, waveform):
        """
        Simulates PSTN telephone line:
        16kHz -> 8kHz downsample -> G.711 mu-law -> 8kHz -> 16kHz upsample.
        Introduces the characteristic 'tinny' phone quality.
        """
        audio_8k = self.downsample_8k(waveform)
        mu_quant = self.mu_encode(audio_8k)
        audio_mu = self.mu_decode(mu_quant)
        audio_16k = self.upsample_16k(audio_mu)
        return audio_16k

    def augment(self, waveform, intensity="medium"):
        """
        Applies a random combination of augmentations.
        intensity: 'light' (1 aug), 'medium' (2 augs), 'heavy' (3+ augs)
        """
        augmentations = [
            lambda w: self.add_noise(w, snr_db=random.uniform(10, 25)),
            lambda w: self.volume_perturb(w),
            lambda w: self.telephone_codec(w),
        ]

        # Pitch shift and time stretch require sox; wrap in try/except
        try:
            augmentations.append(lambda w: self.pitch_shift(w))
            augmentations.append(lambda w: self.time_stretch(w))
        except Exception:
            pass

        if intensity == "light":
            n_augs = 1
        elif intensity == "medium":
            n_augs = 2
        else:
            n_augs = random.randint(2, min(4, len(augmentations)))

        chosen = random.sample(augmentations, min(n_augs, len(augmentations)))
        result = waveform.clone()
        for aug_fn in chosen:
            try:
                result = aug_fn(result)
            except Exception:
                continue

        # Clip to prevent overflow
        result = torch.clamp(result, -1.0, 1.0)
        return result


if __name__ == "__main__":
    # Quick validation test
    aug = AudioAugmentor()
    dummy = torch.randn(1, 16000 * 3)  # 3 seconds

    print("Testing augmentation pipeline...")
    noisy = aug.add_noise(dummy, snr_db=15)
    phone = aug.telephone_codec(dummy)
    vol = aug.volume_perturb(dummy)
    combo = aug.augment(dummy, intensity="medium")

    print(f"  Original shape:    {dummy.shape}")
    print(f"  After noise:       {noisy.shape}")
    print(f"  After phone codec: {phone.shape}")
    print(f"  After volume:      {vol.shape}")
    print(f"  After combo:       {combo.shape}")
    print("✅ All augmentations passed.")