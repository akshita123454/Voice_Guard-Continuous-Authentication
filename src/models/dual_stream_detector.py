# src/models/dual_stream_detector.py
import torch
import torch.nn as nn

class DualStreamDetector(nn.Module):
    def __init__(self, dsp_dim=45):
        super().__init__()
        
        # Stream 1: MLP for 45-dim Acoustic DSP Features (Pitch, Jitter, Shimmer, HNR, MFCC)
        self.dsp_branch = nn.Sequential(
            nn.Linear(dsp_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU()
        )

        # Stream 2: 2D-CNN for Log-Mel Spectrogram (Spatial time-frequency artifacts)
        self.spec_branch = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),  # [32, 40, T/2]

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),  # [64, 20, T/4]

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global pooling -> [128, 1, 1]
        )

        # Fusion Head: Concatenates DSP (64) + Spectrogram (128) = 192 features
        self.fusion_head = nn.Sequential(
            nn.Linear(64 + 128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # [Logits Real (0), Logits Fake (1)]
        )

    def forward(self, dsp_feat, mel_spec):
        # mel_spec: [B, 80, Time] -> unsqueeze to [B, 1, 80, Time]
        mel_spec = mel_spec.unsqueeze(1)

        feat_dsp = self.dsp_branch(dsp_feat)
        feat_spec = self.spec_branch(mel_spec).squeeze(-1).squeeze(-1)

        fused = torch.cat([feat_dsp, feat_spec], dim=1)
        logits = self.fusion_head(fused)
        return logits