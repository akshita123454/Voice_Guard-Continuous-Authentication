# src/models/dual_stream_detector.py
"""
VoiceGuard v3: Physics-First Dual Stream Detector

Architecture:
  CNN Stream:  3-block plain CNN (v1 baseline — proven to generalize)
  DSP Stream:  2-layer MLP with BatchNorm + Dropout (upgraded from v1)
  Fusion:      Late fusion with learnable α (initialized DSP-heavy at 0.6)

Why this design:
  - v2's deeper ResNet + SE + attention fusion OVERFITTED to training attacks
    (eval EER went from 10.62% → 13.49% = regression)
  - Physics-based DSP features (HNR, jitter, shimmer) generalize better than
    learned CNN features for zero-day attack detection
  - Late fusion lets each stream specialize independently
  - Learnable α starts DSP-heavy (0.6) but can adjust during training

Changes from v2:
  ✗ No 4-block ResNet (back to 3-block plain CNN)
  ✗ No SE (Squeeze-and-Excitation) attention
  ✗ No attention-gated fusion
  ✓ 2-layer DSP MLP with BatchNorm + Dropout (stronger physics pathway)
  ✓ Late fusion with learnable α
  ✓ get_stream_scores() for forensic explainability
"""
import torch
import torch.nn as nn


class DualStreamDetector(nn.Module):
    def __init__(self, dsp_dim=45, n_mels=80, num_classes=2):
        super().__init__()

        # =============================================================
        # CNN Stream: 3-block plain CNN (exactly v1 architecture)
        # Input:  (B, 1, 80, T)  log-mel spectrogram
        # Output: (B, 128)       embedding
        # =============================================================
        self.cnn = nn.Sequential(
            # Block 1: (1, 80, T) → (32, 40, T/2)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: (32, 40, T/2) → (64, 20, T/4)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: (64, 20, T/4) → (128, 1, 1)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        # =============================================================
        # DSP Stream: 2-layer MLP (upgraded from v1's single layer)
        # Input:  (B, 45)  Z-score normalized DSP features
        # Output: (B, 64)  embedding
        # =============================================================
        self.dsp_mlp = nn.Sequential(
            nn.Linear(dsp_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # =============================================================
        # Late Fusion with learnable α
        #   logits = α · DSP_logits + (1 − α) · CNN_logits
        #   α = sigmoid(fusion_logit)
        #   Initialized so α ≈ 0.6 (DSP-heavy):  sigmoid(0.405) ≈ 0.6
        # =============================================================
        self.cnn_head = nn.Linear(128, num_classes)
        self.dsp_head = nn.Linear(64, num_classes)
        self.fusion_logit = nn.Parameter(torch.tensor(0.405))

    def forward(self, mel_spec, dsp_features):
        """
        Args:
            mel_spec:     (B, 1, 80, T)  log-mel spectrogram
            dsp_features: (B, 45)        Z-score normalized DSP features
        Returns:
            logits:       (B, 2)         class logits [real, fake]
        """
        # CNN pathway
        cnn_out = self.cnn(mel_spec)
        cnn_embed = cnn_out.view(cnn_out.size(0), -1)    # (B, 128)

        # DSP pathway
        dsp_embed = self.dsp_mlp(dsp_features)            # (B, 64)

        # Late fusion
        alpha = torch.sigmoid(self.fusion_logit)           # DSP weight
        cnn_logits = self.cnn_head(cnn_embed)              # (B, 2)
        dsp_logits = self.dsp_head(dsp_embed)              # (B, 2)

        logits = alpha * dsp_logits + (1.0 - alpha) * cnn_logits
        return logits

    # -----------------------------------------------------------------
    # Monitoring helpers
    # -----------------------------------------------------------------
    def get_fusion_weight(self):
        """Return the current DSP fusion weight α ∈ (0, 1)."""
        return torch.sigmoid(self.fusion_logit).item()

    def get_stream_scores(self, mel_spec, dsp_features):
        """Return individual stream fake-probabilities (for forensic XAI).

        Returns dict with:
          cnn_fake_prob  — CNN stream's P(fake)
          dsp_fake_prob  — DSP stream's P(fake)
          fusion_alpha   — learned DSP weight
        """
        with torch.no_grad():
            cnn_out = self.cnn(mel_spec)
            cnn_embed = cnn_out.view(cnn_out.size(0), -1)
            dsp_embed = self.dsp_mlp(dsp_features)

            cnn_logits = self.cnn_head(cnn_embed)
            dsp_logits = self.dsp_head(dsp_embed)

            cnn_prob = torch.softmax(cnn_logits, dim=1)[:, 1]
            dsp_prob = torch.softmax(dsp_logits, dim=1)[:, 1]
            alpha = torch.sigmoid(self.fusion_logit)

        return {
            "cnn_fake_prob": cnn_prob,
            "dsp_fake_prob": dsp_prob,
            "fusion_alpha": alpha.item(),
        }
