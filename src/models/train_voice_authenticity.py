# src/models/train_voice_authenticity.py
"""
VoiceGuard v3 Training: Physics-First Dual Stream Detector

Training strategy (v1 baseline + minimal hardening):
  - Adam optimizer, lr=1e-3, no warmup, no cosine schedule
  - 10 epochs, NO early stopping (v1 trained all 10 and was best)
  - Gradient clipping max_norm=1.0 (only safeguard from v2 worth keeping)
  - Weighted CrossEntropyLoss (class imbalance correction)
  - Augmentation applied to BOTH classes at 50% probability

Changes from v2:
  ✗ No warmup scheduler
  ✗ No cosine annealing
  ✗ No early stopping (train all 10 epochs like v1)
  ✗ No SpecAugment
  ✓ Gradient clipping max_norm=1.0
  ✓ Augment both classes equally
  ✓ Late fusion with learnable α (logs α each epoch)
  ✓ Saves dsp_mean/dsp_std in checkpoint for inference consistency
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

from voice_auth_dataset import VoiceAuthDataset
from dual_stream_detector import DualStreamDetector


def compute_eer(labels, scores):
    """Compute Equal Error Rate and its threshold."""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer, thresholds[idx]


def train():
    # =================================================================
    # Configuration
    # =================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("VOICEGUARD TRAINING v3: PHYSICS-FIRST DUAL STREAM DETECTOR")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Hardware: {gpu_name}")
    print("=" * 65)

    # Hyperparameters (v1 baseline + gradient clipping)
    EPOCHS        = 10
    BATCH_SIZE    = 64
    LR            = 1e-3
    GRAD_CLIP     = 1.0
    NUM_WORKERS   = 4
    CHECKPOINT    = "checkpoints/best_voice_auth_model.pt"
    os.makedirs("checkpoints", exist_ok=True)

    # =================================================================
    # Data
    # =================================================================
    train_ds = VoiceAuthDataset(split="train", augment=True, augment_prob=0.5)
    dev_ds   = VoiceAuthDataset(split="dev",   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    train_counts = train_ds.df["label"].value_counts()
    dev_counts   = dev_ds.df["label"].value_counts()
    print(f"  [TRAIN] Loaded {len(train_ds)} samples: {dict(train_counts)}")
    print(f"  [DEV]   Loaded {len(dev_ds)} samples: {dict(dev_counts)}")

    # =================================================================
    # Model
    # =================================================================
    model = DualStreamDetector(dsp_dim=45, n_mels=80, num_classes=2).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {n_params:,}")
    print(f"Initial Fusion α(DSP): {model.get_fusion_weight():.3f}")

    # =================================================================
    # Loss & Optimizer
    # =================================================================
    n_fake = train_counts.get("fake", 0)
    n_real = train_counts.get("real", 0)
    weight_ratio = n_fake / max(n_real, 1)
    class_weights = torch.tensor(
        [weight_ratio, 1.0], dtype=torch.float32    # [real_weight, fake_weight]
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class Weights: real={weight_ratio:.2f}, fake=1.00")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # =================================================================
    # Training Loop — 10 epochs, no early stopping
    # =================================================================
    best_eer   = float("inf")
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # ---- Train ----
        model.train()
        running_loss = 0.0
        n_batches    = 0

        for mel, dsp, labels in train_loader:
            mel    = mel.to(device)
            dsp    = dsp.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(mel, dsp)
            loss   = criterion(logits, labels)
            loss.backward()

            # Gradient clipping (only v2 safeguard retained)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)

            optimizer.step()

            running_loss += loss.item()
            n_batches    += 1

        avg_loss = running_loss / max(n_batches, 1)

        # ---- Validate ----
        model.eval()
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for mel, dsp, labels in dev_loader:
                mel = mel.to(device)
                dsp = dsp.to(device)
                logits = model(mel, dsp)
                probs  = torch.softmax(logits, dim=1)[:, 1]   # P(fake)
                all_scores.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        dev_eer, _ = compute_eer(all_labels, all_scores)
        dev_auc    = roc_auc_score(all_labels, all_scores)
        elapsed    = time.time() - t0
        alpha      = model.get_fusion_weight()

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] ({elapsed:.1f}s) | "
            f"Loss: {avg_loss:.4f} | Dev EER: {dev_eer*100:.2f}% | "
            f"AUC: {dev_auc*100:.2f}% | α(DSP): {alpha:.3f}"
        )

        # Save best model by dev EER
        if dev_eer < best_eer:
            best_eer   = dev_eer
            best_epoch = epoch
            ckpt = {
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dev_eer":              dev_eer,
                "dev_auc":              dev_auc,
                "fusion_alpha":         alpha,
                "dsp_mean":             train_ds.dsp_mean,   # for inference
                "dsp_std":              train_ds.dsp_std,     # for inference
                "version":              "v3",
            }
            torch.save(ckpt, CHECKPOINT)
            print(f"  ★ New Best (EER: {dev_eer*100:.2f}%) -> {CHECKPOINT}")

    # =================================================================
    # Summary
    # =================================================================
    print("=" * 65)
    print(f"Training Complete. Best Validation EER: {best_eer*100:.2f}%")
    print(f"Best Epoch: {best_epoch}/{EPOCHS}")
    print(f"Final Fusion α(DSP): {model.get_fusion_weight():.3f}")
    print(f"Model Checkpoint: {CHECKPOINT}")
    print("=" * 65)


if __name__ == "__main__":
    train()
