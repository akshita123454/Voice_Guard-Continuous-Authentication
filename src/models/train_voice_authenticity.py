# src/models/train_voice_authenticity.py
"""
VoiceGuard v4 Training: Physics-First Dual Stream Detector

Training Protocol:
  ✓ Architecture: v1 Proven 3-Block CNN + 2-Layer DSP MLP (110K parameters)
  ✓ Augmentations: Realistic MUSAN noise + RIR room reverb + Telephony (50% prob)
  ✓ Data: 69,765 verified audio files (including 2,500 diverse XTTS-v2 clones)
  ✓ Loss: Weighted CrossEntropyLoss (9.82x weight for real speech protection)
  ✓ Stability: Gradient clipping (max_norm=1.0) prevents training spikes
  ✓ Optimizer: Adam (lr=1e-3), 10 epochs (fast ~45s/epoch on H100)
  ✓ Checkpointing: Stores dsp_mean/dsp_std directly inside checkpoint
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
    """Compute Equal Error Rate (EER) and its operating threshold."""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer, thresholds[idx]


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("VOICEGUARD TRAINING v4: DUAL-STREAM AUTHENTICITY DETECTOR")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Hardware: {gpu_name}")
    print("=" * 65)

    EPOCHS      = 10
    BATCH_SIZE  = 64
    LR          = 1e-3
    GRAD_CLIP   = 1.0
    NUM_WORKERS = 4
    CHECKPOINT  = "checkpoints/best_voice_auth_model.pt"
    os.makedirs("checkpoints", exist_ok=True)

    # -----------------------------------------------------------------
    # Datasets & Loaders
    # -----------------------------------------------------------------
    print("Initializing datasets...")
    train_ds = VoiceAuthDataset(split="train", augment=True, augment_prob=0.50)
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
    print(f"  [TRAIN] Loaded {len(train_ds):,} samples: {dict(train_counts)}")
    print(f"  [DEV]   Loaded {len(dev_ds):,} samples: {dict(dev_counts)}")

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------
    model = DualStreamDetector(dsp_dim=45, n_mels=80, num_classes=2).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {n_params:,}")

    # -----------------------------------------------------------------
    # Loss & Class Imbalance Balancing
    # -----------------------------------------------------------------
    n_fake = train_counts.get("fake", 0)
    n_real = train_counts.get("real", 0)
    weight_ratio = float(n_fake) / max(float(n_real), 1.0)
    class_weights = torch.tensor(
        [weight_ratio, 1.0], dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Banking Class Weights: real={weight_ratio:.2f}x, fake=1.00x")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # -----------------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------------
    best_eer   = float("inf")
    best_epoch = 0

    print("\nStarting 10-Epoch Optimization...")
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

            # Prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)

            optimizer.step()

            running_loss += loss.item()
            n_batches    += 1

        avg_loss = running_loss / max(n_batches, 1)

        # ---- Validation ----
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

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] ({elapsed:.1f}s) | "
            f"Train Loss: {avg_loss:.4f} | Dev EER: {dev_eer*100:.2f}% | "
            f"Dev ROC-AUC: {dev_auc*100:.2f}%"
        )

        if dev_eer < best_eer:
            best_eer   = dev_eer
            best_epoch = epoch
            ckpt = {
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dev_eer":              dev_eer,
                "dev_auc":              dev_auc,
                "dsp_mean":             train_ds.dsp_mean,
                "dsp_std":              train_ds.dsp_std,
                "version":              "v4_musan_rir_xtts",
            }
            torch.save(ckpt, CHECKPOINT)
            print(f"  ★ New Best Model Saved (EER: {dev_eer*100:.2f}%) -> {CHECKPOINT}")

    print("=" * 65)
    print(f"Training Complete. Best Validation EER: {best_eer*100:.2f}% (Epoch {best_epoch})")
    print(f"Model Checkpoint: {CHECKPOINT}")
    print("=" * 65)


if __name__ == "__main__":
    train()
