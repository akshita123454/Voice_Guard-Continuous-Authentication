# src/models/train_voice_authenticity.py
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from src.models.voice_auth_dataset import VoiceAuthDataset
from src.models.dual_stream_detector import DualStreamDetector

def compute_eer(labels, scores):
    """
    Computes Equal Error Rate (EER) - the gold standard metric for biometric anti-spoofing.
    scores: probabilities of being SPOOF (class 1).
    labels: 0 for real, 1 for spoof.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    # EER is point where FPR == FNR
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer * 100.0

def train_and_evaluate():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"VOICEGUARD TRAINING: DUAL-STREAM AUTHENTICITY DETECTOR")
    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print("=" * 65)

    # 1. Datasets
    train_dataset = VoiceAuthDataset(split="train", augment=True)
    dev_dataset = VoiceAuthDataset(split="dev", augment=False)

    # H100 has 80GB VRAM -> batch_size=128 runs smoothly with high GPU utilization
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        prefetch_factor=2
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )

    # 2. Class Weighting to neutralize the 8.85:1 imbalance
    # Penalty for missing a real sample is 8.85x higher
    class_weights = torch.tensor([8.85, 1.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 3. Model, Optimizer, Scheduler
    model = DualStreamDetector(dsp_dim=45).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    epochs = 10
    best_dev_eer = float("inf")
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/best_voice_auth_model.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        for dsp, mel, labels in train_loader:
            dsp = dsp.to(device, non_blocking=True)
            mel = mel.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Native H100 BFloat16 mixed precision
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(dsp, mel)
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * dsp.size(0)

        epoch_loss = running_loss / len(train_dataset)
        scheduler.step()

        # --- Validation on Unseen Dev Set (24,844 samples) ---
        model.eval()
        dev_scores = []
        dev_labels = []

        with torch.no_grad():
            for dsp, mel, labels in dev_loader:
                dsp = dsp.to(device, non_blocking=True)
                mel = mel.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(dsp, mel)
                    probs = torch.softmax(logits, dim=1)[:, 1] # Probability of being SPOOF

                dev_scores.extend(probs.cpu().float().numpy())
                dev_labels.extend(labels.numpy())

        dev_scores = np.array(dev_scores)
        dev_labels = np.array(dev_labels)

        # Compute EER & AUC
        dev_eer = compute_eer(dev_labels, dev_scores)
        dev_auc = roc_auc_score(dev_labels, dev_scores) * 100.0
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | Train Loss: {epoch_loss:.4f} | Dev EER: {dev_eer:.2f}% | Dev ROC-AUC: {dev_auc:.2f}%")

        if dev_eer < best_dev_eer:
            best_dev_eer = dev_eer
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dev_eer': dev_eer,
                'dev_auc': dev_auc
            }, checkpoint_path)
            print(f"  ★ New Best Model Saved (EER: {dev_eer:.2f}%) -> {checkpoint_path}")

    print("=" * 65)
    print(f"Training Complete. Best Validation EER: {best_dev_eer:.2f}%")
    print(f"Model Checkpoint: {checkpoint_path}")
    print("=" * 65)

if __name__ == "__main__":
    train_and_evaluate()