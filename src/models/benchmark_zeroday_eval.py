# src/models/benchmark_zeroday_eval.py
"""
VoiceGuard Zero-Day Robustness Benchmark (Unseen Eval Split A07-A19)

Strict Banking Evaluation:
  ✓ Evaluates 17,001 unseen samples (unknown attack algorithms)
  ✓ Computes Equal Error Rate (EER) and Area Under ROC (AUC)
  ✓ Calculates exact False Rejection Rate (FRR) at FAR <= 1.0% (Wire Transfer)
  ✓ Calculates exact False Rejection Rate (FRR) at FAR <= 0.1% (High-Value RTGS)
"""
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

from voice_auth_dataset import VoiceAuthDataset
from dual_stream_detector import DualStreamDetector


def compute_eer(labels, scores):
    """Compute Equal Error Rate and its operating threshold."""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer, thresholds[idx]


def find_threshold_at_far(labels, scores, target_far):
    """Find operating threshold where False Acceptance Rate <= target_far."""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    valid = np.where(fpr <= target_far)[0]
    if len(valid) == 0:
        return thresholds[-1], fnr[-1], fpr[-1]
    idx = valid[-1]
    return thresholds[idx], fnr[idx], fpr[idx]


def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "checkpoints/best_voice_auth_model.pt"

    print("=" * 65)
    print("VOICEGUARD ZERO-DAY ROBUSTNESS BENCHMARK (UNSEEN EVAL SPLIT)")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Hardware: {gpu_name}")
    print("=" * 65)

    # 1. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DualStreamDetector(dsp_dim=45, n_mels=80, num_classes=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    dev_eer = checkpoint.get("dev_eer", 0.0)
    version = checkpoint.get("version", "v4")
    print(f"Loaded [{version}] checkpoint from Epoch {epoch} (Dev EER was {dev_eer*100:.2f}%)")

    # 2. Load Evaluation Dataset
    eval_ds = VoiceAuthDataset(split="eval", augment=False)
    eval_loader = DataLoader(
        eval_ds, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    eval_counts = eval_ds.df["label"].value_counts()
    print(f"  [EVAL] Loaded {len(eval_ds):,} samples: {dict(eval_counts)}")
    print(f"\nEvaluating {len(eval_ds):,} files across unknown attack algorithms...")

    # 3. Benchmark Inference
    t0 = time.time()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for mel, dsp, labels in eval_loader:
            mel = mel.to(device)
            dsp = dsp.to(device)
            logits = model(mel, dsp)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(fake)
            all_scores.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    elapsed = time.time() - t0
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # 4. Metrics & Threshold Tables
    eer, eer_thresh = compute_eer(all_labels, all_scores)
    auc = roc_auc_score(all_labels, all_scores)
    throughput = len(eval_ds) / elapsed

    std_thresh, std_frr, std_far = find_threshold_at_far(all_labels, all_scores, 0.01)
    rtgs_thresh, rtgs_frr, rtgs_far = find_threshold_at_far(all_labels, all_scores, 0.001)

    print(f"\n{'='*65}")
    print(f"           ZERO-DAY EVALUATION RESULTS ({len(eval_ds):,} SAMPLES)")
    print(f"{'='*65}")
    print(f"Total Evaluated:       {len(eval_ds):,} files in {elapsed:.1f} seconds")
    print(f"Throughput:            {throughput:.1f} calls/sec")
    print(f"Equal Error Rate (EER): {eer*100:.2f}%")
    print(f"Area Under ROC (AUC):  {auc*100:.2f}%")
    print(f"-" * 65)
    print(f"BANKING OPERATING THRESHOLD TABLE:")
    print(f"  • Balanced Policy (EER Threshold):       Score >= {eer_thresh:.4f} -> FRR: {eer*100:.2f}%, FAR: {eer*100:.2f}%")
    print(f"  • Standard Wire Policy (FAR <= 1.0%):     Score >= {std_thresh:.4f} -> FRR: {std_frr*100:.2f}%, FAR: {std_far*100:.2f}%")
    print(f"  • High-Value RTGS Policy (FAR <= 0.1%):   Score >= {rtgs_thresh:.4f} -> FRR: {rtgs_frr*100:.2f}%, FAR: {rtgs_far*100:.2f}%")
    print(f"{'='*65}")


if __name__ == "__main__":
    benchmark()
