# src/models/benchmark_zeroday_eval.py
"""
Zero-Day Attack Robustness Benchmark.
Evaluates the trained Dual-Stream model on the 17,001 unseen 'eval' split files.
Tests resistance against unknown vocoders, neural speech synthesizers, and voice conversion tools.
"""
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from src.models.voice_auth_dataset import VoiceAuthDataset
from src.models.dual_stream_detector import DualStreamDetector

def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer * 100.0, fpr, tpr, thresholds

def run_zeroday_benchmark():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "checkpoints/best_voice_auth_model.pt"

    print("=" * 65)
    print("VOICEGUARD ZERO-DAY ROBUSTNESS BENCHMARK (UNSEEN EVAL SPLIT)")
    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print("=" * 65)

    # 1. Load Model
    model = DualStreamDetector(dsp_dim=45).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from Epoch {checkpoint.get('epoch', 'N/A')} (Dev EER was {checkpoint.get('dev_eer', 0.0):.2f}%)")

    # 2. Load Evaluation Dataset (17,001 files)
    eval_dataset = VoiceAuthDataset(split="eval", augment=False)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )

    # 3. Run Inference on H100
    t0 = time.time()
    all_scores = []
    all_labels = []

    print(f"\nEvaluating {len(eval_dataset)} files across unknown attack algorithms...")
    with torch.no_grad():
        for dsp, mel, labels in eval_loader:
            dsp = dsp.to(device, non_blocking=True)
            mel = mel.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(dsp, mel)
                probs = torch.softmax(logits, dim=1)[:, 1] # Probability of being SPOOF

            all_scores.extend(probs.cpu().float().numpy())
            all_labels.extend(labels.numpy())

    elapsed = time.time() - t0
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # 4. Global Metrics
    eval_eer, fpr, tpr, thresholds = compute_eer(all_labels, all_scores)
    eval_auc = roc_auc_score(all_labels, all_scores) * 100.0

    # 5. Operating Point Analysis for Banking
    # Find operating thresholds at specific False Acceptance Rates (FAR)
    idx_far_01 = np.where(fpr <= 0.01)[0][-1]  # FAR <= 1.0% (Strict)
    idx_far_001 = np.where(fpr <= 0.001)[0][-1] if np.any(fpr <= 0.001) else 0 # FAR <= 0.1% (Critical)

    print("\n" + "=" * 65)
    print("           ZERO-DAY EVALUATION RESULTS (17,001 SAMPLES)")
    print("=" * 65)
    print(f"Total Evaluated:       {len(all_labels)} files in {elapsed:.1f} seconds")
    print(f"Throughput:            {len(all_labels)/elapsed:.1f} calls/sec")
    print(f"Equal Error Rate (EER): {eval_eer:.2f}%")
    print(f"Area Under ROC (AUC):  {eval_auc:.2f}%")
    print("-" * 65)
    print("BANKING OPERATING THRESHOLD TABLE:")
    print(f"  • Balanced Policy (EER Threshold):       Score >= {thresholds[np.argmin(np.abs(fpr - (1 - tpr)))]:.4f} -> FRR: {eval_eer:.2f}%, FAR: {eval_eer:.2f}%")
    print(f"  • Standard Wire Policy (FAR <= 1.0%):     Score >= {thresholds[idx_far_01]:.4f} -> FRR: {(1-tpr[idx_far_01])*100:.2f}%, FAR: {fpr[idx_far_01]*100:.2f}%")
    if idx_far_001 > 0:
        print(f"  • High-Value RTGS Policy (FAR <= 0.1%):   Score >= {thresholds[idx_far_001]:.4f} -> FRR: {(1-tpr[idx_far_001])*100:.2f}%, FAR: {fpr[idx_far_001]*100:.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    run_zeroday_benchmark()