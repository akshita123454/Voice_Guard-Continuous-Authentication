"""
High-Speed Synthetic Voice Generator using Coqui TTS on NVIDIA H100.
Generates diverse AI-synthesized speech for the VoiceGuard training dataset.
"""

import os
import time
import csv
import soundfile as sf
from tqdm import tqdm
from TTS.api import TTS

# Target output directory
OUTPUT_DIR = "/data/user3/voice_guard/data/synthetic_voice"
MANIFEST_PATH = "/data/user3/voice_guard/data/synthetic_manifest.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Banking & Fraud Scenario Sentences (Diverse intent corpus)
# -------------------------------------------------------------
PROMPTS = [
    # Critical Fraud / Social Engineering Intent
    ("fraud_otp", "Sir, I am calling from the fraud prevention department. Your account is flagged. Please share the six-digit OTP immediately."),
    ("fraud_urgency", "Your debit card has been suspended due to suspicious activity. Act now or your account will be permanently closed."),
    ("fraud_rbi_police", "This is Officer Sharma from the Cyber Crime cell. A money laundering case has been registered under your Aadhaar number."),
    ("fraud_remote_access", "To fix your mobile banking issue, please install AnyDesk or QuickSupport from the Play Store and provide the nine-digit access code."),
    ("fraud_beneficiary", "Transfer the security deposit of two lakh rupees to this reserve account to avoid immediate tax department penalties."),
    ("fraud_pin_request", "To reverse the unauthorized charge of forty-five thousand rupees, confirm your ATM PIN and the CVV on the back of your card."),
    ("fraud_kyc_update", "Dear customer, your banking KYC has expired. Click the link sent on SMS and verify your net banking password right now."),
    ("fraud_lottery_prize", "Congratulations, you have won a cash reward of twenty-five lakh rupees from the bank anniversary lucky draw. Pay the processing fee to claim."),
    ("fraud_refund_lure", "We are trying to process your electricity bill refund of two thousand rupees. Kindly open Google Pay and enter your UPI PIN."),
    ("fraud_crypto_threat", "Illegal crypto transactions were detected from your savings account. You must liquidate the balance to the secured vault."),

    # Benign Routine Banking Intent (Normal Customers)
    ("benign_balance", "Hello, I would like to check the available balance in my salary account, please."),
    ("benign_statement", "Can you send the bank account statement for the last financial year to my registered email address?"),
    ("benign_cheque_book", "I want to place a request for a new cheque book with twenty-five leaves for my current account."),
    ("benign_card_upgrade", "Could you explain the annual fees and reward points for upgrading to the Platinum debit card?"),
    ("benign_branch_hours", "What are the working hours and lunch timings for the Indiranagar branch on second Saturdays?"),
    ("benign_fixed_deposit", "I want to create a fixed deposit of five lakh rupees for a tenure of three years at the current interest rate."),
    ("benign_home_loan", "Can you guide me on the documents required to apply for a pre-approved home loan?"),
    ("benign_address_change", "I have relocated to a new address. How do I update my communication address on my credit card?"),
    ("benign_international_tx", "Please enable international usage on my card as I am travelling overseas next week."),
    ("benign_closing_query", "Thank you so much for your assistance, that was very helpful. Have a great day ahead.")
]

# -------------------------------------------------------------
# 2. TTS Models to evaluate (Diverse AI architectures)
# -------------------------------------------------------------
MODELS_TO_RUN = [
    {
        "name": "tts_models/en/ljspeech/vits",
        "tag": "vits_single",
        "speakers": [None]
    },
    {
        "name": "tts_models/en/ljspeech/glow-tts",
        "tag": "glowtts_single",
        "speakers": [None]
    },
    {
        "name": "tts_models/en/ljspeech/tacotron2-DDC",
        "tag": "tacotron2_single",
        "speakers": [None]
    },
    {
        "name": "tts_models/en/vctk/vits",
        "tag": "vits_multispeaker",
        # Diverse speaker IDs from VCTK corpus (male/female, varying accents)
        "speakers": ["p225", "p226", "p227", "p228", "p230"]
    }
]

def main():
    print("=" * 70)
    print("VoiceGuard Synthetic Voice Generation Pipeline (NVIDIA H100)")
    print("=" * 70)

    manifest_rows = []
    total_generated = 0
    start_time = time.time()

    for model_cfg in MODELS_TO_RUN:
        model_name = model_cfg["name"]
        model_tag = model_cfg["tag"]
        speakers = model_cfg["speakers"]

        print(f"\n--> Loading Model: {model_name} onto GPU...")
        try:
            # Load model onto GPU (H100)
            tts = TTS(model_name=model_name, progress_bar=False, gpu=True)
        except Exception as e:
            print(f"[ERROR] Failed to load {model_name}: {e}")
            continue

        for speaker in speakers:
            spk_label = speaker if speaker else "default"
            print(f"    Synthesizing with Speaker: {spk_label}")

            for idx, (category, text) in enumerate(tqdm(PROMPTS, desc=f"    {model_tag}_{spk_label}")):
                filename = f"synth_{model_tag}_{spk_label}_{idx:03d}_{category}.wav"
                filepath = os.path.join(OUTPUT_DIR, filename)

                # Skip if already synthesized
                if not os.path.exists(filepath):
                    try:
                        if speaker:
                            tts.tts_to_file(text=text, speaker=speaker, file_path=filepath)
                        else:
                            tts.tts_to_file(text=text, file_path=filepath)
                    except Exception as e:
                        print(f"\n[Warning] Synthesis failed for '{text[:25]}...': {e}")
                        continue

                # Read audio duration
                try:
                    info = sf.info(filepath)
                    duration = round(info.duration, 2)
                except Exception:
                    duration = 0.0

                manifest_rows.append({
                    "audio_path": filepath,
                    "filename": filename,
                    "label": "fake",
                    "attack_type": model_tag,
                    "speaker_id": spk_label,
                    "category": category,
                    "duration_sec": duration,
                    "transcript": text
                })
                total_generated += 1

    # Write manifest CSV
    fieldnames = ["audio_path", "filename", "label", "attack_type", "speaker_id", "category", "duration_sec", "transcript"]
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print(f"Generation Complete!")
    print(f"Total Synthetic Audio Files: {total_generated}")
    print(f"Saved Directory: {OUTPUT_DIR}")
    print(f"Manifest Saved: {MANIFEST_PATH}")
    print(f"Elapsed Time: {elapsed} seconds")
    print("=" * 70)

if __name__ == "__main__":
    main()