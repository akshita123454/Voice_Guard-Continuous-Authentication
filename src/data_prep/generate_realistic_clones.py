"""
Ultra-Realistic Voice Cloning using XTTS-v2 on NVIDIA H100.
Clones real speakers from LibriSpeech to produce human-parity deepfakes.
"""

import os
import glob
import soundfile as sf
from tqdm import tqdm
from TTS.api import TTS

OUTPUT_DIR = "/data/user3/voice_guard/data/synthetic_voice/realistic_clones"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Grab 3 real human reference voices from your downloaded LibriSpeech dataset
LIBRISPEECH_DIR = "/data/user3/voice_guard/data/librispeech/LibriSpeech/train-clean-100"
reference_audio_files = glob.glob(f"{LIBRISPEECH_DIR}/*/*/*.flac")

if len(reference_audio_files) == 0:
    print(f"Error: No LibriSpeech audio found in {LIBRISPEECH_DIR}. Checking dev-clean...")
    reference_audio_files = glob.glob("/data/user3/voice_guard/data/librispeech/LibriSpeech/*/*/*/*.flac")

# Select 3 distinct real human reference samples
reference_samples = reference_audio_files[:3]
print(f"Selected {len(reference_samples)} real human voices to clone:")
for ref in reference_samples:
    print(f"  - {os.path.basename(ref)}")

# 2. Ultra-realistic conversational banking scripts (with pauses and natural phrasing)
REALISTIC_SCRIPTS = [
    ("fraud_otp", "Hello? Yes, ma'am, I'm calling from the bank's security unit. We just flagged an unauthorized payment of forty-five thousand rupees. Please read out the OTP you just received so I can cancel the transaction immediately."),
    ("fraud_urgent_transfer", "Listen to me very carefully. Your account is facing a temporary legal lien. You need to move your remaining balance of two lakh rupees to our safe escrow account right now, otherwise it will be frozen for six months."),
    ("fraud_remote_access", "Sir, your mobile banking app is malfunctioning because of a corrupted security certificate. Kindly open the Play Store, download AnyDesk, and give me the code displayed on your screen so our engineers can fix it."),
    ("benign_card_query", "Hi there, good morning. I noticed a small international transaction fee on my monthly statement, and I just wanted to verify whether my debit card has international roaming enabled by default?"),
    ("benign_loan_status", "Hello, I submitted an application for a pre-approved car loan two days ago at the Indiranagar branch. Could you please check the current status and let me know if any more documents are required?")
]

def main():
    print("\n--> Loading XTTS-v2 (State-of-the-Art Neural Voice Cloner) onto H100...")
    # XTTS-v2 produces near-human speech
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

    print("\n--> Generating Human-Parity Deepfakes...")
    generated_count = 0

    for speaker_idx, speaker_wav in enumerate(reference_samples):
        speaker_id = os.path.basename(speaker_wav).split("-")[0]

        for tag, text in tqdm(REALISTIC_SCRIPTS, desc=f"Cloning Speaker {speaker_id}"):
            out_filename = f"xtts_clone_spk{speaker_id}_{tag}.wav"
            out_filepath = os.path.join(OUTPUT_DIR, out_filename)

            if os.path.exists(out_filepath):
                continue

            try:
                # XTTS clones the reference audio's exact voice, tone, and accent
                tts.tts_to_file(
                    text=text,
                    speaker_wav=speaker_wav,
                    language="en",
                    file_path=out_filepath
                )
                generated_count += 1
            except Exception as e:
                print(f"Failed to synthesize: {e}")

    print("\n" + "=" * 65)
    print(f"Finished! Generated {generated_count} ultra-realistic cloned audios.")
    print(f"Saved in: {OUTPUT_DIR}")
    print("=" * 65)

if __name__ == "__main__":
    main()
    