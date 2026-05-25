"""
prepare_audio.py

Reads every WAV file from sounds/raw/, normalises peak amplitude,
and writes to sounds/clean/ at natural length — no trimming or padding.

Run once before training:
    python prepare_audio.py
"""

import os
import numpy as np
import soundfile as sf

RAW_DIR   = "sounds/raw"
CLEAN_DIR = "sounds/clean"
TARGET_SR = 48000


def process_file(src_path: str, dst_path: str) -> float:
    """Returns duration in seconds."""
    audio, sr = sf.read(src_path, always_2d=False)

    # Mix down to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != TARGET_SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    audio = audio.astype(np.float32)

    # Peak normalise — skip silent files
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak

    sf.write(dst_path, audio, TARGET_SR, subtype="PCM_24")
    return len(audio) / TARGET_SR


def main():
    os.makedirs(CLEAN_DIR, exist_ok=True)

    files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".wav")]
    if not files:
        print(f"No WAV files found in {RAW_DIR}")
        return

    print(f"Processing {len(files)} files...")
    skipped  = []
    durations = []

    for fname in sorted(files):
        src = os.path.join(RAW_DIR, fname)
        dst = os.path.join(CLEAN_DIR, fname)
        try:
            dur = process_file(src, dst)
            durations.append(dur)
            print(f"  OK  {fname}  ({dur:.2f}s)")
        except Exception as e:
            print(f"  SKIP {fname} — {e}")
            skipped.append(fname)

    print(f"\nDone. {len(files) - len(skipped)} processed, {len(skipped)} skipped.")
    if durations:
        print(f"Duration — min: {min(durations):.2f}s  "
              f"max: {max(durations):.2f}s  "
              f"mean: {sum(durations)/len(durations):.2f}s")


if __name__ == "__main__":
    main()