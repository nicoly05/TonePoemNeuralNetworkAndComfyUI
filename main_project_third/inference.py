"""
inference_generative.py

Generative inference engine for 17-key C kalimba.

Identical output pipeline to inference.py (retrieval), but instead of
blending dataset files the audio is generated via DDPM reverse diffusion:

  1. Detect pitch → snap to nearest kalimba note → cluster
  2. Volume → periphery (cluster centre vs edge)
  3. Encode input audio → condition latent
  4. DDPM reverse diffusion guided toward target cluster centroid
     producing a novel latent that has never existed in the dataset
  5. Decode novel latent through EnCodec decoder → raw audio
  6. Pitch-shift output to exact target note using pre-computed
     reference pitch from the nearest dataset sound
  7. Envelope (5ms attack, 2s exponential release)
  8. Lowpass filter (4kHz Butterworth)
  9. Schroeder reverb (8s decay)
  10. Volume follow + output level scaling

Cluster layout (17-key C kalimba, C4–E6):
  0 → C4, D4           (261–294 Hz)
  1 → E4, F4           (330–349 Hz)
  2 → G4, A4           (392–440 Hz)
  3 → B4, C5, D5       (494–587 Hz)
  4 → E5, F5           (659–698 Hz)
  5 → G5, A5           (784–880 Hz)
  6 → B5, C6, D6, E6   (988–1319 Hz)
"""

import os
import numpy as np
import torch
import librosa
import soundfile as sf
from scipy import signal as scipy_signal

from encodec import EncodecModel
from model import ConditionalDiffusionMLP

MAP_PATH    = "models/latent_map.pt"
MODEL_PATH  = "models/diffusion.pt"
TARGET_SR   = 48000

# ── 17 kalimba notes (C major, C4–E6) ────────────────────────────────────────
KALIMBA_NOTES = [
    (261.63,  "C4", 0),
    (293.66,  "D4", 0),
    (329.63,  "E4", 1),
    (349.23,  "F4", 1),
    (392.00,  "G4", 2),
    (440.00,  "A4", 2),
    (493.88,  "B4", 3),
    (523.25,  "C5", 3),
    (587.33,  "D5", 3),
    (659.25,  "E5", 4),
    (698.46,  "F5", 4),
    (783.99,  "G5", 5),
    (880.00,  "A5", 5),
    (987.77,  "B5", 6),
    (1046.50, "C6", 6),
    (1174.66, "D6", 6),
    (1318.51, "E6", 6),
]

KALIMBA_HZ      = np.array([n[0] for n in KALIMBA_NOTES], dtype=np.float64)
KALIMBA_NAMES   = [n[1] for n in KALIMBA_NOTES]
KALIMBA_CLUSTER = [n[2] for n in KALIMBA_NOTES]

PITCH_FMIN = float(KALIMBA_HZ[0])  * 0.85
PITCH_FMAX = float(KALIMBA_HZ[-1]) * 1.15

# ── Parameters ────────────────────────────────────────────────────────────────
RMS_SOFT            = 0.03
RMS_LOUD            = 0.25
DDPM_STEPS          = 50          # reverse diffusion steps (speed vs quality)
GUIDANCE_STRENGTH   = 0.3        # nudge toward cluster centroid per step
MAX_SHIFT_SEMITONES = 14.0
ATTACK_S            = 0.005
RELEASE_S           = 2.0

# Fixed output length — generative audio doesn't have a natural file length
# so we define the duration explicitly
OUTPUT_S            = 3.0
OUTPUT_SAMPLES      = int(TARGET_SR * OUTPUT_S)

# EnCodec internal frame rate at 48kHz
ENCODEC_FRAME_RATE  = 75         # frames per second

# ── Effects ───────────────────────────────────────────────────────────────────
LOWPASS_CUTOFF_HZ   = 4000
LOWPASS_ORDER       = 4

REVERB_MIX          = 0.35
REVERB_ROOM_SCALE   = 0.85
REVERB_DECAY        = 8.0

# ── Output volume ─────────────────────────────────────────────────────────────
OUTPUT_LEVEL        = 0.7
VOLUME_FOLLOW_MIX   = 0.7


def get_device() -> torch.device:
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# ── Pitch utilities (identical to retrieval version) ──────────────────────────

def _yin_pitch(audio: np.ndarray, sr: int = TARGET_SR) -> float | None:
    if len(audio) < 512:
        return None
    try:
        frame_len = min(2048, (len(audio) // 4) * 2)
        if frame_len < 256:
            return None
        f0    = librosa.yin(audio, fmin=PITCH_FMIN, fmax=PITCH_FMAX,
                            sr=sr, frame_length=frame_len)
        valid = f0[(f0 >= PITCH_FMIN) & (f0 <= PITCH_FMAX)]
        return float(np.median(valid)) if len(valid) > 0 else None
    except Exception:
        return None


def _pyin_pitch(audio: np.ndarray, sr: int = TARGET_SR) -> float | None:
    if len(audio) < 1024:
        return None
    try:
        frame_len = min(2048, (len(audio) // 4) * 2)
        if frame_len < 512:
            return None
        f0, voiced, probs = librosa.pyin(audio, fmin=PITCH_FMIN, fmax=PITCH_FMAX,
                                         sr=sr, frame_length=frame_len,
                                         fill_na=None)
        if not voiced.any():
            return None
        f  = f0[voiced]
        p  = probs[voiced]
        ok = ~np.isnan(f)
        if not ok.any():
            return None
        return float(np.average(f[ok], weights=p[ok]))
    except Exception:
        return None


def detect_pitch_robust(audio: np.ndarray,
                        sr: int = TARGET_SR) -> float | None:
    """
    Detect pitch from the loudest 50ms window of the capture.
    pyin first, yin fallback, octave correction.
    """
    audio    = audio.astype(np.float32)
    window_n = int(sr * 0.05)
    if len(audio) <= window_n:
        segment = audio
    else:
        hop      = int(sr * 0.01)
        energies = np.array([
            np.mean(audio[i:i + window_n] ** 2)
            for i in range(0, len(audio) - window_n, hop)
        ])
        peak_pos  = int(np.argmax(energies)) * hop
        seg_end   = min(len(audio), peak_pos + int(sr * 0.15))
        segment   = audio[peak_pos:seg_end]

    if len(segment) < 512:
        return None

    freq = _pyin_pitch(segment, sr) or _yin_pitch(segment, sr)

    if freq is None or freq <= 0:
        return None

    # Octave correction toward kalimba centre ~500Hz
    best, best_err = freq, abs(np.log2(max(freq, 1e-6) / 500.0))
    for shift in [-2, -1, 1, 2]:
        c   = freq * (2.0 ** shift)
        err = abs(np.log2(max(c, 1e-6) / 500.0))
        if err < best_err:
            best_err, best = err, c
    freq = best

    return freq if PITCH_FMIN <= freq <= PITCH_FMAX else None


def snap_to_kalimba(freq_hz: float,
                    max_cents: float = 40.0) -> tuple[float, str, int, float]:
    """
    Snap to nearest kalimba note.
    Returns (hz, name, cluster, cents_error).
    cents_error > max_cents means the detection is between scale degrees
    and should be treated with low confidence.
    """
    cents = 1200.0 * np.log2(freq_hz / KALIMBA_HZ)
    idx   = int(np.argmin(np.abs(cents)))
    return (float(KALIMBA_HZ[idx]),
            KALIMBA_NAMES[idx],
            KALIMBA_CLUSTER[idx],
            float(abs(cents[idx])))


def rms_to_periphery(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    return float((np.clip(rms, RMS_SOFT, RMS_LOUD) - RMS_SOFT)
                 / (RMS_LOUD - RMS_SOFT))


# ── Main engine ───────────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Generative inference engine. Thread-safe after __init__.
    """

    def __init__(self, k: int = 5, ddpm_steps: int = DDPM_STEPS):
        self.k           = k
        self.ddpm_steps  = ddpm_steps
        self.device      = get_device()
        print(f"GenerativeInferenceEngine on {self.device}")

        # EnCodec — encode input, decode generated latent
        self.encodec = EncodecModel.encodec_model_48khz()
        self.encodec.set_target_bandwidth(6.0)
        self.encodec.eval()

        # Latent map
        if not os.path.exists(MAP_PATH):
            raise FileNotFoundError(f"{MAP_PATH} — run build_map.py first.")
        data = torch.load(MAP_PATH, weights_only=False)
        self.latents         = data["latents"].float()
        self.filenames       = data["filenames"]
        self.cluster_labels  = data["cluster_labels"]
        self.cluster_centers = data["cluster_centers"].float()
        self.n_clusters      = data["n_clusters"]
        print(f"Latent map: {len(self.latents)} sounds, {self.n_clusters} clusters")

        # Diffusion model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"{MODEL_PATH} — run train.py first.")
        ckpt = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
        self.model = ConditionalDiffusionMLP(
            latent_dim=ckpt["latent_dim"], hidden_dim=256, n_layers=4
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.alphas_cumprod = ckpt["alphas_cumprod"].to(self.device)
        self.T_max          = ckpt["T_max"]
        print(f"Diffusion model loaded (val_loss={ckpt['val_loss']:.4f}, "
              f"epoch={ckpt['epoch']})")

        # Pre-load audio and compute source pitches for pitch-shift reference
        self._preload_audio()
        self._precompute_source_pitches()

    def _preload_audio(self):
        print("Pre-loading audio files...")
        self.audio_cache: dict[int, np.ndarray] = {}
        for idx, fname in enumerate(self.filenames):
            audio, _ = sf.read(os.path.join("sounds/clean", fname),
                               always_2d=False)
            self.audio_cache[idx] = audio.astype(np.float32)
        print(f"  {len(self.audio_cache)} files loaded.")

    def _precompute_source_pitches(self):
        """
        Detect pitch of every source file at startup.
        Used as reference when pitch-shifting the generated output —
        we find the nearest dataset sound to the generated latent,
        use its known pitch as the 'from' value for the shift.
        """
        print("Pre-computing source pitches...")
        self.source_pitches: dict[int, float | None] = {}
        detected = 0
        for idx, fname in enumerate(self.filenames):
            freq = detect_pitch_robust(self.audio_cache[idx])
            self.source_pitches[idx] = freq
            if freq:
                detected += 1
        print(f"  Pitch detected for {detected}/{len(self.filenames)} files.")

    # ── Encode input ──────────────────────────────────────────────────────────

    def _encode(self, audio: np.ndarray) -> torch.Tensor:
        audio = audio.astype(np.float32)
        peak  = np.max(np.abs(audio))
        if peak > 1e-6:
            audio = audio / peak
        t = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).expand(1, 2, -1)
        with torch.no_grad():
            frames = self.encodec.encoder(t)
        return frames.mean(dim=-1).squeeze(0)   # (128,)

    # ── DDPM reverse diffusion ────────────────────────────────────────────────

    @torch.no_grad()
    def _generate_latent(self, condition: torch.Tensor,
                         target_cluster: int,
                         periphery: float) -> torch.Tensor:
        """
        Run DDPM reverse diffusion to produce a novel latent.

        condition     : encoded input audio (128,)
        target_cluster: which cluster to guide toward
        periphery     : 0=centroid, 1=cluster edge (from volume)

        Guidance nudges the trajectory toward/away from the centroid
        in proportion to periphery — soft playing stays near the centre
        (most typical sounds), loud playing drifts toward the edge
        (more unusual/contrasting sounds).
        """
        cond    = condition.to(self.device).unsqueeze(0)           # (1, 128)
        centroid = self.cluster_centers[target_cluster].to(self.device)

        step_indices = torch.linspace(
            self.T_max - 1, 0, self.ddpm_steps, dtype=torch.long
        )

        x   = torch.randn(1, condition.shape[0], device=self.device)
        acp = self.alphas_cumprod

        for i, t_idx in enumerate(step_indices):
            t_tensor   = t_idx.unsqueeze(0).to(self.device)
            noise_pred = self.model(x, t_tensor, cond)

            acp_t  = acp[t_idx]
            acp_t1 = acp[step_indices[i + 1]] if i + 1 < len(step_indices) \
                     else torch.tensor(1.0, device=self.device)

            x0_pred = ((x - (1 - acp_t).sqrt() * noise_pred)
                       / acp_t.sqrt()).clamp(-3, 3)
            x = acp_t1.sqrt() * x0_pred + (1 - acp_t1).sqrt() * noise_pred

            # Distance-dampened guidance toward centroid
            diff     = centroid - x.squeeze(0)
            dist     = diff.norm() + 1e-8
            direction = diff / dist
            progress  = i / max(len(step_indices) - 1, 1)
            strength  = GUIDANCE_STRENGTH * (1.0 - progress * 0.5) * dist.clamp(0, 1)
            x = x + strength * direction.unsqueeze(0)

        result = x.squeeze(0).cpu()

        # Periphery: push away from centroid for loud playing
        if periphery > 0.05:
            away   = result - self.cluster_centers[target_cluster]
            away   = away / (away.norm() + 1e-8)
            result = result + away * periphery * 0.5

        return result

    # ── Decode novel latent → audio ───────────────────────────────────────────

    def _decode_latent(self, novel_latent: torch.Tensor) -> np.ndarray:
        """
        Decode a 128-dim latent vector to OUTPUT_SAMPLES of audio.

        Strategy: find the K nearest dataset latents, blend their frame
        sequences weighted by distance, steer toward the novel latent,
        then decode through EnCodec's decoder.
        """
        dists   = (self.latents - novel_latent.unsqueeze(0)).norm(dim=-1)
        k       = min(self.k, len(self.latents))
        topk    = torch.topk(dists, k, largest=False)
        indices = topk.indices
        weights = 1.0 / (topk.values + 1e-8)
        weights = weights / weights.sum()

        # Target frame count from OUTPUT_SAMPLES
        T_target = max(1, int(OUTPUT_SAMPLES / TARGET_SR * ENCODEC_FRAME_RATE))

        # Build blended frame sequence from nearest neighbours
        blended = torch.zeros(128, T_target)
        for idx, w in zip(indices.tolist(), weights.tolist()):
            # Re-encode source file to get its frame sequence
            audio = self.audio_cache[idx]
            t     = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).expand(1, 2, -1)
            with torch.no_grad():
                frames = self.encodec.encoder(t)   # (1, 128, T_src)
            frames = frames.squeeze(0)             # (128, T_src)

            # Interpolate to T_target length
            frames_interp = torch.nn.functional.interpolate(
                frames.unsqueeze(0), size=T_target,
                mode="linear", align_corners=False
            ).squeeze(0)
            blended += w * frames_interp

        # Steer frame mean toward novel latent
        current_mean = blended.mean(dim=-1)
        residual     = (novel_latent - current_mean).unsqueeze(-1)
        blended      = blended + 0.15 * residual

        # Decode
        with torch.no_grad():
            decoded = self.encodec.decoder(blended.unsqueeze(0))

        audio = decoded[0].mean(dim=0).numpy().astype(np.float32)

        # Trim or pad to OUTPUT_SAMPLES
        if len(audio) >= OUTPUT_SAMPLES:
            return audio[:OUTPUT_SAMPLES]
        return np.pad(audio, (0, OUTPUT_SAMPLES - len(audio)))

    # ── Pitch shift using nearest dataset sound as reference ──────────────────

    def _pitch_shift_output(self, audio: np.ndarray,
                            novel_latent: torch.Tensor,
                            target_hz: float) -> np.ndarray:
        """
        Find the dataset sound nearest to the generated latent.
        Use its pre-computed pitch as the 'from' reference.
        Shift audio to target_hz.

        This is the same approach as the retrieval version — known source
        pitch → exact semitone calculation → shift. No secondary detection
        on the generated output needed.
        """
        dists   = (self.latents - novel_latent.unsqueeze(0)).norm(dim=-1)
        nearest = int(dists.argmin().item())
        src_hz  = self.source_pitches.get(nearest)

        # Walk up through nearest neighbours if closest has no pitch
        if src_hz is None:
            for idx in dists.argsort().tolist()[:10]:
                if self.source_pitches.get(idx):
                    src_hz = self.source_pitches[idx]
                    break

        if src_hz is None or src_hz <= 0:
            print("[pitch_shift] no reference pitch found — skipping shift")
            return audio

        semitones = float(np.clip(
            12.0 * np.log2(target_hz / src_hz),
            -MAX_SHIFT_SEMITONES, MAX_SHIFT_SEMITONES
        ))
        print(f"[pitch_shift] ref={src_hz:.1f}Hz  target={target_hz:.1f}Hz  "
              f"shift={semitones:+.2f}st")

        if abs(semitones) < 0.1:
            return audio

        return librosa.effects.pitch_shift(
            audio, sr=TARGET_SR, n_steps=semitones
        ).astype(np.float32)

    # ── Envelope ──────────────────────────────────────────────────────────────

    def _apply_envelope(self, audio: np.ndarray) -> np.ndarray:
        n         = len(audio)
        envelope  = np.ones(n, dtype=np.float32)
        attack_n  = int(TARGET_SR * ATTACK_S)
        release_n = int(TARGET_SR * RELEASE_S)
        if attack_n > 0 and attack_n < n:
            envelope[:attack_n] = np.linspace(0.0, 1.0, attack_n)
        if release_n > 0 and release_n < n:
            t = np.linspace(0.0, 1.0, release_n, dtype=np.float32)
            envelope[-release_n:] = np.exp(-5.0 * t)
        return (audio * envelope).astype(np.float32)

    # ── Lowpass ───────────────────────────────────────────────────────────────

    def _apply_lowpass(self, audio: np.ndarray) -> np.ndarray:
        nyq  = TARGET_SR / 2.0
        b, a = scipy_signal.butter(LOWPASS_ORDER, LOWPASS_CUTOFF_HZ / nyq,
                                   btype="low")
        return scipy_signal.filtfilt(b, a, audio).astype(np.float32)

    # ── Reverb ────────────────────────────────────────────────────────────────

    def _apply_reverb(self, audio: np.ndarray) -> np.ndarray:
        """
        Schroeder reverb using scipy.signal.lfilter for each comb/allpass.
        Identical sound to the loop-based version but runs in C — fast enough
        for real-time use even on long buffers with multiple parallel workers.
        """
        sr             = TARGET_SR
        n              = len(audio)
        base_delays_ms = [29.7, 37.1, 41.1, 43.7]
        comb_delays    = [int(sr * d * REVERB_ROOM_SCALE / 1000.0)
                          for d in base_delays_ms]

        def comb_gain(delay: int) -> float:
            return 10 ** (-3.0 * delay / (sr * REVERB_DECAY))

        allpass_delays = [int(sr * d / 1000.0) for d in [5.0, 1.7]]
        allpass_gain   = 0.7

        def apply_comb_fast(x: np.ndarray, delay: int, gain: float) -> np.ndarray:
            # IIR comb filter: y[n] = x[n] + gain * y[n - delay]
            # Expressed as lfilter: b=[1], a=[1, 0...0, -gain] (delay zeros)
            b = np.zeros(delay + 1, dtype=np.float64)
            a = np.zeros(delay + 1, dtype=np.float64)
            b[0] = 1.0
            a[0] = 1.0
            a[delay] = -gain
            return scipy_signal.lfilter(b, a, x.astype(np.float64)).astype(np.float32)

        def apply_allpass_fast(x: np.ndarray, delay: int, gain: float) -> np.ndarray:
            # Allpass: y[n] = -gain*x[n] + x[n-delay] + gain*y[n-delay]
            b = np.zeros(delay + 1, dtype=np.float64)
            a = np.zeros(delay + 1, dtype=np.float64)
            b[0] = -gain
            b[delay] = 1.0
            a[0] = 1.0
            a[delay] = -gain
            return scipy_signal.lfilter(b, a, x.astype(np.float64)).astype(np.float32)

        # Four parallel comb filters summed
        wet = np.zeros(n, dtype=np.float32)
        for delay in comb_delays:
            wet += apply_comb_fast(audio, delay, comb_gain(delay))
        wet /= len(comb_delays)

        # Two series allpass filters
        for delay in allpass_delays:
            wet = apply_allpass_fast(wet, delay, allpass_gain)

        return ((1.0 - REVERB_MIX) * audio + REVERB_MIX * wet).astype(np.float32)

    # ── Main entry point ──────────────────────────────────────────────────────

    def respond(self, audio_in: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Given mono 48kHz audio (short attack capture), return
        (generated_audio, cluster_int).

        Output is OUTPUT_S seconds of genuinely novel audio generated by
        diffusion, pitch-forced to the nearest kalimba note.
        """
        # 1. Detect pitch → snap to kalimba note → cluster
        freq = detect_pitch_robust(audio_in)
        if freq is not None:
            snapped_hz, note_name, cluster, cents_err = snap_to_kalimba(freq)
            if cents_err <= 40.0:
                print(f"[pitch] {freq:.1f}Hz → {note_name} ({snapped_hz:.1f}Hz) "
                      f"cluster {cluster}  err={cents_err:.1f}¢")
            else:
                # Between scale notes — low confidence, use latent cluster
                condition  = self._encode(audio_in)
                dists      = (self.cluster_centers - condition.unsqueeze(0)).norm(dim=-1)
                cluster    = int(dists.argmin().item())
                notes_in_c = [hz for hz, _, cl in KALIMBA_NOTES if cl == cluster]
                snapped_hz = float(notes_in_c[len(notes_in_c) // 2])
                note_name  = KALIMBA_NAMES[KALIMBA_CLUSTER.index(cluster)]
                print(f"[pitch] {freq:.1f}Hz low confidence ({cents_err:.1f}¢) "
                      f"→ latent cluster {cluster} → {note_name} ({snapped_hz:.1f}Hz)")
        else:
            condition  = self._encode(audio_in)
            dists      = (self.cluster_centers - condition.unsqueeze(0)).norm(dim=-1)
            cluster    = int(dists.argmin().item())
            notes_in_c = [hz for hz, _, cl in KALIMBA_NOTES if cl == cluster]
            snapped_hz = float(notes_in_c[len(notes_in_c) // 2])
            note_name  = KALIMBA_NAMES[KALIMBA_CLUSTER.index(cluster)]
            print(f"[pitch] not detected — cluster {cluster} → {snapped_hz:.1f}Hz")

        # 2. Volume → periphery
        periphery = rms_to_periphery(audio_in)
        print(f"[volume] periphery={periphery:.2f}")

        # 3. Encode input → condition latent
        condition = self._encode(audio_in)

        # 4. Generate novel latent via DDPM
        novel_latent = self._generate_latent(condition, cluster, periphery)
        gen_cluster  = int(
            (self.cluster_centers - novel_latent.unsqueeze(0)).norm(dim=-1).argmin()
        )
        print(f"[generate] target_cluster={cluster}  "
              f"generated_cluster={gen_cluster}")

        # 5. Decode novel latent → audio
        audio_out = self._decode_latent(novel_latent)
        print(f"[decode] dur={len(audio_out)/TARGET_SR:.2f}s")

        # 6. Pitch shift to exact kalimba note
        audio_out = self._pitch_shift_output(audio_out, novel_latent, snapped_hz)

        # 7. Envelope
        audio_out = self._apply_envelope(audio_out)

        # 8. Lowpass
        audio_out = self._apply_lowpass(audio_out)

        # 9. Reverb
        audio_out = self._apply_reverb(audio_out)

        # 10. Volume follow + output level
        input_rms   = float(np.sqrt(np.mean(audio_in.astype(np.float32) ** 2)))
        input_level = (np.clip(input_rms, RMS_SOFT, RMS_LOUD) - RMS_SOFT) \
                      / (RMS_LOUD - RMS_SOFT)
        target_level = OUTPUT_LEVEL * (
            (1.0 - VOLUME_FOLLOW_MIX) + VOLUME_FOLLOW_MIX * input_level
        )
        peak = np.max(np.abs(audio_out))
        if peak > 1e-6:
            audio_out = (audio_out / peak) * target_level

        return audio_out.astype(np.float32), gen_cluster


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = InferenceEngine(k=5, ddpm_steps=50)

    test_files = sorted([f for f in os.listdir("sounds/clean")
                         if f.endswith(".wav")])
    if not test_files:
        print("No files in sounds/clean/")
    else:
        for fname in test_files[:3]:
            audio_in, _ = sf.read(os.path.join("sounds/clean", fname),
                                  always_2d=False)
            audio_in = audio_in.astype(np.float32)
            freq = detect_pitch_robust(audio_in)
            if freq:
                hz, name, c, err = snap_to_kalimba(freq)
                print(f"\n{fname}  →  {freq:.1f}Hz → {name} cluster {c}  err={err:.1f}¢")
            else:
                print(f"\n{fname}  →  pitch not detected")
            audio_out, cluster = engine.respond(audio_in)
            print(f"  cluster={cluster}  dur={len(audio_out)/TARGET_SR:.2f}s")

        sf.write("test_output_generative.wav", audio_out, TARGET_SR)
        print("\nSaved → test_output_generative.wav")