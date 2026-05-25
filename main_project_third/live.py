"""
live_generative.py

Per-note polyphonic echo pipeline kalimba


Controls:
    q      — quit
    t / T  — raise / lower onset threshold
    f / F  — raise / lower flux sensitivity
    + / -  — increase / decrease K (neighbour blend count for decoding)
    d / D  — increase / decrease echo delay
    i      — print current settings
"""

import argparse
import threading
import queue
import time
import sys
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from pythonosc import udp_client

from inference import InferenceEngine, TARGET_SR

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_K          = 8
DEFAULT_THRESHOLD  = 0.015
DEFAULT_FLUX_SENS  = 6.0
DEFAULT_ECHO_DELAY = 0.5
DEFAULT_OSC_IP     = "127.0.0.1"
DEFAULT_OSC_PORT   = 9000
INPUT_DEVICE_HINT  = "Microfone"
OUTPUT_DEVICE_HINT = "Alto-falantes"

INPUT_BUF_SECONDS  = 4.0
INPUT_BUF_SAMPLES  = int(TARGET_SR * INPUT_BUF_SECONDS)

CAPTURE_S          = 0.08
CAPTURE_SAMPLES    = int(TARGET_SR * CAPTURE_S)

MIN_NOTE_GAP_S     = 0.08

RING_SECONDS       = 20.0
RING_SAMPLES       = int(TARGET_SR * RING_SECONDS)
OUTPUT_BLOCKSIZE   = 512

N_WORKERS          = 4
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class NoteJob:
    segment:      np.ndarray
    arrival_time: float
    note_idx:     int


def find_device(hint: str, kind: str) -> int | None:
    for i, d in enumerate(sd.query_devices()):
        if hint.lower() in d["name"].lower():
            if kind == "input"  and d["max_input_channels"]  > 0: return i
            if kind == "output" and d["max_output_channels"] > 0: return i
    return None


def list_devices():
    print("\nAvailable audio devices:")
    for i, d in enumerate(sd.query_devices()):
        tag = []
        if d["max_input_channels"]  > 0: tag.append("IN")
        if d["max_output_channels"] > 0: tag.append("OUT")
        print(f"  [{i:2d}] {'/'.join(tag):6s}  {d['name']}")
    print()


class LiveSystem:

    def __init__(self, k, threshold, flux_sens, echo_delay,
                 osc_ip, osc_port, input_device, output_device):

        self.k             = k
        self.threshold     = threshold
        self.flux_sens     = flux_sens
        self.echo_delay    = echo_delay
        self.input_device  = input_device
        self.output_device = output_device
        self.running       = False
        self.note_counter  = 0

        self.osc = udp_client.SimpleUDPClient(osc_ip, osc_port)
        print(f"OSC → {osc_ip}:{osc_port}  (/cluster)")

        self.engine = InferenceEngine(k=self.k)

        self.input_buf      = np.zeros(INPUT_BUF_SAMPLES, dtype=np.float32)
        self.input_buf_lock = threading.Lock()

        self.job_queue: queue.Queue[NoteJob] = queue.Queue()

        self.ring_buf      = np.zeros(RING_SAMPLES, dtype=np.float32)
        self.ring_read_pos = 0
        self.ring_lock     = threading.Lock()

        self.current_cluster = -1

    # ── Output callback ───────────────────────────────────────────────────────

    def _output_callback(self, outdata: np.ndarray, frames: int,
                         time_info, status):
        if status:
            print(f"[output] {status}")
        with self.ring_lock:
            end = self.ring_read_pos + frames
            if end <= RING_SAMPLES:
                chunk = self.ring_buf[self.ring_read_pos:end].copy()
                self.ring_buf[self.ring_read_pos:end] = 0.0
            else:
                part1 = self.ring_buf[self.ring_read_pos:].copy()
                part2 = self.ring_buf[:end - RING_SAMPLES].copy()
                self.ring_buf[self.ring_read_pos:] = 0.0
                self.ring_buf[:end - RING_SAMPLES] = 0.0
                chunk = np.concatenate([part1, part2])
            self.ring_read_pos = end % RING_SAMPLES
        outdata[:, 0] = np.tanh(chunk)

    def _write_to_ring(self, audio: np.ndarray, delay_samples: int):
        n = len(audio)
        with self.ring_lock:
            write_pos = (self.ring_read_pos + delay_samples) % RING_SAMPLES
            end = write_pos + n
            if end <= RING_SAMPLES:
                self.ring_buf[write_pos:end] += audio
            else:
                split = RING_SAMPLES - write_pos
                self.ring_buf[write_pos:] += audio[:split]
                self.ring_buf[:end - RING_SAMPLES] += audio[split:]

    # ── Input callback ────────────────────────────────────────────────────────

    def _input_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[input] {status}")
        mono = indata[:, 0].astype(np.float32)
        with self.input_buf_lock:
            self.input_buf = np.roll(self.input_buf, -len(mono))
            self.input_buf[-len(mono):] = mono

    # ── Onset detector (spectral flux) ────────────────────────────────────────

    def _onset_thread(self):

        print("[onset] Listening (spectral flux)...")

        frame_n      = int(TARGET_SR * 0.020)
        hop_n        = int(TARGET_SR * 0.010)
        history_len  = 20

        prev_mag     = None
        flux_history = np.zeros(history_len, dtype=np.float32)
        last_note_t  = 0.0

        while self.running:
            time.sleep(0.010)

            with self.input_buf_lock:
                frame = self.input_buf[-frame_n:].copy()
                buf   = self.input_buf.copy()

            rms = float(np.sqrt(np.mean(frame ** 2)))
            if rms < self.threshold:
                prev_mag     = None
                flux_history = np.zeros(history_len, dtype=np.float32)
                continue

            windowed = frame * np.hanning(len(frame))
            mag      = np.abs(np.fft.rfft(windowed, n=2048))

            if prev_mag is not None:
                flux = float(np.sum(np.maximum(mag - prev_mag, 0.0)))
            else:
                flux = 0.0

            prev_mag = mag

            flux_history = np.roll(flux_history, -1)
            flux_history[-1] = flux
            avg_flux = float(flux_history[:-1].mean()) + 1e-8

            now = time.time()
            is_onset = (
                flux > self.flux_sens * avg_flux
                and (now - last_note_t) > MIN_NOTE_GAP_S
            )

            if is_onset:
                last_note_t = now
                self.note_counter += 1
                note_idx = self.note_counter

                segment = buf[-CAPTURE_SAMPLES:].copy()
                self.job_queue.put(NoteJob(
                    segment=segment,
                    arrival_time=now,
                    note_idx=note_idx,
                ))
                print(f"[onset] #{note_idx}  "
                      f"flux={flux:.1f}  ratio={flux/avg_flux:.1f}×  "
                      f"rms={rms:.4f}")

    # ── Inference workers ─────────────────────────────────────────────────────

    def _inference_worker(self, worker_id: int):

        print(f"[worker {worker_id}] Ready")
        while self.running:
            try:
                job: NoteJob = self.job_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            t0 = time.time()
            audio_out, cluster = self.engine.respond(job.segment)
            elapsed = time.time() - t0

            self.current_cluster = cluster
            self.osc.send_message("/cluster", cluster)

            play_at       = job.arrival_time + self.echo_delay
            delay_s       = max(play_at - time.time(), 0.0)
            delay_samples = int(delay_s * TARGET_SR)

            print(f"[worker {worker_id}] #{job.note_idx}  "
                  f"infer={elapsed:.2f}s  cluster={cluster}  "
                  f"dur={len(audio_out)/TARGET_SR:.2f}s  "
                  f"plays_in={delay_s:.2f}s")

            self._write_to_ring(audio_out, delay_samples)

    # ── Keyboard thread ───────────────────────────────────────────────────────

    def _keyboard_thread(self):
        import tty, termios
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self.running:
                ch = sys.stdin.read(1)
                if ch == "q":
                    print("\n[control] Quitting...")
                    self.running = False
                elif ch == "+":
                    self.k = min(self.k + 1, 20)
                    self.engine.k = self.k
                    print(f"\n[control] K → {self.k}")
                elif ch == "-":
                    self.k = max(self.k - 1, 1)
                    self.engine.k = self.k
                    print(f"\n[control] K → {self.k}")
                elif ch == "t":
                    self.threshold = min(self.threshold + 0.005, 0.5)
                    print(f"\n[control] threshold → {self.threshold:.3f}")
                elif ch == "T":
                    self.threshold = max(self.threshold - 0.005, 0.001)
                    print(f"\n[control] threshold → {self.threshold:.3f}")
                elif ch == "f":
                    self.flux_sens = min(self.flux_sens + 0.5, 20.0)
                    print(f"\n[control] flux_sens → {self.flux_sens:.1f}×")
                elif ch == "F":
                    self.flux_sens = max(self.flux_sens - 0.5, 1.5)
                    print(f"\n[control] flux_sens → {self.flux_sens:.1f}×")
                elif ch == "d":
                    self.echo_delay = min(self.echo_delay + 0.1, 10.0)
                    print(f"\n[control] echo_delay → {self.echo_delay:.1f}s")
                elif ch == "D":
                    self.echo_delay = max(self.echo_delay - 0.1, 0.1)
                    print(f"\n[control] echo_delay → {self.echo_delay:.2f}s")
                elif ch == "i":
                    print(f"\n  K={self.k}  threshold={self.threshold:.3f}  "
                          f"flux_sens={self.flux_sens:.1f}×  "
                          f"echo_delay={self.echo_delay:.2f}s  "
                          f"cluster={self.current_cluster}\n")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        self.running = True

        threads = [
            threading.Thread(target=self._onset_thread,    daemon=True),
            threading.Thread(target=self._keyboard_thread, daemon=True),
        ]
        for i in range(N_WORKERS):
            threads.append(threading.Thread(
                target=self._inference_worker, args=(i,), daemon=True))
        for t in threads:
            t.start()

        print(f"\n[live_generative] K={self.k}  threshold={self.threshold:.3f}  "
              f"flux_sens={self.flux_sens:.1f}×  echo_delay={self.echo_delay:.2f}s  "
              f"workers={N_WORKERS}")
        print("[live] q=quit  +/-=K  t/T=threshold  f/F=flux  d/D=delay  i=info\n")
        print("NOTE: generative inference takes ~0.5–1s per note.")
        print(f"      echo_delay >= 1.0s recommended. Current: {self.echo_delay:.2f}s\n")

        try:
            with sd.InputStream(
                device=self.input_device,
                channels=1,
                samplerate=TARGET_SR,
                blocksize=1024,
                dtype="float32",
                callback=self._input_callback,
            ), sd.OutputStream(
                device=self.output_device,
                channels=1,
                samplerate=TARGET_SR,
                blocksize=OUTPUT_BLOCKSIZE,
                dtype="float32",
                callback=self._output_callback,
            ):
                while self.running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            self.running = False

        print("[live_generative] Stopped.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",             type=int,   default=DEFAULT_K)
    parser.add_argument("--threshold",     type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--flux-sens",     type=float, default=DEFAULT_FLUX_SENS)
    parser.add_argument("--echo-delay",    type=float, default=DEFAULT_ECHO_DELAY)
    parser.add_argument("--osc-ip",        type=str,   default=DEFAULT_OSC_IP)
    parser.add_argument("--osc-port",      type=int,   default=DEFAULT_OSC_PORT)
    parser.add_argument("--input-device",  type=int,   default=None)
    parser.add_argument("--output-device", type=int,   default=None)
    parser.add_argument("--list-devices",  action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    in_dev  = args.input_device  or find_device(INPUT_DEVICE_HINT,  "input")
    out_dev = args.output_device or find_device(OUTPUT_DEVICE_HINT, "output")

    if in_dev is None:
        list_devices()
        in_dev = int(input("Enter input device index: "))
    else:
        print(f"[device] Input  → [{in_dev}] {sd.query_devices(in_dev)['name']}")

    if out_dev is None:
        list_devices()
        out_dev = int(input("Enter output device index: "))
    else:
        print(f"[device] Output → [{out_dev}] {sd.query_devices(out_dev)['name']}")

    LiveSystem(
        k=args.k,
        threshold=args.threshold,
        flux_sens=args.flux_sens,
        echo_delay=args.echo_delay,
        osc_ip=args.osc_ip,
        osc_port=args.osc_port,
        input_device=in_dev,
        output_device=out_dev,
    ).run()


if __name__ == "__main__":
    main()
