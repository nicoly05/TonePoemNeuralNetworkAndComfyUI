"""
dataset.py

Loads all processed WAV files and encodes them with the
EnCodec 48kHz model, and returns the continuous latent embeddings.

"""

import os
import torch
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset
from encodec import EncodecModel

CLEAN_DIR  = "sounds/clean"
CACHE_PATH = "sounds/latents.pt"
TARGET_SR  = 48000


def load_encodec() -> EncodecModel:
    model = EncodecModel.encodec_model_48khz()
    model.set_target_bandwidth(6.0)
    model.eval()
    return model


def encode_dataset(model: EncodecModel) -> dict:

    files = sorted([
        f for f in os.listdir(CLEAN_DIR) if f.lower().endswith(".wav")
    ])
    if not files:
        raise RuntimeError(f"No WAV files found in {CLEAN_DIR}")

    latents   = []
    lengths   = []
    print(f"Encoding {len(files)} files with EnCodec...")

    with torch.no_grad():
        for fname in files:
            path = os.path.join(CLEAN_DIR, fname)
            audio, sr = sf.read(path, always_2d=False)
            audio = audio.astype(np.float32)

            n_samples = len(audio)

            # EnCodec expects (1, 2, T) stereo
            audio_t = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
            audio_t = audio_t.expand(1, 2, -1)

            frames = model.encoder(audio_t)            
            vec    = frames.mean(dim=-1).squeeze(0)   

            latents.append(vec)
            lengths.append(n_samples)
            print(f"  {fname}  {n_samples/TARGET_SR:.2f}s  →  {vec.shape}")

    result = {
        "latents":   torch.stack(latents, dim=0),    
        "filenames": files,
        "lengths":   torch.tensor(lengths, dtype=torch.long),
    }
    print(f"\nEncoded {len(files)} files, latent dim = {result['latents'].shape[1]}")
    return result


class LatentDataset(Dataset):

    def __init__(self, force_encode: bool = False):
        if not force_encode and os.path.exists(CACHE_PATH):
            print(f"Loading cached latents from {CACHE_PATH}")
            data = torch.load(CACHE_PATH, weights_only=False)
        else:
            model = load_encodec()
            data  = encode_dataset(model)
            torch.save(data, CACHE_PATH)
            print(f"Latents cached to {CACHE_PATH}")

        self.latents   = data["latents"]   
        self.files     = data["filenames"]
        self.lengths   = data["lengths"]   

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.latents[idx]

    @property
    def latent_dim(self) -> int:
        return self.latents.shape[1]


if __name__ == "__main__":
    ds = LatentDataset(force_encode=True)
    print(f"\nDataset size : {len(ds)}")
    print(f"Latent dim   : {ds.latent_dim}")
    min_len = ds.lengths.min().item()
    max_len = ds.lengths.max().item()
    mean_len = ds.lengths.float().mean().item()
    print(f"Length range : {min_len/48000:.2f}s – {max_len/48000:.2f}s "
          f"(mean {mean_len/48000:.2f}s)")
