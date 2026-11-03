"""Generate reference fixtures for the ggml port from the PyTorch reference.

Usage: make_fixtures.py <model dir> <voices dir> <out dir>

<model dir> holds NVIDIA's code (models/, utils/, recipes/) plus model.safetensors.
Uses the same code path as run.py, plus forward hooks on the sub-modules.

Fixture format (one directory per fixture):
  manifest.txt  one line per tensor: "<name> f32 <ne0> <ne1> ..." (ggml order, innermost first)
  <name>.f32    raw little-endian float32 data
  meta.txt      "key value" lines: sample_rate, n_fft, hop, win, n_samples

Layout rules: STFT-domain tensors (stft_mag, stft_pha, out_mag) are written
frequency-innermost, i.e. ggml ne {F, T}. All other tensors keep the reverse of
the torch shape with the batch dim dropped. Decoder taps (mag_out, pha_r, pha_i)
are the raw hooked module outputs before cropping.
"""
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from safetensors.torch import load_file

MODEL, VOICES, OUT = sys.argv[1:4]
sys.path.insert(0, MODEL)
from models.stfts import mag_phase_stft, mag_phase_istft  # noqa: E402
from models.generator_SEMamba_time_d4 import SEMamba  # noqa: E402
from utils.util import load_config, pad_or_trim_to_match  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
RECIPE = "recipes/USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml"
cfg = load_config(os.path.join(MODEL, RECIPE))
m = SEMamba(cfg=cfg)
m.load_state_dict(load_file(os.path.join(MODEL, "model.safetensors")), strict=True)
m.eval()
RELU = nn.ReLU()


def make_even(v):
    v = int(round(v))
    return v if v % 2 == 0 else v + 1


n_fft, hop, win = (cfg["stft_cfg"][k] for k in ("n_fft", "hop_size", "win_size"))
cf, sr0 = cfg["model_cfg"]["compress_factor"], cfg["stft_cfg"]["sampling_rate"]

taps = {}


def hook(name):
    def fn(_mod, _inp, out):
        taps[name] = out.detach().clone()
    return fn


m.dense_encoder.register_forward_hook(hook("enc_out"))
m.TSMamba[0].register_forward_hook(hook("block1_out"))
m.TSMamba[29].register_forward_hook(hook("block30_out"))
m.mask_decoder.register_forward_hook(hook("mag_out"))
m.phase_decoder.phase_conv_r.register_forward_hook(hook("pha_r"))
m.phase_decoder.phase_conv_i.register_forward_hook(hook("pha_i"))


def enhance(wav, sr):
    """wav: [1, N] float32. Same path as run.py; returns dict of tensors and meta."""
    nf, hp, wn = make_even(n_fft * sr // sr0), make_even(hop * sr // sr0), make_even(win * sr // sr0)
    taps.clear()
    with torch.no_grad():
        mag, pha, _ = mag_phase_stft(wav, n_fft=nf, hop_size=hp, win_size=wn, compress_factor=cf, center=True, addeps=False)
        amp_g, pha_g, _ = m(mag, pha)
        mg = torch.expm1(RELU(amp_g))
        zp = torch.sum(mg == 0, 1) / mg.shape[1]
        amp_g[:, :, (zp > 0.5)[0]] = 0
        out = mag_phase_istft(amp_g, pha_g, nf, hp, wn, cf)
        out = pad_or_trim_to_match(wav, out, pad_value=1e-8)
    t = dict(taps)
    t["input"] = wav[0]
    # STFT-domain tensors: torch [F, T] -> written as [T, F] so that F is innermost.
    t["stft_mag"] = mag[0].T
    t["stft_pha"] = pha[0].T
    t["out_mag"] = amp_g[0].T
    t["output"] = out[0]
    meta = dict(sample_rate=sr, n_fft=nf, hop=hp, win=wn, n_samples=wav.shape[1])
    return t, meta


def write_fixture(name, t, meta, names):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    lines = []
    for n in names:
        x = t[n]
        if n in ("enc_out", "block1_out", "block30_out", "mag_out", "pha_r", "pha_i"):
            x = x[0]  # drop batch dim
        x = x.contiguous().float().cpu()
        ne = list(reversed(x.shape))
        x.numpy().astype("<f4").tofile(os.path.join(d, n + ".f32"))
        lines.append(" ".join([n, "f32"] + [str(v) for v in ne]))
    with open(os.path.join(d, "manifest.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(d, "meta.txt"), "w") as f:
        f.write("".join(f"{k} {v}\n" for k, v in meta.items()))
    print(name, {n: tuple(t[n].shape) for n in names})


def read_mono(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T[:1].copy()), sr


def first_voiced(wav, sr):
    """Sample index of the first voiced 10 ms frame (RMS above 10% of the loudest frame)."""
    fl = sr // 100
    x = wav[0, : (wav.shape[1] // fl) * fl].reshape(-1, fl)
    rms = x.pow(2).mean(1).sqrt()
    return int((rms > 0.1 * rms.max()).nonzero()[0]) * fl


def cut(wav, sr, seconds):
    s = first_voiced(wav, sr)
    n = int(seconds * sr)
    assert s + n <= wav.shape[1], "clip too short"
    return wav[:, s:s + n].contiguous()


import torchaudio.functional as AF  # noqa: E402

ALL = ["input", "stft_mag", "stft_pha", "enc_out", "block1_out", "block30_out",
       "mag_out", "pha_r", "pha_i", "out_mag", "output"]
LITE = ["input", "out_mag", "output"]

tobi, sr = read_mono(os.path.join(VOICES, "tobi2.wav"))
petra, srp = read_mono(os.path.join(VOICES, "petra.wav"))
assert sr == 24000 and srp == 24000
tobi4 = cut(tobi, sr, 4.0)
petra4 = cut(petra, srp, 4.0)

DUR_FULL = float(os.environ.get("FULL_SECONDS", "1.0"))
for name, wav, s, names in [
    ("tobi-1s-24k", tobi4[:, : int(DUR_FULL * sr)], 24000, ALL),
    ("tobi-4s-24k", tobi4, 24000, LITE),
    ("petra-4s-24k", petra4, 24000, LITE),
    ("tobi-3s-16k", AF.resample(tobi4[:, : 3 * sr], 24000, 16000), 16000, LITE),
    ("tobi-2s-44k1", AF.resample(tobi4[:, : 2 * sr], 24000, 44100), 44100, LITE),
]:
    t, meta = enhance(wav.contiguous(), s)
    write_fixture(name, t, meta, names)
