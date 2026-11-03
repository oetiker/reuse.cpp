import sys, time, os
import numpy as np, soundfile as sf, torch, torch.nn as nn
from safetensors.torch import load_file
MODEL = sys.argv[1]
sys.path.insert(0, MODEL)
from models.stfts import mag_phase_stft, mag_phase_istft
from models.generator_SEMamba_time_d4 import SEMamba
from utils.util import load_config, pad_or_trim_to_match
torch.set_num_threads(int(os.environ.get("THREADS", "1")))
cfg = load_config(os.path.join(MODEL, "recipes/USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml"))
m = SEMamba(cfg=cfg)
res = m.load_state_dict(load_file(os.path.join(MODEL, "model.safetensors")), strict=True)
print("load_state_dict:", res, "params:", sum(p.numel() for p in m.parameters()))
DEV = torch.device(os.environ.get("DEV","cpu")); m.eval().to(DEV)
RELU = nn.ReLU()
def make_even(v):
    v = int(round(v)); return v if v % 2 == 0 else v + 1
n_fft, hop, win = (cfg['stft_cfg'][k] for k in ('n_fft', 'hop_size', 'win_size'))
cf, sr0 = cfg['model_cfg']['compress_factor'], cfg['stft_cfg']['sampling_rate']
for inp, outp in zip(sys.argv[2::2], sys.argv[3::2]):
    data, sr = sf.read(inp, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T[:1].copy()).to(DEV)
    nf, hp, wn = make_even(n_fft*sr//sr0), make_even(hop*sr//sr0), make_even(win*sr//sr0)
    t0 = time.time()
    with torch.no_grad():
        mag, pha, _ = mag_phase_stft(wav, n_fft=nf, hop_size=hp, win_size=wn, compress_factor=cf, center=True, addeps=False)
        amp_g, pha_g, _ = m(mag, pha)
        mg = torch.expm1(RELU(amp_g))
        zp = torch.sum(mg == 0, 1) / mg.shape[1]
        amp_g[:, :, (zp > 0.5)[0]] = 0
        out = mag_phase_istft(amp_g, pha_g, nf, hp, wn, cf)
        out = pad_or_trim_to_match(wav, out, pad_value=1e-8)
    print(f"{inp}: sr={sr} stft n_fft={nf} hop={hp} frames={mag.shape[-1]} time={time.time()-t0:.1f}s")
    sf.write(outp, out[0].cpu().numpy(), sr, subtype="PCM_16")
