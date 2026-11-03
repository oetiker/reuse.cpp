# Test tolerances

Measured against the PyTorch fixtures (`tests/fixtures/`, produced by `tools/make_fixtures.py`).
Relative max error = `max |ref - got| / max |ref|`.

| Tap | Fixture | Device | Measured rel. max err | max_abs_diff | SNR | Tolerance |
|-----|---------|--------|-----------------------|--------------|-----|-----------|
| `enc_out` (`TOL_ENC`) | tobi-1s-24k | CPU (Apple M, ggml CPU backend, 4 threads) | 7.58e-07 | 6.2e-06 | 122.2 dB | 1e-5 |
| `block1_out` (`TOL_BLOCK1`) | tobi-1s-24k | CPU (Apple M, ggml CPU backend, 4 threads) | 1.03e-06 | 6.7e-06 | 120.3 dB | 1e-5 |
| `block30_out` (`TOL_BLOCK30`) | tobi-1s-24k | CPU (Apple M, ggml CPU backend, 4 threads) | 1.86e-06 | 1.7e-05 | 117.7 dB | 2e-5 |
| `mag_out` (`TOL_DEC`, from fixture `block30_out`) | tobi-1s-24k | CPU (Apple M, ggml CPU backend, 4 threads) | 9.52e-07 | 3.1e-06 | 124.5 dB | 4e-5 |
| `pha_r` (`TOL_DEC`, from fixture `block30_out`) | tobi-1s-24k | CPU (Apple M, ggml CPU backend, 4 threads) | 3.02e-06 | 1.9e-05 | 122.9 dB | 4e-5 |
| `pha_i` (`TOL_DEC`, from fixture `block30_out`) | tobi-1s-24k | CPU (Apple M, ggml CPU backend, 4 threads) | 3.80e-06 | 2.7e-05 | 122.9 dB | 4e-5 |

`TOL_ENC` started at 1e-4 per the plan and was tightened to 1e-5 (~13x headroom over the
first CPU measurement) so a semantic regression cannot hide inside float noise.

`TOL_BLOCK1` / `TOL_BLOCK30` keep ~10x headroom over the first CPU measurement. The 30 blocks
show almost no drift (1.0e-6 after block 1, 1.9e-6 after block 30): each MambaBlock ends in a
LayerNorm, which keeps the activations bounded. Negative controls on the same fixture: leaving
the backward branch unflipped gives rel. err 0.30 (block 1); applying softplus to dt before
`ggml_ssm_scan` (which applies it itself) gives 0.62. Both fail far above the tolerances.

`TOL_DEC` keeps ~10x headroom over the worst decoder tap (`pha_i`). The decoder taps are
compared before the crop to the STFT size (482 x 204 for 1 s at 24 kHz), as hooked in torch.

## End to end (`reuse_enhance` vs. fixture `output`, `SNR_MIN`)

`output` is the CPU-path `tools/run.py` result. SNR = `10 log10(sum ref^2 / sum (ref - got)^2)`.

| Fixture | Device | Measured SNR | max_abs_diff | Wall time | `SNR_MIN` |
|---------|--------|--------------|--------------|-----------|-----------|
| tobi-4s-24k | CPU (Apple M, 4 threads) | 57.6 dB | 2.2e-3 | 37.6 s | 40 dB |
| petra-4s-24k | CPU (Apple M, 4 threads) | 64.9 dB | 1.9e-3 | 37.6 s | 40 dB |
| tobi-3s-16k | CPU (Apple M, 4 threads) | 58.3 dB | 2.3e-3 | 18.6 s | 40 dB |
| tobi-2s-44k1 | CPU (Apple M, 4 threads) | 53.8 dB | 2.2e-3 | 34.7 s | 40 dB |
| tobi-1s-24k (diagnostic only) | CPU (Apple M, 4 threads) | 48.5 dB | | | |

Why end to end is ~60 dB while every tap is ~120 dB: the model is ill-conditioned with respect
to its *input phase*. In low-energy STFT bins the phase is dominated by FFT rounding noise
(torch: float32; ours: double), and bins with a near-negative-real value can even land on
-pi vs. +pi (max phase difference vs. `stft_pha` is 2 pi). The model takes that phase as an
input channel. Measured on tobi-1s-24k (CPU):

- fixture `stft_mag`/`stft_pha` -> our graph -> our CPU glue: 114.7 dB vs. `output`
  (our STFT magnitude + fixture phase: 115.5 dB; fixture magnitude + our phase: 48.5 dB);
- fixture `out_mag` + phase from fixture `pha_r`/`pha_i` -> our iSTFT: 138.8 dB (glue floor);
- the port against itself with the input perturbed by +-1e-7 (far below one 16-bit LSB): 54.6 dB.

So the end-to-end gap is the reference's own input-phase noise, not a port error; matching it
would mean reproducing torch's float32 FFT bit for bit. `SNR_MIN` stays at 40 dB (> 13 dB
below the worst CPU measurement): other devices and thread counts perturb the computation by
more than 1e-7, and the self-sensitivity above shows that alone costs ~55 dB.

At 44.1 kHz the last 200 samples of both the reference and the port are exactly 1e-8 (the
tail torch.istft leaves uncovered, padded by `pad_or_trim_to_match`); the test checks the
count matches.

Silence (24000 zeros, 24 kHz): PyTorch gives max |y| = 0.04242 (the model predicts positive
log magnitudes in nearly every bin, so the zero-frame rule never fires); the port gives
0.0429. The test requires finite output within 0.004 of the PyTorch level.
