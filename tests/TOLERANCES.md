# Test tolerances

Measured against the PyTorch fixtures (`tests/fixtures/`, produced by `tools/make_fixtures.py`).
Relative max error = `max |ref - got| / max |ref|`.

| Tap | Fixture | Device | Measured rel. max err | max_abs_diff | SNR | Tolerance |
|-----|---------|--------|-----------------------|--------------|-----|-----------|
| `enc_out` (`TOL_ENC`) | tobi-1s-24k | CPU (Apple M, ggml CPU backend, 4 threads) | 7.58e-07 | 6.2e-06 | 122.2 dB | 1e-5 |
| `block1_out` (`TOL_BLOCK1`) | tobi-1s-24k | CPU (Apple M, ggml CPU backend, 4 threads) | 1.03e-06 | 6.7e-06 | 120.3 dB | 1e-5 |
| `block30_out` (`TOL_BLOCK30`) | tobi-1s-24k | CPU (Apple M, ggml CPU backend, 4 threads) | 1.86e-06 | 1.7e-05 | 117.7 dB | 2e-5 |

`TOL_ENC` started at 1e-4 per the plan and was tightened to 1e-5 (~13x headroom over the
first CPU measurement) so a semantic regression cannot hide inside float noise.

`TOL_BLOCK1` / `TOL_BLOCK30` keep ~10x headroom over the first CPU measurement. The 30 blocks
show almost no drift (1.0e-6 after block 1, 1.9e-6 after block 30): each MambaBlock ends in a
LayerNorm, which keeps the activations bounded. Negative controls on the same fixture: leaving
the backward branch unflipped gives rel. err 0.30 (block 1); applying softplus to dt before
`ggml_ssm_scan` (which applies it itself) gives 0.62. Both fail far above the tolerances.
