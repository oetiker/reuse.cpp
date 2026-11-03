# Test tolerances

Measured against the PyTorch fixtures (`tests/fixtures/`, produced by `tools/make_fixtures.py`).
Relative max error = `max |ref - got| / max |ref|`.

| Tap | Fixture | Device | Measured rel. max err | max_abs_diff | SNR | Tolerance |
|-----|---------|--------|-----------------------|--------------|-----|-----------|
| `enc_out` (`TOL_ENC`) | tobi-1s-24k | CPU (Apple M, ggml CPU backend, 4 threads) | 7.58e-07 | 6.2e-06 | 122.2 dB | 1e-5 |

`TOL_ENC` started at 1e-4 per the plan and was tightened to 1e-5 (~13x headroom over the
first CPU measurement) so a semantic regression cannot hide inside float noise.
