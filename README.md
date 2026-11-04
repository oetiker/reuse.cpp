# reuse.cpp

A C++/[ggml](https://github.com/ggml-org/ggml) port of NVIDIA RE-USE speech
enhancement (`nvidia/NVIDIA-NemotronLabs-RE-USE`, OpenMDW-1.1), intended for
CPU inference inside the Rust TTS project lectover.

Status: work in progress. This branch (`lectover/main`) currently contains the
PyTorch reference tooling and test fixtures; the C++ library follows.

## Layout

- `tools/nvidia/` - NVIDIA's unmodified model code and recipe (OpenMDW-1.1).
- `tools/run.py` - reference enhancement script (pure-PyTorch).
- `tools/mamba_ssm/` - plain-PyTorch stand-in for `mamba_ssm.Mamba` (no CUDA needed).
- `tools/make_fixtures.py` - writes intermediate-tensor fixtures to `tests/fixtures/`.
- `tests/fixtures/` - reference tensors; format described in the header of `make_fixtures.py`.

## Reference tooling

The model weights are not part of this repository. Download
`model.safetensors` from the Hugging Face repo (revision
`31f82f2efad1095034492f236e8dac9239ba713b`) and assemble a model directory
containing `tools/nvidia/*` plus `model.safetensors`:

    python tools/run.py <model dir> in.wav out.wav
    THREADS=8 python tools/make_fixtures.py <model dir> <voices dir> tests/fixtures

`run.py` finds the `mamba_ssm` stand-in next to itself (do not use `python -I`).

## Long inputs

`reuse_enhance` runs the model as one graph, and compute memory grows linearly with the clip
(about 2.6 GB per 10 s at 24 kHz). Inputs up to 20 s are processed in one pass. Longer inputs are
split into overlapping windows (at most 20 s each, 2 s overlap, evenly distributed so there is
no tiny last window), each window is enhanced on its own and the overlaps are joined with a
raised-cosine crossfade. Peak memory is thus that of one window for any input length. The result
differs slightly from a single-pass run near the seams (InstanceNorm and the bidirectional Mamba
see one window at a time). `reuse-bench` reports the largest compute buffer over all windows.

## Licence

OpenMDW-1.1, see `LICENSE` and `NOTICE`. Not affiliated with NVIDIA.
