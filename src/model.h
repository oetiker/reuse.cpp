#pragma once
// Model definition (hyperparameters, weights, graph) lives here; filled in by later tasks.
#include "weights.h"

#include <string>

struct BackendHandle {
    ggml_backend_t be = nullptr;
    BackendHandle() = default;
    BackendHandle(const BackendHandle &) = delete;
    BackendHandle & operator=(const BackendHandle &) = delete;
    ~BackendHandle() { if (be) ggml_backend_free(be); }
};

struct reuse_ctx {
    int n_threads = 1;
    BackendHandle backend;  // declared before weights: destroyed after them
    Weights weights;
    size_t last_compute_bytes = 0;  // compute buffer size of the last reuse_enhance graph (reuse-bench)
};

// reuse_enhance without the exception guard (src/enhance.cpp): validates arguments (returning
// the status via reuse_fail), runs STFT -> model graph -> zero-frame rule -> iSTFT and writes
// exactly n samples to out (which may alias in). Throws on internal/backend failure.
int enhance_impl(reuse_ctx * ctx, const float * in, size_t n, int sample_rate, float * out);
