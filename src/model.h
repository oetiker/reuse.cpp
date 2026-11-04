#pragma once
// Model definition (hyperparameters, weights, graph) lives here; filled in by later tasks.
#include "reuse.h"
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
    size_t last_compute_bytes = 0;  // largest compute buffer of the last reuse_enhance's graphs (reuse-bench)
    // Windowing of long inputs (src/window.h). Internal: tests shrink these; not part of the C API.
    double window_seconds = 20.0;   // inputs up to this long run as one graph
    double overlap_seconds = 2.0;   // crossfade length between consecutive windows
    reuse_progress_fn progress_fn = nullptr;  // reuse_set_progress
    void * progress_user = nullptr;
    int last_window_count = 0;      // windows used by the last reuse_enhance
};

// reuse_enhance without the exception guard (src/enhance.cpp): validates arguments (returning
// the status via reuse_fail), runs STFT -> model graph -> zero-frame rule -> iSTFT and writes
// exactly n samples to out (which may alias in). Throws on internal/backend failure.
// Single-window path (no windowing): the former whole-clip enhance_impl. Argument validation is
// the caller's job. Raises ctx->last_compute_bytes to this graph's size if larger.
// With a progress callback set, the graph runs in slices and reports the fraction
// p0 + (p1 - p0) * slice_fraction.
void enhance_window(reuse_ctx * ctx, const float * in, size_t n, int sample_rate, float * out,
                    double p0 = 0.0, double p1 = 1.0);

int enhance_impl(reuse_ctx * ctx, const float * in, size_t n, int sample_rate, float * out);
