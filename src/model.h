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
};
