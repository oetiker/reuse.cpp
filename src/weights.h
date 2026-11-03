#pragma once
// Model weights loaded from a reuse GGUF into one ggml backend buffer.
//
// Access:
//   w.get("<torch name>")        tensor with the torch name unchanged (ne = reversed torch shape).
//                                Throws std::runtime_error naming the tensor if it does not exist.
//   w.has("<torch name>")        existence test, never throws.
//   w.get("<prefix>.A")          for every "<prefix>.A_log" tensor there is a derived tensor
//                                "<prefix>.A" = -exp(A_log), computed once at load (same shape,
//                                F32, same backend buffer kind). E.g.
//                                "TSMamba.0.time_mamba.forward_blocks.A".
// Assumes a little-endian host (GGUF and safetensors are little-endian; data is copied verbatim).
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <map>
#include <string>

struct Hparams {
    int num_tfmamba = 0;
    int hid_feature = 0;
    int d_state = 0;
    int d_conv = 0;
    int expand = 0;
    int dt_rank = 0;
    int base_rate = 0;
    int n_fft = 0;
    int hop = 0;
    int win = 0;
    int d_inner() const { return hid_feature * expand; }
};

struct Weights {
    Weights() = default;
    ~Weights();
    Weights(const Weights &) = delete;
    Weights & operator=(const Weights &) = delete;

    // Loads `path` into buffers of `backend`. Returns false and fills `err` (naming the file) on failure.
    bool load(const std::string & path, ggml_backend_t backend, std::string & err);

    const ggml_tensor * get(const std::string & name) const;
    bool has(const std::string & name) const { return tensors_.count(name) != 0; }
    size_t size() const { return tensors_.size(); }
    size_t nbytes() const { return nbytes_; }

    Hparams hp;

private:
    void release();
    ggml_context * ctx_ = nullptr;      // tensors from the GGUF
    ggml_context * ctx_a_ = nullptr;    // derived A = -exp(A_log) tensors
    ggml_backend_buffer_t buf_ = nullptr;
    ggml_backend_buffer_t buf_a_ = nullptr;
    std::map<std::string, ggml_tensor *> tensors_;
    size_t nbytes_ = 0;
};
