#include "weights.h"

#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

struct FileCloser { void operator()(FILE * f) const { if (f) std::fclose(f); } };
using FilePtr = std::unique_ptr<FILE, FileCloser>;

struct GgufGuard {
    gguf_context * g = nullptr;
    ~GgufGuard() { if (g) gguf_free(g); }
};

bool get_u32(const gguf_context * g, const char * key, int & out, std::string & err, const std::string & path) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_UINT32) {
        err = path + ": missing or non-u32 GGUF key '" + key + "'";
        return false;
    }
    out = (int) gguf_get_val_u32(g, id);
    return true;
}

} // namespace

Weights::~Weights() { release(); }

void Weights::release() {
    tensors_.clear();
    if (buf_a_) { ggml_backend_buffer_free(buf_a_); buf_a_ = nullptr; }
    if (buf_) { ggml_backend_buffer_free(buf_); buf_ = nullptr; }
    if (ctx_a_) { ggml_free(ctx_a_); ctx_a_ = nullptr; }
    if (ctx_) { ggml_free(ctx_); ctx_ = nullptr; }
    nbytes_ = 0;
}

const ggml_tensor * Weights::get(const std::string & name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("missing weight tensor '" + name + "'");
    return it->second;
}

bool Weights::load(const std::string & path, ggml_backend_t backend, std::string & err) {
    release();

    GgufGuard gg;
    gguf_init_params ip = { /*no_alloc=*/ true, /*ctx=*/ &ctx_ };
    gg.g = gguf_init_from_file(path.c_str(), ip);
    if (!gg.g || !ctx_) {
        err = path + ": cannot read GGUF file";
        release();
        return false;
    }

    int version = 0;
    Hparams h;
    if (!get_u32(gg.g, "reuse.version", version, err, path) ||
        !get_u32(gg.g, "reuse.num_tfmamba", h.num_tfmamba, err, path) ||
        !get_u32(gg.g, "reuse.hid_feature", h.hid_feature, err, path) ||
        !get_u32(gg.g, "reuse.d_state", h.d_state, err, path) ||
        !get_u32(gg.g, "reuse.d_conv", h.d_conv, err, path) ||
        !get_u32(gg.g, "reuse.expand", h.expand, err, path) ||
        !get_u32(gg.g, "reuse.dt_rank", h.dt_rank, err, path) ||
        !get_u32(gg.g, "reuse.base_rate", h.base_rate, err, path) ||
        !get_u32(gg.g, "reuse.n_fft", h.n_fft, err, path) ||
        !get_u32(gg.g, "reuse.hop", h.hop, err, path) ||
        !get_u32(gg.g, "reuse.win", h.win, err, path)) {
        release();
        return false;
    }
    if (version != 1) {
        err = path + ": unsupported reuse.version " + std::to_string(version) + " (expected 1)";
        release();
        return false;
    }
    hp = h;

    const int64_t n = gguf_get_n_tensors(gg.g);
    std::vector<ggml_tensor *> a_logs;  // A_log tensors, in order
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_); t; t = ggml_get_next_tensor(ctx_, t)) {
        if (t->type != GGML_TYPE_F32) {
            err = path + ": tensor '" + t->name + "' is not F32";
            release();
            return false;
        }
        tensors_[t->name] = t;
        const size_t len = std::strlen(t->name);
        if (len > 6 && std::strcmp(t->name + len - 6, ".A_log") == 0) a_logs.push_back(t);
    }
    if ((int64_t) tensors_.size() != n) {
        err = path + ": tensor table inconsistent";
        release();
        return false;
    }

    // Derived tensors A = -exp(A_log).
    if (!a_logs.empty()) {
        ggml_init_params cp = { ggml_tensor_overhead() * (a_logs.size() + 1), nullptr, /*no_alloc=*/ true };
        ctx_a_ = ggml_init(cp);
        if (!ctx_a_) { err = path + ": ggml_init failed"; release(); return false; }
        for (ggml_tensor * src : a_logs) {
            std::string nm(src->name);
            nm.replace(nm.size() - 5, 5, "A");  // "...A_log" -> "...A"
            if (nm.size() >= GGML_MAX_NAME || tensors_.count(nm)) {
                err = path + ": cannot derive tensor name for '" + src->name + "'";
                release();
                return false;
            }
            ggml_tensor * a = ggml_new_tensor(ctx_a_, GGML_TYPE_F32, GGML_MAX_DIMS, src->ne);
            ggml_set_name(a, nm.c_str());
        }
    }

    buf_ = ggml_backend_alloc_ctx_tensors(ctx_, backend);
    if (!buf_) { err = path + ": cannot allocate weight buffer on backend"; release(); return false; }
    if (ctx_a_) {
        buf_a_ = ggml_backend_alloc_ctx_tensors(ctx_a_, backend);
        if (!buf_a_) { err = path + ": cannot allocate derived-weight buffer on backend"; release(); return false; }
    }

    // Upload tensor data.
    FilePtr f(std::fopen(path.c_str(), "rb"));
    if (!f) { err = path + ": cannot open for reading"; release(); return false; }
    const size_t data_off = gguf_get_data_offset(gg.g);
    std::vector<char> tmp;
    for (int64_t i = 0; i < n; ++i) {
        ggml_tensor * t = tensors_.at(gguf_get_tensor_name(gg.g, i));
        const size_t sz = ggml_nbytes(t);
        const size_t off = data_off + gguf_get_tensor_offset(gg.g, i);
        tmp.resize(sz);
        if (fseeko(f.get(), (off_t) off, SEEK_SET) != 0 || std::fread(tmp.data(), 1, sz, f.get()) != sz) {
            err = path + ": truncated GGUF data for tensor '" + t->name + "'";
            release();
            return false;
        }
        ggml_backend_tensor_set(t, tmp.data(), 0, sz);
        nbytes_ += sz;

        const size_t len = std::strlen(t->name);
        if (len > 6 && std::strcmp(t->name + len - 6, ".A_log") == 0) {
            std::string nm(t->name);
            nm.replace(nm.size() - 5, 5, "A");
            ggml_tensor * a = nullptr;
            for (ggml_tensor * c = ggml_get_first_tensor(ctx_a_); c; c = ggml_get_next_tensor(ctx_a_, c)) {
                if (nm == c->name) { a = c; break; }
            }
            if (!a) { err = path + ": internal error deriving " + nm; release(); return false; }
            float * v = reinterpret_cast<float *>(tmp.data());
            for (size_t k = 0; k < sz / sizeof(float); ++k) v[k] = -std::exp(v[k]);
            ggml_backend_tensor_set(a, tmp.data(), 0, sz);
            nbytes_ += sz;
            tensors_[nm] = a;
        }
    }
    return true;
}
