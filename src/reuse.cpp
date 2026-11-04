#include "reuse.h"
#include "error.h"
#include "model.h"

#include "ggml-backend.h"

#include <memory>
#include <mutex>
#include <string>

static thread_local std::string g_last_error;

int reuse_fail(int status, const std::string & msg) {
    g_last_error = msg;
    return status;
}

static ggml_backend_dev_t find_device(const std::string & name) {
    if (ggml_backend_dev_t d = ggml_backend_dev_by_name(name.c_str())) return d;
    // Dynamically loaded backends (GGML_BACKEND_DL) only register after load_all; try once.
    static std::once_flag once;
    std::call_once(once, [] { ggml_backend_load_all(); });
    return ggml_backend_dev_by_name(name.c_str());
}

static reuse_ctx * load_impl(const char * gguf_path, const char * device, int n_threads) {
    if (!gguf_path) { reuse_fail(REUSE_ERR_ARG, "reuse_load: NULL gguf path"); return nullptr; }
    if (n_threads < 1) { reuse_fail(REUSE_ERR_ARG, "reuse_load: n_threads must be >= 1"); return nullptr; }

    const std::string dev_name = device ? device : "CPU";
    ggml_backend_dev_t dev = find_device(dev_name);
    if (!dev) {
        std::string avail;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            avail += (avail.empty() ? "" : ", ");
            avail += ggml_backend_dev_name(ggml_backend_dev_get(i));
        }
        reuse_fail(REUSE_ERR_BACKEND, "backend: unknown device '" + dev_name + "' (available: " + avail + ")");
        return nullptr;
    }

    std::unique_ptr<reuse_ctx> ctx(new reuse_ctx());
    ctx->n_threads = n_threads;
    ctx->backend.be = ggml_backend_dev_init(dev, nullptr);
    if (!ctx->backend.be) {
        reuse_fail(REUSE_ERR_BACKEND, "backend: cannot initialise device '" + dev_name + "'");
        return nullptr;
    }
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        auto set_threads = (ggml_backend_set_n_threads_t)
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_set_n_threads");
        if (set_threads) set_threads(ctx->backend.be, n_threads);
    }

    std::string err;
    if (!ctx->weights.load(gguf_path, ctx->backend.be, err)) {
        reuse_fail(REUSE_ERR_MODEL, err);
        return nullptr;
    }
    return ctx.release();
}

extern "C" {

reuse_ctx * reuse_load(const char * gguf_path, const char * device, int n_threads) {
    return reuse_guard<reuse_ctx *>(nullptr, REUSE_ERR_MODEL, [&] { return load_impl(gguf_path, device, n_threads); });
}

int reuse_enhance(reuse_ctx * ctx, const float * in, size_t n, int sample_rate, float * out) {
    // Internal failures here are graph allocation / compute errors on the backend.
    return reuse_guard<int>(REUSE_ERR_BACKEND, REUSE_ERR_BACKEND,
                            [&] { return enhance_impl(ctx, in, n, sample_rate, out); });
}

void reuse_set_progress(reuse_ctx * ctx, reuse_progress_fn fn, void * user) {
    if (!ctx) return;
    ctx->progress_fn = fn;
    ctx->progress_user = user;
}

void reuse_free(reuse_ctx * ctx) {
    reuse_guard<int>(0, REUSE_ERR_MODEL, [&] { delete ctx; return 0; });
}

const char * reuse_last_error(void) {
    return g_last_error.c_str();
}

} // extern "C"
