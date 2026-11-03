#include "reuse.h"
#include "weights.h"

#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

static int g_failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "  CHECK failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__); ++g_failures; } } while (0)

static std::string temp_dir() {
    if (const char * e = std::getenv("REUSE_TEST_TMP")) return e;
    return ".";
}

static const char * safetensors() { return std::getenv("REUSE_SAFETENSORS"); }

// Converts once per process; returns the GGUF path ("" on failure).
static std::string converted() {
    static std::string path;
    if (path.empty()) {
        const std::string out = temp_dir() + "/test_weights.gguf";
        if (reuse_convert(safetensors(), out.c_str()) == REUSE_OK) path = out;
        else std::fprintf(stderr, "  convert failed: %s\n", reuse_last_error());
    }
    return path;
}

static void convert_writes_all_tensors() {
    if (!safetensors()) { std::printf("  SKIP: REUSE_SAFETENSORS not set\n"); return; }
    const std::string out = converted();
    CHECK(!out.empty());
    if (out.empty()) return;
    ggml_context * ctx = nullptr;
    gguf_init_params ip = { true, &ctx };
    gguf_context * g = gguf_init_from_file(out.c_str(), ip);
    CHECK(g != nullptr);
    if (!g) return;
    CHECK(gguf_get_n_tensors(g) == 1416);
    int64_t k = gguf_find_key(g, "reuse.num_tfmamba");
    CHECK(k >= 0 && gguf_get_val_u32(g, k) == 30);
    k = gguf_find_key(g, "reuse.source_revision");
    CHECK(k >= 0 && std::strcmp(gguf_get_val_str(g, k), "31f82f2efad1095034492f236e8dac9239ba713b") == 0);
    k = gguf_find_key(g, "reuse.hop");
    CHECK(k >= 0 && gguf_get_val_u32(g, k) == 40);
    ggml_tensor * t = ggml_get_tensor(ctx, "dense_encoder.dense_block.dense_block.3.0.weight");
    CHECK(t != nullptr);
    if (t) CHECK(t->ne[0] == 3 && t->ne[1] == 3 && t->ne[2] == 256 && t->ne[3] == 64);
    gguf_free(g);
    ggml_free(ctx);
}

static void convert_refuses_wrong_file() {
    if (!safetensors()) { std::printf("  SKIP: REUSE_SAFETENSORS not set\n"); return; }
    const std::string trunc = temp_dir() + "/truncated.safetensors";
    {
        std::ifstream in(safetensors(), std::ios::binary);
        std::vector<char> buf(1 << 20);
        in.read(buf.data(), (std::streamsize) buf.size());
        std::ofstream o(trunc, std::ios::binary);
        o.write(buf.data(), in.gcount());
    }
    const std::string out = temp_dir() + "/should_not_exist.gguf";
    std::remove(out.c_str());
    CHECK(reuse_convert(trunc.c_str(), out.c_str()) != REUSE_OK);
    CHECK(std::string(reuse_last_error()).find("truncated.safetensors") != std::string::npos);
    CHECK(!std::ifstream(out).good());
    CHECK(reuse_convert((temp_dir() + "/no_such_file").c_str(), out.c_str()) != REUSE_OK);
    CHECK(reuse_convert(nullptr, out.c_str()) == REUSE_ERR_ARG);
    std::remove(trunc.c_str());
}

// A file with a valid header followed by a huge sparse data section makes the converter's
// data allocation throw std::bad_alloc; the C API must turn that into an error status.
static void convert_contains_exceptions() {
    if (!safetensors()) { std::printf("  SKIP: REUSE_SAFETENSORS not set\n"); return; }
    const std::string huge = temp_dir() + "/huge_sparse.safetensors";
    {
        std::ifstream in(safetensors(), std::ios::binary);
        unsigned char lb[8];
        in.read(reinterpret_cast<char *>(lb), 8);
        uint64_t hlen = 0;
        for (int i = 7; i >= 0; --i) hlen = (hlen << 8) | lb[i];
        std::vector<char> pre(8 + hlen);
        in.seekg(0);
        in.read(pre.data(), (std::streamsize) pre.size());
        std::ofstream o(huge, std::ios::binary);
        o.write(pre.data(), (std::streamsize) pre.size());
    }
    std::error_code ec;
    std::filesystem::resize_file(huge, std::uintmax_t(1) << 60, ec);  // exceeds the address space, sparse
    if (ec) { std::printf("  SKIP: cannot create sparse file\n"); std::remove(huge.c_str()); return; }
    const int rc = reuse_convert(huge.c_str(), (temp_dir() + "/huge_out.gguf").c_str());
    CHECK(rc != REUSE_OK);
    CHECK(std::string(reuse_last_error()).find("internal error") != std::string::npos);
    std::remove(huge.c_str());
}

static void load_on_cpu() {
    if (!safetensors()) { std::printf("  SKIP: REUSE_SAFETENSORS not set\n"); return; }
    const std::string gguf = converted();
    if (gguf.empty()) { CHECK(false); return; }
    reuse_ctx * c = reuse_load(gguf.c_str(), "CPU", 4);
    CHECK(c != nullptr);
    reuse_free(c);
    c = reuse_load(gguf.c_str(), nullptr, 2);  // NULL = CPU
    CHECK(c != nullptr);
    reuse_free(c);
    c = reuse_load(gguf.c_str(), "NoSuchDev0", 4);
    CHECK(c == nullptr);
    CHECK(std::string(reuse_last_error()).find("NoSuchDev0") != std::string::npos);
    CHECK(reuse_load((temp_dir() + "/no_such.gguf").c_str(), "CPU", 1) == nullptr);
    CHECK(std::string(reuse_last_error()).find("no_such.gguf") != std::string::npos);
    CHECK(reuse_load(gguf.c_str(), "CPU", 0) == nullptr);
    reuse_free(nullptr);
}

static void weights_values_and_derived_a() {
    if (!safetensors()) { std::printf("  SKIP: REUSE_SAFETENSORS not set\n"); return; }
    const std::string gguf = converted();
    if (gguf.empty()) { CHECK(false); return; }
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(cpu != nullptr);
    if (!cpu) return;
    {
        Weights w;
        std::string err;
        CHECK(w.load(gguf, cpu, err));
        if (!err.empty()) std::fprintf(stderr, "  %s\n", err.c_str());
        CHECK(w.hp.num_tfmamba == 30 && w.hp.hid_feature == 64 && w.hp.d_inner() == 256);
        CHECK(w.size() == 1416 + 120);
        const std::string p = "TSMamba.0.time_mamba.forward_blocks.";
        CHECK(w.has(p + "A_log") && w.has(p + "A"));
        CHECK(!w.has("nope"));
        bool threw = false;
        try { w.get("nope"); } catch (const std::exception & e) { threw = std::string(e.what()).find("nope") != std::string::npos; }
        CHECK(threw);
        const ggml_tensor * al = w.get(p + "A_log");
        const ggml_tensor * a = w.get(p + "A");
        CHECK(al->ne[0] == 16 && al->ne[1] == 256 && a->ne[0] == 16 && a->ne[1] == 256);
        std::vector<float> vl(16 * 256), va(16 * 256);
        ggml_backend_tensor_get(al, vl.data(), 0, ggml_nbytes(al));
        ggml_backend_tensor_get(a, va.data(), 0, ggml_nbytes(a));
        double maxerr = 0;
        bool all_neg = true;
        for (size_t i = 0; i < vl.size(); ++i) {
            maxerr = std::fmax(maxerr, std::fabs(va[i] + std::exp((double) vl[i])));
            all_neg = all_neg && va[i] < 0;
        }
        CHECK(all_neg);
        CHECK(maxerr < 1e-6);
    }
    ggml_backend_free(cpu);
}

int main() {
    struct T { const char * name; std::function<void()> fn; };
    const T tests[] = {
        {"convert_writes_all_tensors", convert_writes_all_tensors},
        {"convert_refuses_wrong_file", convert_refuses_wrong_file},
        {"convert_contains_exceptions", convert_contains_exceptions},
        {"load_on_cpu", load_on_cpu},
        {"weights_values_and_derived_a", weights_values_and_derived_a},
    };
    for (const auto & t : tests) {
        const int before = g_failures;
        std::printf("[ RUN  ] %s\n", t.name);
        t.fn();
        std::printf("[ %s ] %s\n", g_failures == before ? "  OK" : "FAIL", t.name);
    }
    std::printf(g_failures ? "%d check(s) failed\n" : "all passed\n", g_failures);
    return g_failures ? 1 : 0;
}
