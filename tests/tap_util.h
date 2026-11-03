#pragma once
// Shared helpers for the graph tap tests (test_encoder, test_mamba): fixture location, device
// selection, one model per process, error metrics and a tiny CHECK/test runner.
#include "fixture.h"
#include "graph.h"
#include "model.h"
#include "reuse.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

static int g_failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "  CHECK failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__); ++g_failures; } } while (0)

static std::string fixtures_dir() {
    if (const char * e = std::getenv("REUSE_FIXTURES")) return e;
    if (const char * e = std::getenv("REUSE_FIXTURES_DEFAULT")) return e;
    return "tests/fixtures";
}

static const char * device() {
    const char * d = std::getenv("REUSE_DEVICE");
    return d ? d : "CPU";
}

// Loads the model once per process (nullptr if REUSE_GGUF is unset or loading failed).
static reuse_ctx * model() {
    static reuse_ctx * ctx = nullptr;
    static bool tried = false;
    if (!tried) {
        tried = true;
        ctx = reuse_load(std::getenv("REUSE_GGUF"), device(), 4);
        if (!ctx) std::fprintf(stderr, "  reuse_load failed: %s\n", reuse_last_error());
    }
    return ctx;
}

// max |ref - got| / max |ref|
static double rel_max_err(const std::vector<float> & ref, const std::vector<float> & got) {
    double peak = 0;
    for (float v : ref) peak = std::fmax(peak, std::fabs((double) v));
    return max_abs_diff(ref.data(), got.data(), ref.size()) / (peak > 0 ? peak : 1.0);
}

// Runs `tap` on the tobi-1s-24k STFT input and compares against fixture `ref_name`; asserts
// shape equality and rel max error < tol. Prints errors and wall time. Skips without REUSE_GGUF.
static void check_tap(const std::string & tap, const std::string & ref_name, double tol) {
    if (!std::getenv("REUSE_GGUF")) { std::printf("  SKIP: REUSE_GGUF not set\n"); return; }
    reuse_ctx * ctx = model();
    CHECK(ctx != nullptr);
    if (!ctx) return;
    auto fx = load_fixture(fixtures_dir() + "/tobi-1s-24k");
    const Tensor & mag = fx.at("stft_mag");
    const Tensor & pha = fx.at("stft_pha");
    const Tensor & ref = fx.at(ref_name);
    std::vector<int64_t> ne;
    std::vector<float> got;
    const auto t0 = std::chrono::steady_clock::now();
    try {
        got = run_tap(ctx, tap, mag.data, pha.data, (int) mag.ne[0], (int) mag.ne[1], &ne);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "  run_tap threw: %s\n", e.what());
        CHECK(false);
        return;
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(ne.size() >= 3 && ne[0] == ref.ne[0] && ne[1] == ref.ne[1] && ne[2] == ref.ne[2]);
    CHECK(got.size() == ref.data.size());
    if (got.size() != ref.data.size()) return;
    const double mad = max_abs_diff(ref.data.data(), got.data(), got.size());
    const double rel = rel_max_err(ref.data, got);
    std::printf("  %s [%s]: max_abs_diff %.3g, rel max err %.3g, snr %.1f dB, %.3f s (incl. graph build)\n",
                tap.c_str(), device(), mad, rel, snr_db(ref.data.data(), got.data(), got.size()), secs);
    CHECK(rel < tol);
}

struct TapTest { const char * name; std::function<void()> fn; };

template <size_t N>
static int run_tests(const TapTest (&tests)[N]) {
    for (const auto & t : tests) {
        const int before = g_failures;
        std::printf("[ RUN  ] %s\n", t.name);
        t.fn();
        std::printf("[ %s ] %s\n", g_failures == before ? "  OK" : "FAIL", t.name);
    }
    if (std::getenv("REUSE_GGUF") && model()) reuse_free(model());
    std::printf(g_failures ? "%d check(s) failed\n" : "all passed\n", g_failures);
    return g_failures ? 1 : 0;
}
