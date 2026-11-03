// Graph tap tests against the PyTorch fixtures (need REUSE_GGUF; skip otherwise).
// REUSE_DEVICE selects the backend device (default CPU).
#include "fixture.h"
#include "graph.h"
#include "model.h"
#include "reuse.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

static int g_failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "  CHECK failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__); ++g_failures; } } while (0)

// Relative max error tolerance for enc_out; see tests/TOLERANCES.md.
static const double TOL_ENC = 1e-5;

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

static void encoder_matches_fixture() {
    if (!std::getenv("REUSE_GGUF")) { std::printf("  SKIP: REUSE_GGUF not set\n"); return; }
    reuse_ctx * ctx = model();
    CHECK(ctx != nullptr);
    if (!ctx) return;
    auto fx = load_fixture(fixtures_dir() + "/tobi-1s-24k");
    const Tensor & mag = fx.at("stft_mag");
    const Tensor & pha = fx.at("stft_pha");
    const Tensor & ref = fx.at("enc_out");
    std::vector<int64_t> ne;
    std::vector<float> got;
    try {
        got = run_tap(ctx, "enc_out", mag.data, pha.data, (int) mag.ne[0], (int) mag.ne[1], &ne);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "  run_tap threw: %s\n", e.what());
        CHECK(false);
        return;
    }
    CHECK(ne.size() >= 3 && ne[0] == ref.ne[0] && ne[1] == ref.ne[1] && ne[2] == ref.ne[2]);
    CHECK(got.size() == ref.data.size());
    if (got.size() != ref.data.size()) return;
    const double mad = max_abs_diff(ref.data.data(), got.data(), got.size());
    const double rel = rel_max_err(ref.data, got);
    std::printf("  enc_out [%s]: max_abs_diff %.3g, rel max err %.3g, snr %.1f dB\n",
                device(), mad, rel, snr_db(ref.data.data(), got.data(), got.size()));
    CHECK(rel < TOL_ENC);
}

static void run_tap_rejects_unknown_tap() {
    if (!std::getenv("REUSE_GGUF")) { std::printf("  SKIP: REUSE_GGUF not set\n"); return; }
    reuse_ctx * ctx = model();
    if (!ctx) { CHECK(false); return; }
    std::vector<float> z(16 * 8, 0.f);
    bool threw = false;
    try { run_tap(ctx, "no_such_tap", z, z, 16, 8); }
    catch (const std::exception & e) { threw = std::string(e.what()).find("no_such_tap") != std::string::npos; }
    CHECK(threw);
}

int main() {
    struct T { const char * name; std::function<void()> fn; };
    const T tests[] = {
        {"encoder_matches_fixture", encoder_matches_fixture},
        {"run_tap_rejects_unknown_tap", run_tap_rejects_unknown_tap},
    };
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
