// Graph tap tests against the PyTorch fixtures (need REUSE_GGUF; skip otherwise).
// REUSE_DEVICE selects the backend device (default CPU).
#include "tap_util.h"

// Relative max error tolerance for enc_out; see tests/TOLERANCES.md.
static const double TOL_ENC = 1e-5;

static void encoder_matches_fixture() { check_tap("enc_out", "enc_out", TOL_ENC); }

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
    const TapTest tests[] = {
        {"encoder_matches_fixture", encoder_matches_fixture},
        {"run_tap_rejects_unknown_tap", run_tap_rejects_unknown_tap},
    };
    return run_tests(tests);
}
