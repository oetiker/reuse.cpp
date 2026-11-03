// Decoder and end-to-end reuse_enhance tests against the PyTorch fixtures (need REUSE_GGUF for
// everything that touches the model; skip otherwise). REUSE_DEVICE selects the backend device
// (default CPU).
#include "tap_util.h"

#include <algorithm>
#include <cstring>

// Tolerances; see tests/TOLERANCES.md.
static const double TOL_DEC = 4e-5;   // decoder taps, rel. max error
static const double SNR_MIN = 40.0;   // end-to-end output vs fixture `output`, dB

#define NEED_MODEL()                                                                 \
    if (!std::getenv("REUSE_GGUF")) { std::printf("  SKIP: REUSE_GGUF not set\n"); return; } \
    reuse_ctx * ctx = model();                                                       \
    if (!ctx) { CHECK(false); return; }

// block30_out (fixture) -> mask / phase decoders -> mag_out, pha_r, pha_i (pre-crop).
static void decoders_match_fixture() {
    NEED_MODEL();
    auto fx = load_fixture(fixtures_dir() + "/tobi-1s-24k");
    const Tensor & x = fx.at("block30_out");
    for (const char * tap : {"mag_out", "pha_r", "pha_i"}) {
        const Tensor & ref = fx.at(tap);
        std::vector<int64_t> ne;
        std::vector<float> got;
        const auto t0 = std::chrono::steady_clock::now();
        try {
            got = run_decoder_tap(ctx, tap, x.data, x.ne[0], x.ne[1], x.ne[2], &ne);
        } catch (const std::exception & e) {
            std::fprintf(stderr, "  run_decoder_tap threw: %s\n", e.what());
            CHECK(false);
            continue;
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        CHECK(ne.size() >= 3 && ne[0] == ref.ne[0] && ne[1] == ref.ne[1] && ne[2] == ref.ne[2]);
        CHECK(got.size() == ref.data.size());
        if (got.size() != ref.data.size()) continue;
        const double rel = rel_max_err(ref.data, got);
        std::printf("  %s [%s]: max_abs_diff %.3g, rel max err %.3g, snr %.1f dB, %.3f s\n", tap, device(),
                    max_abs_diff(ref.data.data(), got.data(), got.size()), rel,
                    snr_db(ref.data.data(), got.data(), got.size()), secs);
        CHECK(rel < TOL_DEC);
    }
}

// Runs reuse_enhance on fixture `name`'s input into `out` (resized to n + guard, guard filled
// with a sentinel that must survive). Returns the SNR vs the fixture output (-1 on failure).
static const size_t kGuard = 64;
static const float kSentinel = 12345.f;

static double enhance_fixture(reuse_ctx * ctx, const std::string & name, std::vector<float> & out,
                              std::vector<float> * ref_out = nullptr) {
    const std::string dir = fixtures_dir() + "/" + name;
    auto fx = load_fixture(dir);
    auto meta = load_meta(dir);
    const Tensor & in = fx.at("input");
    const Tensor & ref = fx.at("output");
    const size_t n = in.data.size();
    CHECK(ref.data.size() == n);
    out.resize(n + kGuard);
    std::fill(out.begin() + (std::ptrdiff_t) n, out.end(), kSentinel);
    const auto t0 = std::chrono::steady_clock::now();
    const int st = reuse_enhance(ctx, in.data.data(), n, meta.at("sample_rate"), out.data());
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (st != REUSE_OK) std::fprintf(stderr, "  reuse_enhance(%s) = %d: %s\n", name.c_str(), st, reuse_last_error());
    CHECK(st == REUSE_OK);
    if (st != REUSE_OK) return -1;
    bool guard_ok = true, finite = true;
    for (size_t i = n; i < n + kGuard; ++i) guard_ok = guard_ok && out[i] == kSentinel;
    for (size_t i = 0; i < n; ++i) finite = finite && std::isfinite(out[i]);
    CHECK(guard_ok);  // wrote exactly n samples
    CHECK(finite);
    const double snr = snr_db(ref.data.data(), out.data(), n);
    std::printf("  %s [%s]: %zu samples @ %d Hz, snr %.1f dB, max_abs_diff %.3g, %.1f s\n", name.c_str(), device(),
                n, meta.at("sample_rate"), snr, max_abs_diff(ref.data.data(), out.data(), n), secs);
    if (ref_out) *ref_out = ref.data;
    return snr;
}

// Same ctx, two different inputs, one output buffer: each result must match its own reference
// and not the other one (a no-op that leaves the previous output in place fails the second).
static void enhance_matches_reference_twice() {
    NEED_MODEL();
    std::vector<float> out, ref_tobi, ref_petra;
    const double s1 = enhance_fixture(ctx, "tobi-4s-24k", out, &ref_tobi);
    CHECK(s1 > SNR_MIN);
    std::vector<float> out_tobi(out.begin(), out.begin() + (std::ptrdiff_t) ref_tobi.size());
    const double s2 = enhance_fixture(ctx, "petra-4s-24k", out, &ref_petra);
    CHECK(s2 > SNR_MIN);
    if (s1 < 0 || s2 < 0 || ref_tobi.size() != ref_petra.size()) { CHECK(ref_tobi.size() == ref_petra.size()); return; }
    const size_t n = ref_tobi.size();
    const double x1 = snr_db(ref_petra.data(), out_tobi.data(), n);
    const double x2 = snr_db(ref_tobi.data(), out.data(), n);
    std::printf("  cross: tobi output vs petra ref %.1f dB, petra output vs tobi ref %.1f dB\n", x1, x2);
    CHECK(x1 < 3.0);
    CHECK(x2 < 3.0);
}

static void any_rate_keeps_length() {
    NEED_MODEL();
    std::vector<float> out, ref;
    CHECK(enhance_fixture(ctx, "tobi-3s-16k", out) > SNR_MIN);
    CHECK(enhance_fixture(ctx, "tobi-2s-44k1", out, &ref) > SNR_MIN);
    // torch.istft leaves the tail after the last full frame uncovered; run.py pads it with 1e-8.
    size_t tail_ref = 0, tail_got = 0;
    for (size_t i = ref.size(); i-- > 0 && ref[i] == 1e-8f;) ++tail_ref;
    for (size_t i = ref.size(); i-- > 0 && out[i] == 1e-8f;) ++tail_got;
    std::printf("  44.1 kHz 1e-8 tail: reference %zu, got %zu samples\n", tail_ref, tail_got);
    CHECK(tail_got == tail_ref);
}

static void too_short_input_is_refused() {
    NEED_MODEL();
    std::vector<float> in(1000, 0.1f), out(1000, 0.f);
    for (size_t n : {(size_t) 0, (size_t) 100, (size_t) 480}) {  // 24 kHz: n_fft 960, needs n >= 481
        const int st = reuse_enhance(ctx, in.data(), n, 24000, out.data());
        std::printf("  n=%zu: status %d, \"%s\"\n", n, st, reuse_last_error());
        CHECK(st == REUSE_ERR_TOO_SHORT);
        CHECK(std::strlen(reuse_last_error()) > 0);
    }
    CHECK(reuse_enhance(ctx, in.data(), 481, 24000, out.data()) == REUSE_OK);
}

static void bad_arguments_are_refused() {
    std::vector<float> in(24000, 0.1f), out(24000, 0.f);
    CHECK(reuse_enhance(nullptr, in.data(), in.size(), 24000, out.data()) == REUSE_ERR_ARG);
    CHECK(std::strlen(reuse_last_error()) > 0);
    NEED_MODEL();
    for (int sr : {0, 199, 7999, 192001, -24000}) {
        const int st = reuse_enhance(ctx, in.data(), in.size(), sr, out.data());
        std::printf("  sr=%d: status %d, \"%s\"\n", sr, st, reuse_last_error());
        CHECK(st == REUSE_ERR_ARG);
    }
    CHECK(reuse_enhance(ctx, nullptr, in.size(), 24000, out.data()) == REUSE_ERR_ARG);
    CHECK(reuse_enhance(ctx, in.data(), in.size(), 24000, nullptr) == REUSE_ERR_ARG);
}

// All-zero input: the STFT input equals torch's exactly (magnitude 0, phase atan2(0,0) = 0), but
// the model still predicts positive log magnitudes in nearly every bin, so the zero-frame rule
// does not fire and the output is low-level noise. PyTorch (tools/run.py path, CPU) gives
// max |y| = 0.04242 for 24000 zeros at 24 kHz; the port must be finite and match that level.
static void silence_stays_finite_at_reference_level() {
    NEED_MODEL();
    std::vector<float> in(24000, 0.f), out(24000, 1.f);
    CHECK(reuse_enhance(ctx, in.data(), in.size(), 24000, out.data()) == REUSE_OK);
    double peak = 0;
    bool finite = true;
    for (float v : out) { finite = finite && std::isfinite(v); peak = std::fmax(peak, std::fabs((double) v)); }
    std::printf("  silence: max |y| = %.4g (PyTorch reference 0.04242)\n", peak);
    CHECK(finite);
    CHECK(std::fabs(peak - 0.04242) < 0.004);
}

int main() {
    const TapTest tests[] = {
        {"bad_arguments_are_refused", bad_arguments_are_refused},
        {"too_short_input_is_refused", too_short_input_is_refused},
        {"decoders_match_fixture", decoders_match_fixture},
        {"silence_stays_finite_at_reference_level", silence_stays_finite_at_reference_level},
        {"enhance_matches_reference_twice", enhance_matches_reference_twice},
        {"any_rate_keeps_length", any_rate_keeps_length},
    };
    return run_tests(tests);
}
