// Windowed reuse_enhance for long inputs (src/window.h, src/enhance.cpp). The window and overlap
// lengths are overridden through reuse_ctx so short fixtures exercise the windowing. Model tests
// need REUSE_GGUF (skip otherwise); REUSE_DEVICE selects the backend device (default CPU).
#include "tap_util.h"
#include "window.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#define NEED_MODEL()                                                                 \
    if (!std::getenv("REUSE_GGUF")) { std::printf("  SKIP: REUSE_GGUF not set\n"); return; } \
    reuse_ctx * ctx = model();                                                       \
    if (!ctx) { CHECK(false); return; }

static const double SNR_WINDOWED_MIN = 15.0;  // windowed vs single-pass output, dB (see TOLERANCES.md)
static const double SEAM_FACTOR = 5.0;        // |diff| at a seam vs 99th percentile of |diff|

// RAII: restores the default window lengths.
struct WindowOverride {
    reuse_ctx * c;
    WindowOverride(reuse_ctx * ctx, double w, double o) : c(ctx) { c->window_seconds = w; c->overlap_seconds = o; }
    ~WindowOverride() { c->window_seconds = 20.0; c->overlap_seconds = 2.0; }
};

static void window_plan_properties() {
    struct Case { size_t n, window, overlap, min_len; };
    const Case cases[] = {
        {1000, 1000, 100, 481},        // n == window: single window
        {500, 1000, 100, 481},         // n < window
        {1001, 1000, 100, 100},        // just above: 2 windows
        {48000, 48000, 4800, 481},
        {480001, 480000, 48000, 481},  // 20 s / 2 s at 24 kHz, one sample over
        {1440000, 480000, 48000, 481}, // 60 s
        {1440000 + 7, 480000, 48000, 481},
        {96000, 48000, 12000, 481},
        {96001, 48000, 12000, 481},
        {1234567, 480000, 48000, 481},
        {441000 * 7 + 3, 441000, 88200, 1103},  // 44.1 kHz
        {160000 * 13 + 11, 320000, 32000, 241}, // 16 kHz
    };
    for (const Case & c : cases) {
        std::vector<Window> w;
        try {
            w = plan_windows(c.n, c.window, c.overlap, c.min_len);
        } catch (const std::exception & e) {
            std::fprintf(stderr, "  plan(%zu,%zu,%zu) threw: %s\n", c.n, c.window, c.overlap, e.what());
            CHECK(false);
            continue;
        }
        CHECK(!w.empty());
        if (w.empty()) continue;
        bool ok = w.front().start == 0 && w.back().start + w.back().len == c.n;
        if (c.n <= c.window) ok = ok && w.size() == 1;
        size_t shortest = c.n, longest = 0;
        for (size_t i = 0; i < w.size(); ++i) {
            ok = ok && w[i].len <= c.window && w[i].len >= c.min_len;
            if (w.size() > 1) ok = ok && w[i].len >= c.overlap + c.min_len;
            shortest = std::min(shortest, w[i].len);
            longest = std::max(longest, w[i].len);
            if (i + 1 < w.size()) {  // overlap exactly `overlap`, no gap
                ok = ok && w[i].start + w[i].len == w[i + 1].start + c.overlap;
                ok = ok && w[i + 1].start > w[i].start;
            }
        }
        CHECK(ok);
        // Even distribution: the last window is at most k-1 samples shorter than the others.
        CHECK(longest - shortest < w.size());
        std::printf("  n=%zu window=%zu overlap=%zu: %zu windows, len %zu..%zu\n", c.n, c.window, c.overlap, w.size(),
                    shortest, longest);
    }
    // Invalid configurations are refused (window too small for the overlap), not mis-planned.
    bool threw = false;
    try { plan_windows(5000, 1000, 600, 100); } catch (const std::invalid_argument &) { threw = true; }
    CHECK(threw);
}

// The crossfade is amplitude-complementary: windows that all carry the same signal x (constant,
// ramp, sine) add up to x again, the weights are monotone across an overlap, and the weights of
// a window's head and tail never meet. No model needed; catches a missing or wrong crossfade.
static void crossfade_reconstructs_signal() {
    struct Case { size_t n, window, overlap; };
    const Case cases[] = {{1001, 1000, 100}, {96000, 48000, 12000}, {1234567 / 10, 48000, 4800}, {20000, 4000, 1000}};
    for (const Case & c : cases) {
        const std::vector<Window> plan = plan_windows(c.n, c.window, c.overlap, 100);
        for (int kind = 0; kind < 3; ++kind) {
            std::vector<float> x(c.n), acc(c.n, 0.f);
            for (size_t i = 0; i < c.n; ++i)
                x[i] = kind == 0 ? 1.f : kind == 1 ? (float) i / (float) c.n : std::sin(0.01f * (float) i);
            for (size_t k = 0; k < plan.size(); ++k)
                accumulate_window(acc.data(), x.data() + plan[k].start, plan[k], k > 0, k + 1 < plan.size(), c.overlap);
            double err = 0;
            for (size_t i = 0; i < c.n; ++i) err = std::fmax(err, std::fabs((double) acc[i] - x[i]));
            CHECK(err < 1e-5);
            if (kind == 0) std::printf("  n=%zu: %zu windows, constant 1 reconstructed within %.2g\n", c.n, plan.size(), err);
        }
        // Windows carrying different signals (alternating +1 / -1) are joined without a step:
        // the largest sample-to-sample change is that of the raised cosine, pi/overlap.
        {
            std::vector<float> acc(c.n, 0.f);
            for (size_t k = 0; k < plan.size(); ++k) {
                const std::vector<float> y(plan[k].len, k % 2 ? -1.f : 1.f);
                accumulate_window(acc.data(), y.data(), plan[k], k > 0, k + 1 < plan.size(), c.overlap);
            }
            double step = 0;
            for (size_t i = 0; i + 1 < c.n; ++i) step = std::fmax(step, std::fabs((double) acc[i + 1] - acc[i]));
            CHECK(step <= 3.2 / (double) c.overlap);
            if (plan.size() > 1) CHECK(step > 1.0 / (double) c.overlap);  // the crossfade is really there
        }
        // Window 0's weight falls monotonically from 1 towards 0 across the first overlap.
        if (plan.size() > 1) {
            std::vector<float> one(c.n, 1.f), acc(c.n, 0.f);
            accumulate_window(acc.data(), one.data(), plan[0], false, true, c.overlap);
            const size_t o = plan[0].len - c.overlap;
            bool mono = acc[o] == 1.f && acc[plan[0].len - 1] < 0.01f;
            for (size_t i = o; i + 1 < plan[0].len; ++i) mono = mono && acc[i + 1] <= acc[i];
            CHECK(mono);
        }
    }
}

static std::vector<float> run(reuse_ctx * ctx, const std::vector<float> & in, int sr) {
    std::vector<float> out(in.size());
    const int st = reuse_enhance(ctx, in.data(), in.size(), sr, out.data());
    if (st != REUSE_OK) std::fprintf(stderr, "  reuse_enhance = %d: %s\n", st, reuse_last_error());
    CHECK(st == REUSE_OK);
    return out;
}

static bool all_finite(const std::vector<float> & v) {
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

// Sequence of the two tests below shares one expensive single-pass + windowed run.
struct WindowedRun {
    std::vector<float> single, windowed;
    size_t n = 0;
    int sr = 0;
    int windows = 0;
    size_t overlap = 0;
    bool ok = false;
};

static WindowedRun & windowed_run(reuse_ctx * ctx) {
    static WindowedRun r;
    static bool done = false;
    if (done) return r;
    done = true;
    const std::string dir = fixtures_dir() + "/tobi-4s-24k";
    auto fx = load_fixture(dir);
    r.sr = load_meta(dir).at("sample_rate");
    const std::vector<float> in = fx.at("input").data;
    r.n = in.size();
    r.single = run(ctx, in, r.sr);
    CHECK(ctx->last_window_count == 1);
    {
        WindowOverride ov(ctx, 2.0, 0.5);
        r.windowed = run(ctx, in, r.sr);
        r.windows = ctx->last_window_count;
    }
    r.overlap = (size_t) std::llround(0.5 * r.sr);
    r.ok = true;
    return r;
}

static void windowed_keeps_length_and_is_close() {
    NEED_MODEL();
    WindowedRun & r = windowed_run(ctx);
    if (!r.ok) return;
    CHECK(r.windowed.size() == r.n);
    CHECK(r.windows > 1);
    CHECK(all_finite(r.windowed));
    const double snr = snr_db(r.single.data(), r.windowed.data(), r.n);
    std::printf("  tobi-4s-24k [%s]: %d windows (2 s / 0.5 s), windowed vs single-pass snr %.1f dB\n", device(),
                r.windows, snr);
    CHECK(snr > SNR_WINDOWED_MIN);
}

// Sanity check on real model output. Speech is wideband, so |diff| at a seam is not far below
// the clip's 99th percentile and this check is weak (a hard cut passes too); the deterministic
// proof that the crossfade is smooth is crossfade_reconstructs_signal.
static void windowed_has_no_seam_jump() {
    NEED_MODEL();
    WindowedRun & r = windowed_run(ctx);
    if (!r.ok) return;
    const size_t window = (size_t) std::llround(2.0 * r.sr);
    const std::vector<Window> plan = plan_windows(r.n, window, r.overlap, min_input_samples(r.sr));
    CHECK(plan.size() > 1);
    std::vector<float> d(r.n - 1);
    for (size_t i = 0; i + 1 < r.n; ++i) d[i] = std::fabs(r.windowed[i + 1] - r.windowed[i]);
    std::vector<float> sorted = d;
    std::sort(sorted.begin(), sorted.end());
    const double p99 = sorted[(size_t) (0.99 * (double) (sorted.size() - 1))];
    double worst = 0;
    for (size_t i = 1; i < plan.size(); ++i) {
        const size_t centre = plan[i].start + r.overlap / 2;
        // Largest |diff| within +-2 samples of the seam centre.
        double m = 0;
        for (size_t k = centre - 2; k <= centre + 2; ++k) m = std::fmax(m, d[k]);
        worst = std::fmax(worst, m);
        std::printf("  seam %zu at %zu: |diff| %.3g (p99 over clip %.3g)\n", i, centre, m, p99);
    }
    CHECK(worst <= SEAM_FACTOR * p99);
}

static void short_input_unchanged() {
    NEED_MODEL();
    const std::string dir = fixtures_dir() + "/tobi-1s-24k";
    auto fx = load_fixture(dir);
    const int sr = load_meta(dir).at("sample_rate");
    const std::vector<float> in = fx.at("input").data;
    // Reference: the single-window path called directly, bypassing the window planner.
    std::vector<float> direct(in.size());
    enhance_window(ctx, in.data(), in.size(), sr, direct.data());
    const std::vector<float> got = run(ctx, in, sr);  // defaults: 20 s window
    CHECK(ctx->last_window_count == 1);
    CHECK(got.size() == direct.size() && std::memcmp(got.data(), direct.data(), got.size() * sizeof(float)) == 0);
    // Exactly one window long is still one window.
    {
        WindowOverride ov(ctx, (double) in.size() / sr, 0.25);
        const std::vector<float> got2 = run(ctx, in, sr);
        CHECK(ctx->last_window_count == 1);
        CHECK(std::memcmp(got2.data(), direct.data(), got2.size() * sizeof(float)) == 0);
    }
    // Aliased in/out stays valid on the windowed path.
    {
        WindowOverride ov(ctx, 0.5, 0.125);
        std::vector<float> buf = in;
        CHECK(reuse_enhance(ctx, buf.data(), buf.size(), sr, buf.data()) == REUSE_OK);
        CHECK(ctx->last_window_count > 1);
        CHECK(all_finite(buf));
    }
}

int main() {
    const TapTest tests[] = {
        {"window_plan_properties", window_plan_properties},
        {"crossfade_reconstructs_signal", crossfade_reconstructs_signal},
        {"short_input_unchanged", short_input_unchanged},
        {"windowed_keeps_length_and_is_close", windowed_keeps_length_and_is_close},
        {"windowed_has_no_seam_jump", windowed_has_no_seam_jump},
    };
    return run_tests(tests);
}
