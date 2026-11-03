#include "fixture.h"
#include "stft.h"

#include <algorithm>
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

static void stft_sizes_follow_make_even() {
    struct C { int sr, n_fft, hop, win; };
    const C cs[] = {{24000, 960, 120, 960}, {16000, 640, 80, 640}, {44100, 1764, 220, 1764},
                    {8000, 320, 40, 320}, {11025, 442, 56, 442}};
    for (const C & c : cs) {
        const StftSizes s = stft_sizes(c.sr);
        CHECK(s.n_fft == c.n_fft && s.hop == c.hop && s.win == c.win);
    }
}

static void stft_matches_fixture() {
    const std::string dir = fixtures_dir() + "/tobi-1s-24k";
    auto meta = load_meta(dir);
    auto fx = load_fixture(dir);
    const StftSizes s = stft_sizes(meta["sample_rate"]);
    CHECK(s.n_fft == meta["n_fft"] && s.hop == meta["hop"] && s.win == meta["win"]);
    const Tensor & in = fx["input"];
    std::vector<float> mag, pha;
    int nf = 0, nt = 0;
    stft(in.data.data(), in.data.size(), s, mag, pha, nf, nt);
    CHECK(nf == fx["stft_mag"].ne[0] && nt == fx["stft_mag"].ne[1]);
    if (nf != fx["stft_mag"].ne[0] || nt != fx["stft_mag"].ne[1]) return;
    const size_t n = (size_t) nf * nt;
    const double dm = max_abs_diff(mag.data(), fx["stft_mag"].data.data(), n);
    double dp = 0, dpt[4] = {0,0,0,0};
    const double tier[4] = {1e-4, 1e-3, 1e-2, 1e-1};
    for (size_t i = 0; i < n; ++i) {
        if (std::expm1(fx["stft_mag"].data[i]) <= 1e-4) continue;  // phase is meaningless near zero
        double d = std::fabs((double) pha[i] - fx["stft_pha"].data[i]);
        if (d > M_PI) d = std::fabs(d - 2 * M_PI);
        { const double lm = std::expm1(fx["stft_mag"].data[i]); for (int j = 0; j < 4; ++j) if (lm > tier[j]) dpt[j] = std::fmax(dpt[j], d); }
        dp = std::fmax(dp, d);
    }
    std::printf("  mag max diff %.3g, phase max diff %.3g\n", dm, dp);
    std::printf("  phase max diff by linear-mag floor: >1e-4 %.3g, >1e-3 %.3g, >1e-2 %.3g, >1e-1 %.3g\n", dpt[0], dpt[1], dpt[2], dpt[3]);
    CHECK(dm < 1e-4);
    // The float32 torch reference has ~3e-7 absolute FFT noise, so the phase error of a bin scales
    // as 3e-7/|X|: 3e-3 rad at |X|~2e-4, but < 1e-3 rad above |X|=1e-3. Gate accordingly.
    CHECK(dpt[1] < 1e-3);
}

static void roundtrip(const char * name) {
    const std::string dir = fixtures_dir() + "/" + name;
    auto meta = load_meta(dir);
    auto fx = load_fixture(dir);
    const Tensor & in = fx["input"];
    const StftSizes s = stft_sizes(meta["sample_rate"]);
    std::vector<float> mag, pha;
    int nf = 0, nt = 0;
    stft(in.data.data(), in.data.size(), s, mag, pha, nf, nt);
    std::vector<float> c(pha.size()), sn(pha.size());
    for (size_t i = 0; i < pha.size(); ++i) { c[i] = std::cos(pha[i]); sn[i] = std::sin(pha[i]); }
    auto out = istft(mag.data(), c.data(), sn.data(), nf, nt, s, in.data.size());
    CHECK(out.size() == in.data.size());
    if (out.size() != in.data.size()) return;
    // torch.istft yields hop*(n_frames-1) samples; a tail shorter than one hop is never
    // reconstructed (the reference pads it with 1e-8), so SNR is measured on the covered part.
    const size_t covered = std::min((size_t) s.hop * (nt - 1), in.data.size());
    const double snr = snr_db(in.data.data(), out.data(), covered);
    std::printf("  %s roundtrip SNR %.1f dB (%zu of %zu samples covered)\n", name, snr, covered, in.data.size());
    for (size_t i = covered; i < out.size(); ++i) CHECK(out[i] == 1e-8f);
    CHECK(snr > 80.0);
}

static void istft_roundtrip() {
    roundtrip("tobi-1s-24k");
    roundtrip("tobi-2s-44k1");
}

int main() {
    struct T { const char * name; std::function<void()> fn; };
    const T tests[] = {
        {"stft_sizes_follow_make_even", stft_sizes_follow_make_even},
        {"stft_matches_fixture", stft_matches_fixture},
        {"istft_roundtrip", istft_roundtrip},
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
