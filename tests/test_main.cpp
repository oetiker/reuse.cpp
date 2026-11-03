#include "fixture.h"
#include "reuse.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static void fixture_reader_reads_manifest() {
    const std::string dir = fixtures_dir() + "/tobi-1s-24k";
    auto meta = load_meta(dir);
    auto fx = load_fixture(dir);
    CHECK(meta.count("n_samples") == 1);
    CHECK(fx.count("input") == 1 && fx.count("enc_out") == 1);
    if (g_failures) return;
    CHECK(fx["input"].ne == std::vector<int64_t>{meta["n_samples"]});
    CHECK(fx["input"].data.size() == (size_t) meta["n_samples"]);
    CHECK(fx["enc_out"].ne.size() == 3);
    CHECK(fx["enc_out"].data.size() == (size_t) (241 * 51 * 64));
}

static void metrics_helpers() {
    const float a[3] = {1.f, 2.f, 3.f};
    const float b[3] = {1.f, 2.5f, 3.f};
    CHECK(max_abs_diff(a, b, 3) == 0.5);
    CHECK(snr_db(a, a, 3) > 100.0);
    CHECK(snr_db(a, b, 3) < 100.0);
}

static void enhance_stub_reports_not_implemented() {
    CHECK(reuse_enhance(nullptr, nullptr, 0, 16000, nullptr) == REUSE_ERR_MODEL);
    CHECK(std::strcmp(reuse_last_error(), "not implemented") == 0);
}

static void gguf_tests() {
    if (!std::getenv("REUSE_GGUF")) {
        std::printf("  SKIP: REUSE_GGUF not set\n");
        return;
    }
    // Later tasks add model-dependent tests here.
}

int main() {
    struct T { const char * name; std::function<void()> fn; };
    const T tests[] = {
        {"fixture_reader_reads_manifest", fixture_reader_reads_manifest},
        {"metrics_helpers", metrics_helpers},
        {"enhance_stub_reports_not_implemented", enhance_stub_reports_not_implemented},
        {"gguf_tests", gguf_tests},
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
