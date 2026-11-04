// reuse-bench: load time, enhance wall time / real-time factor and memory use of reuse.cpp.
// Usage: reuse-bench [--threads N] <gguf> <device> <wav> [repeat=3] [out.wav]
#include "model.h"
#include "reuse.h"

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

uint32_t rd32(const unsigned char * p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t) p[3] << 24; }
uint16_t rd16(const unsigned char * p) { return (uint16_t) (p[0] | p[1] << 8); }

// Minimal RIFF/WAVE reader: PCM16 or float32, any channel count (averaged to mono), skips unknown chunks.
bool read_wav(const std::string & path, std::vector<float> & mono, int & rate, std::string & err) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) { err = path + ": cannot open"; return false; }
    std::vector<unsigned char> b;
    unsigned char buf[65536];
    size_t k;
    while ((k = std::fread(buf, 1, sizeof buf, f)) > 0) b.insert(b.end(), buf, buf + k);
    std::fclose(f);
    if (b.size() < 12 || std::memcmp(b.data(), "RIFF", 4) || std::memcmp(b.data() + 8, "WAVE", 4)) {
        err = path + ": not a RIFF/WAVE file";
        return false;
    }
    int fmt = 0, ch = 0, bits = 0;
    rate = 0;
    size_t pos = 12;
    while (pos + 8 <= b.size()) {
        const unsigned char * h = &b[pos];
        size_t sz = rd32(h + 4);
        const size_t body = pos + 8;
        if (!std::memcmp(h, "fmt ", 4) && body + 16 <= b.size()) {
            fmt = rd16(&b[body]);
            ch = rd16(&b[body + 2]);
            rate = (int) rd32(&b[body + 4]);
            bits = rd16(&b[body + 14]);
            if (fmt == 0xFFFE && sz >= 26 && body + 26 <= b.size()) fmt = rd16(&b[body + 24]);  // extensible
        } else if (!std::memcmp(h, "data", 4)) {
            if (body + sz > b.size()) sz = b.size() - body;  // truncated / streamed header
            const bool pcm16 = fmt == 1 && bits == 16, f32 = fmt == 3 && bits == 32;
            if (!pcm16 && !f32) { err = path + ": only 16-bit PCM or 32-bit float WAV supported"; return false; }
            if (ch < 1 || rate < 1) { err = path + ": missing or invalid fmt chunk"; return false; }
            const size_t bps = bits / 8, frames = sz / (bps * ch);
            mono.assign(frames, 0.f);
            for (size_t i = 0; i < frames; ++i) {
                float acc = 0;
                for (int c = 0; c < ch; ++c) {
                    const unsigned char * s = &b[body + (i * ch + c) * bps];
                    if (pcm16) acc += (int16_t) rd16(s) / 32768.f;
                    else { uint32_t u = rd32(s); float v; std::memcpy(&v, &u, 4); acc += v; }
                }
                mono[i] = acc / ch;
            }
            return true;
        }
        pos = body + sz + (sz & 1);
    }
    err = path + ": no data chunk";
    return false;
}

void put32(std::vector<unsigned char> & v, uint32_t x) { for (int i = 0; i < 4; ++i) v.push_back((x >> (8 * i)) & 255); }
void put16(std::vector<unsigned char> & v, uint16_t x) { v.push_back(x & 255); v.push_back(x >> 8); }

bool write_wav(const std::string & path, const std::vector<float> & s, int rate) {
    std::vector<unsigned char> v;
    const uint32_t dbytes = (uint32_t) (s.size() * 4);
    v.insert(v.end(), {'R', 'I', 'F', 'F'}); put32(v, 36 + dbytes);
    v.insert(v.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '}); put32(v, 16);
    put16(v, 3); put16(v, 1); put32(v, (uint32_t) rate); put32(v, (uint32_t) rate * 4); put16(v, 4); put16(v, 32);
    v.insert(v.end(), {'d', 'a', 't', 'a'}); put32(v, dbytes);
    const unsigned char * p = (const unsigned char *) s.data();
    v.insert(v.end(), p, p + dbytes);  // little-endian host assumed
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(v.data(), 1, v.size(), f) == v.size();
    return std::fclose(f) == 0 && ok;
}

double max_rss_mib() {
    rusage ru;
    getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    return ru.ru_maxrss / 1048576.0;  // bytes
#else
    return ru.ru_maxrss / 1024.0;     // KiB
#endif
}

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char ** argv) {
    int threads = (int) std::min(8u, std::max(1u, std::thread::hardware_concurrency()));
    std::vector<std::string> a;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::max(1, std::atoi(argv[++i]));
        else a.push_back(argv[i]);
    }
    if (a.size() < 3 || a.size() > 5) {
        std::fprintf(stderr, "usage: %s [--threads N] <gguf> <device> <wav> [repeat=3] [out.wav]\n", argv[0]);
        return 2;
    }
    const int repeat = a.size() > 3 ? std::max(1, std::atoi(a[3].c_str())) : 3;

    std::vector<float> in;
    int rate = 0;
    std::string err;
    if (!read_wav(a[2], in, rate, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    const double audio_s = (double) in.size() / rate;
    std::printf("input: %s  %.2f s @ %d Hz, device %s, threads %d\n", a[2].c_str(), audio_s, rate, a[1].c_str(), threads);

    double t0 = now();
    reuse_ctx * ctx = reuse_load(a[0].c_str(), a[1].c_str(), threads);
    if (!ctx) { std::fprintf(stderr, "error: reuse_load: %s\n", reuse_last_error()); return 1; }
    std::printf("load: %.3f s\n", now() - t0);

    std::vector<float> out(in.size());
    for (int r = 0; r < repeat; ++r) {
        t0 = now();
        int rc = reuse_enhance(ctx, in.data(), in.size(), rate, out.data());
        const double dt = now() - t0;
        if (rc != REUSE_OK) {
            std::fprintf(stderr, "error: reuse_enhance: %s\n", reuse_last_error());
            reuse_free(ctx);
            return 1;
        }
        std::printf("run %d: %.3f s  RTF %.2fx  compute buffer %.1f MiB\n", r + 1, dt, audio_s / dt,
                    ctx->last_compute_bytes / 1048576.0);
    }
    std::printf("weights buffer: %.1f MiB\nmax RSS: %.1f MiB\n", ctx->weights.nbytes() / 1048576.0, max_rss_mib());

    int rc = 0;
    if (a.size() > 4) {
        if (!write_wav(a[4], out, rate)) { std::fprintf(stderr, "error: cannot write %s\n", a[4].c_str()); rc = 1; }
        else std::printf("wrote %s\n", a[4].c_str());
    }
    reuse_free(ctx);
    return rc;
}
