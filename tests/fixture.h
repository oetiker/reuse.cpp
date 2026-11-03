#pragma once
// Reader for fixtures written by tools/make_fixtures.py.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Tensor {
    std::vector<int64_t> ne;  // ggml order, innermost first
    std::vector<float> data;  // little-endian float32 (host assumed little-endian)
};

inline std::map<std::string, int> load_meta(const std::string & dir) {
    std::ifstream f(dir + "/meta.txt");
    if (!f) throw std::runtime_error("cannot open " + dir + "/meta.txt");
    std::map<std::string, int> m;
    std::string key; int v;
    while (f >> key >> v) m[key] = v;
    return m;
}

inline std::map<std::string, Tensor> load_fixture(const std::string & dir) {
    std::ifstream mf(dir + "/manifest.txt");
    if (!mf) throw std::runtime_error("cannot open " + dir + "/manifest.txt");
    std::map<std::string, Tensor> out;
    std::string line;
    while (std::getline(mf, line)) {
        std::istringstream ls(line);
        std::string name, type;
        if (!(ls >> name >> type)) continue;
        if (type != "f32") throw std::runtime_error("unsupported type " + type + " for " + name);
        Tensor t;
        int64_t d, n = 1;
        while (ls >> d) { t.ne.push_back(d); n *= d; }
        const std::string path = dir + "/" + name + ".f32";
        std::ifstream df(path, std::ios::binary);
        if (!df) throw std::runtime_error("cannot open " + path);
        t.data.resize((size_t) n);
        df.read(reinterpret_cast<char *>(t.data.data()), (std::streamsize) (n * sizeof(float)));
        if (df.gcount() != (std::streamsize) (n * sizeof(float))) throw std::runtime_error("short read: " + path);
        out[name] = std::move(t);
    }
    return out;
}

inline double max_abs_diff(const float * a, const float * b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) m = std::fmax(m, std::fabs((double) a[i] - (double) b[i]));
    return m;
}

// 10*log10(sum(ref^2) / sum((ref-got)^2)); capped at 200 dB for identical signals.
inline double snr_db(const float * ref, const float * got, size_t n) {
    double s = 0, e = 0;
    for (size_t i = 0; i < n; ++i) {
        const double r = ref[i], d = r - got[i];
        s += r * r; e += d * d;
    }
    if (e <= 0) return 200.0;
    return 10.0 * std::log10(s / e);
}
