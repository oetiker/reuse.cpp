// safetensors -> GGUF converter for the NVIDIA RE-USE model.
// Tensor names are kept as in torch; data is F32 copied verbatim (little-endian host assumed);
// the torch shape is reversed into ggml's ne order (innermost first).
#include "reuse.h"
#include "error.h"

#include "ggml.h"
#include "gguf.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

// Model constants (validated against the safetensors).
constexpr int    kNumTfmamba = 30;
constexpr int    kHidFeature = 64;
constexpr int    kDState     = 16;
constexpr int    kDConv      = 4;
constexpr int    kExpand     = 4;
constexpr int    kDtRank     = 4;
constexpr int    kBaseRate   = 8000;
constexpr int    kNFft       = 320;
constexpr int    kHop        = 40;
constexpr int    kWin        = 320;
constexpr size_t kNumTensors = 1416;
constexpr uint64_t kNumParams = 9607747;
constexpr const char * kSourceRevision = "31f82f2efad1095034492f236e8dac9239ba713b";

struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;  // torch order
    uint64_t begin = 0, end = 0; // relative to the data section
};

// Minimal JSON reader for the safetensors header.
struct Json {
    const char * p;
    const char * e;
    std::string err;

    bool fail(const char * msg) { if (err.empty()) err = msg; return false; }
    void ws() { while (p < e && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) ++p; }
    bool eat(char c) { ws(); if (p < e && *p == c) { ++p; return true; } return false; }
    bool peek(char c) { ws(); return p < e && *p == c; }

    bool str(std::string & out) {
        ws();
        if (p >= e || *p != '"') return fail("expected string");
        ++p;
        out.clear();
        while (p < e && *p != '"') {
            if (*p == '\\') {
                if (++p >= e) break;
                switch (*p) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'u': return fail("unsupported \\u escape");
                    default:  out += *p; break;
                }
            } else {
                out += *p;
            }
            ++p;
        }
        if (p >= e) return fail("unterminated string");
        ++p;
        return true;
    }
    bool u64(uint64_t & out) {
        ws();
        if (p >= e || *p < '0' || *p > '9') return fail("expected unsigned integer");
        uint64_t v = 0;
        while (p < e && *p >= '0' && *p <= '9') {
            const uint64_t d = (uint64_t) (*p - '0');
            if (v > (UINT64_MAX - d) / 10) return fail("integer overflow");
            v = v * 10 + d;
            ++p;
        }
        out = v;
        return true;
    }
    bool skip() {  // any JSON value
        ws();
        if (p >= e) return fail("unexpected end");
        if (*p == '"') { std::string s; return str(s); }
        if (*p == '{' || *p == '[') {
            const char open = *p, close = open == '{' ? '}' : ']';
            ++p;
            if (eat(close)) return true;
            for (;;) {
                if (open == '{') { std::string k; if (!str(k) || !eat(':')) return fail("bad object"); }
                if (!skip()) return false;
                if (eat(',')) continue;
                if (eat(close)) return true;
                return fail("bad container");
            }
        }
        while (p < e && *p != ',' && *p != '}' && *p != ']' && *p != ' ' && *p != '\n') ++p;  // number/literal
        return true;
    }
    bool u64_array(std::vector<uint64_t> & out) {
        out.clear();
        if (!eat('[')) return fail("expected array");
        if (eat(']')) return true;
        for (;;) {
            uint64_t v;
            if (!u64(v)) return false;
            out.push_back(v);
            if (eat(',')) continue;
            if (eat(']')) return true;
            return fail("bad array");
        }
    }
};

bool parse_header(const char * s, size_t n, std::vector<TensorInfo> & out, std::string & err) {
    Json j{s, s + n, {}};
    auto bad = [&](const std::string & what) { err = "bad safetensors header: " + what + (j.err.empty() ? "" : " (" + j.err + ")"); return false; };
    if (!j.eat('{')) return bad("expected '{'");
    if (j.eat('}')) return true;
    std::map<std::string, bool> seen;
    for (;;) {
        TensorInfo t;
        if (!j.str(t.name) || !j.eat(':')) return bad("tensor key");
        if (t.name == "__metadata__") {
            if (!j.skip()) return bad("__metadata__");
        } else {
            std::string dtype;
            std::vector<uint64_t> shape, offs;
            bool have_dtype = false, have_shape = false, have_offs = false;
            if (!j.eat('{')) return bad("tensor '" + t.name + "' is not an object");
            if (!j.eat('}')) {
                for (;;) {
                    std::string k;
                    if (!j.str(k) || !j.eat(':')) return bad("field of '" + t.name + "'");
                    if (k == "dtype") { have_dtype = j.str(dtype); if (!have_dtype) return bad("dtype"); }
                    else if (k == "shape") { have_shape = j.u64_array(shape); if (!have_shape) return bad("shape"); }
                    else if (k == "data_offsets") { have_offs = j.u64_array(offs); if (!have_offs) return bad("data_offsets"); }
                    else if (!j.skip()) return bad("field value");
                    if (j.eat(',')) continue;
                    if (j.eat('}')) break;
                    return bad("tensor object");
                }
            }
            if (!have_dtype || !have_shape || !have_offs || offs.size() != 2)
                return bad("tensor '" + t.name + "' lacks dtype/shape/data_offsets");
            if (dtype != "F32") { err = "tensor '" + t.name + "' has dtype " + dtype + ", expected F32"; return false; }
            if (seen.count(t.name)) return bad("duplicate tensor '" + t.name + "'");
            seen[t.name] = true;
            for (uint64_t d : shape) t.shape.push_back((int64_t) d);
            t.begin = offs[0];
            t.end = offs[1];
            out.push_back(std::move(t));
        }
        if (j.eat(',')) continue;
        if (j.eat('}')) return true;
        return bad("expected ',' or '}'");
    }
}

struct Checker {
    const std::map<std::string, const TensorInfo *> & by_name;
    std::string err;
    void shape(const std::string & name, std::vector<int64_t> want) {
        if (!err.empty()) return;
        auto it = by_name.find(name);
        if (it == by_name.end()) { err = "missing tensor '" + name + "'"; return; }
        if (it->second->shape != want) {
            std::string s = "[";
            for (size_t i = 0; i < it->second->shape.size(); ++i) s += (i ? "," : "") + std::to_string(it->second->shape[i]);
            err = "tensor '" + name + "' has shape " + s + "] which differs from the expected RE-USE model";
        }
    }
};

struct FileCloser { void operator()(FILE * f) const { if (f) std::fclose(f); } };
using FilePtr = std::unique_ptr<FILE, FileCloser>;

} // namespace

extern "C" int reuse_convert(const char * st_path, const char * out_path) {
    if (!st_path || !out_path) return reuse_fail(REUSE_ERR_ARG, "reuse_convert: NULL path");
    const std::string src(st_path), dst(out_path);

    FilePtr f(std::fopen(st_path, "rb"));
    if (!f) return reuse_fail(REUSE_ERR_MODEL, src + ": cannot open for reading");
    std::fseek(f.get(), 0, SEEK_END);
    const long fsize_l = std::ftell(f.get());
    std::fseek(f.get(), 0, SEEK_SET);
    if (fsize_l < 8) return reuse_fail(REUSE_ERR_MODEL, src + ": too short for a safetensors file");
    const uint64_t fsize = (uint64_t) fsize_l;

    unsigned char lenb[8];
    if (std::fread(lenb, 1, 8, f.get()) != 8) return reuse_fail(REUSE_ERR_MODEL, src + ": cannot read header length");
    uint64_t hlen = 0;
    for (int i = 7; i >= 0; --i) hlen = (hlen << 8) | lenb[i];  // little-endian u64
    if (hlen == 0 || hlen > fsize - 8 || hlen > (64u << 20))
        return reuse_fail(REUSE_ERR_MODEL, src + ": invalid or truncated safetensors header");
    std::string header((size_t) hlen, '\0');
    if (std::fread(&header[0], 1, header.size(), f.get()) != header.size())
        return reuse_fail(REUSE_ERR_MODEL, src + ": cannot read header");

    std::vector<TensorInfo> infos;
    std::string err;
    if (!parse_header(header.data(), header.size(), infos, err)) return reuse_fail(REUSE_ERR_MODEL, src + ": " + err);

    const uint64_t data_start = 8 + hlen;
    const uint64_t data_size = fsize - data_start;
    uint64_t params = 0;
    std::map<std::string, const TensorInfo *> by_name;
    for (const TensorInfo & t : infos) {
        uint64_t n = 1;
        for (int64_t d : t.shape) n *= (uint64_t) d;
        if (t.shape.size() > GGML_MAX_DIMS) return reuse_fail(REUSE_ERR_MODEL, src + ": tensor '" + t.name + "' has too many dimensions");
        if (t.end < t.begin || t.end - t.begin != n * 4)
            return reuse_fail(REUSE_ERR_MODEL, src + ": tensor '" + t.name + "' has inconsistent data_offsets");
        if (t.end > data_size)
            return reuse_fail(REUSE_ERR_MODEL, src + ": truncated file, tensor '" + t.name + "' lies beyond the end of the data");
        if (t.name.size() >= GGML_MAX_NAME)
            return reuse_fail(REUSE_ERR_MODEL, src + ": tensor name '" + t.name + "' longer than ggml's limit of " + std::to_string(GGML_MAX_NAME - 1));
        params += n;
        by_name[t.name] = &t;
    }
    if (infos.size() != kNumTensors || params != kNumParams)
        return reuse_fail(REUSE_ERR_MODEL, src + ": not the RE-USE model (" + std::to_string(infos.size()) + " tensors, "
                          + std::to_string(params) + " parameters; expected " + std::to_string(kNumTensors) + " and " + std::to_string(kNumParams) + ")");

    // Architecture check on the Mamba blocks and one encoder conv.
    const int64_t di = (int64_t) kHidFeature * kExpand;
    Checker c{by_name, {}};
    c.shape("dense_encoder.dense_block.dense_block.3.0.weight", {64, 256, 3, 3});
    for (int i = 0; i < kNumTfmamba; ++i) {
        for (const char * ax : {"time", "freq"}) {
            for (const char * dir : {"forward", "backward"}) {
                const std::string p = "TSMamba." + std::to_string(i) + "." + ax + "_mamba." + dir + "_blocks.";
                c.shape(p + "A_log", {di, kDState});
                c.shape(p + "D", {di});
                c.shape(p + "conv1d.weight", {di, 1, kDConv});
                c.shape(p + "dt_proj.weight", {di, kDtRank});
                c.shape(p + "in_proj.weight", {2 * di, kHidFeature});
                c.shape(p + "x_proj.weight", {kDtRank + 2 * kDState, di});
                c.shape(p + "out_proj.weight", {kHidFeature, di});
            }
        }
    }
    if (!c.err.empty()) return reuse_fail(REUSE_ERR_MODEL, src + ": " + c.err);

    // Read all tensor data.
    std::vector<char> data((size_t) data_size);
    if (std::fseek(f.get(), (long) data_start, SEEK_SET) != 0 || std::fread(data.data(), 1, data.size(), f.get()) != data.size())
        return reuse_fail(REUSE_ERR_MODEL, src + ": truncated file, cannot read tensor data");
    f.reset();

    ggml_init_params cp = { ggml_tensor_overhead() * (infos.size() + 8), nullptr, /*no_alloc=*/ true };
    ggml_context * ctx = ggml_init(cp);
    gguf_context * g = gguf_init_empty();
    struct Cleanup { ggml_context * c; gguf_context * g; ~Cleanup() { if (g) gguf_free(g); if (c) ggml_free(c); } } cleanup{ctx, g};
    if (!ctx || !g) return reuse_fail(REUSE_ERR_MODEL, "reuse_convert: out of memory");

    gguf_set_val_u32(g, "reuse.version", 1);
    gguf_set_val_u32(g, "reuse.num_tfmamba", kNumTfmamba);
    gguf_set_val_u32(g, "reuse.hid_feature", kHidFeature);
    gguf_set_val_u32(g, "reuse.d_state", kDState);
    gguf_set_val_u32(g, "reuse.d_conv", kDConv);
    gguf_set_val_u32(g, "reuse.expand", kExpand);
    gguf_set_val_u32(g, "reuse.dt_rank", kDtRank);
    gguf_set_val_u32(g, "reuse.base_rate", kBaseRate);
    gguf_set_val_u32(g, "reuse.n_fft", kNFft);
    gguf_set_val_u32(g, "reuse.hop", kHop);
    gguf_set_val_u32(g, "reuse.win", kWin);
    gguf_set_val_str(g, "reuse.source_revision", kSourceRevision);

    for (const TensorInfo & t : infos) {
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        const int nd = t.shape.empty() ? 1 : (int) t.shape.size();
        for (int i = 0; i < (int) t.shape.size(); ++i) ne[i] = t.shape[t.shape.size() - 1 - i];
        ggml_tensor * gt = ggml_new_tensor(ctx, GGML_TYPE_F32, nd, ne);
        if (!gt) return reuse_fail(REUSE_ERR_MODEL, "reuse_convert: out of memory");
        ggml_set_name(gt, t.name.c_str());
        gguf_add_tensor(g, gt);
        gguf_set_tensor_data(g, t.name.c_str(), data.data() + t.begin);
    }

    if (!gguf_write_to_file(g, out_path, false))
        return reuse_fail(REUSE_ERR_MODEL, dst + ": cannot write GGUF file");
    return REUSE_OK;
}
