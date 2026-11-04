// reuse_enhance: STFT -> model graph -> zero-frame rule -> iSTFT, as tools/run.py does.
#include "error.h"
#include "graph.h"
#include "model.h"
#include "reuse.h"
#include "stft.h"
#include "window.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Reads the [F2, T2] decoder output `t` and crops it to [F, T] (torch denoised_x[:, :F, :T]).
std::vector<float> read_cropped(ggml_tensor * t, int F, int T) {
    const int64_t F2 = t->ne[0], T2 = t->ne[1];
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || t->ne[2] != 1 || t->ne[3] != 1 || F2 < F || T2 < T)
        throw std::runtime_error("enhance: unexpected decoder output shape");
    std::vector<float> full((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, full.data(), 0, ggml_nbytes(t));
    std::vector<float> out((size_t) F * (size_t) T);
    for (int ti = 0; ti < T; ++ti)
        std::memcpy(&out[(size_t) ti * F], &full[(size_t) ti * F2], (size_t) F * sizeof(float));
    return out;
}

}  // namespace

void enhance_window(reuse_ctx * ctx, const float * in, size_t n, int sample_rate, float * out) {
    const StftSizes s = stft_sizes(sample_rate);
    // STFT (log1p magnitude, phase), [F, T] frequency innermost.
    std::vector<float> mag, pha;
    int F = 0, T = 0;
    stft(in, n, s, mag, pha, F, T);
    const size_t nft = (size_t) F * (size_t) T;

    // Model graph on [F, T, 2] -> pre-crop magnitude and phase (r, i) maps.
    ggml_backend_t be = ctx->backend.be;
    GraphRun g(graph_max_nodes(ctx->weights.hp.num_tfmamba));
    ggml_tensor * x = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, F, T, 2);
    ggml_set_name(x, "mag_pha");
    ggml_set_input(x);
    const DecoderOut d = forward(g.ctx, ctx->weights, x);
    g.alloc(be, {d.mag, d.pha_r, d.pha_i});  // also uploads the graph constants
    ctx->last_compute_bytes = std::max(ctx->last_compute_bytes, g.compute_bytes());
    ggml_backend_tensor_set(x, mag.data(), 0, nft * sizeof(float));
    ggml_backend_tensor_set(x, pha.data(), nft * sizeof(float), nft * sizeof(float));
    g.compute(be);

    std::vector<float> amp = read_cropped(d.mag, F, T);
    const std::vector<float> re = read_cropped(d.pha_r, F, T);
    const std::vector<float> im = read_cropped(d.pha_i, F, T);

    // Zero-frame rule (run.py): a frame where more than half the bins have
    // expm1(relu(mag)) == 0 gets all its magnitudes set to 0.
    for (int t = 0; t < T; ++t) {
        float * m = &amp[(size_t) t * F];
        int zeros = 0;
        for (int f = 0; f < F; ++f) zeros += std::expm1(std::max(0.f, m[f])) == 0.f;
        if (2 * zeros > F) std::fill(m, m + F, 0.f);
    }

    // Phase = atan2(i, r), needed only as cos/sin: (r, i) / |z|; atan2(0, 0) = 0 -> (1, 0).
    std::vector<float> cs(nft), sn(nft);
    for (size_t k = 0; k < nft; ++k) {
        const double r = re[k], i = im[k], z = std::sqrt(r * r + i * i);
        cs[k] = z > 0 ? (float) (r / z) : 1.f;
        sn[k] = z > 0 ? (float) (i / z) : 0.f;
    }

    const std::vector<float> y = istft(amp.data(), cs.data(), sn.data(), F, T, s, n);
    std::memcpy(out, y.data(), n * sizeof(float));  // `in` is no longer read: out may alias it
}

int enhance_impl(reuse_ctx * ctx, const float * in, size_t n, int sample_rate, float * out) {
    if (!ctx) return reuse_fail(REUSE_ERR_ARG, "reuse_enhance: NULL context");
    // Checked before stft_sizes: below ~200 Hz the hop rounds to 0.
    if (sample_rate < 8000 || sample_rate > 192000)
        return reuse_fail(REUSE_ERR_ARG, "reuse_enhance: sample_rate " + std::to_string(sample_rate) +
                                             " outside [8000, 192000]");
    const StftSizes s = stft_sizes(sample_rate);
    const size_t min_n = (size_t) s.n_fft / 2 + 1;  // torch.stft reflect padding needs n > n_fft/2
    if (n < min_n)
        return reuse_fail(REUSE_ERR_TOO_SHORT, "reuse_enhance: input of " + std::to_string(n) +
                                                   " samples is too short; at least " + std::to_string(min_n) +
                                                   " are needed at " + std::to_string(sample_rate) + " Hz");
    if (!in || !out) return reuse_fail(REUSE_ERR_ARG, "reuse_enhance: NULL input or output buffer");
    if (n / (size_t) s.hop >= (size_t) INT_MAX)
        return reuse_fail(REUSE_ERR_ARG, "reuse_enhance: input too long");

    ctx->last_compute_bytes = 0;
    const size_t window = (size_t) std::llround(ctx->window_seconds * sample_rate);
    if (n <= window) {
        ctx->last_window_count = 1;
        enhance_window(ctx, in, n, sample_rate, out);
        return REUSE_OK;
    }

    // Long input: overlapping windows, each through the single-window path, joined by a
    // raised-cosine crossfade (amplitude-complementary). `in` may alias `out`, so the result is
    // accumulated in a separate buffer and copied out at the end.
    const size_t overlap = (size_t) std::llround(ctx->overlap_seconds * sample_rate);
    const std::vector<Window> plan = plan_windows(n, window, overlap, min_n);
    ctx->last_window_count = (int) plan.size();
    std::vector<float> acc(n, 0.f), y;
    for (size_t i = 0; i < plan.size(); ++i) {
        const Window & w = plan[i];
        y.resize(w.len);
        enhance_window(ctx, in + w.start, w.len, sample_rate, y.data());
        accumulate_window(acc.data(), y.data(), w, i > 0, i + 1 < plan.size(), overlap);
    }
    std::memcpy(out, acc.data(), n * sizeof(float));
    return REUSE_OK;
}
