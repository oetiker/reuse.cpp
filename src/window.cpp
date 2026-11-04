#include "stft.h"
#include "window.h"

#include <cmath>
#include <stdexcept>
#include <string>

std::vector<Window> plan_windows(size_t n, size_t window, size_t overlap, size_t min_len) {
    if (n <= window) return {{0, n}};
    if (window < 2 * overlap + min_len)
        throw std::invalid_argument("window of " + std::to_string(window) + " samples too short for overlap " +
                                    std::to_string(overlap) + " and minimum length " + std::to_string(min_len));
    const size_t stride_max = window - overlap;                  // > 0 given the check above
    const size_t k = (n - overlap + stride_max - 1) / stride_max;  // >= 2 since n > window
    const size_t len = (n + (k - 1) * overlap + k - 1) / k;        // ceil, <= window
    const size_t stride = len - overlap;
    std::vector<Window> w;
    w.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        const size_t start = i * stride;
        w.push_back({start, i + 1 < k ? len : n - start});
    }
    if (w.back().len < overlap + min_len)
        throw std::invalid_argument("last window of " + std::to_string(w.back().len) + " samples is too short");
    return w;
}

size_t min_input_samples(int sample_rate) { return (size_t) stft_sizes(sample_rate).n_fft / 2 + 1; }

void accumulate_window(float * acc, const float * y, const Window & w, bool has_prev, bool has_next, size_t overlap) {
    const double pi = std::acos(-1.0);
    auto w_out = [&](size_t t) { return 0.5 * (1.0 + std::cos(pi * (double) t / (double) overlap)); };
    for (size_t j = 0; j < w.len; ++j) {
        double g = 1.0;
        if (has_prev && j < overlap) g *= 1.0 - w_out(j);
        if (has_next && j >= w.len - overlap) g *= w_out(j - (w.len - overlap));
        acc[w.start + j] += (float) (g * y[j]);
    }
}
