#pragma once
// Window plan for long inputs (internal): reuse_enhance cleans inputs longer than one window in
// overlapping windows so peak compute memory is bounded by the window length.
#include <cstddef>
#include <vector>

struct Window { size_t start, len; };

// Splits [0, n) into windows. If n <= window: a single window {0, n}. Otherwise k =
// ceil((n - overlap) / (window - overlap)) windows are distributed evenly: every window has
// length L = ceil((n + (k-1) * overlap) / k) <= window, consecutive windows start L - overlap
// apart (overlap is therefore exact) and the last one is shortened to end at n; it is at most
// k-1 samples shorter than L. Requires window >= 2 * overlap + min_len (so the head and tail
// crossfades of a window never meet and every window is at least min_len long); throws
// std::invalid_argument otherwise, or if the last window would end up shorter than
// overlap + min_len.
std::vector<Window> plan_windows(size_t n, size_t window, size_t overlap, size_t min_len);

// Shortest input (samples) the STFT accepts at `sample_rate`: n_fft/2 + 1 (reflect padding).
size_t min_input_samples(int sample_rate);

// Adds window `w`'s output y (w.len samples) into acc (indexed by absolute sample position),
// weighted by the raised-cosine crossfade: the first `overlap` samples of a window that has a
// predecessor get 1 - w_out(t), the last `overlap` samples of one that has a successor get
// w_out(t), with w_out(t) = 0.5 (1 + cos(pi t / overlap)). The two weights sum to exactly 1.
void accumulate_window(float * acc, const float * y, const Window & w, bool has_prev, bool has_next, size_t overlap);
