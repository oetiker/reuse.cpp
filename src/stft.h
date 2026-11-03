#pragma once
// CPU STFT / iSTFT matching tools/nvidia/models/stfts.py (relu_log1p, center=True, addeps=False).
#include <cstddef>
#include <vector>

struct StftSizes { int n_fft, hop, win; };

// Scale the 24 kHz reference sizes (960/120/960) to `sample_rate`, like make_even in tools/run.py.
StftSizes stft_sizes(int sample_rate);

// Output layout [n_freq, n_frames], freq innermost. mag_log1p = log1p(|X|), pha = angle(X).
void stft(const float * x, size_t n, StftSizes s, std::vector<float> & mag_log1p,
          std::vector<float> & pha, int & n_freq, int & n_frames);

// Inverse of stft(): expm1(max(0, m)) * (cos, sin), torch.istft semantics, then pad (1e-8) / trim to n_out.
std::vector<float> istft(const float * mag_log1p, const float * cos_pha, const float * sin_pha,
                         int n_freq, int n_frames, StftSizes s, size_t n_out);
