#include "stft.h"

#include "pocketfft_hdronly.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace {

const double kPi = 3.14159265358979323846;

int make_even(long v) { return (int) (v % 2 == 0 ? v : v + 1); }

// torch.hann_window(win) (periodic), centre-padded with zeros to n_fft as torch.stft/istft do.
std::vector<double> padded_window(const StftSizes & s) {
    std::vector<double> w((size_t) s.n_fft, 0.0);
    const int left = (s.n_fft - s.win) / 2;
    for (int i = 0; i < s.win; ++i) w[(size_t) (left + i)] = 0.5 - 0.5 * std::cos(2.0 * kPi * i / s.win);
    return w;
}

// Reflect index into [0, n) (torch 'reflect' padding, no edge repeat).
size_t reflect(long i, long n) {
    if (i < 0) i = -i;
    if (i >= n) i = 2 * (n - 1) - i;
    return (size_t) i;
}

}  // namespace

StftSizes stft_sizes(int sample_rate) {
    // Same integer arithmetic as tools/run.py: make_even(v * sr // 24000) on 960/120/960.
    const long sr = sample_rate;
    return { make_even(960 * sr / 24000), make_even(120 * sr / 24000), make_even(960 * sr / 24000) };
}

void stft(const float * x, size_t n, StftSizes s, std::vector<float> & mag_log1p,
          std::vector<float> & pha, int & n_freq, int & n_frames) {
    const long pad = s.n_fft / 2;
    if (n == 0 || (long) n <= pad) throw std::invalid_argument("stft: input too short for reflect padding");
    n_freq = s.n_fft / 2 + 1;
    n_frames = (int) (n / (size_t) s.hop) + 1;
    mag_log1p.assign((size_t) n_freq * n_frames, 0.f);
    pha.assign((size_t) n_freq * n_frames, 0.f);

    const std::vector<double> win = padded_window(s);
    std::vector<double> frame((size_t) s.n_fft);
    std::vector<std::complex<double>> spec((size_t) n_freq);
    const pocketfft::shape_t shape{(size_t) s.n_fft};
    const pocketfft::stride_t sin_{(ptrdiff_t) sizeof(double)}, sout{(ptrdiff_t) sizeof(std::complex<double>)};
    for (int t = 0; t < n_frames; ++t) {
        for (int i = 0; i < s.n_fft; ++i) {
            const long p = (long) t * s.hop + i - pad;  // index in the unpadded signal
            frame[(size_t) i] = (double) x[reflect(p, (long) n)] * win[(size_t) i];
        }
        pocketfft::r2c(shape, sin_, sout, 0, true, frame.data(), spec.data(), 1.0);
        for (int k = 0; k < n_freq; ++k) {
            const std::complex<double> & c = spec[(size_t) k];
            const size_t o = (size_t) t * n_freq + k;
            mag_log1p[o] = (float) std::log1p(std::abs(c));
            pha[o] = (float) std::atan2(c.imag(), c.real());
        }
    }
}

std::vector<float> istft(const float * mag_log1p, const float * cos_pha, const float * sin_pha,
                         int n_freq, int n_frames, StftSizes s, size_t n_out) {
    if (n_freq != s.n_fft / 2 + 1) throw std::invalid_argument("istft: n_freq does not match n_fft");
    const std::vector<double> win = padded_window(s);
    const size_t total = (size_t) s.hop * (size_t) (n_frames - 1) + (size_t) s.n_fft;
    std::vector<double> y(total, 0.0), env(total, 0.0), frame((size_t) s.n_fft);
    std::vector<std::complex<double>> spec((size_t) n_freq);
    const pocketfft::shape_t shape{(size_t) s.n_fft};
    const pocketfft::stride_t sin_{(ptrdiff_t) sizeof(std::complex<double>)}, sout{(ptrdiff_t) sizeof(double)};
    for (int t = 0; t < n_frames; ++t) {
        for (int k = 0; k < n_freq; ++k) {
            const size_t o = (size_t) t * n_freq + k;
            const double m = std::expm1(std::max(0.0, (double) mag_log1p[o]));
            spec[(size_t) k] = {m * cos_pha[o], m * sin_pha[o]};
        }
        pocketfft::c2r(shape, sin_, sout, 0, false, spec.data(), frame.data(), 1.0 / s.n_fft);
        const size_t base = (size_t) t * s.hop;
        for (int i = 0; i < s.n_fft; ++i) {
            const double w = win[(size_t) i];
            y[base + i] += frame[(size_t) i] * w;
            env[base + i] += w * w;
        }
    }
    // Centre trim (n_fft/2 each side), normalise by the squared-window envelope.
    // torch errors where the envelope is ~0; here such samples are left undivided.
    const size_t pad = (size_t) s.n_fft / 2;
    const size_t len = total - 2 * pad;
    std::vector<float> out(n_out, 1e-8f);
    for (size_t i = 0; i < std::min(len, n_out); ++i) {
        const double e = env[pad + i];
        out[i] = (float) (e > 1e-11 ? y[pad + i] / e : y[pad + i]);
    }
    return out;
}
