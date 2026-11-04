#ifndef REUSE_H
#define REUSE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct reuse_ctx reuse_ctx;

enum reuse_status {
    REUSE_OK = 0,
    REUSE_ERR_ARG = 1,
    REUSE_ERR_TOO_SHORT = 2,
    REUSE_ERR_MODEL = 3,
    REUSE_ERR_BACKEND = 4
};

/* device: ggml device name ("CPU", "MTL0", "ROCm0", "Vulkan0", "CUDA0"); NULL = CPU. */
reuse_ctx * reuse_load(const char * gguf_path, const char * device, int n_threads);
/* Enhances n mono samples at sample_rate (8000..192000 Hz) into out (n floats; may alias in).
 * REUSE_ERR_TOO_SHORT if n < n_fft/2 + 1 (481 at 24 kHz); REUSE_ERR_ARG for a NULL pointer or
 * a sample rate out of range; REUSE_ERR_BACKEND if the graph cannot be allocated or computed.
 * Calls sharing one ctx must be serialised.
 *
 * Long inputs are cleaned in windows so peak compute memory stays that of a 20 s window
 * (about 2.6 GB per 10 s of audio at 24 kHz) regardless of length. Inputs up to 20 s run as one
 * graph. Longer ones are split into evenly sized windows of at most 20 s that overlap by 2 s;
 * the overlaps are joined with a raised-cosine crossfade. Results therefore differ slightly from
 * a single-pass run around the seams, because InstanceNorm statistics and the bidirectional
 * Mamba state only see one window at a time. The output always has n samples. */
/* Optional progress reporting for reuse_enhance. fn is called from the calling thread with a
 * fraction in [0, 1], non-decreasing across the whole call (all windows of a long input), ending
 * with exactly 1.0f when reuse_enhance succeeds. NULL disables it. The output is bit-identical
 * with and without a callback; with one, each window's graph runs in about 32 consecutive slices. */
typedef void (*reuse_progress_fn)(float fraction, void * user);
void        reuse_set_progress(reuse_ctx * ctx, reuse_progress_fn fn, void * user);
int         reuse_enhance(reuse_ctx * ctx, const float * in, size_t n, int sample_rate, float * out);
void        reuse_free(reuse_ctx * ctx);
/* Thread-local; never NULL. */
const char * reuse_last_error(void);
int         reuse_convert(const char * safetensors_path, const char * gguf_out_path);

#ifdef __cplusplus
}
#endif

#endif /* REUSE_H */
