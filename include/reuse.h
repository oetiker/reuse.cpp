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
 * Calls sharing one ctx must be serialised. */
int         reuse_enhance(reuse_ctx * ctx, const float * in, size_t n, int sample_rate, float * out);
void        reuse_free(reuse_ctx * ctx);
/* Thread-local; never NULL. */
const char * reuse_last_error(void);
int         reuse_convert(const char * safetensors_path, const char * gguf_out_path);

#ifdef __cplusplus
}
#endif

#endif /* REUSE_H */
