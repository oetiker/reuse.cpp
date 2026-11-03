#include "reuse.h"
#include "model.h"

#include <string>

static thread_local std::string g_last_error;

static int fail(int status, const char * msg) {
    g_last_error = msg;
    return status;
}

extern "C" {

reuse_ctx * reuse_load(const char *, const char *, int) {
    fail(REUSE_ERR_MODEL, "not implemented");
    return nullptr;
}

int reuse_enhance(reuse_ctx *, const float *, size_t, int, float *) {
    return fail(REUSE_ERR_MODEL, "not implemented");
}

void reuse_free(reuse_ctx * ctx) {
    delete ctx;
}

const char * reuse_last_error(void) {
    return g_last_error.c_str();
}

int reuse_convert(const char *, const char *) {
    return fail(REUSE_ERR_MODEL, "not implemented");
}

} // extern "C"
