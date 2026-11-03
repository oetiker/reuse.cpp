#pragma once
// Thread-local last-error shared by the C API implementation files.
#include <string>

// Stores `msg` as the thread-local last error and returns `status` (for `return fail(...)`).
int reuse_fail(int status, const std::string & msg);
