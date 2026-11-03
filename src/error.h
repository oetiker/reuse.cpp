#pragma once
// Thread-local last-error shared by the C API implementation files.
#include <string>

// Stores `msg` as the thread-local last error and returns `status` (for `return fail(...)`).
int reuse_fail(int status, const std::string & msg);

#include <exception>
#include <utility>

// Runs `f` and contains every C++ exception: records it as the last error and returns
// `on_error` instead. Every extern "C" entry point must go through this (exceptions must
// never unwind into the C caller).
template <class R, class F>
R reuse_guard(R on_error, int status, F && f) noexcept {
    try {
        return std::forward<F>(f)();
    } catch (const std::exception & e) {
        try { reuse_fail(status, std::string("internal error: ") + e.what()); } catch (...) {}
    } catch (...) {
        try { reuse_fail(status, "internal error: unknown exception"); } catch (...) {}
    }
    return on_error;
}
