cmake_minimum_required(VERSION 3.14)
project(reuse_parent LANGUAGES C CXX)
# GGML_DIR and REUSE_SRC are passed on the command line.
add_subdirectory(${GGML_DIR} ${CMAKE_BINARY_DIR}/ggml)
set(REUSE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${REUSE_SRC} ${CMAKE_BINARY_DIR}/reuse)
