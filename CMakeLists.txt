cmake_minimum_required(VERSION 3.14)
project(reuse LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(REUSE_TOPLEVEL ON)
else()
    set(REUSE_TOPLEVEL OFF)
endif()

option(REUSE_BUILD_TESTS "Build reuse.cpp tests" ${REUSE_TOPLEVEL})
set(REUSE_GGML_DIR "" CACHE PATH "Path to llama.cpp/ggml (used only when no ggml target exists)")

# Reuse the parent's ggml when embedded; otherwise build our own.
if(NOT TARGET ggml)
    if(NOT REUSE_GGML_DIR)
        message(FATAL_ERROR "No ggml target found: set REUSE_GGML_DIR to the llama.cpp/ggml directory")
    endif()
    add_subdirectory(${REUSE_GGML_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ggml)
endif()

add_library(reuse STATIC src/reuse.cpp src/convert.cpp src/weights.cpp src/stft.cpp
    src/graph_encoder.cpp src/graph_mamba.cpp src/graph_run.cpp)
target_include_directories(reuse
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(reuse PUBLIC ggml)

if(REUSE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
