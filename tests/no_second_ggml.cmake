# Usage: cmake -DREUSE_SRC=<reuse.cpp> -DPARENT_BIN=<dir> -DGGML_DIR=<ggml> -P no_second_ggml.cmake
# Configures tests/parent (ggml first, then reuse.cpp) and asserts reuse.cpp's
# binary dir contains no ggml build directory.
foreach(v REUSE_SRC PARENT_BIN GGML_DIR)
    if(NOT ${v})
        message(FATAL_ERROR "${v} not set")
    endif()
endforeach()
file(REMOVE_RECURSE ${PARENT_BIN})
execute_process(
    COMMAND ${CMAKE_COMMAND} -S ${REUSE_SRC}/tests/parent -B ${PARENT_BIN}
            -DGGML_DIR=${GGML_DIR} -DREUSE_SRC=${REUSE_SRC} -DGGML_METAL=OFF
    RESULT_VARIABLE rc OUTPUT_QUIET ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "parent configure failed:\n${err}")
endif()
if(NOT IS_DIRECTORY ${PARENT_BIN}/ggml)
    message(FATAL_ERROR "parent's own ggml build dir missing: ${PARENT_BIN}/ggml")
endif()
file(GLOB_RECURSE all LIST_DIRECTORIES true RELATIVE ${PARENT_BIN}/reuse ${PARENT_BIN}/reuse/*)
set(hits ${all})
list(FILTER hits INCLUDE REGEX "ggml")
if(hits)
    message(FATAL_ERROR "second ggml found under reuse.cpp's binary dir:\n${hits}")
endif()
message(STATUS "OK: no second ggml under ${PARENT_BIN}/reuse")
