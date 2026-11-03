add_executable(reuse_tests test_main.cpp)
target_link_libraries(reuse_tests PRIVATE reuse)

if(DEFINED ENV{REUSE_FIXTURES})
    set(REUSE_FIXTURES_DIR $ENV{REUSE_FIXTURES})
else()
    set(REUSE_FIXTURES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/fixtures)
endif()

# REUSE_FIXTURES / REUSE_GGUF / REUSE_DEVICE are read from the environment at run time;
# the fixtures default is provided here when the variable is unset.
add_test(NAME reuse_tests COMMAND reuse_tests)
set_tests_properties(reuse_tests PROPERTIES
    ENVIRONMENT "REUSE_FIXTURES_DEFAULT=${REUSE_FIXTURES_DIR}")

# Embedding check: a parent project that owns ggml must not get a second ggml build.
add_test(NAME no_second_ggml
    COMMAND ${CMAKE_COMMAND}
        -DREUSE_SRC=${PROJECT_SOURCE_DIR}
        -DPARENT_BIN=${PROJECT_SOURCE_DIR}/build-parent
        -DGGML_DIR=${REUSE_GGML_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/no_second_ggml.cmake)
