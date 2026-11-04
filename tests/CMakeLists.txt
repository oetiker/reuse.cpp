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

# STFT/iSTFT tests (select with ctest -R stft)
add_executable(test_stft test_stft.cpp)
target_include_directories(test_stft PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test_stft PRIVATE reuse)
add_test(NAME stft COMMAND test_stft)
set_tests_properties(stft PROPERTIES
    ENVIRONMENT "REUSE_FIXTURES_DEFAULT=${REUSE_FIXTURES_DIR}")

# Converter / loader tests (need REUSE_SAFETENSORS; skip otherwise). GGUFs are written to the build dir.
add_executable(test_weights test_weights.cpp)
target_include_directories(test_weights PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test_weights PRIVATE reuse)
add_test(NAME weights COMMAND test_weights)
set_tests_properties(weights PROPERTIES
    ENVIRONMENT "REUSE_TEST_TMP=${CMAKE_CURRENT_BINARY_DIR}")

# Graph tap tests vs. PyTorch fixtures (need REUSE_GGUF; skip otherwise; REUSE_DEVICE picks the device).
add_executable(test_encoder test_encoder.cpp)
target_include_directories(test_encoder PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test_encoder PRIVATE reuse)
add_test(NAME encoder COMMAND test_encoder)
set_tests_properties(encoder PROPERTIES
    ENVIRONMENT "REUSE_FIXTURES_DEFAULT=${REUSE_FIXTURES_DIR}")

# TF-Mamba block tests vs. PyTorch fixtures (need REUSE_GGUF; skip otherwise).
add_executable(test_mamba test_mamba.cpp)
target_include_directories(test_mamba PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test_mamba PRIVATE reuse)
add_test(NAME mamba COMMAND test_mamba)
set_tests_properties(mamba PROPERTIES
    ENVIRONMENT "REUSE_FIXTURES_DEFAULT=${REUSE_FIXTURES_DIR}")

# Decoder + end-to-end reuse_enhance tests vs. PyTorch fixtures (need REUSE_GGUF; skip otherwise).
# On CPU the full model runs at roughly 9 s per second of audio, so this takes a few minutes.
add_executable(test_enhance test_enhance.cpp)
target_include_directories(test_enhance PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test_enhance PRIVATE reuse)
add_test(NAME enhance COMMAND test_enhance)
set_tests_properties(enhance PROPERTIES
    TIMEOUT 1800
    ENVIRONMENT "REUSE_FIXTURES_DEFAULT=${REUSE_FIXTURES_DIR}")

# Windowed reuse_enhance (long inputs); window lengths overridden through reuse_ctx so short
# fixtures exercise it. Needs REUSE_GGUF for the model tests; the plan test runs without.
add_executable(test_window test_window.cpp)
target_include_directories(test_window PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test_window PRIVATE reuse)
add_test(NAME window COMMAND test_window)
set_tests_properties(window PROPERTIES
    TIMEOUT 1800
    ENVIRONMENT "REUSE_FIXTURES_DEFAULT=${REUSE_FIXTURES_DIR}")
