
cmake_minimum_required(VERSION 3.24)

project(BrahmaEngine VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Brahma Engine configuration
# ---------------------------------------------------------------------------

option(BRAHMA_ENGINE_BUILD_TESTS
    "Build Brahma Engine tests" ON)

option(BRAHMA_ENGINE_ENABLE_OPENMP
    "Enable OpenMP parallel CPU kernels for the native engine" ON)

option(BRAHMA_ENGINE_USE_LLAMA_CPP
    "Build Brahma Engine with the llama.cpp inference backend" ON)

option(BRAHMA_ENGINE_ENABLE_CUDA
    "Enable CUDA acceleration in the llama.cpp backend" OFF)

# ---------------------------------------------------------------------------
# Existing Brahma native inference core
# ---------------------------------------------------------------------------

add_library(brahma_engine_core
    src/gguf_reader.cpp
    src/tensor_validator.cpp
    src/quantization.cpp
    src/tensor_ops.cpp
    src/tensor_loader.cpp
    src/tensor_registry.cpp
    src/qwen2_schema.cpp
    src/qwen2_block.cpp
    src/qwen2_runtime.cpp
    src/qwen2_tokenizer.cpp
)

target_include_directories(brahma_engine_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_features(brahma_engine_core
    PUBLIC cxx_std_20
)

if(BRAHMA_ENGINE_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
endif()

if(BRAHMA_ENGINE_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(brahma_engine_core
        PUBLIC OpenMP::OpenMP_CXX
    )

    target_compile_definitions(brahma_engine_core
        PRIVATE BRAHMA_ENGINE_USE_OPENMP=1
    )
endif()

if(MSVC)
    target_compile_options(brahma_engine_core
        PRIVATE /W4 /permissive- /EHsc
    )
else()
    target_compile_options(brahma_engine_core
        PRIVATE -Wall -Wextra -Wpedantic -Wconversion
    )
endif()

# ---------------------------------------------------------------------------
# Brahma Engine: llama.cpp-backed inference
# ---------------------------------------------------------------------------

if(BRAHMA_ENGINE_USE_LLAMA_CPP)

    set(BRAHMA_LLAMA_CPP_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp"
    )

    if(NOT EXISTS "${BRAHMA_LLAMA_CPP_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "llama.cpp was not found at: ${BRAHMA_LLAMA_CPP_DIR}. "
            "Initialize the third_party/llama.cpp Git submodule."
        )
    endif()

    # Configure the llama.cpp compute backend.
    set(GGML_CUDA ${BRAHMA_ENGINE_ENABLE_CUDA} CACHE BOOL
        "Enable CUDA support in llama.cpp" FORCE)

    set(LLAMA_BUILD_COMMON OFF CACHE BOOL
        "Disable upstream common utilities" FORCE)

    set(LLAMA_BUILD_TESTS OFF CACHE BOOL
        "Disable upstream tests in the embedded build" FORCE)

    set(LLAMA_BUILD_TOOLS OFF CACHE BOOL
        "Disable upstream tools in the embedded build" FORCE)

    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL
        "Disable upstream examples in the embedded build" FORCE)

    set(LLAMA_BUILD_SERVER OFF CACHE BOOL
        "Disable the upstream server in the embedded build" FORCE)

    add_subdirectory(
        "${BRAHMA_LLAMA_CPP_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp"
        EXCLUDE_FROM_ALL
    )

    # User-facing executable. This target uses the library API directly;
    # it does not launch llama-cli.exe as a subprocess.
    add_executable(brahma-engine
        src/llama_backend_main.cpp
    )

    target_compile_features(brahma-engine
        PRIVATE cxx_std_20
    )

    target_link_libraries(brahma-engine
        PRIVATE llama
    )

    if(MSVC)
        target_compile_options(brahma-engine
            PRIVATE /W4 /EHsc
        )
    else()
        target_compile_options(brahma-engine
            PRIVATE -Wall -Wextra -Wpedantic
        )
    endif()

    # Put the executable in a predictable location for Windows builds.
    set_target_properties(brahma-engine PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            "${CMAKE_CURRENT_BINARY_DIR}/bin"
    )

    foreach(config Debug Release RelWithDebInfo MinSizeRel)
        set_target_properties(brahma-engine PROPERTIES
            "RUNTIME_OUTPUT_DIRECTORY_${config}"
                "${CMAKE_CURRENT_BINARY_DIR}/bin/${config}"
        )
    endforeach()

endif()

# ---------------------------------------------------------------------------
# Existing Brahma Engine tests
# ---------------------------------------------------------------------------

if(BRAHMA_ENGINE_BUILD_TESTS)

    enable_testing()

    add_executable(brahma-engine-tests
        tests/gguf_reader_tests.cpp
    )

    target_link_libraries(brahma-engine-tests
        PRIVATE brahma_engine_core
    )

    add_test(
        NAME gguf_reader_tests
        COMMAND brahma-engine-tests
    )

    add_executable(brahma-tensor-ops-tests
        tests/tensor_ops_tests.cpp
    )

    target_link_libraries(brahma-tensor-ops-tests
        PRIVATE brahma_engine_core
    )

    add_test(
        NAME tensor_ops_tests
        COMMAND brahma-tensor-ops-tests
    )

    add_executable(brahma-tensor-loader-tests
        tests/tensor_loader_tests.cpp
    )

    target_link_libraries(brahma-tensor-loader-tests
        PRIVATE brahma_engine_core
    )

    add_test(
        NAME tensor_loader_tests
        COMMAND brahma-tensor-loader-tests
    )

    add_executable(brahma-qwen2-schema-tests
        tests/qwen2_schema_tests.cpp
    )

    target_link_libraries(brahma-qwen2-schema-tests
        PRIVATE brahma_engine_core
    )

    add_test(
        NAME qwen2_schema_tests
        COMMAND brahma-qwen2-schema-tests
    )

    add_executable(brahma-qwen2-block-tests
        tests/qwen2_block_tests.cpp
    )

    target_link_libraries(brahma-qwen2-block-tests
        PRIVATE brahma_engine_core
    )

    add_test(
        NAME qwen2_block_tests
        COMMAND brahma-qwen2-block-tests
    )

    add_executable(brahma-qwen2-tokenizer-tests
        tests/qwen2_tokenizer_tests.cpp
    )

    target_link_libraries(brahma-qwen2-tokenizer-tests
        PRIVATE brahma_engine_core
    )

    add_test(
        NAME qwen2_tokenizer_tests
        COMMAND brahma-qwen2-tokenizer-tests
    )

    add_executable(brahma-quantization-tests
        tests/quantization_tests.cpp
    )

    target_link_libraries(brahma-quantization-tests
        PRIVATE brahma_engine_core
    )

    add_test(
        NAME quantization_tests
        COMMAND brahma-quantization-tests
    )

endif()
