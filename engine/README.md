# Brahma Engine Core

Native C++20 runtime project. Initial milestone: independent GGUF metadata and tensor-descriptor parser. This does not yet execute inference and does not link to llama.cpp/GGML.

Build from repository root in Visual Studio 2022 Developer PowerShell:

```powershell
cmake -S engine -B engine/build -A x64
cmake --build engine/build --config Release
ctest --test-dir engine/build -C Release --output-on-failure
```

Upstream format references: [gguf.h](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/gguf.h) and [gguf.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/gguf.cpp). The parser is an independent implementation, not a dependency on llama.cpp.
