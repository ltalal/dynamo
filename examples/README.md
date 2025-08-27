# Examples

This directory collects runnable examples for Dynamo.

- Engines: vLLM, SGLang, TensorRT-LLM, llama.cpp
- Environments: local single-node (Python module entrypoints) and Kubernetes (Operator + CRDs)

Engine example entry points:

- vLLM: examples/engines/vllm/README.md
- SGLang: examples/engines/sglang/README.md
- TensorRT-LLM: examples/engines/trtllm/README.md
- llama.cpp: components/backends/llama_cpp/README.md

Coming next: per-engine subfolders under `examples/engines/{engine}/` with:

- single-node/: minimal `python -m dynamo.frontend` + `python -m dynamo.<engine>` flows
- kubernetes/: Operator + CRDs flows for the same scenarios
