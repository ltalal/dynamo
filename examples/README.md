# Dynamo Examples

Practical, task-oriented examples for deploying and using Dynamo for distributed LLM inference. Each example includes setup instructions, configs, and explanations to help you learn deployment patterns and apply them quickly.

Want a specific example? Open a GitHub issue to request it, or open a pull request to contribute your own.

## Prerequisites (concise)
- etcd + NATS: start locally with `docker compose -f tooling/docker-compose.yml up -d`
- Python 3.10+: install `ai-dynamo[<engine>]` and use `python -m` entrypoints
- CUDA-capable GPU: required for LLM inference (runtime hello_world is CPU-only)
- Kubernetes cluster: for any Kubernetes examples (Operator + CRDs)

## Basics & Tutorials
Learn core Dynamo concepts with minimal setups:
- Quickstart: examples/basics/quickstart
- Disaggregated Serving: examples/basics/disaggregated_serving
- Multi-node: examples/basics/multinode

## Engine Workflows
Framework-specific examples for [SGLang](engines/sglang/), [TRT-LLM](engines/trtllm/), and [vLLM](engines/vllm/). 
Each engine has:
- single-node/: local scripts using `python -m dynamo.frontend` + `python -m dynamo.<engine>`
- kubernetes/: CRDs for Kubernetes deployments using Dynamo operator

## Deployment Guides (Platforms)
Production-focused walkthroughs by platform:
- Amazon EKS: examples/deployments/EKS
- Router Standalone: examples/deployments/router_standalone
- Azure AKS: Coming soon
- Google GKE: Coming soon
- Amazon ECS: Coming soon
- Ray: Coming soon
- NVIDIA Cloud Functions (NVCF): Coming soon

## Runtime Examples (Python↔Rust)
Low-level runtime examples using Dynamo’s Python bindings:
- Hello World: src/lib/bindings/python/examples/hello_world

## Getting Started
1) Choose a path: start with Quickstart for an easy local deployment, or Disaggregated Serving for advanced architectures.
2) Start services: `docker compose -f tooling/docker-compose.yml up -d`
3) Follow the engine example: run `python -m dynamo.frontend` and the corresponding `python -m dynamo.<engine>` worker, then validate with curl.

## Framework Support
These examples demonstrate Dynamo’s engine-agnostic design across major inference engines. For deeper, framework-specific guidance:
- Engine examples: examples/engines/{vllm,sglang,trtllm}
- Source internals: src/components/backends/{vllm,sglang,trtllm,llama_cpp}
