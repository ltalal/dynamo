# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**NVIDIA Dynamo** is a distributed LLM serving orchestration framework designed for multi-node, multi-GPU inference. It coordinates KV cache across workers and memory hierarchies (GPU → CPU → Disk), provides intelligent request routing, and supports disaggregated prefill/decode serving.

**Key Architecture**:
- **Rust Core**: Performance-critical components (networking, routing, KV cache management, tokenization)
- **Python Layer**: Engine integrations (vLLM, SGLang, TensorRT-LLM), configuration, multimodal preprocessing
- **Two Python Packages**:
  - `ai-dynamo-runtime` (from `lib/bindings/python/`) - General Dynamo runtime
  - `kvbm` (from `lib/kvbm/`) - vLLM integration and KV cache block manager

## Build Commands

### Initial Setup (Required)

Install system dependencies (requires sudo):
```bash
sudo ./install_deps.sh
```

This installs UCX, NIXL, gdrcopy, and other system libraries required for cargo build. After installation, set environment variables:
```bash
export NIXL_PREFIX=/opt/nvidia/nvda_nixl
export NIXL_LIB_DIR=/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=${NIXL_LIB_DIR}:${NIXL_LIB_DIR}/plugins:/usr/local/ucx/lib:/usr/local/ucx/lib/ucx:${LD_LIBRARY_PATH}
```

### Development Setup

Create Python virtual environment:
```bash
uv venv dynamo
source dynamo/bin/activate
uv pip install pip maturin
```

### Building Rust Components

Build all Rust libraries:
```bash
cargo build --locked --profile dev
```

Build with release optimizations:
```bash
cargo build --locked --release
```

Build specific crate:
```bash
cargo build -p dynamo-llm
cargo build -p dynamo-runtime
```

### Building Python Bindings

**For general Dynamo runtime** (`ai-dynamo-runtime`):
```bash
cd lib/bindings/python
maturin develop --uv
```

**For KVBM/vLLM integration** (`kvbm`):
```bash
cd lib/kvbm
maturin develop --uv
```

Quick rebuild script (for KVBM development):
```bash
./build_kvbm_dev.sh
```

### Installing Python Package

Install Dynamo in editable mode:
```bash
uv pip install -e .
```

Install with specific engine support:
```bash
uv pip install -e ".[vllm]"
uv pip install -e ".[sglang]"
uv pip install -e ".[trtllm]"
```

## Running Services

### Prerequisites

Start etcd and NATS (required for all Dynamo services):
```bash
docker compose -f deploy/docker-compose.yml up -d
```

Or run them manually:
```bash
# Terminal 1: etcd
./etcd

# Terminal 2: NATS with JetStream
nats-server -js
```

### Running Frontend and Workers

Start HTTP frontend with router:
```bash
python -m dynamo.frontend --http-port 8000
```

Start SGLang worker:
```bash
python -m dynamo.sglang --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

Start vLLM worker:
```bash
python -m dynamo.vllm --model <model-name>
```

Start TensorRT-LLM worker:
```bash
python -m dynamo.trtllm --help
```

### Environment Variables

- `DYN_LOG`: Set Rust logging level (same syntax as `RUST_LOG`)
  ```bash
  export DYN_LOG=debug
  export DYN_LOG=info,dynamo_runtime=debug
  ```
- `CUDA_VISIBLE_DEVICES`: Select which GPUs to use
- `NIXL_PREFIX`, `NIXL_LIB_DIR`: Required for NIXL transfers

## Testing

### Running Tests

Run all Python tests:
```bash
pytest
```

Run specific test categories:
```bash
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests
pytest -m gpu_1         # Tests requiring 1 GPU
pytest -m gpu_2         # Tests requiring 2 GPUs
pytest -m vllm          # vLLM-specific tests
pytest -m kvbm          # KVBM tests
```

Run specific test file:
```bash
pytest tests/router/test_kv_router.py
```

Run Rust tests:
```bash
cargo test                    # All tests
cargo test -p dynamo-llm      # Specific crate
cargo test --features testing-cuda  # With CUDA features
```

Run a single Rust test:
```bash
cargo test test_name
cargo test test_name -- --nocapture  # Show output
```

## Architecture

### High-Level Component Flow

```
HTTP Client → Frontend (Python)
              ├─ HTTP Server (Rust)
              ├─ Preprocessor (Rust: tokenization, templates)
              └─ Router (Rust: KV-aware or round-robin)
                   ↓ NATS pub/sub
              Worker Discovery (etcd)
                   ↓
              Workers (Prefill/Decode/Unified)
              ├─ vLLM Engine
              ├─ SGLang Engine
              └─ TensorRT-LLM Engine
                   ↓
              KVBM System (KV Cache Management)
              ├─ Leader (Scheduler)
              ├─ Workers (Executors)
              ├─ Transfer Engine
              └─ Scheduler (Two-phase commit)
                   ↓
              Storage Tiers: [GPU] [CPU] [Disk]
```

### Key Libraries (`lib/`)

- **`runtime/`**: Distributed runtime foundation
  - Component system: service registration, discovery, client generation
  - Pipeline: stream processing with context propagation
  - Transports: NATS (pub/sub), etcd (watch/lock), TCP
  - Metrics: Prometheus integration

- **`llm/`**: LLM-specific primitives
  - `block_manager/`: KV cache block allocation and transfer orchestration
  - `kv_router/`: Prefix matching (radix tree), worker scoring algorithms
  - `preprocessor/`: Chat templates (Jinja2), tokenization, multimodal
  - `protocols/`: OpenAI API compatibility layer
  - `backend.rs`: Token decoding and stop condition handling

- **`kvbm/`**: Advanced distributed KV cache
  - Leader-Worker coordination via ZMQ
  - Multi-tier storage (Device → Host → Disk)
  - Transfer scheduler with two-phase commit
  - Lock-free completion tracking (atomic counters)
  - Per-request state machines (see DYNAMO_CONNECTOR_DATAFLOW.md for details)

- **`config/`**: Figment-based configuration (env/file/CLI sources)
- **`tokens/`**: Token ID handling and vocabulary utilities
- **`parsers/`**: OpenAI request/response parsing, tool calling
- **`async-openai/`**: OpenAI client with custom tokenizer injection (BYOT)
- **`bindings/python/`**: PyO3 bindings exposing Rust to Python (stable ABI)

### Rust-Python Integration

**Division of Responsibilities**:
- **Rust**: Network I/O, pipeline processing, routing, KV cache management, tokenization, all hot-path operations
- **Python**: Engine integrations (vLLM/SGLang/TRT-LLM APIs), configuration parsing, multimodal preprocessing, health checks

**Integration Pattern**:
```python
# Python imports Rust modules via PyO3 bindings
from dynamo.runtime import DistributedRuntime
from dynamo.llm import make_engine, register_llm

# Python registers callbacks for Rust to invoke
register_engine_metrics_callback(llm.get_stats)
```

## Development Workflow

### For vLLM/KVBM Integration

1. **Edit Rust code**: `vim lib/kvbm/src/block_manager/vllm/connector/worker.rs`
2. **Rebuild**: `cd lib/kvbm && maturin develop` (or `./build_kvbm_dev.sh`)
3. **Restart vLLM**: The running vLLM process has the old `.so` loaded in memory - must restart

Verify correct package is loaded:
```bash
python3 -c "import kvbm._core; print(kvbm._core.__file__)"
# Expected: /workspace/lib/kvbm/python/kvbm/_core.abi3.so
```

### For General Dynamo Runtime

1. **Edit Rust code**: `vim lib/runtime/src/pipeline.rs`
2. **Rebuild**: `cd lib/bindings/python && maturin develop`
3. **Restart service**: Restart frontend or other service using the runtime

### Code Patterns

**Service Registration**:
```rust
Component::builder()
    .namespace("llm.default.model-a")
    .endpoint("prefill_worker")
    .serve(handler)
```

**Pipeline Composition**:
```rust
// Chain operators: HTTP → Preprocessor → Router → Backend
AsyncEngine<Input, Output> trait with operator chaining
Context<T> carries metadata through the pipeline
```

**Discovery-Driven Configuration**:
- Services register in etcd: `/llm/{namespace}/{model}/{role}/{instance_id}`
- Clients watch for changes and update routing tables dynamically

**Error Handling**:
- Use `anyhow::Result<T>` for most functions
- Structured errors with `thiserror` for library code
- Context propagation: `.context("operation failed")?`

## Important Files

- **`DYNAMO_CONNECTOR_DATAFLOW.md`**: Detailed data flow analysis of KVBM connector system (leader-worker coordination, state machines, transfer protocols)
- **`DEV.md`**: Development guide for Python package structure and KVBM workflow
- **`README.md`**: Installation instructions and feature matrix
- **`Cargo.toml`**: Workspace configuration and dependencies
- **`pyproject.toml`**: Python package configuration and dependencies

## Common Issues

### Changes not appearing in vLLM
- Ensure you built from `lib/kvbm` (not `lib/bindings/python`)
- Restart vLLM after rebuilding

### Wrong .so file loaded
Check with: `python3 -c "import kvbm._core; print(kvbm._core.__file__)"`
Should show `/workspace/lib/kvbm/python/kvbm/_core.abi3.so`
If it shows something in site-packages, reinstall with: `cd lib/kvbm && maturin develop`

### Cargo build fails
- Run `sudo ./install_deps.sh` to install UCX, NIXL, gdrcopy
- Set environment variables for NIXL (see "Initial Setup" section)
- Check that CUDA is installed at `/usr/local/cuda`

### NATS/etcd connection errors
- Ensure services are running: `docker compose -f deploy/docker-compose.yml up -d`
- Check ports: etcd (2379), NATS (4222)
