# Dynamo Development Guide

## Python Package Structure

Dynamo has **two separate Python packages**:

1. **`ai-dynamo-runtime`** (`lib/bindings/python/`)
   - Module: `dynamo._core`
   - Use for: General Dynamo runtime functionality

2. **`kvbm`** (`lib/kvbm/`)
   - Module: `kvbm._core`
   - Use for: vLLM integration, KV cache block manager
   - **This is what vLLM uses!**

## Initial Setup (Required Every Container Restart)

⚠️ **IMPORTANT**: The venv at `/opt/dynamo/venv/` is NOT persisted between container restarts.

Every time you start the dev container with `./run_devc.sh`, you must run:

```bash
./build_kvbm_dev.sh
```

This installs `kvbm` in editable mode, so Python loads your development code from `/workspace/lib/kvbm/`.

## Development Workflow for vLLM Integration

### 1. Make Changes to Rust Code

Edit files in `lib/kvbm/src/`:
```bash
vim lib/kvbm/src/block_manager/vllm/connector/worker.rs
# or any other Rust file in lib/kvbm/src/
```

### 2. Rebuild the KVBM Package

**IMPORTANT**: Build from the `lib/kvbm` directory, NOT `lib/bindings/python`!

```bash
cd lib/kvbm && maturin develop
```

This compiles the Rust code and builds:
```
lib/kvbm/python/kvbm/_core.abi3.so
```

### 3. Restart vLLM

The running vLLM process has the OLD `_core.so` loaded in memory. You must restart it:

```bash
# Press Ctrl+C to stop the old server
# Then start it again:
python3 -m vllm.entrypoints.openai.api_server \
  --model YOUR_MODEL \
  --kv-transfer-config '{"kv_connector": "DynamoConnector", ...}'
```

## Complete Development Cycle

### On Container Restart
```bash
# 1. Start dev container
./run_devc.sh

# 2. Set up editable installation (required every time)
./build_kvbm_dev.sh
```

### During Development
```bash
# Edit → Rebuild → Restart vLLM
vim lib/kvbm/src/block_manager/vllm/connector/worker.rs
cd lib/kvbm && maturin develop  # or: ./build_kvbm_dev.sh
# Kill vLLM (Ctrl+C) and restart it
```

## Verification

### Verify Correct Package is Loaded

After running `./build_kvbm_dev.sh`, verify the editable installation:

```bash
# Check kvbm installation (should show "Editable project location")
pip show kvbm

# Verify .so location (should be in workspace, not site-packages)
python3 -c "import kvbm._core; print(kvbm._core.__file__)"
# Expected: /workspace/lib/kvbm/python/kvbm/_core.abi3.so
```

If you forgot to run `./build_kvbm_dev.sh`, the `.so` will load from `/opt/dynamo/venv/` instead of `/workspace/`, and your changes won't be reflected.

### Check Build Cache

Maturin/Cargo caches are stored in:
- **Build artifacts**: `lib/kvbm/target/` (~2GB)
- **Downloaded crates**: `~/.cargo/registry/` and `~/.cargo/git/`

The dev container (`run_devc.sh`) preserves both caches between runs.

## Common Issues

### Issue: Changes not appearing in vLLM
**Solution**: Make sure you:
1. Built from `lib/kvbm` (not `lib/bindings/python`)
2. Restarted vLLM after rebuilding

### Issue: Wrong .so file loaded
**Solution**: Check with:
```bash
python3 -c "import kvbm._core; print(kvbm._core.__file__)"
```
Should show `/workspace/lib/kvbm/python/kvbm/_core.abi3.so`

If it shows something in `/opt/dynamo/venv/lib/python3.12/site-packages/`, the package is not in editable mode. Reinstall with:
```bash
cd lib/kvbm && maturin develop
```

## Building for ai-dynamo-runtime Package

If you need to work on the general Dynamo runtime (not vLLM integration):

```bash
cd lib/bindings/python && maturin develop
```

This builds `dynamo._core` instead of `kvbm._core`.
