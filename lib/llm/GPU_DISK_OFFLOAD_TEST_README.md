# GPU->Disk Offload Performance Test

## Overview

A comprehensive performance test has been added to measure GPU->Disk direct offloading speed and latency under different concurrency configurations.

**Location**: `lib/llm/src/block_manager/offload.rs` (lines 2504-2760)
**Test Name**: `test_gpu_disk_offload_performance`

## Features

The test measures:
- **Throughput**: MB/s and GB/s transfer rates
- **Latency**: Average latency per block transfer (ms)
- **Blocks per second**: Transfer rate in blocks/sec
- **First block latency**: Time to complete first transfer

## Configuration Parameters

Edit these parameters directly in the test code (around line 2550):

```rust
let num_blocks = 128;                    // Number of blocks to transfer
let max_concurrent_transfers = 8;        // Concurrent transfer limit (try: 1, 2, 4, 8, 16)
let max_transfer_batch_size = 16;       // Batch size (try: 1, 8, 16, 32)
let disk_path = std::env::var("DISK_PATH").unwrap_or_else(|_| "/tmp".to_string());
```

## How to Run

### Prerequisites

1. **Install protoc** (Protocol Buffers compiler):
   ```bash
   sudo apt-get install protobuf-compiler
   ```

2. **NVIDIA GPU** with CUDA support

3. **Disk space**: The test transfers approximately 0.5MB per block (default: 128 blocks = ~64MB)

### Running the Test

**Default configuration** (uses /tmp):
```bash
cd lib/llm
cargo test test_gpu_disk_offload_performance -- --ignored --nocapture
```

**Custom disk path** (e.g., /mnt/model-storage/kvbm_cache/testcache.tmp):
```bash
cd lib/llm
DISK_PATH=/mnt/model-storage/kvbm_cache/testcache.tmp cargo test test_gpu_disk_offload_performance -- --ignored --nocapture
```

## Testing Different Concurrency Settings

To benchmark different `MAX_CONCURRENT_TRANSFERS` and `MAX_TRANSFER_BATCH_SIZE` values:

1. Edit `lib/llm/src/block_manager/offload.rs` around line 2550-2552
2. Change the values:
   - `max_concurrent_transfers`: 1, 2, 4, 8, 16
   - `max_transfer_batch_size`: 1, 8, 16, 32
3. Run the test
4. Record the performance metrics
5. Repeat with different values

### Example Test Matrix

| MAX_CONCURRENT_TRANSFERS | MAX_TRANSFER_BATCH_SIZE | Throughput (GB/s) | Avg Latency (ms) |
|-------------------------|------------------------|-------------------|------------------|
| 1                       | 1                      | ?                 | ?                |
| 1                       | 16                     | ?                 | ?                |
| 4                       | 16                     | ?                 | ?                |
| 8                       | 16                     | ?                 | ?                |
| 16                      | 16                     | ?                 | ?                |

## Implementation Details

- **Direct GPU->Disk transfers**: Uses `bypass_cpu_mem: true` to enable GPUDirect Storage
- **No host memory involved**: Transfers go directly from GPU to disk
- **GDS alignment**: Uses 4KB alignment for optimal GDS performance
- **Data verification**: All blocks are verified after transfer to ensure correctness
- **Configurable transfer settings**: Allows testing different concurrency and batch size configurations

## Output Example

```
=== GPU->Disk Offload Performance Test ===
Configuration:
  Num blocks: 128
  MAX_CONCURRENT_TRANSFERS: 8
  MAX_TRANSFER_BATCH_SIZE: 16
  Block size (tokens): 4
  Disk path: /mnt/model-storage/kvbm_cache/testcache.tmp
  Num layers: 8

Allocating and populating 128 blocks on GPU...
Block size: 524288 bytes (0.50 MB)
Total data to transfer: 67108864 bytes (64.00 MB)

Starting GPU->Disk offload...

Verifying transferred blocks...
All blocks verified successfully!

=== Performance Results ===
Total duration: 0.523 seconds
Throughput: 122.37 MB/s (0.12 GB/s)
Average latency per block: 4.09 ms
Blocks per second: 244.74
First block latency: 4.09 ms

=== Configuration Used ===
MAX_CONCURRENT_TRANSFERS: 8
MAX_TRANSFER_BATCH_SIZE: 16
Block size: 524288 bytes
Total blocks: 128
```

## Notes

- The test is marked with `#[ignore]` so it won't run during normal `cargo test` executions
- Must be explicitly run with the `--ignored` flag
- Requires GPU and disk I/O, so it's not suitable for CI environments
- The disk path must exist before running the test
- GDS (GPUDirect Storage) support recommended for optimal performance
