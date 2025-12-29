#!/bin/bash
CUFILE_EXPERIMENTAL_FS=true CUFILE_ENV_PATH_JSON=cufile.json RUST_BACKTRACE=full RUST_LIB_BACKTRACE=full \
DYN_KVBM_DISK_CACHE_DIR=/mnt/weka/kvbm_cache \
DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER=true \
DYN_KVBM_DISK_CACHE_GB=4096 \
DYN_KVBM_METRICS=true \
  python3 -m vllm.entrypoints.openai.api_server \
    --model wangqia0309/Captain-Eris_Violet-V0.420-12B-FP8-KV \
    --served-model-name flowgpt/Captain-Eris_Violet-V0.420-12B \
    --trust-remote-code \
    --disable-log-requests \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 256 \
    --max-model-len 128000 \
    --block-size 32 \
    --enable-prefix-caching \
    --kv-transfer-config '{"kv_connector": "DynamoConnector", "kv_role": "kv_both", "kv_connector_module_path": "kvbm.vllm_integration.connector"}'