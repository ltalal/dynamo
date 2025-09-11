#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Add error handling
set -e
set -u
trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

# Check if SCRIPTS_DIR is set, if not try to infer it or exit
if [ -z "${SCRIPTS_DIR:-}" ]; then
    # Try to infer SCRIPTS_DIR from the current script location
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SCRIPTS_DIR="$(dirname "$SCRIPT_DIR")"
    echo "SCRIPTS_DIR not set, inferred as: ${SCRIPTS_DIR}"
fi

# Verify SCRIPTS_DIR exists and contains expected structure
if [ ! -d "${SCRIPTS_DIR}/scripts/bench" ]; then
    echo "Error: SCRIPTS_DIR (${SCRIPTS_DIR}) does not contain expected structure"
    echo "Expected: ${SCRIPTS_DIR}/scripts/bench to exist"
    exit 1
fi

WAIT_TIME=300

model=${1}
multi_round=${2}
num_gen_servers=${3}
concurrency_list=${4}
streaming=${5}
log_path=${6}
total_gpus=${7}
model_path=${8}
isl=${9}
osl=${10}
kind=${11}

if [ "$#" -ne 11 ]; then
    echo "Error: Expected 11 arguments, got $#"
    echo "Usage: $0 <model> <multi_round> <num_gen_servers> <concurrency_list> <streaming> <log_path> <total_gpus> <model_path> <isl> <osl> <kind>"
    exit 1
fi

echo "Arguments:"
echo "  model: $model"
echo "  multi_round: $multi_round"
echo "  num_gen_servers: $num_gen_servers"
echo "  concurrency_list: $concurrency_list"
echo "  streaming: $streaming"
echo "  log_path: $log_path"
echo "  total_gpus: $total_gpus"
echo "  model_path: $model_path"
echo "  isl: $isl"
echo "  osl: $osl"
echo "  kind: $kind"



# check process id is not 0
if [[ ${SLURM_PROCID} != "0" ]]; then
    echo "Process id is ${SLURM_PROCID} for loadgen, exiting"
    exit 0
fi

set -x
config_file=${log_path}/config.yaml


hostname=$HEAD_NODE
port=8000

echo "Hostname: ${hostname}, Port: ${port}"

apt update
apt install curl


# try client

do_get_logs(){
    worker_log_path=$1
    output_folder=$2
    grep -a "'num_ctx_requests': 0, 'num_ctx_tokens': 0" ${worker_log_path} > ${output_folder}/gen_only.txt || true
    grep -a "'num_generation_tokens': 0" ${worker_log_path} > ${output_folder}/ctx_only.txt || true
}

# The configuration is dumped to a JSON file which hold details of the OAI service
# being benchmarked.
deployment_config=$(cat << EOF
{
  "kind": "${kind}",
  "model": "${model}",
  "total_gpus": "${total_gpus}"
}
EOF
)

mkdir -p "${log_path}"
if [ -f "${log_path}/deployment_config.json" ]; then
  echo "Deployment configuration already exists. Overwriting..."
  rm -f "${log_path}/deployment_config.json"
fi
echo "${deployment_config}" > "${log_path}/deployment_config.json"

# Wait for server to load (up to 50 attempts)
failed=true
for ((i=1; i<=50; i++)); do
    sleep $((i == 1 ? WAIT_TIME : 20))
    response=$(curl -s -w "\n%{http_code}" "${hostname}:${port}/v1/models")
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')

    if [[ "$http_code" == "200" ]]; then
        echo "$body"
        if echo "$body" | grep -q "\"id\":\"${model}\""; then
            echo "Model check succeeded on attempt $i"
            failed=false
            break
        else
            echo "Attempt $i: Model '${model}' not found in /v1/models response."
        fi
    else
        echo "Attempt $i failed: /v1/models not ready (HTTP $http_code)."
    fi
done

if [[ "$failed" == "true" ]]; then
    echo "Server did not respond with healthy status after 50 attempts."
    exit 1
fi

curl -v  -w "%{http_code}" "${hostname}:${port}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
  "model": "'${model}'",
  "messages": [
  {
    "role": "user",
    "content": "Tell me a story as if we were playing dungeons and dragons."
  }
  ],
  "stream": true,
  "max_tokens": 30
}'

cp ${log_path}/output_workers.log ${log_path}/workers_start.log

# WAR for TRTLLM custom dataset
SCRIPTS_DIR="/lustre/share/coreai_comparch_trtllm/iputterman/iputterman/TensorRT-LLM/tensorrt_llm/serve/scripts"
#python3 ${SCRIPTS_DIR}/scripts/bench/benchmark_serving.py \
python3 ${SCRIPTS_DIR}/benchmark_serving.py \
        --served-model-name ${model} \
        --model ${model_path} \
        --dataset-name "trtllm_custom" \
        --dataset-path /lustre/share/coreai_comparch_trtllm/iputterman/iputterman/deepseek-r1-1024-1024-200000-ratio-08_for_serve.json \
        --num-prompts "${multi_round}" \
        --random-input-len ${isl} \
        --random-output-len ${osl} \
        --random-range-ratio 0.8 \
        --ignore-eos \
        --backend "openai" \
        --endpoint "/v1/completions" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --max-concurrency "1" \
        --host ${hostname} \
        --port ${port}

echo "Starting benchmark..."
for concurrency in ${concurrency_list}; do
    concurrency=$((concurrency * num_gen_servers))
    num_prompts=$((concurrency * multi_round))
    echo "Benchmarking with concurrency ${concurrency} ... ${num_prompts} prompts"
    mkdir -p ${log_path}/concurrency_${concurrency}

    # NOTES:
    # - double check --dataset name/path
    # - double check endpoint (chat vs completions)
    # - double check --use-chat-template on client-side vs server side
    #python3 ${SCRIPTS_DIR}/scripts/bench/benchmark_serving.py \
    python3 ${SCRIPTS_DIR}/benchmark_serving.py \
        --served-model-name ${model} \
        --model ${model_path} \
        --dataset-name "trtllm_custom" \
        --dataset-path /lustre/share/coreai_comparch_trtllm/iputterman/iputterman/deepseek-r1-1024-1024-200000-ratio-08_for_serve.json \
        --num-prompts "$num_prompts" \
        --random-input-len ${isl} \
        --random-output-len ${osl} \
        --random-range-ratio 0.8 \
        --ignore-eos \
        --backend "openai" \
        --endpoint "/v1/completions" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --max-concurrency "$concurrency" \
        --host ${hostname} \
        --port ${port}

# Try removing this in case we are double tokenizing on both client and server
# and possibly negatively impacting the MTP Acceptance Rate
#        --use-chat-template \

# chat completions
#        --backend "dynamo" \
#        --endpoint "/v1/chat/completions" \

# completions
#        --use-chat-template \
#        --backend "openai" \
#        --endpoint "/v1/completions" \

# dataset considerations
# (1) random data generated
#        --dataset-name random \
# (2) trtllm dataset (also random?)
#        --dataset-name "trtllm_custom" \
#        --dataset-path /lustre/share/coreai_comparch_trtllm/iputterman/iputterman/deepseek-r1-1024-1024-200000-ratio-08_for_serve.json
#
# original location:
#        --dataset-path /lustre/fsw/coreai_comparch_trtllm/kevxie/tensorrt_llm_scripts/xianjie/dataset/deepseek-r1-1024-1024-200000-ratio-08_for_serve.json

    echo "Benchmark with concurrency ${concurrency} done"
    do_get_logs ${log_path}/output_workers.log ${log_path}/concurrency_${concurrency}
    echo -n "" > ${log_path}/output_workers.log
done


job_id=${SLURM_JOB_ID}
if [ -n "${job_id}" ]; then
    echo "${SLURM_JOB_NODELIST}" > ${log_path}/job_${job_id}.txt
fi
