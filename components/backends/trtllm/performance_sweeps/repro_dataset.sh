#!/bin/bash
export ISL=1024 
export OSL=1024

export SLURM_PARTITION="36x2-a01r"
export SLURM_ACCOUNT="$(sacctmgr -nP show assoc where user=$(whoami) format=account)"
export SLURM_JOB_NAME="${SLURM_ACCOUNT}-dynamo_trtllm:sa"


# NOTE: Old image, don't use
#export IMAGE="/lustre/fsw/core_dlfw_ci/rmccormick/images/tanmay-dynamo-trtllm-v2.sqsh"

# Based on tanmayv25/dynamo-trtllm:v3 (dockerhub): https://hub.docker.com/r/tanmayv25/dynamo-trtllm/tags
# - trtllm 1.1.0rc2
# - dynamo based on https://github.com/ai-dynamo/dynamo/tree/release/0.5.1-rc0.post1
export IMAGE="/lustre/fsw/core_dlfw_ci/rmccormick/images/tanmay-dynamo-trtllm-v3-trtllm1.1.0rc2.sqsh"


# pretyche paths
export MODEL_PATH="/lustre/fsw/core_dlfw_ci/rmccormick/DeepSeek-R1-0528-FP4-v2/"
export SERVED_MODEL_NAME="nvidia/DeepSeek-R1-FP4"

# 16GPU means decode will be EP16
# prefill will be 4 gpus by default
# So need 20 gpus (5 nodes of GB200)

# ORIGINAL:
#./submit_disagg_dataset.sh mtp=on 16GPU

# Itay's request:
#./submit_disagg_dataset.sh mtp=on 8GPU

# Izzy experiments
#./submit_disagg_dataset.sh mtp=on 16GPU
./submit_disagg_dataset.sh mtp=off 16GPU

# NOTES:
#   The logs directory has `output_serve.log` and `output_workers.log` files. 
# 
#   It also contains yaml configuration files for prefill and decode workers being used.
#
#   tail -f output_workers.log to get a live view of worker progress
#   tail -f bench.log to get a live view of benchmark progress
