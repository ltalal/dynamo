#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -ex

trap 'echo Cleaning up...; kill 0' EXIT

MODEL_NAME="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

# run ingress
python -m dynamo.frontend &

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || realpath "$(dirname "$0")/../../..")

# run processor
python3 "$REPO_ROOT/src/components/multimodal/components/processor.py" --model $MODEL_NAME --prompt-template "<|image|>\n<prompt>" &
# LLama 4 doesn't support image embedding input, so the prefill worker will also
# handle image encoding.
# run EP/D workers
python3 "$REPO_ROOT/src/components/multimodal/components/worker.py" --model $MODEL_NAME --worker-type encode_prefill --tensor-parallel-size=8 --max-model-len=208960 &

# Wait for all background processes to complete
wait
