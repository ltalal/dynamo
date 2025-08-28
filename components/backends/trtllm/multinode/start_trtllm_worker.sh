#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if [[ -z ${MODEL_PATH} ]]; then
    echo "ERROR: MODEL_PATH was not set."
    echo "ERROR: MODEL_PATH must be set to either the HuggingFace ID or locally " \
         "downloaded path to the model weights. Since Deepseek R1 is large, it is " \
         "recommended to pre-download them to a shared location and provide the path."
    exit 1
fi

if [[ -z ${SERVED_MODEL_NAME} ]]; then
    echo "WARNING: SERVED_MODEL_NAME was not set. It will be derived from MODEL_PATH."
fi



if [[ -z ${ENGINE_CONFIG} ]]; then
    echo "ERROR: ENGINE_CONFIG was not set."
    echo "ERROR: ENGINE_CONFIG must be set to a valid Dynamo+TRTLLM engine config file."
    exit 1
fi

EXTRA_ARGS=""
if [[ -n ${DISAGGREGATION_MODE} ]]; then
  EXTRA_ARGS+="--disaggregation-mode ${DISAGGREGATION_MODE} "
fi

if [[ -n ${DISAGGREGATION_STRATEGY} ]]; then
  EXTRA_ARGS+="--disaggregation-strategy ${DISAGGREGATION_STRATEGY} "
fi

if [[ -n ${MODALITY} ]]; then
  EXTRA_ARGS+="--modality ${MODALITY} "
fi

# FIXME: yaml engine config files should take precedence, but don't seem to be,
# so set max num tokens here as a workaround
if [[ -n ${MAX_NUM_TOKENS} ]]; then
  EXTRA_ARGS+="--max-num-tokens ${MAX_NUM_TOKENS} "
fi

if [[ -n ${MAX_BATCH_SIZE} ]]; then
  EXTRA_ARGS+="--max-batch-size ${MAX_BATCH_SIZE} "
fi



trtllm-llmapi-launch \
  python3 -m dynamo.trtllm \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --extra-engine-args "${ENGINE_CONFIG}" \
    ${EXTRA_ARGS}
