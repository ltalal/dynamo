#!/bin/bash -x

HOST=localhost

# Sanity check
#genai-perf profile -m ${SERVED_MODEL_NAME} \
#--endpoint-type=chat \
#--synthetic-input-tokens-mean 128   \
#--synthetic-input-tokens-stddev 0   \
#--output-tokens-mean 100   \
#--output-tokens-stddev 0   \
#--url $HOST:8000   \
#--streaming   \
#--request-count 10   \
#--warmup-request-count 2 \
#--tokenizer ${MODEL_PATH}

genai-perf profile -m ${SERVED_MODEL_NAME} \
--endpoint-type=chat \
--synthetic-input-tokens-mean 20480   \
--synthetic-input-tokens-stddev 5120   \
--output-tokens-mean 1024   \
--output-tokens-stddev 256  \
--url $HOST:8000   \
--streaming   \
--request-count 2500   \
--concurrency 6
--warmup-request-count 6 \
--tokenizer ${MODEL_PATH}
