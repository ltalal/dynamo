#!/bin/bash
set -e

curl -s ptyche0037:8000/v1/completions \
-H "Content-Type: application/json" \
-d @completions_payload.json 2>&1 | tee completions_output_streaming.log
