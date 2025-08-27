# TensorRT-LLM Single-Node (Local)

Run Dynamo locally with the TensorRT-LLM engine using Python module entrypoints.

## Prerequisites
- Python 3.10+
- Docker + Docker Compose
- CUDA-capable GPU and drivers (for LLM inference)
- System packages: `libopenmpi-dev`

## 1) Start control-plane services

```
docker compose -f tooling/docker-compose.yml up -d
```

## 2) Install the TRT-LLM extra

Depending on your CUDA stack, you may need additional pins (see root README TRT-LLM section). Minimal flow:

```
uv venv venv
source venv/bin/activate
uv pip install pip
uv pip install ai-dynamo[trtllm]
```

On some systems you may also need:
```
sudo apt-get -y install libopenmpi-dev
```

## 3) Run frontend and worker

Terminal A (OpenAI-compatible HTTP server):
```
python -m dynamo.frontend --http-port 8080
```

Terminal B (TRT-LLM worker; replace model/build as needed):
```
python -m dynamo.trtllm --help
```

Refer to engine flags if you already have a TRT-LLM engine build.

## 4) Validate

```
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false,
    "max_tokens": 64
  }'
```

## Troubleshooting
- Run diagnostics: `python tooling/dynamo_check.py`
- Ensure etcd (2379) and NATS (4222) are up from step 1.

## Notes
- Set `CUDA_VISIBLE_DEVICES` to choose GPUs.
- Single-node shell scripts in this folder are legacy helpers; prefer `python -m` flows above.

