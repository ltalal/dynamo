# SGLang Single-Node (Local)

Run Dynamo locally with the SGLang engine using Python module entrypoints.

## Prerequisites
- Python 3.10+
- Docker + Docker Compose
- CUDA-capable GPU and drivers (for LLM inference)

## 1) Start control-plane services

```
docker compose -f tooling/docker-compose.yml up -d
```

## 2) Install the SGLang extra

```
uv venv venv
source venv/bin/activate
uv pip install pip
uv pip install "ai-dynamo[sglang]"
```

## 3) Run frontend and worker

Terminal A (OpenAI-compatible HTTP server):
```
python -m dynamo.frontend --http-port 8080
```

Terminal B (SGLang worker; replace model as needed):
```
python -m dynamo.sglang \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --skip-tokenizer-init
```

## 4) Validate

```
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
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

