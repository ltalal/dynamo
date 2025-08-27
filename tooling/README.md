# Tooling

Utility assets for local development and validation.

- `docker-compose.yml`: Starts etcd and NATS locally for Dynamo.
- `dynamo_check.py`: Diagnostics script to verify connectivity, ports, and environment.

Quick start

1) Launch control-plane services:

```
docker compose -f tooling/docker-compose.yml up -d
```

2) Run diagnostics (optional but recommended):

```
python tooling/dynamo_check.py
```

3) Start Dynamo components using Python module entrypoints:

```
python -m dynamo.frontend --http-port 8080
python -m dynamo.<engine> ...
```

