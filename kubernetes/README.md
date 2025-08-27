# Dynamo Deployment
For local development helpers (starting etcd/NATS and environment diagnostics), use the canonical tooling under the repository root:

- Local services: `docker compose -f tooling/docker-compose.yml up -d`
- Diagnostics: `python tooling/dynamo_check.py`

This ensures a single source of truth for local setups and troubleshooting.
➡️ View the full [guide](../docs/guides/dynamo_deploy/README.md)
