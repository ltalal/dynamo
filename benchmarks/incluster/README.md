# In-Cluster Benchmarking

This directory contains a simple Kubernetes Job manifest for running Dynamo benchmarks directly within a Kubernetes cluster, eliminating the need for port forwarding.

## Overview

The in-cluster benchmarking solution consists of:

- **Kubernetes Job manifest** (`benchmark_job.yaml`) - Simple wrapper around the existing benchmarking infrastructure
- **Documentation** - Instructions for using the job manifest

## Key Benefits

- **No port forwarding required** - Uses Kubernetes service DNS for direct communication
- **Better resource utilization** - Runs within the cluster alongside your deployments
- **Persistent results** - Uses `dynamo-pvc` for storing manifests and results
- **Simple deployment** - Just edit the YAML and run `kubectl apply -f`
- **Uses existing infrastructure** - Leverages the existing `/benchmarks` and `/benchmarks/utils` code

## Prerequisites

1. **Kubernetes cluster** with NVIDIA GPUs and Dynamo namespace setup (see [Dynamo Cloud/Platform docs](../../docs/guides/dynamo_deploy/README.md))
2. **dynamo-pvc** PersistentVolumeClaim configured (see [deploy/utils README](../../deploy/utils/README.md))
3. **Service account** (`dynamo-sa`) with appropriate permissions (see [deploy/utils README](../../deploy/utils/README.md))
4. **Docker image** containing the Dynamo benchmarking tools

## Quick Start

### 1. Edit the Job Manifest

Edit `benchmark_job.yaml` to specify your benchmark inputs:

```yaml
# Set your namespace and docker image
NAMESPACE: your-namespace
DOCKER_IMAGE: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.5.0

# Add your --input arguments
INPUT_ARGS: --input agg=/manifests/agg.yaml --input disagg=/manifests/disagg.yaml
```

### 2. Deploy and Run

```bash
# Deploy the benchmark job
kubectl apply -f benchmark_job.yaml

# With a custom namespace
NAMESPACE=hannahz envsubst < benchmark_job.yaml | kubectl apply -f -

# Monitor the job
kubectl logs -f job/dynamo-benchmark -n your-namespace

# Check job status
kubectl get jobs -n your-namespace
```

### 3. Retrieve Results

```bash
# Copy results from PVC
kubectl cp <pod-name>:/results ./benchmark_results -n your-namespace
```

## Configuration

### Environment Variables

The job manifest supports these environment variables with reasonable defaults:

- `NAMESPACE` - Kubernetes namespace (default: benchmarking)
- `DOCKER_IMAGE` - Docker image to use (default: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.5.0)
- `MODEL` - Model name (default: Qwen/Qwen3-0.6B)
- `ISL` - Input sequence length (default: 2000)
- `STD` - Input sequence standard deviation (default: 10)
- `OSL` - Output sequence length (default: 256)
- `INPUT_ARGS` - Your --input arguments (no default - you must specify these)

### Input Arguments

Specify your benchmark inputs using the `INPUT_ARGS` variable:

```yaml
# Compare DynamoGraphDeployment manifests
INPUT_ARGS: --input agg=/manifests/agg.yaml --input disagg=/manifests/disagg.yaml

# Compare Dynamo vs external services
INPUT_ARGS: --input dynamo=/manifests/dynamo.yaml --input external=http://external-service:8000

# Single deployment benchmark
INPUT_ARGS: --input my-deployment=/manifests/my-config.yaml
```

## Usage Examples

### Compare Multiple Dynamo Deployments

```yaml
# In benchmark_job.yaml
NAMESPACE: benchmarking
DOCKER_IMAGE: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.5.0
INPUT_ARGS: --input vllm-agg=/manifests/vllm-agg.yaml --input vllm-disagg=/manifests/vllm-disagg.yaml --input trtllm-agg=/manifests/trtllm-agg.yaml
```

```bash
kubectl apply -f benchmark_job.yaml
kubectl logs -f job/dynamo-benchmark -n benchmarking
```

### Compare Dynamo vs External Services

```yaml
# In benchmark_job.yaml
NAMESPACE: benchmarking
DOCKER_IMAGE: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.5.0
INPUT_ARGS: --input dynamo=/manifests/dynamo-config.yaml --input external-vllm=http://external-vllm-service:8000 --input external-trtllm=http://external-trtllm-service:8000
```

```bash
kubectl apply -f benchmark_job.yaml
kubectl logs -f job/dynamo-benchmark -n benchmarking
```

### Custom Model and Parameters

```yaml
# In benchmark_job.yaml
NAMESPACE: benchmarking
DOCKER_IMAGE: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.5.0
MODEL: meta-llama/Meta-Llama-3-8B
ISL: 1500
OSL: 200
INPUT_ARGS: --input my-deployment=/manifests/my-config.yaml
```

```bash
kubectl apply -f benchmark_job.yaml
kubectl logs -f job/dynamo-benchmark -n benchmarking
```

## Job Manifest Structure

The job manifest includes:

- **Service Account**: Uses `dynamo-sa` for Kubernetes API access
- **Resource Limits**: Configurable CPU/memory limits
- **Environment Variables**: HuggingFace token, NATS, ETCD endpoints
- **Volume Mounts**:
  - `/manifests`: Read-only access to deployment manifests
  - `/results`: Read-write access for benchmark results
- **Cleanup**: Automatic job cleanup after 1 hour

## Monitoring and Debugging

### Check Job Status

```bash
kubectl get jobs -n <namespace>
kubectl describe job dynamo-benchmark -n <namespace>
```

### View Logs

```bash
# Follow logs in real-time
kubectl logs -f job/dynamo-benchmark -n <namespace>

# Get logs from specific container
kubectl logs job/dynamo-benchmark -c benchmark-runner -n <namespace>
```

### Debug Failed Jobs

```bash
# Check pod status
kubectl get pods -n <namespace> -l job-name=dynamo-benchmark

# Describe failed pod
kubectl describe pod <pod-name> -n <namespace>

# Get events
kubectl get events -n <namespace> --sort-by='.lastTimestamp'
```

## Results Structure

Results are organized in the `/results` directory:

```
/results/
├── plots/                           # Performance visualization plots
│   ├── SUMMARY.txt                  # Human-readable benchmark summary
│   ├── p50_inter_token_latency_vs_concurrency.png
│   ├── avg_inter_token_latency_vs_concurrency.png
│   ├── request_throughput_vs_concurrency.png
│   ├── efficiency_tok_s_gpu_vs_user.png
│   └── avg_time_to_first_token_vs_concurrency.png
├── <label-1>/                       # Results for first input
│   ├── c1/                          # Concurrency level 1
│   │   └── profile_export_genai_perf.json
│   ├── c2/                          # Concurrency level 2
│   └── ...                          # Other concurrency levels
└── <label-N>/                       # Results for additional inputs
    └── c*/                          # Same structure as above
```

## Troubleshooting

### Common Issues

1. **Manifest not found**: Ensure manifests are copied to `/manifests` in the PVC
2. **Service account permissions**: Verify `dynamo-sa` has necessary RBAC permissions
3. **PVC access**: Check that `dynamo-pvc` is properly configured and accessible
4. **Image pull issues**: Ensure the Docker image is accessible from the cluster
5. **Resource constraints**: Adjust resource limits if the job is being evicted

### Debug Commands

```bash
# Check PVC status
kubectl get pvc dynamo-pvc -n <namespace>

# Verify service account
kubectl get sa dynamo-sa -n <namespace>

# Check RBAC permissions
kubectl auth can-i create dynamographdeployments --as=system:serviceaccount:<namespace>:dynamo-sa -n <namespace>
```

## Comparison with Local Benchmarking

| Feature | Local Benchmarking | In-Cluster Benchmarking |
|---------|-------------------|------------------------|
| Port Forwarding | Required | Not needed |
| Resource Usage | Local machine | Cluster resources |
| Network Latency | Higher (port-forward) | Lower (direct service) |
| Scalability | Limited | High |
| Isolation | Shared environment | Isolated job |
| Results Storage | Local filesystem | Persistent PVC |

The in-cluster approach is recommended for:
- Production benchmarking
- Multiple deployment comparisons
- Resource-constrained environments
- Automated CI/CD pipelines