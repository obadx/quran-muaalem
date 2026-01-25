# Triton Inference Server – Local Setup Guide

This part of the repository uses **NVIDIA Triton Inference Server** to serve the ASR model efficiently on GPU with support for:

- Dynamic batching
- gRPC / HTTP inference
- Prometheus metrics
- Production-grade deployment

This guide explains how to set up Triton **locally**, both **with Docker (recommended)** and **without Docker**, so contributors and users can run the system even if a prebuilt image is not available.

---

## 1. Prerequisites

### Hardware

- NVIDIA GPU (compute capability ≥ 7.0 recommended)
- At least **8 GB VRAM** (depends on model size)

### Software

- Linux (Ubuntu 20.04 / 22.04 recommended)
- NVIDIA Driver ≥ **535**
- CUDA ≥ **12.x**

Verify GPU access:

```bash
nvidia-smi
```

## 2. Triton Model Repository Structure

Triton requires a **strict directory structure**.

``` bash
triton/models/
└── muaalem/
    ├── 1/
    │   └── model.pt
    └── config.pbtxt
```
### Important Rules

- Folder name (muaalem) must match the Triton model_name

- A version folder (1/) is mandatory

- config.pbtxt defines inputs, outputs, batching, and device placement

## 3. Run Triton with Docker

This is the **easiest and safest** way to run Triton.

### 3.1. Install Docker

```bash
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable --now docker
```

### 3.2. Install NVIDIA Container Toolkit

``` bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU access inside Docker:

``` bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### 3.3. Run Triton Server

``` bash
docker run --rm --gpus all \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v $(pwd)/triton/models:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver \
    --model-repository=/models \
    --http-port=8000 \
    --grpc-port=8001 \
    --metrics-port=8002 \
    --log-verbose=1
```

#### Exposed Ports

| Port | Purpose |
| -------- | -------- |
| 8000 | HTTP inference |
| 8001 | gRPC inference |
| 8002 | Prometheus metrics |

## 4. Verify Triton Is Running

Check readiness:

```bash
curl http://localhost:8000/v2/health/ready
```

Expected response:

```json
{"ready":true}
```

List loaded models:

```bash
curl http://localhost:8000/v2/models
```

## 6. Dynamic Batching

Dynamic batching is configured inside ```config.pbtxt```:

``` protobuf
dynamic_batching {
  preferred_batch_size: [4, 8, 16]
  max_queue_delay_microseconds: 5000
}
```

### Notes

- Dynamic batching only activates under concurrent load

- Single-request testing will not trigger batching

- Monitor batching behavior via Triton metrics

## 7. Monitoring & Metrics

### Prometheus Metrics Endpoint

```http://localhost:8002/metrics```

#### Includes:

- Request count

- Queue latency

- Inference latency

- Batch sizes

- GPU utilization

## 8. Logs & Debugging

View Triton logs:

```bash
docker logs -f <triton_container_id>
```

Increase verbosity:

```bash
--log-verbose=1
```

## 9. Common Issues

### Model Fails to Load

- Input/output names mismatch

- Incorrect dtype (FP16 vs BF16)

### Model is not TorchScript

- Dynamic Batching Not Triggering

- Concurrency too low

- Queue delay too small

- Check batching metrics

### CUDA / Driver Mismatch

- Prefer Docker to avoid environment issues

- Ensure driver ≥ CUDA required by Triton

## 10. Recommended Workflow

1. Export model → TorchScript (model.pt)

2. Validate model locally (PyTorch)

3. Load into Triton

4. Test via gRPC client

5. Benchmark with concurrency

6. Tune dynamic batching

7. Monitor metrics

## 11. Why Triton?

- High-performance GPU inference

- Dynamic batching

- Production observability

- Multi-framework support

- Scales from local to production