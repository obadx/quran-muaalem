#!/usr/bin/env bash
set -euo pipefail

echo "=== DEBUG: repo before export ==="
find /models -maxdepth 3 -print

mkdir -p /models/muaalem/1

echo "=== Exporting to /models/muaalem/1/model.pt ==="
python3 /workspace/engine/export_model_pt.py \
  --model-id obadx/muaalem-model-v3_2 \
  --output /models/muaalem/1/model.pt \
  --device cuda \
  --dtype fp16 \
  --example-seconds 3

echo "=== DEBUG: repo after export ==="
ls -lah /models/muaalem
ls -lah /models/muaalem/1

# hard fail if missing
test -f /models/muaalem/1/model.pt

exec tritonserver \
  --model-repository=/models \
  --grpc-port=8001 \
  --http-port=8000 \
  --metrics-port=8002 \
  --exit-on-error=false \
  --log-verbose=1