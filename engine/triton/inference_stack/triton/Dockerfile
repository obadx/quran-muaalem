FROM nvcr.io/nvidia/tritonserver:24.01-py3

# -----------------------------
# Python dependencies
# -----------------------------
RUN pip install --no-cache-dir \
    torch \
    transformers \
    accelerate \
    safetensors \
    quran-muaalem 


# -----------------------------
# Recreate repo structure
# -----------------------------
WORKDIR /workspace/engine

COPY engine/ engine/
COPY engine/tensorrt/trt_runtime.py /workspace/engine/
COPY engine/export_model_pt.py /workspace/engine/

COPY engine/triton/inference_stack/triton/models /models

# Make repo-root importable
ENV PYTHONPATH=/workspace/engine

# -----------------------------
# Export model at build time
# -----------------------------
WORKDIR /workspace/engine

RUN python3 export_model_pt.py \
    --model-id obadx/muaalem-model-v3_2 \
    --output /models/muaalem/1/model.pt

# -----------------------------
# Run Triton
# -----------------------------
CMD ["tritonserver", \
     "--model-repository=/models", \
     "--grpc-port=8001", \
     "--http-port=8000", \
     "--metrics-port=8002"]
