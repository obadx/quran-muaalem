FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch \
    transformers \
    accelerate \
    safetensors \
    quran-muaalem 

COPY /engine/triton/inference_stack/client/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY engine/triton/inference_stack/client/app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
