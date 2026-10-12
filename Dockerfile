# SGLang image with CUDA 13 (required by B200 hosts) and MiniMax-H3 diffusion support.
FROM lmsysorg/sglang:v0.5.21-cu130

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/huggingface \
    HF_XET_HIGH_PERFORMANCE=1 \
    H3_MODEL_DIR=/runpod-volume/models/MiniMax-H3 \
    H3_LORA_DIR=/runpod-volume/loras

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY h3worker ./h3worker

ENTRYPOINT []
CMD ["python3", "-m", "h3worker.handler"]
