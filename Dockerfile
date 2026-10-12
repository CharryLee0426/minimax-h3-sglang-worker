# SGLang image with CUDA 13 (required by B200 hosts) and MiniMax-H3 diffusion support.
FROM lmsysorg/sglang:v0.5.21-cu130

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/huggingface \
    HF_XET_HIGH_PERFORMANCE=1 \
    H3_MODEL_DIR=/runpod-volume/models/MiniMax-H3 \
    H3_LORA_DIR=/runpod-volume/loras

WORKDIR /app
COPY requirements.txt .
# Hold every package already in the image at its installed version so the Runpod SDK cannot replace
# SGLang's pinned dependencies. If that is impossible, fall back to a plain install and let the
# smoke test below decide whether the image still works.
RUN pip list --format=freeze > /tmp/constraints.txt \
 && (pip install --no-cache-dir -c /tmp/constraints.txt -r requirements.txt \
     || (echo "constrained install failed; installing without constraints" \
         && pip install --no-cache-dir -r requirements.txt)) \
 && rm /tmp/constraints.txt
COPY h3worker ./h3worker
RUN python3 -c "import runpod, h3worker.handler" && sglang serve --help > /dev/null

ENTRYPOINT []
CMD ["python3", "-m", "h3worker.handler"]
