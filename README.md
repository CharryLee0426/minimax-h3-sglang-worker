# minimax-h3-sglang-worker

[Runpod Serverless](https://docs.runpod.io/serverless/overview) worker that serves
[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) **Reference-to-Audio-Video (Ref2VA)**
through [SGLang](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3), with an optional LoRA.

One image serves many endpoints; each endpoint is configured with environment variables.

> MiniMax-H3 is released under the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE),
> including its Acceptable Use Policy and a territory restriction (the EU, UK, South Korea and the US are
> excluded from the grant). You are responsible for complying with it. This repository contains only glue code (MIT).

## What the worker does

1. On start it downloads (once, to the network volume) the `Ref2VA/` partition of the model and the optional LoRA.
2. It launches `sglang serve --model-variant ref2va` across all GPUs of the worker (`--ulysses-degree = #GPUs`; 1, 2, 4 or 8).
3. For each job it submits a `ref2va` request, waits for the MP4, uploads it to a presigned `PUT` URL and returns metadata.
   Progress is reported through Runpod's `progress_update` (`loading_model`, `submitting`, `generating`, `uploading`),
   at most every 15 s per stage.
4. Cancelling a job in Runpod stops the SGLang process and exits the worker, because SGLang cannot abort a generation;
   Runpod starts a fresh worker for the next job (a cold start). A generation that exceeds `H3_JOB_TIMEOUT_S` (default 3000 s)
   also recycles the worker.

## Job input

```json
{
  "input": {
    "prompt": "<Picture 1> walks through a rainy neon street, camera follows.",
    "references": [{ "url": "https://.../ref-1.png" }],
    "duration_seconds": 5,
    "aspect_ratio": "16:9",
    "seed": 1234,
    "num_inference_steps": 50,
    "flow_shift": 12,
    "audio_flow_shift": 3,
    "output": { "put_url": "https://.../video.mp4?signature", "headers": {}, "key": "videos/abc.mp4" }
  }
}
```

* `references`: 1-9 HTTPS image URLs. Prompts should address them as `<Picture 1>`, `<Picture 2>`, ...; if no tag is present
  the worker prepends one.
* `duration_seconds`: 4-15. `aspect_ratio`: `auto`, `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, `9:16`. Output is 768p, 24 fps, stereo audio.
* Output: `{ "video_key", "size_bytes", "resolution", "generation_seconds", "total_seconds", "seed", "effective_prompt" }`.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `H3_MODEL_DIR` | `/runpod-volume/models/MiniMax-H3` | Where the `Ref2VA/` weights live (network volume) |
| `H3_LORA_REPO`, `H3_LORA_FILE` | unset | Hugging Face repo and safetensors file of an optional LoRA |
| `H3_LORA_SCALE` | `1.0` | LoRA strength |

Kohya-style H3 LoRAs (`lora_unet_blocks_*` keys) are converted on start to the native key names SGLang v0.5.21 matches
(`<file>.native.safetensors` next to the original). If a LoRA still applies to 0 layers the worker refuses to serve, so base-model
output is never returned as LoRA output.

| Variable | Default | Meaning |
| --- | --- | --- |
| `H3_LORA_MERGE_MODE` | `auto` | `auto`, `merge` or `dynamic` (SGLang `--lora-merge-mode`) |
| `H3_NUM_GPUS` | autodetect | 1, 2, 4 or 8 |
| `H3_ALLOWED_URL_SUFFIXES` | unset | Comma list (for example `.blob.vercel-storage.com`) restricting reference and upload hosts |
| `H3_IDLE_EXIT_S` | `150` | Exit the worker after this many seconds ready and idle (0 disables), so GPUs are released even if the platform leaves idle workers running |
| `H3_EXTRA_ARGS` | unset | Extra `sglang serve` flags |
| `H3_AUTO_DOWNLOAD` | `1` | Download missing weights on start; set `0` when the volume is pre-populated |
| `HF_TOKEN` | unset | Only needed for gated/private repos |

Pre-populate the volume on a cheap CPU pod instead of paying for GPU time:
`pip install huggingface_hub && python -m h3worker.bootstrap` with the same env vars.

## Development

```bash
pip install -r requirements-dev.txt
PYTHONPATH=. pytest -q
```

Tests run the handler against a fake SGLang server; real GPU behaviour is not covered.
