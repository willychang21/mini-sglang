# Part 0 · Lesson 0.2 — Deploy Your First SGLang Server

> Tutorial: `part0/Part0-Deploy-Your-First-SGLang-Server.md` · Lab: none (GPU lesson; a CPU
> equivalent with mini-sglang is run below) · Related code: `python/minisgl/server/args.py`,
> `server/launch.py`, `server/api_server.py`, `README.md` (quick start)

## Key ideas

The lesson is a recipe: install SGLang, start a server for Qwen3-0.6B, send it an
OpenAI-style request. Each step touches something the rest of the course explains.

| Step | What it actually does | Explained later in |
|---|---|---|
| `nvidia-smi` | Checks that the driver sees the GPU. The "CUDA Version" in the corner is the **newest CUDA runtime the driver supports**. It is not an installed toolkit. It decides which wheels you can use. | I.3 (GPU) |
| conda / venv | Gives SGLang its own environment, so its torch/kernel versions do not clash with other projects. | — |
| `uv pip install --prerelease=allow sglang` | Installs SGLang plus its compiled kernels and a CUDA runtime bundled inside the wheels. That is why no CUDA Toolkit is needed. The default wheels target CUDA 13. | — |
| CUDA 12.x branch | Reinstalls `torch`, `sglang-kernel` and `sgl-deep-gemm` built for cu129, so they run on an older driver. The other fix is a driver ≥ 580. | III.1 (attention backends / kernels) |
| `HF_ENDPOINT` / `SGLANG_USE_MODELSCOPE` | Changes where the weights are downloaded from. It only works in the shell that launches the server. | — |
| `python3 -m sglang.launch_server --model-path … --host 0.0.0.0 --port 30000` | Starts the whole serving stack: HTTP server, tokenizer, scheduler and model worker, and detokenizer. `--host 0.0.0.0` listens on every network interface. Use `127.0.0.1` for local-only access. | II.2 (request path), II.5 (HTTP) |
| "fired up and ready to roll" | Printed after the weights are loaded, the KV pool is allocated and warm-up has finished. | II.4/II.7 (KV pool) |
| `curl /health` | Liveness check. | II.5 |
| `POST /v1/chat/completions` | Uses the OpenAI-compatible API. The server applies the chat template, tokenizes, schedules, generates and detokenizes. | II.2, II.5 |
| `<think>…</think>` | Qwen3 reasons by default. `chat_template_kwargs: {"enable_thinking": false}` changes the **prompt** through the chat template. `--reasoning-parser qwen3` instead splits the reasoning into `reasoning_content` in the **response**. | II.2 (tokenizer / chat template) |

FAQ flags, read as resource decisions:

- `--mem-fraction-static 0.6`: the fraction of GPU memory SGLang takes up front for
  *static* allocations. That is the weights plus the KV-cache pool. The rest is left
  for activations, CUDA graphs and other processes. Lowering it shrinks the KV pool,
  which means fewer concurrent tokens (Part I Ch4, Part II Ch7).
- `--context-length 8192`: caps the maximum sequence length. This bounds the KV that
  one request can hold.
- `--attention-backend triton`: chooses another attention kernel. The default
  (FlashInfer/FA3 family) needs a newer GPU architecture than Turing (T4, RTX 20)
  (Part III Ch1).
- `--port 30001`: works around `address already in use`.

### mini-sglang equivalents

`python -m minisgl` follows the same design with simpler flags
(`python/minisgl/server/args.py:parse_args`):

| SGLang | mini-sglang | Note |
|---|---|---|
| `--model-path` | `--model` / `--model-path` (aliases) | local directory with `config.json` + safetensors |
| `--port 30000` | `--port 1919` (default) | `--host` defaults to `127.0.0.1` |
| `--mem-fraction-static` | `--max-total-tokens` (default 32768) | sets the KV pool size in **tokens** directly, not as a fraction of memory |
| `--context-length` | `--max-seq-len` (default 4096) | the effective limit is `min(max_seq_len, pool tokens, max_position_embeddings)` (`engine/engine.py:Engine.__init__`) |
| `--attention-backend` | — | a single pure-PyTorch backend (`attention/torch_backend.py:TorchAttnBackend`) |
| `--tp-size` | `--tp-size` | Part II Ch9 |
| `--disable-radix-cache` | `--cache-type naive` | Part II Ch8 |
| `--reasoning-parser` | — | not implemented; use `chat_template_kwargs` |

## Where in mini-sglang

| Concept | Code | Note |
|---|---|---|
| CLI flags | `python/minisgl/server/args.py:parse_args`, `ServerArgs.engine_config` | flags are turned into an `EngineConfig` |
| Process startup + "ready" | `python/minisgl/server/launch.py:launch_server`, `start_backend` | the API server waits for a ready ack from the scheduler, tokenizer(s) and detokenizer |
| `/health`, `/v1/models` | `python/minisgl/server/api_server.py:health`, `v1_models` | |
| OpenAI chat endpoint | `python/minisgl/server/api_server.py:v1_chat_completions` → `_handle` | request schema `OpenAICompletionRequest` (has `chat_template_kwargs`) |
| Chat template | `python/minisgl/tokenizer/tokenize.py:TokenizeManager.apply_chat_template` | `enable_thinking` is passed through here |
| KV pool allocated at startup | `python/minisgl/engine/engine.py:Engine.__init__` → `kvcache/__init__.py:create_kvcache_pool` | the size is logged as `kv_tokens=… (… MiB)` |

## Common misconceptions

**Misconception:** "CUDA Version 12.8" in `nvidia-smi` means CUDA 12.8 is installed.
**Actually:** it is the highest runtime version the *driver* supports. The runtime
SGLang uses comes inside its pip wheels. That is why the fix for "driver version is
insufficient" is either other wheels or a newer driver.

**Misconception:** `<think>` output means the server is broken.
**Actually:** it is Qwen3's default chat template. Turn it off per request with
`chat_template_kwargs`, or separate it on the server with `--reasoning-parser`.

**Misconception:** `--mem-fraction-static` limits only the weights.
**Actually:** it covers weights **and** the KV pool. Most of what is left after the
weights becomes KV cache. That is exactly why lowering it fixes OOM, and why it also
lowers the number of concurrent requests (Part I Ch4).

**Misconception:** `--host 0.0.0.0` is needed to reach the server from the same machine.
**Actually:** `127.0.0.1` is enough locally. `0.0.0.0` exposes an unauthenticated
endpoint to the network.

## Lab: GPU commands (Colab-ready) and the CPU equivalent

### NOT EXECUTED HERE (requires a GPU)

The commands below come from the tutorial. They are arranged for a Google Colab GPU
runtime: `!` runs a shell command, and the server runs in the background so the
notebook stays usable. **They have not been run in this environment** (no GPU).

```bash
# --- NOT EXECUTED HERE (requires a GPU) ---------------------------------------
# Colab: Runtime -> Change runtime type -> GPU. A free Colab GPU is usually a T4.
!nvidia-smi                                   # note "CUDA Version" (top right)

!pip install --upgrade pip uv
!uv pip install --system --prerelease=allow sglang
# Only if nvidia-smi shows CUDA Version 12.x (the tutorial's CUDA-12 branch):
!uv pip install --system --force-reinstall torch==2.13.0 torchaudio==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu129
!uv pip install --system --force-reinstall sglang-kernel --index-url https://docs.sglang.ai/whl/cu129/
!uv pip install --system --force-reinstall sgl-deep-gemm --index-url https://docs.sglang.ai/whl/cu129/ --no-deps
!python3 -c "import sglang; print(sglang.__version__)"

# Launch in the background. T4 is a Turing GPU, so the tutorial's FAQ applies:
# add --attention-backend triton. Lower memory use as in the OOM FAQ if needed.
!nohup python3 -m sglang.launch_server --model-path Qwen/Qwen3-0.6B \
    --host 127.0.0.1 --port 30000 --attention-backend triton \
    --mem-fraction-static 0.6 --context-length 8192 > sglang.log 2>&1 &
!for i in $(seq 1 120); do curl -s localhost:30000/health && break; sleep 5; done
!grep -m1 "fired up" sglang.log

!curl -s http://localhost:30000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Introduce yourself in one sentence."}], "max_tokens": 512}'
```

```python
# --- NOT EXECUTED HERE (requires a GPU) ---------------------------------------
# !uv pip install --system openai
import openai
client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")
r = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "List 3 countries and their capitals."}],
    max_tokens=512,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(r.choices[0].message.content)
```

Caveats for Colab (not verified here). Colab images come with their own torch, and
`uv pip install --system` replaces it. Restart the runtime if imports break. The T4
has no native bf16. If startup complains about bfloat16, try adding `--dtype float16`.

### Executed here: the same flow with mini-sglang on CPU

```bash
python -m minisgl --model /tmp/models/Qwen3-0.6B --device cpu --port 19191 --dist-port 29611 &
curl -s http://127.0.0.1:19191/health
curl -s http://127.0.0.1:19191/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Introduce yourself in one sentence."}],
       "max_tokens":40,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
kill %1
```

Output (4-core CPU, real Qwen3-0.6B, fp32). The server was ready about 2 s after the
tokenizers:

```text
[03:30:51 minisgl-TP0-scheduler INFO] Engine ready: Qwen3ForCausalLM layers=28 tp=0/1 device=cpu dtype=torch.float32 kv_tokens=32768 (7168.0 MiB)
[03:30:51 MainProcess INFO] API server is ready to serve on http://127.0.0.1:19191
{"status":"ok"}
{"id":"cmpl-0","object":"chat.completion","created":1794627054,"model":"Qwen3-0.6B","choices":[{"index":0,
 "message":{"role":"assistant","content":"I am a language model designed to assist with various tasks,
 including language learning and problem-solving."},"finish_reason":"stop"}],
 "usage":{"prompt_tokens":19,"completion_tokens":20,"total_tokens":39}}      # 1.7 s wall
```

Without `enable_thinking: false`, the same server starts with the reasoning block
(`"content":"<think>\nOkay, the user wants me to list three countries…"`,
`finish_reason: "length"` at 40 tokens). That is the behaviour the tutorial describes.

What to look for:

- `kv_tokens=32768 (7168.0 MiB)`: the KV pool is allocated at startup. Check it with
  the Part I Ch4 formula: 2 × 28 layers × 8 KV heads × 128 × 4 B (fp32 on CPU) =
  229,376 B/token, and × 32,768 tokens = 7.0 GiB. On a GPU, SGLang sizes this pool
  from `--mem-fraction-static` instead.
- `usage.prompt_tokens: 19` is far more than the 7 words typed: the chat template adds
  role markers and other special tokens.
- Streaming (`"stream": true`) returns SSE `data:` chunks of one token each. Part I
  Ch5's TTFT/ITL are measured from those chunks (`benchmarks/bench_serving.py`).

## Exercises

This lesson has no end-of-chapter exercises. Self-check:

1. *Which steps would change on a machine with CUDA Version 12.4?* Use the CUDA-12
   branch (cu129 wheels need a driver that supports 12.9, which some 12.x drivers do
   not), or upgrade the driver to ≥ 580 and use the default wheels.
2. *The server OOMs at startup on an 8 GB card that also drives a desktop. Which two
   flags, and why do they help?* `--mem-fraction-static` lowers the share of memory
   reserved for weights + KV pool. `--context-length` lowers the per-request maximum.
   Neither shrinks the 1.5 GB of weights. They shrink the KV pool and the activations.
3. *Can mini-sglang's server be queried with the same `openai` client code?* Yes, with
   `base_url="http://127.0.0.1:1919/v1"`. The `model` field is accepted but ignored.
   `extra_body={"chat_template_kwargs": …}` maps to `OpenAICompletionRequest.chat_template_kwargs`.
   (The `openai` package is not installed here, so this was checked with `curl`.)
