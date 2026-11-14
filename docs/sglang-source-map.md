# mini-sglang → SGLang source map

Every module in this repo, the SGLang code that does the same job, what SGLang adds,
and where to read next. Prerequisite: the [Part 0–II study guide](study-guide/README.md); next: the
[Part III/IV notes](README.md).

- **Pinned SGLang commit:** [`c5d7c598049655046ccbdb8ae8d95b8e3aa3d2b3`](https://github.com/sgl-project/sglang/tree/c5d7c598049655046ccbdb8ae8d95b8e3aa3d2b3) (main, 2026-11-13).
- **Notation:** *srt/x/y.py::Name* means `python/sglang/srt/x/y.py`, class or function `Name`;
  *Class.method* names a method; *kernels/…* means `python/sglang/kernels/…`. Paths in the
  mini-sglang column are relative to `python/minisgl/`.
- **Verification:** *python docs/tools/verify_source_map.py /path/to/sglang* checks that
  every path and symbol in this file exists at the pinned commit (see the last section).
- SGLang moves fast. Since the course's reference commit (`1d019ce`, `75878f2`, …), the
  scheduler has been split into `managers/scheduler_components/`, and the radix cache is
  converging on `UnifiedRadixCache`. If a link breaks, search for the class name.

## 0. Big picture: same skeleton, more of everything

| Concern | mini-sglang (this repo) | SGLang (c5d7c59) |
|---|---|---|
| Processes | API server + N tokenizers + 1 detokenizer + 1 scheduler per TP rank (`server/launch.py`) | Same shape: `TokenizerManager` (in the HTTP process), `DetokenizerManager`, one `Scheduler` per (TP, PP, DP) rank, plus an optional `DataParallelController` |
| Loop | `normal_loop`: recv → schedule → forward → process | `Scheduler.event_loop_normal`, `event_loop_overlap` (CPU work of step *i+1* overlaps GPU step *i*), `event_loop_pp` |
| Batch | `Batch` (one phase only) | `ScheduleBatch` (scheduler, CPU) → `ForwardBatch` (model runner, GPU tensors); `ForwardMode` = EXTEND / DECODE / MIXED / IDLE / TARGET_VERIFY / … |
| KV | `MHAKVCache` + `CacheManager.free_slots` + page table | `MHATokenToKVPool` / `MLATokenToKVPool` / hybrid pools + allocator classes + `ReqToTokenPool` |
| Prefix cache | `RadixPrefixCache` (LRU, page-aligned) | `RadixCache`, `UnifiedRadixCache` (Full/SWA/Mamba components, HiCache L2/L3, pluggable eviction) |
| Attention | one pure-PyTorch backend | ~50 backends behind `AttentionBackend` (`init_forward_metadata` / `forward_extend` / `forward_decode`), plus CUDA-graph hooks |
| Execution | eager PyTorch | CUDA graphs for decode (and optionally prefill), torch.compile, custom kernels (`python/sglang/kernels`) |

## 1. Request path and processes

| mini-sglang | SGLang | What SGLang does beyond ours | Read next |
|---|---|---|---|
| `server/api_server.py` (`/v1/chat/completions`, `/v1/completions`, SSE) | `srt/entrypoints/http_server.py::openai_v1_chat_completions`, `srt/entrypoints/openai/serving_chat.py::OpenAIServingChat` | Tool calling, reasoning parsers, logprobs, n>1, structured output, `/v1/responses`, Anthropic/Ollama APIs, `/generate` native API, auth, metrics | `serving_chat.py` → `TokenizerManager.generate_request` |
| `server/launch.py::launch_server`, `start_backend` | `srt/entrypoints/http_server.py::launch_server`, `srt/entrypoints/engine.py::Engine`, `srt/managers/scheduler.py::run_scheduler_process` | Offline `Engine` with the same subprocess launch; DP controller; ports and health checks; multi-node bootstrap | `engine.py::Engine.__init__` |
| `llm/llm.py::LLM` (offline API, in-process `OfflineIO`) | `srt/entrypoints/engine.py::Engine.generate` | The SGLang `Engine` still uses subprocesses and ZMQ. Ours runs the scheduler in-process. | `Engine.generate` |
| `message/__init__.py` (`UserMsg`, `DetokenizeMsg`, pickle) | `srt/managers/io_struct.py` (`TokenizedGenerateReqInput`, `BatchTokenIDOutput`, …) | msgspec serialization (the contribution guide forbids `pickle.loads` on network data), batched outputs | `io_struct.py` |
| `scheduler/io.py::ZmqSchedulerIO` (rank 0 recv + Gloo broadcast) | `srt/managers/scheduler_components/request_receiver.py::SchedulerRequestReceiver` | Same "rank 0 receives, broadcast to TP" idea, plus DP-attention routing and a recv skipper | `request_receiver.py` |

## 2. Tokenizer and detokenizer

| mini-sglang | SGLang | Beyond ours | Read next |
|---|---|---|---|
| `tokenizer/tokenize.py::TokenizeManager`, `tokenizer/server.py::tokenize_worker` | `srt/managers/tokenizer_manager.py::TokenizerManager` (`generate_request`, `_tokenize_one_request`, `handle_loop`), `srt/managers/multi_tokenizer_mixin.py::MultiTokenizerRouter` | Async tokenization, multimodal processors, request state (`ReqState`), abort/timeout, per-request metrics, multiple tokenizer workers | `TokenizerManager._handle_batch_output` |
| `tokenizer/detokenize.py::DetokenizeManager` (incremental decode with read/surr offsets) | `srt/managers/detokenizer_manager.py::DetokenizerManager.handle_batch_token_id_out`, `srt/managers/incremental_detokenize.py::DetokenizeState` | Same surrogate-offset algorithm, batched across requests, `find_printable_text` for CJK, stop strings | `incremental_detokenize.py` |

## 3. Scheduler loop, prefill admission, decode, chunked prefill

| mini-sglang | SGLang | Beyond ours | Read next |
|---|---|---|---|
| `scheduler/scheduler.py` — `normal_loop` | `srt/managers/scheduler.py::Scheduler.event_loop_normal`, `Scheduler.event_loop_overlap` | **Overlap scheduling.** The CPU prepares batch *i+1* and processes the results of *i−1* while the GPU runs *i*. A `FutureMap` holds next-token placeholders. Our profile shows why it matters: see [profiling walkthrough](part4/profiling-walkthrough.md) | `event_loop_overlap`, `srt/managers/overlap_utils.py::FutureMap` |
| `scheduler/scheduler.py` — `_schedule_next_batch` (prefill first, else decode) | `Scheduler.get_next_batch_to_run`, `Scheduler.get_new_batch_prefill`, `Scheduler.update_running_batch` | Merges the last prefill batch into the running batch. Supports **mixed** prefill+decode batches (`--enable-mixed-chunk`), decode retraction when memory runs out, and `new_token_ratio` estimates instead of our worst-case reservation | `get_next_batch_to_run` |
| `scheduler/prefill.py::PrefillAdder` (FCFS, token budget, KV reservation) | `srt/managers/schedule_policy.py::PrefillAdder.add_one_req`, `PrefillAdder.add_chunked_req`, `PrefillAdder.rem_total_tokens` | Same three limits (slots, budget, KV), plus host-cache load-back, Mamba/SWA state slots, and LoRA. Uses an estimated `new_token_ratio` rather than `max_tokens` | `PrefillAdder.add_one_req` |
| FCFS ordering | `srt/managers/schedule_policy.py::SchedulePolicy.calc_priority` (`CacheAwarePolicy`: LPM, DFS-weight; `CacheAgnosticPolicy`: FCFS, LOF, random) | **Longest-prefix-match** ordering, and in-batch prefix dedup (`IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD`, a simulated `waiting_queue_radix_tree`). This fixes the duplicate prefix compute our profile found | `SchedulePolicy.calc_priority` |
| `core.py::ChunkedReq`, chunked prefill in `scheduler/prefill.py` (`_add_one_req`) | `Scheduler.stash_chunked_request`, `Scheduler.chunked_req` (property over `chunked_reqs`), `srt/managers/scheduler_components/dynamic_chunk_sizer.py::DynamicChunkSizer` | One chunked request at a time, a stashed partial prefix in the radix tree, and dynamic chunk sizes | `stash_chunked_request` |
| `scheduler/decode.py::DecodeManager` | `srt/managers/schedule_batch.py::ScheduleBatch.prepare_for_decode`, `ScheduleBatch.check_decode_mem`, `ScheduleBatch.retract_decode`, `srt/managers/scheduler_components/new_token_ratio_tracker.py::NewTokenRatioTracker` | **Retraction:** when decode runs out of KV, requests are evicted back to the waiting queue. Ours reserves the worst case up front, so it never needs to retract but under-admits | `retract_decode` |
| `scheduler/scheduler.py` — `_process_last_data` | `Scheduler.process_batch_result`, `srt/managers/scheduler_components/batch_result_processor.py::SchedulerBatchResultProcessor.process_batch_result_decode` | Stop strings and token ids, logprobs, grammar state, streaming intervals, metrics | `process_batch_result_prefill` |
| `scheduler/table.py::TableManager` (`token_pool`, `page_table`) | `srt/mem_cache/memory_pool.py::ReqToTokenPool` | Ours stores token ids; SGLang's `req_to_token` stores KV slot indices (our `page_table`) | `ReqToTokenPool.alloc` |
| abort (`AbortBackendMsg`) | `Scheduler.abort_request` | Abort in the waiting queue, running batch, chunked req, or PD queues; DP-attention routing (see issue #50137) | `abort_request` |

## 4. Radix cache, KV pool, allocator

| mini-sglang | SGLang | Beyond ours | Read next |
|---|---|---|---|
| `kvcache/radix_cache.py::RadixPrefixCache` (`match_prefix`, `insert_prefix`, `evict`, `lock_handle`) | `srt/mem_cache/radix_cache.py::RadixCache` (`match_prefix`, `insert`, `evict`, `inc_lock_ref`, `dec_lock_ref`, `_split_node`), `RadixKey`, `TreeNode` | Extra-key namespaces (LoRA, cache salt), KV events for routers, pluggable eviction (`srt/mem_cache/evict_policy.py::LRUStrategy`, `LFUStrategy`, `SLRUStrategy`, …) | `test/registered/unit/mem_cache/test_radix_cache_unit.py` (read the tests first) |
| (none) | `srt/mem_cache/unified_radix_cache.py::UnifiedRadixCache` | One tree for full-attention KV, SWA KV and Mamba state; hosts HiCache (`init_hicache`, `load_back`, `write_backup_storage`, `prefetch_from_storage`) | `srt/mem_cache/README.md`, then `unified_cache/components/README.md` |
| `scheduler/cache.py::CacheManager.match_req` / `cache_req` | `srt/mem_cache/common.py::match_kv_cache`, `checkpoint_kv_cache` (= our `cache_req(finished=False)`), `release_kv_cache` (= `cache_req(finished=True)`) | Same lifecycle: match → lock → allocate → insert → unlock → evict. Older SGLang versions (and the course's reference commit) call these `cache_unfinished_req` / `cache_finished_req` | `srt/mem_cache/common.py` |
| `kvcache/mha_pool.py::MHAKVCache` (`[2, L, pages, page, H, D]`, `store_kv`) | `srt/mem_cache/memory_pool.py::MHATokenToKVPool` (`set_kv_buffer`, `get_key_buffer`), `MLATokenToKVPool`, `MHATokenToKVPoolFP4` | FP8/FP4 KV with scales, MLA latent layout, SWA and hybrid-linear pools, layer-split buffers for PD transfer | `MHATokenToKVPool.set_kv_buffer` |
| `CacheManager.free_slots`, `allocate_paged`, `_free`, `lazy_free_region` | `srt/mem_cache/allocator/token.py::TokenToKVPoolAllocator`, `srt/mem_cache/allocator/paged.py::PagedTokenToKVPoolAllocator` (`alloc_extend`, `alloc_decode`, `free`) | GPU-side Triton kernels compute page allocation for a whole batch, and `free_group_begin/end` batches frees (our `lazy_free_region` is the same idea) | `PagedTokenToKVPoolAllocator.alloc_extend` |
| capacity from `max_total_tokens` | `srt/model_executor/pool_configurator.py`, `srt/mem_cache/kv_cache_configurator.py` | Profiles free GPU memory after weights and graphs, then sets `max_total_num_tokens` from `--mem-fraction-static` (Part III Ch2 §3.3) | `pool_configurator.py` |

## 5. Attention backend

| mini-sglang | SGLang | Beyond ours | Read next |
|---|---|---|---|
| `attention/base.py::BaseAttnBackend` (`prepare_metadata`, `forward`) | `srt/layers/attention/base_attn_backend.py::AttentionBackend` (`init_forward_metadata`, `forward`, `forward_extend`, `forward_decode`, CUDA-graph `init_forward_metadata_capture_cuda_graph` / `replay`) | Separate extend/decode entry points; CUDA-graph-safe metadata; `--prefill-attention-backend` / `--decode-attention-backend` | `base_attn_backend.py` |
| `attention/torch_backend.py::TorchAttnBackend` (gather KV via page table, padded SDPA) | `srt/layers/attention/torch_native_backend.py::TorchNativeAttnBackend` | The closest relative. It also gathers per request through `req_to_token`, loops over sequences and calls SDPA. Use it to diff your understanding | `TorchNativeAttnBackend.forward_decode` |
| (no real kernels) | `srt/layers/attention/flashinfer_backend.py::FlashInferAttnBackend`, `triton_backend.py::TritonAttnBackend`, `flashattention_backend.py::FlashAttentionBackend`, `flashmla_backend.py::FlashMLABackend`, `trtllm_mha_backend.py` | Paged KV read in place (no gather), plan/run split, split-KV decode with `merge_state` (`srt/layers/attention/merge_state.py::merge_state`). `labs/part3_ch01_attention_tiling.py` reproduces the math | `triton_backend.py::TritonAttnBackend.forward_decode` (readable), then `kernels/` |
| backend chosen in `Engine.__init__` | `srt/layers/attention/attention_registry.py::register_attention_backend`, `ATTENTION_BACKENDS`; names in `srt/arg_groups/choices.py` | Registry keyed by `--attention-backend` | `attention_registry.py` |
| per-layer call in `models/utils.py::RopeAttn` | `srt/layers/radix_attention.py::RadixAttention.forward` | The model-side layer that hands q/k/v and `layer_id` to `forward_batch.attn_backend` | `radix_attention.py` |

## 6. Engine, forward batch, sampler

| mini-sglang | SGLang | Beyond ours | Read next |
|---|---|---|---|
| `engine/engine.py::Engine.forward_batch` | `srt/managers/tp_worker.py::TpModelWorker.forward_batch_generation`, `srt/model_executor/model_runner.py::ModelRunner.forward`, `ModelRunner.sample` | CUDA graph replay (`srt/model_executor/runner/decode_cuda_graph_runner.py::DecodeCudaGraphRunner`), DP padding, PP proxy tensors, LoRA, hooks | `ModelRunner.forward` → `_forward_raw` |
| `core.py::Batch` fields (`input_ids`, `positions`, `out_loc`, `attn_metadata`) | `srt/model_executor/forward_batch_info.py::ForwardBatch`, `srt/model_executor/forward_mode.py::ForwardMode` | `out_cache_loc` = our `out_loc`; `seq_lens`, `extend_prefix_lens`, spec info, multimodal inputs | `ForwardBatch.init_new` |
| `core.py::Context` (global ctx) | `srt/model_executor/forward_context.py` | | |
| `engine/sample.py::Sampler`, `sample_impl` (temperature, top-k, top-p) | `srt/layers/sampler.py::Sampler.forward`, `srt/sampling/sampling_batch_info.py::SamplingBatchInfo`, `srt/sampling/sampling_params.py::SamplingParams` | FlashInfer sampling kernels, min-p, penalties (`srt/sampling/penaltylib`), logit bias, custom logit processors, grammar masks, deterministic sampling | `Sampler.forward` |
| LM head + `all_gather_last_dim` | `srt/layers/logits_processor.py::LogitsProcessor.forward`, `LogitsProcessor._get_logits` | all-gather of vocab shards (issue #3365 asks for `gather`), logprob extraction, last-token pruning, DP-attention gather | `_get_logits` |

## 7. TP layers, communication, weight loading

| mini-sglang | SGLang | Beyond ours | Read next |
|---|---|---|---|
| `layers/linear.py::ColumnParallelLinear`, `QKVParallelLinear`, `RowParallelLinear` | `srt/layers/linear.py::ColumnParallelLinear`, `MergedColumnParallelLinear`, `QKVParallelLinear`, `RowParallelLinear` (`weight_loader`) | Each parameter carries a `weight_loader` that shards on load (ours shards in `models/weight.py`). Quantized `quant_method` hooks, fused all-reduce + RMSNorm | `QKVParallelLinear.weight_loader` |
| `layers/embedding.py::VocabParallelEmbedding`, `ParallelLMHead` | `srt/layers/vocab_parallel_embedding.py::VocabParallelEmbedding`, `ParallelLMHead` | Padding to multiples of 64, LoRA vocab extension | |
| `distributed/impl.py::DistributedCommunicator` | `srt/distributed/parallel_state.py::GroupCoordinator`, `initialize_model_parallel`, `get_tp_group`; `srt/distributed/communication_op.py::tensor_model_parallel_all_reduce` | Custom all-reduce kernels, PyNccl, symmetric memory, TP/PP/EP/DP-attn/CP groups | `GroupCoordinator.all_reduce` |
| `models/weight.py::load_weight` (shard first, then merge) | `srt/model_loader/loader.py::DefaultModelLoader`, `srt/model_loader/weight_utils.py::safetensors_weights_iterator`, `default_weight_loader`; per-model `srt/models/qwen3.py::Qwen3ForCausalLM.load_weights` (`stacked_params_mapping`) | Many formats (GGUF, bitsandbytes, sharded state, remote instance, ModelOpt), quantized checkpoints, prefetch | `Qwen3ForCausalLM.load_weights` |
| `models/qwen3.py::Qwen3ForCausalLM`, `models/utils.py::RopeAttn`, `GatedMLP` | `srt/models/qwen3.py::Qwen3ForCausalLM`, `Qwen3Attention`, `Qwen3DecoderLayer`, `Qwen3Model` | Same structure (q/k norm, RoPE, GQA); adds PP layer ranges, EAGLE3 aux hidden states, dual-chunk attention | diff it against ours |
| `layers/rotary.py`, `layers/norm.py::RMSNorm` | `srt/layers/rotary_embedding/base.py::RotaryEmbedding`, `srt/layers/layernorm.py::RMSNorm` | Fused kernels, many RoPE scaling variants | |

## 8. Speculative decoding

| mini-sglang | SGLang | Beyond ours | Read next |
|---|---|---|---|
| `speculative/decoder.py::SpeculativeDecoder` (draft → verify one sequence) | `srt/speculative/eagle_worker_v2.py::EAGLEWorkerV2` (`draft`, `verify`, `forward_batch_generation`), `srt/speculative/standalone_worker_v2.py` | Batched, tree drafts (top-k branching), draft model shares the target's KV pool layout, CUDA graphs for draft and verify, overlap scheduler support | `EAGLEWorkerV2.forward_batch_generation` |
| `speculative/worker.py::SpecWorker` (batched rounds in the scheduler, `Batch(phase="verify")`) | `srt/speculative/ngram_worker.py::NGRAMWorker`, `srt/speculative/standalone_worker_v2.py::StandaloneWorkerV2`, `ForwardMode.TARGET_VERIFY` | Tree drafts + custom masks, CUDA-graph verify, overlap via FutureMap; see `docs/speculative-in-scheduler.md` | `ngram_worker.py` |
| `speculative/drafter.py::NgramDrafter` | `srt/speculative/ngram_worker.py::NGRAMWorker._prepare_draft_tokens`, `srt/speculative/cpp_ngram/ngram_corpus.py::NgramCorpus.batch_get` | A C++ trie corpus shared across requests (the root cause of issue #50395) | `ngram_worker.py` |
| `speculative/verify.py::greedy_commit`, `rejection_commit` | `kernels/ops/speculative/tree.py::verify_tree_greedy`, `kernels/ops/speculative/sampling.py::tree_speculative_sampling_target_only`, `srt/speculative/eagle_info.py::EagleVerifyInput` | Tree verification on GPU with a tree attention mask (`srt/layers/attention/verify_mask.py`) | `eagle_info.py` |

## 9. Part III / IV topics with no counterpart here

| Topic (course chapter) | SGLang entry points |
|---|---|
| Quantization (III-2) | `srt/layers/quantization/__init__.py::get_quantization_config` (`BASE_QUANTIZATION_METHODS`), `srt/layers/quantization/fp8.py`, `srt/layers/quantization/awq`, `srt/layers/quantization/modelopt_quant.py`, `srt/layers/quantization/kv_cache.py`; dev guide `docs/docs/developer_guide/quantization_contribution_guide.mdx` |
| HiCache (III-3) | `srt/managers/cache_controller.py::HiCacheController` (`write`, `load`, `prefetch`, `LayerDoneCounter`), `srt/mem_cache/pool_host/mha.py::MHATokenToKVPoolHost`, `srt/mem_cache/hicache_storage.py::HiCacheStorage`, `srt/mem_cache/storage/` |
| DP attention, EP, PP (III-4) | `srt/layers/dp_attention.py`, `srt/managers/data_parallel_controller.py::DataParallelController`, `srt/layers/moe/fused_moe_triton/layer.py::FusedMoE`, `srt/layers/moe/ep_moe/layer.py::DeepEPMoE`, `srt/layers/moe/token_dispatcher/deepep.py`, `srt/layers/moe/topk.py::TopK`, `srt/eplb/`, `srt/managers/scheduler_pp_mixin.py::SchedulerPPMixin.event_loop_pp` |
| PD disaggregation (III-5) | `srt/disaggregation/prefill.py::SchedulerDisaggregationPrefillMixin`, `PrefillBootstrapQueue`; `srt/disaggregation/decode.py::DecodePreallocQueue`, `DecodeTransferQueue`; `srt/disaggregation/base/conn.py::BaseKVSender`, `BaseKVReceiver`, `KVPoll`; transports in `srt/disaggregation/mooncake/conn.py`, `nixl/` |
| Profiling (IV-2) | `srt/managers/scheduler_components/profiler_manager.py::SchedulerProfilerManager`; `/start_profile` and `/stop_profile`; `SGLANG_TORCH_PROFILER_DIR`; `docs/docs/developer_guide/benchmark_and_profiling.mdx` |
| Contribution (IV-3) | `docs/docs/developer_guide/contribution_guide.mdx`, `.github/pull_request_template.md`, `.pre-commit-config.yaml`, `test/registered/unit/README.md`, `.github/MAINTAINER.md`, `.github/CODEOWNERS` |

## Suggested reading order (≈ one evening each)

1. `srt/mem_cache/radix_cache.py` together with its unit test. This is closest to our Chapter 8, and reading the test first works well.
2. `srt/managers/schedule_policy.py::PrefillAdder`, then `Scheduler.get_next_batch_to_run`. Compare with our `scheduler/prefill.py`.
3. `srt/layers/attention/torch_native_backend.py`, then `triton_backend.py`.
4. `srt/managers/scheduler.py::Scheduler.event_loop_overlap` and `overlap_utils.py`.
5. `srt/models/qwen3.py` side by side with ours, then `srt/layers/linear.py` weight loaders.
6. `srt/speculative/eagle_worker_v2.py` once 1–5 are comfortable.

## Verification

[`verify_source_map.py`](tools/verify_source_map.py) extracts every *srt/…py::Symbol* and plain
path from this file. It checks that each file exists and that each symbol is defined
there (`class X` / `def X` / `X =`, method names inside the file). Output at the
pinned commit:

```
sglang commit c5d7c598049655046ccbdb8ae8d95b8e3aa3d2b3: 138 citations OK, 0 missing
```
