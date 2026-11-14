# Readiness self-check: Parts I–IV

45 questions. Try each one before opening the answer. The answers refer to this repo
(`python/minisgl/…`) and to SGLang at `c5d7c59` (see the [source map](sglang-source-map.md)).
A rough bar for "ready to make a first SGLang contribution": ≥35 correct, including all
questions marked ★ (they come up in almost every SGLang PR review).

## Part I — Foundations

**1.** Why does decoding without a KV cache cost O(n²) per generated token (over the whole generation, O(n³) attention work), and what does the KV cache change?
<details><summary>Answer</summary>
Without a cache, step t recomputes K/V for all t previous tokens through every layer and attention over t×t. With a cache, each step computes Q/K/V only for the new token, appends K/V, and attends 1×t. Per-step work becomes linear in context, and memory grows by 2·layers·kv_heads·head_dim·bytes per token.
</details>

**2.** ★ Compute the KV bytes per token for Qwen3-0.6B (28 layers, 8 KV heads, head_dim 128) in bf16. How many tokens fit in 10 GiB?
<details><summary>Answer</summary>
2·28·8·128·2 = 114,688 B = 112 KiB per token. 10 GiB / 112 KiB ≈ 93.6K tokens. `labs/part3/kv_capacity.py` checks this against `MHAKVCache.num_bytes`.
</details>

**3.** Why is prefill usually compute-bound and small-batch decode memory-bound? Give the roofline argument.
<details><summary>Answer</summary>
Arithmetic intensity is FLOPs per byte moved. Prefill multiplies each weight by L tokens (≈2L FLOPs per weight read), which is high intensity, so it is compute-bound. Decode at batch B multiplies each weight by B tokens. For small B the intensity is below the GPU's FLOP/byte ratio (~300 on H100), so time ≈ weight bytes / bandwidth. Our profile: 4× the decode batch costs only 1.2× the time.
</details>

**4.** Define TTFT, TPOT, ITL and throughput. Which one hides a single long stall, and which one reveals it?
<details><summary>Answer</summary>
TTFT: arrival → first token. TPOT: (last − first token time)/(n−1), the per-request average. ITL: each gap between consecutive tokens. Throughput: tokens/s across all requests. The TPOT average hides one long stall; ITL (its max or p99) reveals it.
</details>

**5.** Why must a benchmark comparison fix precision, quantization, input/output length distribution and concurrency?
<details><summary>Answer</summary>
Each of these changes the bottleneck and the work done. A "speedup" obtained by changing any of them is not attributable to the code change (Part IV Ch3 Ex.2).
</details>

**6.** What are HBM, SRAM and SMs, and why does FlashAttention care about the difference between HBM and SRAM?
<details><summary>Answer</summary>
HBM is large and relatively slow device memory. SRAM (shared memory/registers) is small and very fast on-chip storage per SM. SMs are the compute units. FlashAttention keeps tiles of Q/K/V and the running softmax state in SRAM, so the L×L score matrix never round-trips through HBM.
</details>

## Part II — Building mini-sglang

**7.** Name the processes of a mini-sglang server and the message flow for one chat request.
<details><summary>Answer</summary>
The API server (FastAPI) sends the text to a tokenizer worker (ZMQ). The tokenizer sends `UserMsg` with token ids to scheduler rank 0, which broadcasts to the other TP ranks over Gloo. Every rank runs the same batch. Rank 0 sends `DetokenizeMsg` per token to the detokenizer, which sends incremental text (`UserReply`) to the API server, which streams it over SSE (`server/launch.py`, `scheduler/io.py`, `tokenizer/server.py`).
</details>

**8.** ★ In `core.py::Req`, what do `cached_len`, `device_len` and `extend_len` mean, and how do they change after a forward pass?
<details><summary>Answer</summary>
`[0, cached_len)` already has K/V in the pool. `[cached_len, device_len)` is computed this step. `extend_len = device_len − cached_len`. After the forward, `complete_one()` sets `cached_len = device_len` and `device_len += 1` (the sampled token occupies the next position). In SGLang's `Req` (`srt/managers/schedule_batch.py`, c5d7c59) the same roles are `prefix_len` (= `cached_len`), `extend_end` (= `device_len`) and `extend_input_len = extend_end − prefix_len`.
</details>

**9.** Why must `store_kv` run *before* attention reads the history in the same layer?
<details><summary>Answer</summary>
The new tokens attend to themselves and to each other (causal within the chunk). Their K/V must already be in the pool slots that the attention gather or kernel reads, otherwise they would read stale data. The same holds in SGLang's backends (`set_kv_buffer` before the kernel).
</details>

**10.** Why does `CacheManager.match_req` never match the last prompt token?
<details><summary>Answer</summary>
Prefill must compute at least one position to produce logits for the first output token. If the whole prompt were matched, there would be nothing to run and no logits.
</details>

**11.** What is a `ChunkedReq`, and why must it never sample?
<details><summary>Answer</summary>
A request whose prompt is only partially prefilled this step (the prefill budget ran out). The logits at the end of an intermediate chunk predict the next *prompt* token, not an output token. Sampling would emit garbage and corrupt the sequence.
</details>

**12.** ★ `PrefillAdder` enforces three separate limits. Name them and the failure each one prevents.
<details><summary>Answer</summary>
Request slots (`table_manager.available_size`): running out of page-table rows. Token budget (`prefill_budget`): huge prefill steps that stall decode. KV capacity (`estimated_len + reserved_size ≤ available_size`, with `reserved_size` counting what running decodes may still need): running out of KV mid-generation. SGLang uses estimates (`new_token_ratio`) and handles overflow by retracting decodes instead of reserving the worst case.
</details>

**13.** Why does mini-sglang's `_try_allocate_one` (`scheduler/prefill.py`) check capacity twice, around `lock`?
<details><summary>Answer</summary>
Locking the matched prefix moves its tokens from "evictable" to "protected", which *shrinks* `available_size` (evictable cache counts as available). The request may fit before the lock but not after it, so it re-checks and unlocks on failure.
</details>

**14.** Continuous batching vs static batching: what changes, and what is the cost in mini-sglang's loop?
<details><summary>Answer</summary>
Requests join and leave between steps (`DecodeManager.filter_reqs`, `PrefillManager`), so finished slots are reused immediately and no request waits for the longest one in its batch. The cost: a scheduling decision per step, and with prefill priority, prefill steps interrupt running decodes (ITL spikes; profile Finding 4).
</details>

**15.** What does a page table entry store in mini-sglang, and why are pages useful even with `page_size=1`?
<details><summary>Answer</summary>
`page_table[row, pos]` holds the physical KV *slot* index of that token. Indirection means a request's KV need not be contiguous, so there is no fragmentation and no copying on growth, and prefixes can be shared by pointing several rows at the same slots. Larger pages reduce metadata and match kernel tiles, at the cost of internal fragmentation and page-aligned prefix matching.
</details>

**16.** ★ Describe the radix-cache lifecycle (match → lock → allocate → insert → unlock → evict) and why only unreferenced leaves are evicted.
<details><summary>Answer</summary>
Match finds the longest cached prefix. Lock increments `ref_count` along the path so it cannot be evicted while used. Pages are allocated for the uncached suffix. After prefill (or finish) the sequence is inserted (the duplicate part is freed, new nodes own the slots). The old handle is unlocked. Under pressure, LRU unreferenced leaves are evicted. Only leaves can go, because an inner node is a prefix of its children and removing it would orphan them. Referenced nodes are in use by running requests.
</details>

**17.** In `CacheManager.cache_req(finished=False)`, why does mini-sglang rewrite part of the running request's page-table row after insertion?
<details><summary>Answer</summary>
If another request inserted the same tokens meanwhile, insert returns `prefix_len > old.cached_len`, and our own duplicate slots for that range are freed. They can be reallocated, so the running request must point at the cache's copy of those tokens, or it would read someone else's future data.
</details>

**18.** TP: which projections are column-parallel and which are row-parallel in a Qwen3 layer, and how many all-reduces per layer?
<details><summary>Answer</summary>
Column-parallel (shard the output dim, no communication): `qkv_proj`, `gate_up_proj`. Row-parallel (shard the input dim, then all-reduce): `o_proj`, `down_proj`. That is 2 all-reduces per layer, plus a vocab all-gather at the LM head.
</details>

**19.** Why must weight loading "shard first, then merge" for `qkv_proj`?
<details><summary>Answer</summary>
Each rank needs its own heads of Q, K and V. Concatenating full q|k|v and then chunking evenly would give rank 0 all of Q and part of K, which is the wrong heads. So shard q, k and v separately, then concatenate the local slices (`models/weight.py`). SGLang does this per parameter in `QKVParallelLinear.weight_loader`.
</details>

**20.** How do all TP ranks agree on the sampled token in mini-sglang without broadcasting it?
<details><summary>Answer</summary>
Every rank all-gathers the full logits and uses the same seeded generator, so they compute identical samples. (SGLang issue #3365 discusses replacing this with gather to rank 0 plus a token broadcast.)
</details>

**21.** ★ Greedy speculative decoding: why is the output identical to plain greedy decoding regardless of the drafter, and what can break that guarantee in practice?
<details><summary>Answer</summary>
Verify accepts a draft token only if it equals the target's argmax at that position, and always appends the target's own token at the first mismatch, so the committed sequence is exactly the target's greedy path. It breaks if the verify forward is not numerically identical to a 1-token decode (different kernels or reduction orders in batched/tree verify). That is SGLang issue #50395 (NGRAM under deterministic mode).
</details>

**22.** Why must the KV of a rejected draft token never be reused, and how does mini-sglang guarantee it?
<details><summary>Answer</summary>
That K/V was computed for a token that is not in the final sequence, so later tokens would attend to a wrong history. mini-sglang keeps the invariant "the target cache holds `seq[:-1]`", truncates both caches to the confirmed path after each round, and recomputes the newest confirmed (correction or bonus) token in the next round (`speculative/decoder.py`).
</details>

## Part III — Advanced topics

**23.** Mechanism, kernel, backend: one sentence each.
<details><summary>Answer</summary>
Mechanism: the model's attention math (MHA/GQA/MLA, masks). Kernel: one GPU program implementing a computation. Backend: the engine adapter that builds metadata, writes K/V and calls a suitable kernel per phase (`AttentionBackend.init_forward_metadata`/`forward_extend`/`forward_decode`).
</details>

**24.** ★ What is split-KV, why does decode need it, and how are partial results merged?
<details><summary>Answer</summary>
The long KV sequence is cut into segments processed by different CTAs, each producing a partial output and its log-sum-exp. Merged output = Σ softmax-weights(lse_i)·o_i. Decode has one query per request, so (batch × heads) can be far below the SM count; splitting adds parallelism (`merge_state.py`; `labs/part3/attention_tiling.py`).
</details>

**25.** Does FlashAttention make prefill linear in sequence length?
<details><summary>Answer</summary>
No. FLOPs stay O(L²). It removes the O(L²) score-matrix *memory* and HBM traffic through tiling and online softmax.
</details>

**26.** Why did mini-sglang's decode attention spend as much time in `aten::index` as in SDPA, and what do paged kernels do instead?
<details><summary>Answer</summary>
`TorchAttnBackend` gathers every request's full history into a padded dense tensor per layer per step (~763 MB/step in the profile). Paged kernels (FlashInfer/Triton) read the KV in place through index lists or page tables.
</details>

**27.** Why does W4A16 speed up batch-1 decode but barely help long prefill?
<details><summary>Answer</summary>
Decode time ≈ weight bytes / bandwidth, and 4-bit weights cut bytes ~4×. Prefill is compute-bound, and W4A16 still computes in BF16 (after dequant), so FLOPs and attention are unchanged.
</details>

**28.** Why is an "8B" INT4 checkpoint ~6.1 GB rather than 4 GB?
<details><summary>Answer</summary>
Embedding and lm_head stay 16-bit (~2.5 GB), plus per-group scales and zero points, plus norms.
</details>

**29.** ★ Group size vs bits: in the lab, why did NVFP4 (FP4, group 16) beat INT4 per-channel on weight error despite having fewer distinct values?
<details><summary>Answer</summary>
Smaller groups mean an outlier only inflates the scale of 16 values instead of the whole row of 1024, so the other values keep resolution. Error is dominated by scale coarseness from outliers, not just bit count (relative error 0.10 vs 0.20).
</details>

**30.** Which file tells you a checkpoint is NVFP4 with an FP8 KV cache, and which modules are excluded?
<details><summary>Answer</summary>
For ModelOpt checkpoints, `hf_quant_config.json` (`quant_algo: NVFP4`, `group_size: 16`, `kv_cache_quant_algo: FP8`, `exclude_modules`). For `nvidia/DeepSeek-R1-0528-NVFP4`, `exclude_modules` lists lm_head and every `self_attn`.
</details>

**31.** HiCache: recompute 150 ms, H2D 30 ms, storage read +200 ms. Which lower-tier hits pay off, and what would rescue the other?
<details><summary>Answer</summary>
L2 (30 < 150) pays off. L3 (230 > 150) does not, unless prefetch hides ≥ 80 ms of the storage read during queueing, or the prefix is longer (recompute grows superlinearly).
</details>

**32.** Why can layer-wise restore start computing before all layers' KV has arrived, and what must it wait for?
<details><summary>Answer</summary>
Layer i's attention needs only layer i's history. It must wait until layer i's copy is complete (`LayerDoneCounter.wait_until`), with slots allocated, the page table written, and the streams ordered by events.
</details>

**33.** GQA with 8 KV heads at TP=16: KV bytes per GPU vs TP=8?
<details><summary>Answer</summary>
Unchanged. Each rank still holds one full KV head because heads are replicated 2× (`_shard_tensor`: `head_idx = rank * 8 // 16`).
</details>

**34.** ★ DP attention vs ordinary DP: what is shared and what is not?
<details><summary>Answer</summary>
Ordinary DP: full independent model replicas. DP attention: each group owns its requests' attention and KV (no KV replication), but the FFN/MoE is still shared across groups (TP or EP), with tokens gathered and scattered around it.
</details>

**35.** In EP, a token on GPU 0 routes to experts on GPUs 1 and 3. What moves over the network, and what doesn't?
<details><summary>Answer</summary>
The token's hidden activation is dispatched to GPUs 1 and 3, and the expert outputs are combined back to GPU 0. Its KV history never moves; attention stays local to GPU 0.
</details>

**36.** Why can't PP fill its pipeline with a single request's future tokens?
<details><summary>Answer</summary>
Token t+1's input is the token sampled after token t has passed through *all* stages. Autoregressive dependency serialises a request, so only different requests can be micro-batches.
</details>

**37.** PD: where is the first token sampled, and what crosses the P→D link?
<details><summary>Answer</summary>
On P, at the end of prefill. The prompt's KV for all layers, the first token and the request state cross the link into slots that D pre-allocated (`DecodePreallocQueue`). Both sides run all layers.
</details>

**38.** 16K-token prompt, Qwen3-8B bf16 KV, 20 GiB/s link: transfer time? With 8 concurrent transfers?
<details><summary>Answer</summary>
144 KiB × 16384 = 2.25 GiB, so ≈ 112 ms. Sharing the link 8 ways, ≈ 900 ms each (`labs/part3/pd_transfer.py`).
</details>

## Part IV — Practice

**39.** ★ Benchmark vs profile: what does each answer, and when should the profiler be off?
<details><summary>Answer</summary>
A benchmark answers *what* and *how much* (TTFT, TPOT, throughput). A profile answers *why* (which ops, gaps, overlaps). Turn the profiler off for the numbers you report: it adds overhead (~10% in our CPU run).
</details>

**40.** A kernel segment got 16.7% faster but throughput rose 1%. Explain, and say what you would report.
<details><summary>Answer</summary>
Amdahl: the segment is a small share of the critical path. Report the local trace evidence, the end-to-end numbers at several concurrencies with repeats and ranges, the correctness check, and what was not verified.
</details>

**41.** In mini-sglang's CPU profile, scheduler overhead was 0.7% of a decode step. Why would the same code be a major cost on a GPU, and what does SGLang do about it?
<details><summary>Answer</summary>
The GPU forward is ~100× faster, but the Python scheduling (~1.75 ms/step), the `.to("cpu")` sync and kernel launches are not. The loop is sequential, so the GPU idles during them. SGLang uses overlap scheduling (`event_loop_overlap` + `FutureMap`) and CUDA graphs for decode.
</details>

**42.** The profile showed only 1536 of 1792 possible prefix tokens reused. Why, and what SGLang mechanism addresses it?
<details><summary>Answer</summary>
Two requests sharing the prefix were admitted in the same prefill batch, before the first one's prefix was inserted, so both computed it. SGLang's scheduling policy has in-batch prefix caching (`IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD`, a simulated waiting-queue radix tree) and LPM ordering.
</details>

**43.** ★ Write a valid SGLang PR title for a fix to the draft model loader, and list the PR template sections.
<details><summary>Answer</summary>
`[Fix] loader: Strip target-only extra config for draft loader`. Sections: Motivation, Modifications, Accuracy Tests, Speed Tests and Profiling, Checklist (plus the Review and Merge Process notes).
</details>

**44.** Your PR has been open two days and CI never ran. Why, and what do you do?
<details><summary>Answer</summary>
GPU CI needs the `run-ci` label, added by users in `CI_PERMISSIONS.json`. Run pre-commit and the unit tests locally, then ask the Merge Oncall or a module reviewer (or Slack `#pull-request`). Don't push empty commits or spam reruns. As the author you may use `/rerun-failed-ci` after a failed run.
</details>

**45.** ★ Where does a unit test for `srt/mem_cache/radix_cache.py` go, what must the file contain to run in CI, and how do you run it the way CI does?
<details><summary>Answer</summary>
Under `test/registered/unit/mem_cache/`. It needs a `register_cpu_ci(...)` (or the CUDA/AMD equivalent) call at the top, test classes deriving from `CustomTestCase`, and a `if __name__ == "__main__": unittest.main()` entry. Run it with `python3 test/registered/unit/mem_cache/test_radix_cache_unit.py` (or `pytest` on the directory).
</details>

## Practical checks (no answer key; just do them)

- [ ] Explain, from memory, the path of one token through `normal_loop` and then through `Scheduler.event_loop_overlap`.
- [ ] Run `python benchmarks/profile.py` and find, in Perfetto, the `aten::index` inside one decode `attn.forward`.
- [ ] Run `python docs/tools/verify_source_map.py <sglang checkout>` on a newer SGLang commit and fix one broken pointer.
- [ ] Build SGLang from source, then run `python3 test/registered/unit/mem_cache/test_radix_cache_unit.py` and `pre-commit run --all-files` on a clean branch.
- [ ] Reproduce one of the five issues in `contributing-to-sglang.md` on your own hardware.
