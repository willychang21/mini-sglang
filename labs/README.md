# Labs (Part III & IV, CPU-only)

All labs run on CPU. The ones that load a model expect Qwen3-0.6B at
`/tmp/models/Qwen3-0.6B` (override with `--model`). Run them from the repo root after
`pip install -e ".[test]"`.

| lab | chapter | what it shows | runtime |
|---|---|---|---|
| [`part3/attention_tiling.py`](part3/attention_tiling.py) | III-1 | online-softmax tiling == naive attention; split-KV + LSE merge; mini-sglang's mask | ~2 s |
| [`part3/quant_error.py`](part3/quant_error.py) | III-2 | rounding error of BF16/FP8/INT8/INT4/NVFP4-like schemes on real weights; outliers; end-to-end greedy agreement and KL | ~60 s (`--skip-e2e`: 5 s) |
| [`part3/kv_capacity.py`](part3/kv_capacity.py) | III-2, III-4 | KV tokens/requests per precision (reproduces Fig. 5); per-rank KV under TP with GQA | <5 s |
| [`part3/hicache_cost_model.py`](part3/hicache_cost_model.py) | III-3 | measured restore vs recompute; GPU cost model; prefetch and layer-wise overlap; Ex. 3 | ~15 s |
| [`part3/parallel_comm.py`](part3/parallel_comm.py) | III-4 | bytes/collectives per step for TP, DP-attn, EP, PP | <2 s |
| [`part3/pd_transfer.py`](part3/pd_transfer.py) | III-5 | KV transfer size/time vs prompt length and link; contention; prefix on D; measured bytes | ~10 s |
| [`part4/ab_compare.py`](part4/ab_compare.py) | IV-1 Ex.2, IV-2 Ex.3 | credible one-setting A/B with repeats, ranges, correctness | ~45 s |
| [`../benchmarks/profile.py`](../benchmarks/profile.py) | IV-2 | torch.profiler trace with scheduler phases + summary | ~20 s, ~5 GB RAM |
