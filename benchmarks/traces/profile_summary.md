# mini-sglang profile summary

workload: 8 requests, shared prefix 256 tokens + 32-256 unique, output 32 tokens, prefill_budget=512, cache=radix, device=cpu, threads=4

prefix cache: 3342 prompt tokens admitted, 1536 reused from the cache (46%), 1806 computed in prefill

wall time (profiled run) 12.99 s, 35 busy steps (+0 idle), 256 output tokens, 19.7 tok/s under the profiler


## prefill: 4 steps, batch size mean 2.8, tokens/step mean 452

| phase | total ms | per step ms | % of step |
|---|---:|---:|---:|
| sched.recv | 0.0 | 0.01 | 0.0% |
| sched.prefill_admission | 0.8 | 0.20 | 0.0% |
| kv.allocate_paged | 0.2 | 0.04 | 0.0% |
| attn.prepare_metadata | 1.4 | 0.35 | 0.0% |
| sampler.prepare | 0.0 | 0.01 | 0.0% |
| sched.prepare_batch | 2.2 | 0.55 | 0.0% |
| sched.schedule | 3.1 | 0.78 | 0.1% |
| model.forward | 4992.0 | 1248.00 | 99.8% |
| sampler.sample | 1.7 | 0.42 | 0.0% |
| sched.forward | 4995.2 | 1248.79 | 99.9% |
| kv.cache_req | 0.9 | 0.23 | 0.0% |
| sched.process_results | 1.5 | 0.38 | 0.0% |
| step | 5000.7 | 1250.17 | 100.0% |

non-model time (scheduler + bookkeeping) = 1.75 ms/step = 0.1% of prefill step time

## decode: 31 steps, batch size mean 8.0, tokens/step mean 8

| phase | total ms | per step ms | % of step |
|---|---:|---:|---:|
| sched.recv | 0.2 | 0.01 | 0.0% |
| sched.prefill_admission | 0.0 | 0.00 | 0.0% |
| kv.allocate_paged | 3.6 | 0.12 | 0.0% |
| attn.prepare_metadata | 21.5 | 0.69 | 0.3% |
| sampler.prepare | 0.3 | 0.01 | 0.0% |
| sched.prepare_batch | 31.4 | 1.01 | 0.4% |
| sched.schedule | 32.4 | 1.04 | 0.4% |
| model.forward | 7900.6 | 254.86 | 98.9% |
| sampler.sample | 37.1 | 1.20 | 0.5% |
| sched.forward | 7950.1 | 256.46 | 99.5% |
| kv.cache_req | 0.7 | 0.02 | 0.0% |
| sched.process_results | 5.8 | 0.19 | 0.1% |
| step | 7992.0 | 257.81 | 100.0% |

non-model time (scheduler + bookkeeping) = 1.75 ms/step = 0.7% of decode step time

## decode step time by batch size

| batch size | steps | step ms | model.forward ms | non-model ms | attn.prepare_metadata ms |
|---:|---:|---:|---:|---:|---:|
| 8 | 31 | 257.8 | 254.9 | 1.75 | 0.69 |

## top operators by self CPU time (whole run)

| op | calls | self CPU ms | % |
|---|---:|---:|---:|
| `aten::mm` | 3955 | 7493.1 | 61.1% |
| `aten::_scaled_dot_product_flash_attention_for_cpu` | 980 | 2140.7 | 17.5% |
| `aten::index` | 5009 | 1515.4 | 12.4% |
| `aten::mul` | 16730 | 214.6 | 1.7% |
| `aten::_index_put_impl_` | 2975 | 104.6 | 0.9% |
| `aten::pow` | 3955 | 89.2 | 0.7% |
| `aten::add` | 7910 | 83.4 | 0.7% |
| `aten::silu` | 980 | 56.1 | 0.5% |
| `aten::sum` | 3955 | 44.2 | 0.4% |
| `aten::argmax` | 35 | 38.0 | 0.3% |
| `aten::cat` | 2329 | 37.6 | 0.3% |
| `aten::as_strided` | 40316 | 33.8 | 0.3% |
| `aten::div_` | 3955 | 28.8 | 0.2% |
| `aten::where` | 1960 | 27.4 | 0.2% |
| `aten::sub` | 1995 | 26.5 | 0.2% |

## attention (`attn.forward`) share

attn.forward: 980 calls, 4115 ms total = 31.9% of model.forward

trace: `benchmarks/traces/minisgl_trace.json` (66.9 MiB)

## gaps between scheduler steps (from the trace)

35 step ranges; total gap 0.8 ms; largest 0.05 ms, 0.04 ms, 0.03 ms, 0.03 ms, 0.03 ms
