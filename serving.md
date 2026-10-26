# Kernel swaps in inference engines

Start with **one dense attention replacement in vLLM**, then test DeepInfer's sparse MLA path separately. Keep the CuTe/NCCL harness for isolated math, communication and backward. Serving validates forward behavior only.

## Evidence saved here

- [trajectory.json](trajectory.json) and [trajectory.md](trajectory.md): session `ses_e8d0b6bb0ffeygfOvr5vV6FFpX`, 433 messages across nine API pages. JSON retains user messages, attachments, assistant content and tool results. Exported read-only from the running OpenCode service; its newer API contained the session absent from legacy SQLite tables. No session was resumed.
- [upstream.json](upstream.json): export hash, clone URLs, commits and DeepInfer vendor gitlinks/checkouts.
- [upstream/vllm](upstream/vllm): `a7407ab44557802ca32a0ba2c2707362d7d21563`.
- [upstream/DeepInfer](upstream/DeepInfer): `553741557a57df44a46fb3ab72b6aafeaf945337`.
- These are shallow source clones. A matching vLLM wheel was installed and exercised remotely; DeepInfer remains source-only. Full trajectories and clones are Git-ignored. No external model weights were downloaded; the vLLM smoke creates a tiny seeded random checkpoint.
- [server_experiment.py](server_experiment.py): two reversible source edits exposing three callable selections; applied locally, defaulting to stock. Other commits or unrelated edits are refused. A slow paged-GQA reference replacement and a real HTTP smoke live in the same file.
- [test_server_experiment.py](test_server_experiment.py): six CPU checks, including dispatch through the actual DeepInfer source with fake tensor/kernel dependencies. These CPU checks do not execute CUDA or load models; separate real GPU smoke results are recorded below.

The relevant final conversation comparison is assistant message `msg_1759bd274001OvU5Q5KejhT8uu`; its preceding cost estimate is `msg_1757527e7001RCy4EgJMZsJzWF`. Historical assertions are not our benchmark results.

## What transfers, and what needs correction

The useful principles are to separate model compute from service behavior, compare a kernel within the same engine, and test cold prefill, cached-prefix prefill, decode and bursty agent traffic independently. Repeated prefixes and tool-idle time make saturated throughput a poor predictor of campaign cost. Reuse native runners before adding services.

Three corrections matter:

- DeepInfer has internal KV storage and cached-prefix/fork logic (`worker/cache_engine.py`, `examples/benchmark.py`). A missing production KV-cache *system* does not mean no cache or mandatory re-prefill.
- The vLLM article's $38.96/GPU-hour is API-price-derived serving value, not rental cost. Its 954K aggregate tokens/s/GPU includes cached input. Its prefill configurations differ across engines, so curve proximity is not a controlled speedup. [Primary article](https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep).
- Old memory arithmetic and allocation probes do not prove model loading, kernel compatibility or current cost. DeepInfer documents NVIDIA SM100. Its pinned FlashMLA explicitly builds SM100a and SM103a, but that alone does not establish whole-engine SM103 support. V4.1 compressed/sparse MLA is also a different operator from the project's V3.2 DSA and original GDN.

## Read these paths

Paths below are relative to each clone.

**vLLM:** `vllm/entrypoints/cli/serve.py` starts the API service. `vllm/v1/engine/core.py` and `vllm/v1/core/sched/scheduler.py` schedule work; GPU workers/model runners prepare model inputs, cache metadata and graph execution. `vllm/v1/attention/selector.py` selects the backend.

For dense/GQA attention read `vllm/v1/attention/backends/flash_attn.py::FlashAttentionImpl.forward`, then `vllm/vllm_flash_attn/flash_attn_interface.py::flash_attn_varlen_func`. Reuse `tests/kernels/attention/{test_flash_attn.py,test_flash_attn_interface.py,test_prefix_prefill.py}` and `vllm/benchmarks/serve.py`.

V4.1 uses `vllm/models/deepseek_v41/nvidia/{model.py,flashmla.py,flash_mla_mega_attn.py,flashinfer_sparse.py}`; it bypasses the dense hook below. For V3.2 start at `vllm/models/deepseek_v32/attention.py` and `vllm/v1/attention/backends/mla/`. Do not force the dense backend onto a sparse model.

**DeepInfer:** `deepinfer/entrypoints/llm.py::LLM` invokes `engine/llm_engine.py::LLMEngine`, then `worker/model_runner.py`. `model_executor/models/v4_modules/attn.py::V4Attention.forward` coordinates projections, compression/indexing, KV storage, CP exchange and attention. `_prefill_attention` / `_decode_attention` call the two functions in `model_executor/layers/attention_backend.py`, which dispatch to FlashMLA including fused paths.

Reuse `examples/offline_inference.py`, `examples/benchmark.py --result_jsonl`, `examples/sweep/run_sweep.sh`, and `tests/models/{test_batch_invariant.py,test_offline_inference.py,test_offline_inference_long.py}`. Investigate numerical changes before refreshing existing PPL baselines. DeepInfer's public release is an offline engine; a complete service requires additional components. [Upstream scope](https://github.com/deepseek-ai/DeepInfer).

Its NVIDIA vendor gitlinks pin FlashMLA, DeepEP, DeepGEMM and DeepSelect; requirements also pin TileKernels/TileLang. The four NVIDIA vendor source checkouts are now fetched at their exact parent gitlink commits. Build them with native scripts in a separate engine environment when needed. The pinned FlashMLA and DeepEP require toolkit 13.1 or newer (FlashMLA recommends 13.2); the existing research image's toolkit 13.0.2 is insufficient for those builds. DeepEP also requires NCCL >=2.32.3 and a matching loaded library. They are not laptop-harness dependencies. Ascend vendor entries are outside this B200 investigation.

## Where our ideas fit

**Dense CP:** prove local math through vLLM first. Fusing communication and compute requires replacing the enclosing backend, not only its local FA call. `vllm/v1/attention/backends/registry.py::register_backend` supports custom backends; `vllm/plugins/__init__.py` loads general plugins in workers. Reuse metadata/cache builders only where ownership/layout matches.

At this pin `FlashAttentionImpl` inherits `supports_pcp=False`. Setting `--prefill-context-parallel-size 4` is not a valid dense CP baseline. Existing `_forward_with_dcp` and `vllm/v1/attention/ops/dcp.py` use different cache ownership from contiguous training shards. Read `vllm/v1/worker/gpu/pcp_manager.py` and `vllm/v1/attention/ops/pcp.py` before opting a new backend into PCP. Preserve global positions, cache ownership, output order and LSE logarithm base.

**DeepInfer CP:** `model_executor/parallel_utils/cp.py::CPManager` uses DeepEP BucketBuffer sessions. Gathered tensors alias reusable session storage; waits must finish within a session. `V4Attention.forward` launches gathers before Q projection and consumes/reorders results using metadata such as `mirror_cp_reverse_index`. The pinned DeepEP BucketBuffer uses NCCL Gin and its all-gather uses copy engines (`num_sms=0`). This is a useful balanced/overlapped baseline: compare against the real copy-engine path, not an assumed SM-consuming transfer. Preserve session/event completion before reusing storage. Its sparse attention hook is downstream of those gathers; swapping it cannot establish their removal. Fused CP must also handle indexer inputs and the enclosing orchestration.

**DSA:** use DeepInfer to test supplied-index sparse math against its actual V4.1 contract. Complete-operator timings must include compression, indexer, selection and routing. Keep V3.2 indexer/backward references separate and use vLLM's V3.2 model path for that integration.

**GDN:** retain recurrence, state-boundary and gradient tests here. Later integrate through a vLLM GDN/Mamba layer/backend, preserving per-request state and resets. DeepInfer's V4.1 path is not a GDN host. Neither engine replaces backward checks in `experiment.py`.

## The small swap mechanism

~~~bash
uv run python server_experiment.py status vllm
uv run python server_experiment.py status DeepInfer
uv run python server_experiment.py apply vllm
uv run python server_experiment.py apply DeepInfer
# Restore only our exact changes:
uv run python server_experiment.py restore vllm
uv run python server_experiment.py restore DeepInfer
~~~

Each selection is read once at module import:

- `DIST_VLLM_ATTENTION=stock` or `kernels:my_vllm_attention`.
- `DIST_DEEPINFER_PREFILL=stock` or `kernels:my_sparse_prefill`.
- `DIST_DEEPINFER_DECODE=stock` or `kernels:my_sparse_decode`.

The example replacement `DIST_VLLM_ATTENTION=server_experiment:reference_attention` is implemented and GPU-tested. It handles small eager BF16 paged causal GQA with explicit bottom-right masking, FP32 PyTorch math and caller-owned output writes. It rejects unsupported features, long contexts and graph capture; it never calls native attention. It is a correctness demonstrator, not an optimized CuTe kernel. The other named candidates remain to be written; add/LSA are not attention kernels. The helper never automatically falls back on import, shape or kernel failures. Candidate code must check supported cases and raise otherwise. Without audit, selection returns the callable directly, with no added per-call Python dispatch wrapper.

Preserve the native ABI. vLLM needs ragged lengths, paged KV/block tables, bottom-right causal alignment for unequal Q/K lengths, optional LSE and caller-owned `out` writes. DeepInfer prefill accepts BF16 Q/KV and indices; its fused path returns quantized `(output, scale)` plus LSE. Decode additionally requires quantized paged cache, scheduler metadata, sinks, index lengths and extra-cache inputs. A dense BF16-only implementation cannot satisfy these contracts. At the pinned FlashMLA revision, Q/K and V dimensions are 512, with 64/128 Q heads. Decode invalid indices must be exactly -1: positive out-of-range indices are not bounds-checked. Prefill also accepts positive indices beyond KV length as invalid. Rebuild decode scheduling metadata when top-k length values change; reusing stale metadata can silently select wrong splits. Attention sinks affect output normalization but not returned LSE. All-invalid rows return zero output and +inf LSE, so this sentinel is an explicit correctness case.

Set `DIST_KERNEL_AUDIT=/absolute/result/directory` for a small **eager** smoke. Per-process JSONL records native source/signature, selected source/hash, input layouts and successful Python returns. No tensor payloads or synchronization are recorded. Check calls from every expected worker and relevant shape; an import-time `selected` entry alone proves no execution. A returned Python call also does not establish device completion or numerical correctness.

Unset audit and restart all workers before timing. Python I/O perturbs latency and can break capture/compilation. Graph replay does not re-enter Python. Both arms must use the same graph/eager settings.

## Run inside isolated engine environments

Keep vLLM, DeepInfer and the research GPU image separate. DeepInfer currently pins Torch 2.13.0+cu132 and FA4 4.0.0b34; our scaffold pins Torch 2.14.1 and CuTe 4.8.1. Do not merge those dependency sets.

On a suitable CUDA host an editable vLLM installation can start with:

~~~bash
uv venv .venv-vllm --python 3.12
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=a7407ab44557802ca32a0ba2c2707362d7d21563 \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
uv pip install --python .venv-vllm/bin/python \
  -e upstream/vllm --torch-backend=auto
~~~

Matching cu130 wheel metadata was fetched to `results/vllm-wheel-metadata.json` (x86_64 and aarch64, version `0.32.1rc1.dev491+ga7407ab44`). Availability is not runtime validation. Precompiled artifacts must match the source/build ABI. Save build logs and versions; use upstream source-build instructions if artifacts do not match. This editable local installation has not been run here. The exact wheel and two checked source overlays were exercised on Modal as described below.

Set `MODEL` to an already staged small dense/GQA checkpoint. Start one GPU:

~~~bash
export DIST_PROJECT_ROOT="$PWD"
export PYTHONPATH="$DIST_PROJECT_ROOT:$DIST_PROJECT_ROOT/upstream/vllm"
export DIST_KERNEL_AUDIT="$DIST_PROJECT_ROOT/results/vllm-stock-audit"
export DIST_VLLM_ATTENTION=stock
.venv-vllm/bin/python -m vllm.entrypoints.cli.main serve "$MODEL" \
  --attention-backend FLASH_ATTN --dtype bfloat16 \
  --enforce-eager --no-enable-prefix-caching \
  --max-model-len 4096 --max-num-seqs 4 --port 8000
~~~

In another terminal, set `MODEL` identically and exercise prefill/decode:

~~~bash
.venv-vllm/bin/python -m vllm.entrypoints.cli.main bench serve \
  --backend vllm --model "$MODEL" --base-url http://127.0.0.1:8000 \
  --dataset-name random --random-input-len 256 --random-output-len 32 \
  --num-prompts 32 --max-concurrency 4 --request-rate inf --seed 0 \
  --num-warmups 4 --ignore-eos --save-result --save-detailed \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --result-dir results/vllm-stock-audit --result-filename requests.json
~~~

This checks routing, not statistical significance. Independently validate a candidate, change the selection/result directory, restart the server and repeat. Compare logits/tensors as well as generated token IDs; matching text alone is weak numerical evidence. For performance disable audit, then test identical graph settings in both arms after changed-input replay passes.

For DeepInfer create a separate Python 3.12 environment on supported hardware and follow its README from inside the clone. `build.sh` uses the active environment and initializes pinned vendor sources. Add the project root and DeepInfer clone to `PYTHONPATH` in every worker/container. With the actual model staged:

~~~bash
# Inside upstream/DeepInfer, in its built environment.
export PYTHONPATH="$DIST_PROJECT_ROOT:$DIST_PROJECT_ROOT/upstream/DeepInfer"
export DEEPINFER_ACCELERATOR=nvidia
export WORLD_SIZE=1 RANK=0
# Set CUDA_VISIBLE_DEVICES to the actual sufficient allocation.
export DEEPINFER_USE_NEXTN_DECODE=0 DEEPINFER_ENABLE_MTP_VARLEN=0
export DEEPINFER_ENFORCE_EAGER=1 PROFILE=0
export DIST_DEEPINFER_PREFILL=stock DIST_DEEPINFER_DECODE=stock
python examples/benchmark.py --model_name "$DEEPINFER_HF_MODEL_DIR" \
  --cp 1 --context_length 4096 --non_cached_tokens 4096 \
  --prefill_batch_size 4096 --decode_num_requests 8 \
  --result_jsonl "$DIST_PROJECT_ROOT/results/deepinfer-stock.jsonl"
~~~

Use audit only for routing, then unset it. Change one DeepInfer selection per experiment. Run existing offline/PPL and long-context regressions before interpreting speed. Full V4.1 inference is not a single-GPU smoke; kernel-only replay cannot be labeled whole-engine validation.

## Statistics and acceptance

Save a manifest per independent run block: engine commit plus dirty patch, helper/kernel hashes, model/tokenizer revisions, request IDs/token inputs/seed/order, GPU topology/driver/clocks, package lock/image digest, parallelism, dtype/layout/cache pages, batching limits, eager/graphs, prefix state, MTP, EPLB, SWA/replay and profiler settings. Retain logs and failures.

1. **Correctness:** independent small FP32/FP64 reference plus stock backend; outputs and LSE (including documented empty-row sentinels), finite ordinary outputs, poisoned output buffers, changed inputs, ragged/padded/page boundaries, non-square causality, document exclusions and graph reuse. Record max absolute and normalized RMS errors with declared tolerances. Relative error near zero is unstable. Training requires gradients separately.
2. **Operator latency:** warm compilation/allocations, then time the entire public call, including conversions, staging, planning, communication and returns. Save rank-by-iteration samples. Distributed latency is `max_rank(time)` for each coordinated iteration **before** summarizing.
3. **Engine/service latency:** vLLM detailed JSON stores durations in seconds (goodput CLI limits use milliseconds) and retains `ttfts`, `itls`, `latencies`, `start_times`, `queue_times`, output lengths and errors. TTFT includes server-side scheduling; client-side queuing is separate. TPOT is `(latency - TTFT)/(output_tokens - 1)` for outputs longer than one token. Report median/p90/p99 TTFT, TPOT, ITL and end-to-end latency with error/completion counts.
4. **Useful throughput/resources:** actual committed output tokens divided by measured wall time; uncached input, cached input and output counts separately, aggregate and per GPU. Record memory (Torch allocated/reserved and native/NCCL buffers), cache capacity/hits, communication bytes and utilization. Unavailable metrics are missing, not zero.
5. **Service impact:** fixed offered rate for latency; a saturation sweep for capacity. Goodput is requests/s meeting predeclared TTFT/TPOT/latency limits with acceptable correctness. Preserve timeouts/OOMs/failures and offered/completed counts. Add frozen multi-turn prefixes and tool-idle gaps only after simple regimes pass; include GPU residency time in cost.

DeepInfer JSONL records `prefill_window_tokens`, `prefill_window_seconds`, `decode_avg_sps` and `decode_max_step_ms`, not all step intervals or HTTP TTFT/p99. Its prefill denominator uses CP size; DP/global aggregation needs explicit accounting. LB-stat-flush intervals are excluded: label this **steady-state clean-window throughput**, not full service cost. Save raw per-step intervals before estimating tails or interval-level uncertainty.

Decode uses `steps/s × requests × (1 + nextn) / CP_size`, a full-acceptance convention. Start with `nextn=0` and variable drafting off. Later measure committed tokens/acceptance directly; do not compare that convention to vLLM's actual output tokens/s.

For a pilot take 30–50 warmed samples per arm and at least five independent paired blocks; increase blocks before a decision. Randomize stock/candidate order within blocks, match hardware/requests, restart/re-warm consistently and save order/wall times. Iterations/requests in one allocation are correlated.

For block b define `R_b = median(stock latency_b)/median(candidate latency_b)`. Report every ratio, medians, spread/CV and `exp(mean(log R_b))`. Estimate a 95% interval by resampling whole paired blocks together. Use block-level uncertainty for service tails too; 32 requests cannot establish p99. Predeclare useful gain and allowed tail/error/memory regression. An interval crossing 1 is inconclusive.

Profile separately. If attention occupies critical-path fraction f and improves by s, the idealized engine bound is `1/(1-f+f/s)`; measure f. Compare dense CP against all-gather + pinned FA4 and a balanced ring/zigzag implementation at fixed global inputs. Training backward is separate. An unbalanced baseline alone is insufficient.

## Verified server swap and next experiment

Run the bounded native-server stage from this root. Its model geometry and prompt lengths are fixed; research-stage size/iteration flags do not configure this smoke, and it requires seed 0. The server image mounts both `server_experiment.py` and `kernels.py`, so candidates in either file can be selected once their imports match the isolated engine stack:

~~~bash
uv run modal run modal_app.py --stage server --gpus 1
uv run modal run modal_app.py --stage server --gpus 1 \
  --selection server_experiment:reference_attention
~~~

Both commands were executed successfully on one NVIDIA B200 (SM100). The server image is isolated from the research extra: vLLM `0.32.1rc1.dev491+ga7407ab44`, Torch `2.14.0+cu130`, Transformers `5.20.2`. The installed attention file's original hash matched the pinned clone before applying its exact overlay. The launcher saves source snapshots, raw worker audits, HTTP responses, CPU reference logprobs and environment; the replacement run includes all 204 installed package versions. The function has no retries, a 600-second limit and a 500-second child-process limit. All smoke apps stopped; no endpoint was deployed.

The checkpoint is a seeded two-layer random Llama/GQA model, BF16 weights, two query heads, one KV head and head dimension 128. Four HTTP completions use prompt lengths 17, 129, 257 and 17, generating four tokens each. Length 257 crosses the configured prefill chunk boundary. The repeated first prompt checks KV storage reuse after intervening requests; prefix caching is disabled. All 128 vocabulary logprobs at each generated position are compared to CPU FP32 eager Transformers with the exact same BF16-rounded weights, using absolute tolerance 0.04.

- Stock: [raw result](results/20261026T012256Z-1690fd/result.json), 36 worker attention calls spanning prefill and decode; maximum logprob error against CPU reference **0.007627**.
- Replacement: [raw result](results/20261026T013256Z-1d2dd2/result.json), 36 calls selecting `server_experiment:reference_attention`; maximum CPU-reference error **0.008932**.
- [Comparison](results/server-comparison.json): identical checkpoint hashes and all 16 generated token IDs; maximum stock/replacement logprob difference **0.008044**.

This establishes real worker routing, paged cache use, changed sequence lengths, incremental decode and an actual replacement computation through HTTP. It does not establish trained-model quality, optimized-kernel speed, CuTe attention correctness, arbitrary shapes, graphs, backward, multi-GPU CP or DeepInfer full-model execution. The original add/LSA research checks remain separate and have not been GPU-executed by these server runs.

The next research step is to replace the reference body with a matching local CuTe kernel and repeat numerical checks before timed blocks. Use the enclosing backend boundary when integrating fused CP; use the separate sparse contract and supported toolchain for DeepInfer. No HTTP wrapper, KV connector or engine-neutral adapter layer is needed for the current experiments.
