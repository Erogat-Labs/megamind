# Experiments

For experiments inside actual inference engines, follow [serving.md](serving.md). Start with the stock vLLM dense path; the sparse DeepInfer path is separate. A deliberately slow paged-GQA reference replacement is GPU-verified through vLLM; optimized CuTe attention and fused CP remain to be implemented.

Work in short, inspectable steps. The immediate question is whether we can compile, launch, and correctly use CuTe and NCCL peer memory on the actual Modal machine. Implement one mechanism at a time.

## First: local compilation

Use the no-GPU compile command in README.md, or `uv run --extra gpu python experiment.py compile` if using the full local GPU extra. This compiles vector addition at lengths 1/129/`--n` and the two-rank LSA sum for `sm_100a` with fake tensors/windows. Pass means compilation only: the result explicitly records `executed: false`. `compile` is a local-only stage; the Modal launcher accepts the five GPU stages below.

## Runnable matrix

Run these in order. Each invocation launches one allocation; nothing automatically sweeps combinations.

1. **Environment: `--stage env`.** GPUs: 1, then 4. Pass: CUDA is available, expected NCCL runtime/bitcode selected, package/device information recorded. Inspect peer-access entries and `nvidia-smi topo -m` in the log. Environment success alone is not a working communication test.
2. **CuTe: `--stage cute`.** GPUs: 1, then 2/4 independently. Float32 addition at lengths 1, 127, 128, 129, and `--n` (default 4099). Pass: exact equality with PyTorch, poisoned output fully overwritten, changed-input repeated launches, and three changed-input CUDA graph replays per shape. Multiple GPUs here test independent local kernels, not distributed computation.
3. **SDPA: `--stage sdpa --gpus 1`.** Hq=32; `--kv-heads` 32/8/1 gives MHA/GQA/MQA. B=1, D=128, BF16, square causal mask. Start at `--tokens 256`; later time 8192 and 32768 separately. Pass: forced FlashAttention backend works, output and dQ/dK/dV agree with the independent FP32 reference at min(tokens, 256). Record actual checked and timed lengths separately. Initial smoke tolerances are output rtol/atol 0.02 and gradients 0.05; inspect saved maximum errors and calibrate tighter workload-specific tolerances before making numerical claims.
4. **NCCL: `--stage nccl --gpus 2`, then 4.** Float32 all-reduce of `--n` elements with rank- and iteration-dependent values. Pass: exact expected sums for every element and iteration. This checks ordinary collectives, not symmetric-window device access.
5. **Symmetric memory: `--stage symmetric --gpus 2`, then 4.** NCCL-owned buffers, collectively registered windows, 16 Float32 elements per rank, custom CuTe LSA reduction. Pass: one LSA team spans all requested ranks and every rank receives exact expected sums across changed-input iterations. Host barriers separate writes, reads, and reuse. This intentionally isolates API/mapping support before implementing asynchronous flags.

Use seeds 0, 1, and 2 after the first successful run. Start with 10 iterations; use 40 and several independent allocations for meaningful measurements. Vector addition and SDPA save per-iteration CUDA-event samples. Communication checks currently report correctness only; do not interpret their total job duration as transport latency.

Example follow-ups:

```bash
uv run modal run modal_app.py --stage cute --gpus 1 --n 1048583 --seed 1
uv run modal run modal_app.py --stage sdpa --kv-heads 32 --tokens 8192 --iterations 40
uv run modal run modal_app.py --stage sdpa --kv-heads 8 --tokens 8192 --iterations 40
uv run modal run modal_app.py --stage sdpa --kv-heads 1 --tokens 8192 --iterations 40
uv run modal run modal_app.py --stage symmetric --gpus 4 --iterations 40
```

These are paid remote commands. Keep the initial investigation within the selected two-hour window; inspect errors instead of repeatedly relaunching the same failure. A per-stage 600-second container timeout is present, but no dollar cap or account-wide budget is configured.

## Where to write code

- Put the next small kernel in `kernels.py`.
- Put its inputs, independent reference, assertions, and result dictionary in `experiment.py`.
- Add a stage to both stage lists and its dispatch in `experiment.py`. Change Modal allocation/setup only if the experiment needs it.
- Keep shape choices explicit in the command. Add another file only when it makes an actual experiment easier to understand.

For a custom attention kernel, start with one GPU and a tiny sequence. Compare against an independently expressed FP32/FP64 operator and PyTorch/FA4 where supported. Add backward only after forward passes. Avoid assuming that another optimized kernel is an independent oracle.

The current code is deliberately a smoke harness: no optimized attention kernel, full distributed attention, device signaling protocol, asynchronous overlap, packed-document planner, or DSA/GDN implementation exists yet.

## Next mechanism: ready / acknowledge

The host-synchronized LSA sum is only the starting point. The next experiment should copy a payload to a peer-owned receive slot, publish a ready flag, consume locally, and acknowledge before slot reuse.

Prove this chain explicitly: payload issue -> copy completion -> required proxy/system ordering -> release-ready -> acquire-poll -> consumer read -> acknowledgement -> producer reuse. Registration alone does not establish that chain.

Start with one producer and one consumer. Use changing payloads, poison receive storage, one writer per signal, bounded waits, and a record of rank/epoch/expected flag on timeout. Add a controlled delayed rank, repeated epochs, and a negative control that deliberately removes a required wait/fence and must fail. Keep the negative control separate from valid runs. Then check device-side epoch advancement under graph replay.

Only after this passes should communication enter the FA4 compute pipeline. Do not reuse a shared-memory stage until both compute and transfer readers are done. Validate actual warp roles and generated resources for the pinned FA4 revision.

## Attention rollout

1. **Single GPU:** pin stock FA4; compare MHA/GQA/MQA under the same geometry and backend rules.
2. **Two-GPU forward:** contiguous shards, explicit global positions, global key prefix, no migration.
3. **Four-GPU forward:** repeated calls, delayed ranks, grid sizes exceeding resident CTAs, and graph replay.
4. **Backward:** owner-directed dK/dV, correct dQ, global loss scaling, fast accumulation versus fixed-order per-source slots. Test checkpoint recomputation and retained forward contexts.
5. **Balancing:** migrate whole output tiles; count query/key sends, output returns, and backward ownership. Show progress without assuming all CTAs run simultaneously.
6. **Documents:** preserve document starts in global coordinates; test tiny, cross-shard, aligned, unaligned, and padded documents. GPU backward planning follows a correct CPU reference planner.
7. **Extensions:** DSA supplied-index core -> indexer/global top-k -> routed operator/backward; GDN sequential recurrence -> chunks -> boundary-state pipeline -> prefix composition/reverse gradients -> full layer.

The first distributed attention shape is global S=32768, W=1/2/4, B=1, Hq=32, Hkv=8, Dqk=Dv=128, BF16. Hold global tensors fixed as W changes. Expand to S=8K/16K/32K/64K/128K/256K only after correctness and memory estimates pass. Hkv=32/1 and D=64 follow later; unequal query/value dimensions, FP16, and larger batches are compatibility follow-ups.

For packed documents, use fixed seeds 0–4 and distributions such as short exponential lengths (mean 512), lognormal lengths (mean about 4096), heavy tails, mixed lengths exceeding a shard, and an adversarial imbalance. Save actual lengths and valid token counts. Frozen FineWeb-Edu manifests can add realistic lengths later.

DSA must include causal/document-constrained global selection, deterministic tie handling, request deduplication semantics, gradient scatter, and indexer costs. Start with k=512/1024/2048. For a trained indexer, pin the reference auxiliary KL loss: detach teacher attention and backbone inputs so auxiliary gradients update the indexer while the main loss updates the backbone. Hard top-k is not its ordinary differentiable training path. Compare with native CuTe DSA and all-gather references.

GDN should start with state dimensions 64x64/128x128 and chunks 64/128. The recurrence is an affine state transformation; ordered shard transforms can be composed, but floating-point composition is not bitwise associative. Include summary construction/exchange, incoming-state correction, reverse gradients, gates/normalization, document resets, and convolution halos. Compare original GDN, FLA, FLA CP, and a simple sequential boundary-state pipeline.

## How to decide whether it works

First inspect `manifest.json`, then `result.json` and `run.log`. Passing means every requested rank passed its actual assertions; unsupported APIs and unavailable windows are failures to investigate, not skipped successes.

Distinguish image/installation failures, runtime/library failures, kernel compilation failures, numerical failures, and timeouts. Fix the first failing stage before increasing shapes or GPU count. Keep unsuccessful results; their source snapshots make failures reproducible.

For performance, compile and warm before measuring. These initial event samples synchronize each iteration and serve smoke measurements. Before publishing throughput, add sustained back-to-back measurement, paired randomized baseline ordering, 30–50 samples and several run blocks, raw rank-by-iteration times, confidence intervals, and power/clock observations.

Compare complete public calls, including layout conversion, staging, selection, synchronization, migration, and output returns. Distributed latency is the slowest rank after a coordinated start. Single-GPU scaling and speedup against the strongest distributed baseline answer different questions. Do not add overlapping phase durations as if sequential.

Track PyTorch and native/NCCL memory separately. At 256K tokens, BF16 K/V alone can be about 1 GiB globally for Hkv=8 and D=128; prefix staging can approach global size on later ranks. Avoid full S-by-S reference masks/scores at long context.

Keep profiling optional. CUDA events and sampled device timestamps can reveal waits and phases. Check instrumentation overhead and trace overflow. GPU clocks are separate timebases. Genuine NCU reports require permitted live capture; isolated local-tile reports explain resources but not distributed overlap. Do not use default profiler replay on mutually waiting kernels.

Stop at the smallest result that answers the current question. A negative speed result with correct code and a clear bottleneck is useful. Kernel throughput, trained-model quality, inter-node scalability, and novelty require separate evidence.
