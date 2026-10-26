# dist-megamind

A small uv project for trying CuTe DSL kernels on B200 GPUs through Modal or an existing Verda cluster. Edit a kernel, run one short experiment, inspect the saved result, repeat.

The inference-engine experiment guide is [serving.md](serving.md): it digests the recovered OpenCode session, maps vLLM/DeepInfer call paths, and provides reversible kernel selections and comparison statistics. Full trajectory exports live in this root; pinned source clones live under `upstream/`. A stock vLLM server and an independent reference replacement have both passed real HTTP inference on one B200; DeepInfer's hooks have CPU coverage.

The research goal is to reproduce in-kernel context-parallel attention on **one machine with four B200s**: split the sequence across GPUs, overlap K/V transfers with attention, and migrate whole output tiles to balance causal work. DeepSeek Sparse Attention (DSA) and original Gated DeltaNet (GDN) are later extensions.

Verda setup is in [verda.md](verda.md): the official Python SDK is installed through uv, with a local `.env`, an inventory client, architecture notes and first single-worker kernel checks. Account authentication awaits your keys.

## Files

- `kernels.py`: the kernels to edit; starts with vector addition and a tiny NCCL LSA reduction.
- `experiment.py`: GPU checks and PyTorch references; one process per GPU.
- `modal_app.py`: isolated research/server images, GPU allocation, bounded launchers and local result download.
- `experiments.md`: runnable experiment matrix and the approach for adding experiments.
- `serving.md` / `server_experiment.py`: native inference paths, minimal source hooks and measurement protocol.
- `test_server_experiment.py`: CPU selection/patching checks.
- `verda_cli.py` / `test_verda_cli.py`: SDK inventory commands and offline HTTP-contract checks.
- `verda.md` / `.env.example`: architecture, setup and credential template.
- `upstream.json`: clone pins and trajectory provenance.
- `pyproject.toml` / `uv.lock`: local and GPU dependencies.
- `.gitignore`: environments and generated results stay out of version control.

No package hierarchy, configuration framework, persistent volumes, deployment, or optimized distributed attention implementation. CLI arguments are the configuration. The previous plan, runbook, review, sources, session log, YAML matrix, and HTML roadmap have been consolidated into README.md and experiments.md; serving.md adds the inference-engine investigation.

## Setup

Requires uv and Python 3.12. Run from this directory:

```bash
uv sync --frozen
uv run modal --version
uv run modal token new
```

The last command connects your Modal account through its browser authentication flow. Skip it if already authenticated. Credentials stay in Modal's normal local configuration; do not put tokens in this repo.

Only Modal is installed by default on the laptop. Install the Verda control client with `uv sync --frozen --extra verda`; follow [verda.md](verda.md) before GPU execution. The Modal image runs `uv sync` with the `gpu` extra to install PyTorch, CuTe DSL, CUDA Python, and NCCL4Py from the same lockfile. A local GPU or CUDA toolkit is unnecessary. For development on a Linux CUDA machine, optionally use `uv sync --frozen --extra gpu` and `uv run --extra gpu python experiment.py cute`.

The candidate stack is Python 3.12, CUDA 13, PyTorch 2.14.1, CuTe DSL 4.8.1, NCCL4Py 0.6.0, and NCCL 2.32.3. The base image is `nvidia/cuda:13.0.2-devel-ubuntu24.04`. The image is also pinned to its registry digest in the launcher.

**Dependency override:** PyTorch pins NCCL 2.30.7; the selected NCCL4Py device interface requires 2.32.3. `tool.uv.override-dependencies` deliberately selects 2.32.3 for both. Dependency resolution succeeds, but runtime compatibility is an experiment. Each worker checks the loaded NCCL version/location, PyTorch's NCCL version, and matching wheel-provided device bitcode before checking kernels. Remove the override only when changing to an explicitly compatible stack.

## Check compilation without a GPU

This installs the compiler/NCCL tools into an isolated uv environment and compiles both kernels for B200 without allocating a GPU:

```bash
uv run --with "nccl4py[cu13]==0.6.0" \
  --with "nvidia-nccl-cu13==2.32.3" \
  --with "nvidia-cutlass-dsl[cu13]==4.8.1" \
  python experiment.py compile
```

Inspect `results/local/rank-0.json`. This fixed local path is overwritten on a repeat; use `--output results/my-compile-check` to keep a separate record. Compilation does not execute either kernel or establish runtime correctness. The current stack emits LLVM data-layout warnings while linking NCCL bitcode; both offline compilations complete successfully. If GPU dependencies are already installed locally, `uv run --extra gpu python experiment.py compile` uses the project extra directly.

## First runs

These commands allocate paid GPUs. Start with one GPU; run each command separately and inspect its result before the next.

```bash
# Inventory, software/library checks, peer-access information.
uv run modal run modal_app.py --stage env --gpus 1

# Compile and validate our CuTe vector-add kernel, including graph replay.
uv run modal run modal_app.py --stage cute --gpus 1

# Fused SDPA: small forward/gradient correctness check plus forward timing.
uv run modal run modal_app.py --stage sdpa --gpus 1 --kv-heads 8 --tokens 256

# Check ordinary NCCL communication, then kernel-side registered-window access.
uv run modal run modal_app.py --stage nccl --gpus 2
uv run modal run modal_app.py --stage symmetric --gpus 2
```

After both communication checks pass, repeat `env`, `cute`, `nccl`, and `symmetric` with `--gpus 4`. One Modal function owns all GPUs, and `torchrun` starts one local rank per GPU. Four separate single-GPU functions would not establish same-machine communication.

For MHA/GQA/MQA baselines, repeat the SDPA command with `--kv-heads 32`, `8`, and `1`. Query heads stay at 32, head dimension at 128, batch at 1, dtype at BF16, and dropout at zero. GQA/MQA describe head sharing within SDPA; they are not separate alternatives to SDPA. The FLASH_ATTENTION backend is forced: unsupported combinations fail instead of silently switching implementations. Stock FA4 is a later, separately installed baseline; this PyTorch check is not an FA4 reproduction.

Use `--n` for vector lengths, `--iterations` for repeat count, and `--seed` for reproducibility. Defaults are 4099 elements, 10 iterations, and seed 0. `--tokens` only changes SDPA's timed sequence length; its independent reference check remains at at most 256 tokens to avoid quadratic reference memory.

## Results and limits

Every launch saves `results/<UTC timestamp>-<id>/` locally:

- `manifest.json`: configuration, source hashes, base image tag, and final status.
- `result.json`: overall status, per-rank environments, correctness details, and raw timing samples where implemented.
- `run.log`: nvidia-smi, topology, worker output, and errors.
- Copies of the three Python files, pyproject, and lockfile used for the run.

A failed correctness check exits nonzero. Worker failures stop sibling ranks through torchrun. The launcher kills the worker process group after 500 seconds; the Modal function has a 600-second timeout and no automatic retries. Image build and scheduling time are outside that worker timeout. These limits bound short experiments, not total account spending.

Results return directly to the CLI; no Modal volume or download command is needed. If the connection or container dies before return, some remote logs may only be available in Modal's dashboard. The local manifest records a launch/remote error when the CLI can catch it. For long experiments, add a volume only when this becomes necessary.

The symmetric probe uses **host barriers** around immutable inputs and validates repeated LSA sums through NCCL-owned registered windows. Passing it establishes a useful API/platform foothold. It does not establish device-side ready/ack ordering, communication/computation overlap, distributed CUDA graph safety, or attention correctness.

## Full project context

The starting point is [this shared design artifact](https://claude.ai/artifact/FDALXXGSUrBoechinsDvQv). It identifies `dist_attention`, branch `in-kernel-attention`, commit `51d96cd`, but its original repository URL has not been obtained. Reported roughly 3.5–3.9x four-GPU scaling is an author-reported comparison target, not our result. The source contains some inconsistent speedup tables; do not treat them as acceptance thresholds.

The selected immediate milestone is a **two-hour feasibility investigation**. This scaffold supports initial environment, arithmetic, baseline, collective, and registered-window checks. It does not complete the whole feasibility investigation until real hardware runs and further ordering/progress probes pass. No dollar ceiling was supplied. Historical planning estimated a multi-week to multi-month implementation and roughly 67–132 four-GPU allocation-hours for the extended scope; those are uncertain estimates, not an approved budget. Check current Modal rates before running.

The intended dense path is:

1. Pin and validate single-GPU FA4 and an independent small high-precision attention reference.
2. Implement contiguous sequence shards and in-kernel K/V push forwarding.
3. Prove epochs, ready signals, acknowledgements, repeated-call reuse, and CUDA graph behavior.
4. Add backward K/V pre-push and owner-directed dK/dV accumulation.
5. Migrate complete output-row tiles, including return traffic and correct gradient ownership.
6. Preserve packed-document boundaries and eventually generate backward work plans on GPU.

Causal work is unequal across contiguous shards; later ranks attend to more history. Load migration must repay query/key transfers, output returns, and scheduling overhead. Compare against all-gather plus FA4 and a strong balanced ring/zigzag baseline. A win over a deliberately unbalanced baseline is insufficient.

Shared infrastructure should use NCCL-owned symmetric windows and same-node load/store-accessible (LSA) peer memory. NCCL symmetric registration supplies addressing, not synchronization; it is not a drop-in NVSHMEM superset. Use CuTe DSL for custom GPU compute and communication kernels. Begin with BF16 training, one stream per context, allocations outside capture, and no dropout. Defer multi-node transport, FP8, dense decode, arbitrary masks, and compiler integration.

DSA requires its learned indexer, exact global top-k, sparse KV routing, and backward. A supplied-index sparse kernel is only a partial operator. Include indexer/routing costs; the indexer may still have quadratic work. Trainable indexers need the reference auxiliary KL objective and explicit detach boundaries.

GDN has a recurrent matrix state and needs a different algorithm: chunked local computation, boundary-state baseline, affine shard-summary composition, reverse-state gradients, document resets, and convolution boundary halos. Compare with existing FLA context parallelism. Share setup and measurement infrastructure, not dense attention's migration algorithm.

Begin with seeded synthetic activations. Later use frozen FineWeb-Edu document lengths and packing manifests; real text lengths with random activations measure packing behavior, not trained-model quality. Small decoder-block training can check gradient plumbing. Quality claims across dense attention, DSA, and GDN require trained/adapted models and controlled training/evaluation budgets.

## Correctness and profiling essentials

- For local queries attending a longer global key prefix, construct masks from global positions and document starts. Unchecked non-square `is_causal=True` can align the wrong causal triangle.
- Compare outputs and gradients, including empty/padded regions, cross-shard documents, cancellation, and repeated calls. Bitwise equality is only meaningful against an identical pinned specialization; four-ULP gradients are a target from the source, not a universal guarantee.
- Before asynchronous reuse, prove payload completion, ordering, ready publication, consumer reads, and acknowledgement. Test delayed ranks and grids larger than available SMs; waiting CTAs can deadlock unscheduled producers.
- Define global summed loss or global valid-token mean explicitly. Do not divide gradients by world size twice.
- Count NCCL/native buffers, staging, deterministic gradient slots, and saved tensors in memory estimates. Context parallelism does not divide every allocation by GPU count.
- For distributed timings, hold global input fixed and report the slowest rank's complete-call latency. Include communication, conversions, planning, and returns. Warm compilation first and save raw samples.
- A real `.ncu-rep` requires live Nsight Compute collection. If Modal denies counters/collection, retain CUDA-event timings and optionally sampled device timestamps or CuTe IKET traces. Timing JSON cannot be converted into an NCU report. Another host's capture must be labeled external evidence. Avoid profiler replay modes that serialize mutually dependent GPU kernels.

## Provenance and current status

The previous package was prepared on 25 October 2026 and recorded an independent design review covering stack compatibility, masks, numerical targets, memory ordering, progress, loss scaling, extension semantics, and profiling. This simplification was made on 26 October 2026. Earlier references to a research-philosophy file pointed outside this folder.

Local checks passed: uv dependency resolution/install, CLI help, Python syntax/lint, compiler/NCCL imports, B200-target compilation of both kernels, and mocked launcher success/failure/timeout/artifact handling. GPU compiler tools were installed in an isolated uv environment; the default project environment contains Modal; the optional `verda` extra adds the Cloud API client. Local validation and installation do not establish remote compatibility. That original scaffold setup did not execute the add/LSA kernels on GPUs. A later inference-engine investigation built the isolated server image and ran stock and reference attention through a real vLLM HTTP server on one B200; both passed full-vocabulary logprob checks against a CPU FP32 model. See serving.md and results/server-comparison.json. Four-GPU topology/communication, custom CuTe attention, training and profiler capture remain unverified.

Useful primary references:

- [Modal GPU allocation](https://modal.com/docs/guide/gpu), [uv image setup](https://modal.com/docs/sdk/py/latest/Image#uv_sync), and [pricing](https://modal.com/pricing).
- [CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html) and [FlashAttention](https://github.com/Dao-AILab/flash-attention).
- [NCCL4Py releases and compatibility](https://github.com/NVIDIA/nccl/releases), [device API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html), and [LSA example](https://github.com/NVIDIA/nccl/blob/nccl4py-v0.6.0/bindings/nccl4py/examples/cute/08_reduce_copy.py).
- [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention), [Ring FlashAttention](https://github.com/zhuzilin/ring-flash-attention), and [Striped Attention](https://arxiv.org/abs/2311.09431).
- [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp), [DSA training description](https://arxiv.org/abs/2512.02556), and [NVIDIA DSA kernels](https://docs.nvidia.com/cudnn/latest/fe-oss-apis/dsa/).
- [Gated DeltaNet](https://github.com/NVlabs/GatedDeltaNet) and [FLA context parallelism](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/cp/README.md).
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), [RULER](https://github.com/NVIDIA/RULER), and [LongBench](https://huggingface.co/datasets/zai-org/LongBench).
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/NsightComputeCli/), [replay constraints](https://docs.nvidia.com/nsight-compute/ProfilingGuide/), and [CuTe IKET](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/iket_profiling.html).
