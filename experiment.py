"""Small experiments launched by torchrun; --help works without GPU packages."""

import argparse
from dataclasses import asdict
from datetime import timedelta
import hashlib
from importlib.metadata import distribution, version
import json
import os
from pathlib import Path
import statistics
import traceback

STAGES = ("env", "cute", "sdpa", "nccl", "symmetric")


def check_compile(args):
    import cutlass
    from cutlass import cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
    from nccl.core.device.cute.runtime import make_fake_window
    from kernels import add, lsa_sum

    stream = make_fake_stream()
    for n in (1, 129, args.n):
        tensor = make_fake_tensor(cutlass.Float32, (n,), (1,))
        cute.compile(add, tensor, tensor, tensor, stream, options="--gpu-arch=sm_100a")
    cute.compile(lsa_sum, make_fake_window(), make_fake_window(), 0, 2,
                 stream, options="--gpu-arch=sm_100a")
    return {"target": "sm_100a", "compiled": ["add", "lsa_sum"], "executed": False}


def environment(rank):
    import torch
    package = distribution("nvidia-nccl-cu13")
    bitcode = next(package.locate_file(f) for f in package.files
                   if str(f).endswith("/libnccl_device.bc"))
    os.environ["NCCL_HOME"] = str(bitcode.parent.parent)
    import nccl.core as nccl

    stack = nccl.get_version()
    if stack.libnccl is None or str(stack.libnccl.version) != "2.32.3":
        raise RuntimeError(f"Expected loaded NCCL 2.32.3, got {stack}")
    if not Path(stack.libnccl.path).resolve().is_relative_to(bitcode.parent.parent.resolve()):
        raise RuntimeError("Loaded NCCL and device bitcode come from different locations")
    if torch.cuda.nccl.version() != (2, 32, 3):
        raise RuntimeError("PyTorch loaded a different NCCL runtime")
    props = torch.cuda.get_device_properties(rank)
    return {
        "packages": {p: version(p) for p in
                     ("torch", "nvidia-cutlass-dsl", "nccl4py", "nvidia-nccl-cu13")},
        "cuda": torch.version.cuda, "nccl": asdict(stack),
        "bitcode": str(bitcode), "bitcode_sha256": hashlib.sha256(bitcode.read_bytes()).hexdigest(),
        "gpu": props.name, "capability": torch.cuda.get_device_capability(rank),
        "memory_bytes": props.total_memory,
        "peer_access": {str(i): torch.cuda.can_device_access_peer(rank, i)
                        for i in range(torch.cuda.device_count()) if i != rank},
    }


def measure(call, iterations):
    import torch
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def check_cute(args):
    import torch
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cuda.bindings.driver import CUstream
    from kernels import add

    cases = []
    for n in dict.fromkeys((1, 127, 128, 129, args.n)):
        a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
        out = torch.full_like(a, float("nan"))
        views = tuple(from_dlpack(t) for t in (a, b, out))
        compiled = cute.compile(add, *views, CUstream(torch.cuda.current_stream().cuda_stream))

        def launch(compiled=compiled, views=views):
            compiled(*views, CUstream(torch.cuda.current_stream().cuda_stream))

        for _ in range(2):
            a.add_(1)
            out.fill_(float("nan"))
            launch()
            torch.cuda.synchronize()
            torch.testing.assert_close(out, a + b, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        for _ in range(3):
            a.add_(1)
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(out, a + b, rtol=0, atol=0)
        cases.append({"n": n, "graph_replays": 3, **measure(launch, args.iterations)})
    return {"cases": cases}


def check_sdpa(args):
    import torch
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    from torch.nn.attention import SDPBackend, sdpa_kernel

    s = min(args.tokens, 256)
    q, k, v = (torch.randn(1, h, s, 128, device="cuda", dtype=torch.bfloat16,
                           requires_grad=True) for h in (32, args.kv_heads, args.kv_heads))
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out = sdpa(q, k, v, is_causal=True, enable_gqa=args.kv_heads != 32)
    qr, kr, vr = (t.detach().float().requires_grad_() for t in (q, k, v))
    kr_full, vr_full = (t.repeat_interleave(32 // args.kv_heads, dim=1) for t in (kr, vr))
    scores = (qr @ kr_full.transpose(-1, -2)) / (128 ** 0.5)
    mask = torch.ones(s, s, device="cuda", dtype=torch.bool).tril()
    ref = scores.masked_fill(~mask, float("-inf")).softmax(-1) @ vr_full
    torch.testing.assert_close(out.float(), ref, rtol=0.02, atol=0.02)
    grad = torch.randn_like(out)
    actual_grads = torch.autograd.grad(out, (q, k, v), grad)
    ref_grads = torch.autograd.grad(ref, (qr, kr, vr), grad.float())
    gradient_errors = []
    for actual, expected in zip(actual_grads, ref_grads, strict=True):
        torch.testing.assert_close(actual.float(), expected, rtol=0.05, atol=0.05)
        gradient_errors.append((actual.float() - expected).abs().max().item())
    q, k, v = (torch.randn(1, h, args.tokens, 128, device="cuda", dtype=torch.bfloat16)
               for h in (32, args.kv_heads, args.kv_heads))
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        timing = measure(lambda: sdpa(q, k, v, is_causal=True,
                                     enable_gqa=args.kv_heads != 32), args.iterations)
    return {"checked_tokens": s, "timed_tokens": args.tokens, "Hq": 32, "Hkv": args.kv_heads,
            "backend": "FLASH_ATTENTION", "forward_max_abs": (out.float() - ref).abs().max().item(),
            "gradient_max_abs": gradient_errors, "forward_timing": timing}


def check_communication(args, rank, world):
    import torch
    import torch.distributed as dist

    dist.init_process_group("nccl" if args.stage == "nccl" else "gloo",
                            timeout=timedelta(seconds=90))
    if args.stage == "nccl":
        value = torch.empty(args.n, device="cuda")
        for epoch in range(args.iterations):
            value.fill_(rank + 1 + epoch)
            dist.all_reduce(value)
            torch.cuda.synchronize()
            expected = world * (world + 1) / 2 + epoch * world
            torch.testing.assert_close(value, torch.full_like(value, expected), rtol=0, atol=0)
        dist.destroy_process_group()
        return {"epochs": args.iterations, "elements": args.n}

    import nccl.core as nccl
    import nccl.core.device.cute as nccl_cute
    from cuda.bindings.driver import CUstream
    from kernels import lsa_sum

    uid = [bytes(nccl.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(uid)
    comm = nccl.Communicator.init(nranks=world, rank=rank,
                                  unique_id=nccl.UniqueId.from_bytes(uid[0]))
    team = comm.team_lsa
    if team.n_ranks != world:
        raise RuntimeError(f"LSA team covers {team.n_ranks}/{world} ranks")
    src, dst = (nccl.torch.empty(16, dtype=torch.float32, device=rank) for _ in range(2))
    src_window, dst_window = comm.register_window(src), comm.register_window(dst)
    if not src_window.is_valid or not dst_window.is_valid:
        raise RuntimeError("NCCL registered windows unavailable")
    for epoch in range(args.iterations):
        src.fill_(team.rank + 1 + epoch)
        dst.fill_(float("nan"))
        torch.cuda.synchronize()
        dist.barrier()  # Finish writes before peer reads.
        lsa_sum(nccl_cute.Window(src_window), nccl_cute.Window(dst_window),
                team.rank, team.n_ranks, CUstream(torch.cuda.current_stream().cuda_stream))
        torch.cuda.synchronize()
        expected = world * (world + 1) / 2 + epoch * world
        torch.testing.assert_close(dst, torch.full_like(dst, expected), rtol=0, atol=0)
        dist.barrier()  # Finish all peer reads before input reuse.
    src_window.close()
    dst_window.close()
    comm.destroy()
    dist.destroy_process_group()
    return {"lsa_ranks": world, "elements": 16, "epochs": args.iterations,
            "synchronization": "host barriers; no device flag/overlap claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("compile", *STAGES))
    parser.add_argument("--n", type=int, default=4099)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--kv-heads", type=int, choices=(32, 8, 1), default=8)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/local"))
    args = parser.parse_args()
    if min(args.n, args.tokens, args.iterations) < 1:
        parser.error("n, tokens and iterations must be positive")
    rank, world = int(os.environ.get("LOCAL_RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"rank-{rank}.json"
    result = {"rank": rank, "world": world, "stage": args.stage, "ok": False,
              "config": vars(args).copy()}
    path.write_text(json.dumps(result, default=str))
    try:
        if args.stage == "compile":
            result["result"] = check_compile(args)
        else:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU required. Use the Modal commands in README.md.")
            torch.cuda.set_device(rank)
            torch.manual_seed(args.seed + rank)
            result["environment"] = environment(rank)
            if args.stage == "cute":
                result["result"] = check_cute(args)
            elif args.stage == "sdpa":
                result["result"] = check_sdpa(args)
            elif args.stage in ("nccl", "symmetric"):
                if world < 2:
                    raise ValueError("Communication checks need at least two ranks")
                result["result"] = check_communication(args, rank, world)
        result["ok"] = True
    except Exception:
        result["error"] = traceback.format_exc()
        raise
    finally:
        path.write_text(json.dumps(result, indent=2, default=str))
        print("rank={} stage={} ok={}".format(rank, args.stage, result["ok"]), flush=True)


if __name__ == "__main__":
    main()
