"""Three explicit kernel swap points in pinned inference source checkouts.

Run --help locally. Runtime selection is read once when an engine module imports.
Audit mode is for eager smoke runs only; leave it unset for measurements.
"""

import argparse
from functools import wraps
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
TARGETS = {
    "vllm": (
        "vllm/v1/attention/backends/flash_attn.py",
        "import vllm.envs as envs\n",
        'from server_experiment import select as _dist_select\n'
        'if is_flash_attn_varlen_func_available():\n'
        '    flash_attn_varlen_func = _dist_select("vllm_attention", flash_attn_varlen_func)\n\n',
    ),
    "DeepInfer": (
        "deepinfer/model_executor/layers/attention_backend.py",
        None,
        '\n\nfrom server_experiment import select as _dist_select\n'
        'prefill_attn_varlen_sparse = _dist_select("deepinfer_prefill", prefill_attn_varlen_sparse)\n'
        'decode_mla_mqa_sparse = _dist_select("deepinfer_decode", decode_mla_mqa_sparse)\n',
    ),
}


def select(target, native):
    """Choose a callable with the exact native ABI, with no automatic fallback."""
    variable = "DIST_" + target.upper()
    spec = os.environ.get(variable, "stock")
    selected = native
    if spec != "stock":
        module, separator, name = spec.partition(":")
        if not separator or not module or not name:
            raise ValueError(f"{variable} must be stock or module:function")
        selected = getattr(importlib.import_module(module), name)
        if not callable(selected):
            raise TypeError(f"{spec} is not callable")
    audit = os.environ.get("DIST_KERNEL_AUDIT")
    if not audit:
        return selected

    directory = Path(audit)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{target}-{os.getpid()}.jsonl"
    source = inspect.getsourcefile(selected)
    record = {
        "target": target, "selection": spec, "pid": os.getpid(),
        "rank": os.environ.get("RANK"), "local_rank": os.environ.get("LOCAL_RANK"),
        "native": f"{native.__module__}:{native.__qualname__}",
        "native_signature": str(inspect.signature(native)),
        "native_source": inspect.getsourcefile(native),
        "selected_source": source,
        "selected_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest()
        if source and Path(source).is_file() else None,
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with path.open("a") as f:
        f.write(json.dumps({"event": "selected", **record}) + "\n")

    def describe(value):
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            stride = getattr(value, "stride", None)
            return {"shape": list(value.shape), "dtype": str(value.dtype),
                    "device": str(getattr(value, "device", None)),
                    "stride": list(stride()) if callable(stride) else None}
        if isinstance(value, dict):
            return {str(k): describe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [describe(v) for v in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return {"type": type(value).__qualname__}

    @wraps(selected)
    def traced(*args, **kwargs):
        result = selected(*args, **kwargs)
        with path.open("a") as f:
            f.write(json.dumps({
                "event": "call_returned", "target": target, "selection": spec,
                "args": describe(args), "kwargs": describe(kwargs),
                "output": describe(result),
            }) + "\n")
        return result

    return traced


def patch(engine, action, root=ROOT):
    """Refuse unknown revisions or local edits; restore only our exact changes."""
    root = Path(root)
    repo = root / "upstream" / engine
    relative, anchor, insertion = TARGETS[engine]
    lock = json.loads((root / "upstream.json").read_text())["repos"][engine]
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != lock["commit"]:
        raise RuntimeError("Checkout differs from upstream.json; re-inspect before updating the pin")
    original = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:{relative}"])
    source = original.decode()
    if anchor:
        if source.count(anchor) != 1:
            raise RuntimeError("Swap anchor is no longer unique")
        modified = source.replace(anchor, insertion + anchor, 1).encode()
    else:
        modified = (source + insertion).encode()
    path = repo / relative
    current = path.read_bytes()
    if current not in (original, modified):
        raise RuntimeError(f"Refusing to overwrite local edits: {path}")
    if action in ("apply", "restore"):
        wanted = modified if action == "apply" else original
        path.write_bytes(wanted)
        current = wanted
    return {"engine": engine, "commit": commit, "path": str(path),
            "patched": current == modified,
            "original_sha256": hashlib.sha256(original).hexdigest(),
            "current_sha256": hashlib.sha256(current).hexdigest()}


def reference_attention(
    *, q, k, v, out, cu_seqlens_q, seqused_k, block_table,
    softmax_scale, causal, window_size=None, softcap=0, **options,
):
    """Slow eager paged-GQA reference for smoke testing a real kernel replacement.

    This deliberately uses FP32 PyTorch math, not the native attention function.
    Replace this body with a CuTe launch after preserving the same ABI.
    """
    import torch

    ignored = {
        "max_seqlen_q", "max_seqlen_k", "scheduler_metadata", "fa_version",
        "q_descale", "k_descale", "v_descale", "num_splits",
    }
    absent = {"alibi_slopes", "dynamic_causal", "s_aux", "mask_mod", "aux_tensors"}
    if set(options) - ignored - absent or any(options.get(key) is not None for key in absent):
        raise NotImplementedError("Reference supports ordinary dense causal attention only")
    if causal is not True or window_size not in (None, [-1, -1], (-1, -1)) or softcap != 0:
        raise NotImplementedError("Reference supports full causal attention without softcap")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise NotImplementedError("Reference supports BF16 Q/K/V; BF16 descales are unused")
    if q.ndim != 3 or k.ndim != 4 or k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        raise NotImplementedError("Expected Q[T,H,D] and paged K/V[P,B,Hkv,D]")
    if q.shape[1] % k.shape[2] or q.shape[0] > 512:
        raise NotImplementedError("Unsupported head sharing or large reference workload")
    if q.is_cuda and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError("This reference intentionally requires eager execution")
    offsets, lengths = cu_seqlens_q.tolist(), seqused_k.tolist()
    if offsets[-1] != q.shape[0] or len(offsets) != len(lengths) + 1:
        raise ValueError("Query offsets do not cover the input")
    page_size, groups = k.shape[1], q.shape[1] // k.shape[2]
    for batch, (start, end, length) in enumerate(zip(offsets[:-1], offsets[1:], lengths, strict=True)):
        if end == start:
            continue
        if not 0 < end - start <= length <= 512:
            raise NotImplementedError("Reference requires nonempty causal prefixes up to 512 tokens")
        pages = block_table[batch, : (length + page_size - 1) // page_size].long()
        keys = k.index_select(0, pages).flatten(0, 1)[:length].transpose(0, 1).float()
        values = v.index_select(0, pages).flatten(0, 1)[:length].transpose(0, 1).float()
        keys, values = (t.repeat_interleave(groups, dim=0) for t in (keys, values))
        scores = q[start:end].transpose(0, 1).float() @ keys.transpose(-1, -2)
        scores *= softmax_scale
        key_positions = torch.arange(length, device=q.device)
        query_positions = torch.arange(end - start, device=q.device) + length - (end - start)
        mask = key_positions[None, :] <= query_positions[:, None]
        result = scores.masked_fill(~mask, float("-inf")).softmax(-1) @ values
        out[start:end].copy_(result.transpose(0, 1).to(out.dtype))
    return out


def smoke(output, selection="stock"):
    """Exercise real HTTP inference and compare all vocabulary logprobs."""
    from importlib.metadata import distributions, version
    import signal
    import sys
    import time
    import traceback
    import urllib.request

    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {"rank": 0, "world": 1, "stage": "server", "ok": False,
              "selection": selection, "performance_claim": False}
    server = None
    try:
        result["environment"] = {
            "packages": {p: version(p) for p in ("torch", "vllm", "transformers")},
            "package_versions": {p.metadata["Name"]: p.version for p in distributions()},
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
        }
        torch.set_num_threads(2)
        torch.manual_seed(0)
        config = LlamaConfig(
            vocab_size=128, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=128, max_position_embeddings=512,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )
        model = LlamaForCausalLM(config).eval().to(torch.bfloat16)
        checkpoint = output / "model"
        model.save_pretrained(checkpoint)
        tokenizer = Tokenizer(models.WordLevel(
            {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
             **{f"t{i}": i for i in range(4, 128)}}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, bos_token="[BOS]", eos_token="[EOS]",
            pad_token="[PAD]", unk_token="[UNK]",
        ).save_pretrained(checkpoint)
        result["model_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in checkpoint.iterdir() if p.is_file()
        }
        model = model.float()
        model.config._attn_implementation = "eager"
        env = os.environ | {
            "DIST_VLLM_ATTENTION": selection,
            "DIST_KERNEL_AUDIT": str(output / "audit"),
            "HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1",
        }
        command = [
            sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(checkpoint),
            "--served-model-name", "tiny-gqa", "--host", "127.0.0.1", "--port", "8000",
            "--attention-backend", "FLASH_ATTN", "--dtype", "bfloat16",
            "--enforce-eager", "--compilation-config", '{"mode":0}',
            "--no-enable-prefix-caching", "--max-model-len", "512",
            "--max-num-batched-tokens", "256", "--max-num-seqs", "4",
            "--gpu-memory-utilization", "0.1", "--num-gpu-blocks-override", "64",
            "--generation-config", "vllm", "--max-logprobs", "128", "--seed", "0",
        ]
        result["command"] = command
        # The outer launcher owns the process group and complete stdout log.
        server = subprocess.Popen(command, env=env)
        deadline = time.monotonic() + 360
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"Server exited before readiness: {server.returncode}")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Server readiness exceeded 360 seconds")
                time.sleep(1)

        cases = result["cases"] = []
        for length in (17, 129, 257, 17):
            prompt = [4 + i % 124 for i in range(length)]
            body = {"model": "tiny-gqa", "prompt": prompt, "temperature": 0,
                    "max_tokens": 4, "ignore_eos": True, "seed": 0,
                    "logprobs": 128, "return_token_ids": True,
                    "return_tokens_as_token_ids": True}
            req = urllib.request.Request(
                "http://127.0.0.1:8000/v1/completions", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=45) as response:
                data = json.load(response)
            choice = data["choices"][0]
            tokens = choice["token_ids"]
            assert len(tokens) == data["usage"]["completion_tokens"] == 4
            actual, reference = [], []
            with torch.inference_mode():
                for step in range(4):
                    logits = model(torch.tensor([prompt + tokens[:step]])).logits[0, -1]
                    reference.append(logits.log_softmax(-1))
                    table = choice["logprobs"]["top_logprobs"][step]
                    actual.append(torch.tensor([table[f"token_id:{i}"] for i in range(128)]))
            actual, reference = torch.stack(actual), torch.stack(reference)
            cases.append({"prompt_tokens": length, "generated_token_ids": tokens,
                          "max_abs_logprob_error": (actual - reference).abs().max().item(),
                          "response": data, "reference_logprobs": reference.tolist()})
            torch.testing.assert_close(actual, reference, rtol=0, atol=0.04)
        assert cases[0]["generated_token_ids"] == cases[-1]["generated_token_ids"]
        audits = {p.name: [json.loads(line) for line in p.read_text().splitlines()]
                  for p in (output / "audit").glob("*.jsonl")}
        calls = [r for records in audits.values() for r in records
                 if r["event"] == "call_returned"]
        phases = {"decode" if r["kwargs"]["max_seqlen_q"] == 1 else "prefill"
                  for r in calls}
        assert phases == {"prefill", "decode"}, f"Missing real kernel calls: {phases}"
        result.update(ok=True, cases=cases, audits=audits, observed_phases=sorted(phases),
                      reference="CPU FP32 eager HF model with the same BF16-rounded weights",
                      logprob_atol=0.04)
    except Exception:
        result["error"] = traceback.format_exc()
        raise
    finally:
        if "audits" not in result:
            result["audits"] = {p.name: p.read_text()
                                for p in (output / "audit").glob("*.jsonl")}
        if server is not None and server.poll() is None:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        (output / "rank-0.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("apply", "restore", "status", "smoke"))
    parser.add_argument("engine", choices=tuple(TARGETS))
    parser.add_argument("--output", default="results/server-local")
    parser.add_argument("--selection", default="stock")
    args = parser.parse_args()
    if args.action == "smoke":
        if args.engine != "vllm":
            parser.error("The HTTP smoke currently targets vLLM")
        print(json.dumps({"ok": smoke(args.output, args.selection)["ok"]}))
    else:
        print(json.dumps(patch(args.engine, args.action), indent=2))
