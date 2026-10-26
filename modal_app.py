"""Launch one short experiment with: uv run modal run modal_app.py --stage cute"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback
import uuid

import modal

ROOT = Path(__file__).resolve().parent
BASE_IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6"
STAGES = ("env", "cute", "sdpa", "nccl", "symmetric", "server")
app = modal.App("dist-megamind")
image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.12")
    .uv_sync(uv_project_dir=str(ROOT), extras=["gpu"], uv_version="0.5.29")
    .env({"OMP_NUM_THREADS": "1", "MAX_JOBS": "2", "NCCL_DEBUG": "WARN",
          "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1"})
    .add_local_file(ROOT / "experiment.py", "/root/experiment.py")
    .add_local_file(ROOT / "kernels.py", "/root/kernels.py")
)


SERVER_WHEEL = (
    "https://wheels.vllm.ai/a7407ab44557802ca32a0ba2c2707362d7d21563/"
    "vllm-0.32.1rc1.dev491%2Bga7407ab44-cp38-abi3-manylinux_2_28_x86_64.whl"
)
server_image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.12")
    .apt_install("libnuma1", "libgomp1")
    .pip_install(SERVER_WHEEL, extra_index_url="https://download.pytorch.org/whl/cu130")
    .env({"OMP_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1",
          "VLLM_NO_USAGE_STATS": "1", "PYTHONPATH": "/root"})
    .add_local_file(ROOT / "server_experiment.py", "/root/server_experiment.py")
    .add_local_file(ROOT / "kernels.py", "/root/kernels.py")
    .add_local_file(
        ROOT / "upstream/vllm/vllm/v1/attention/backends/flash_attn.py",
        "/root/flash_attn_hook.py",
    )
)


def validate(stage, gpus, tokens, kv_heads, n, iterations):
    if stage not in STAGES or gpus not in (1, 2, 4):
        raise ValueError(f"stage must be one of {STAGES}; gpus must be 1, 2 or 4")
    if stage in ("nccl", "symmetric") and gpus < 2:
        raise ValueError("Communication checks need --gpus 2 or --gpus 4")
    if stage in ("sdpa", "server") and gpus != 1:
        raise ValueError("SDPA/server smoke use --gpus 1")
    if min(tokens, n, iterations) < 1 or kv_heads not in (32, 8, 1):
        raise ValueError("Use positive sizes/iterations and --kv-heads 32, 8 or 1")


@app.function(image=image, gpu="B200", cpu=4, memory=16384, timeout=600, retries=0)
def remote(config: dict):
    import os
    import signal
    import subprocess
    import sys
    import tempfile

    output = Path(tempfile.mkdtemp())
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        f"--nproc-per-node={config['gpus']}", "--max-restarts=0",
        "/root/experiment.py", config["stage"], "--output", str(output),
    ]
    for name in ("tokens", "kv_heads", "n", "iterations", "seed"):
        command.extend(["--" + name.replace("_", "-"), str(config[name])])
    if config["stage"] == "server":
        command = [sys.executable, "/root/server_experiment.py", "smoke", "vllm",
                   "--output", str(output), "--selection", config["selection"]]
    timed_out = False
    with (output / "run.log").open("w") as log:
        for args in (["nvidia-smi"], ["nvidia-smi", "topo", "-m"]):
            subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, timeout=20, check=False)
        log.flush()
        worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        try:
            worker.wait(timeout=500)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Also stop surviving grandchildren after a worker or launcher failure.
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            worker.wait()
    ranks = [json.loads(p.read_text()) for p in sorted(output.glob("rank-*.json"))]
    return {
        "ok": worker.returncode == 0 and not timed_out and len(ranks) == config["gpus"]
              and all(r["ok"] for r in ranks),
        "exit_code": worker.returncode, "timed_out": timed_out, "ranks": ranks,
        "log": (output / "run.log").read_text(), "command": command,
    }


@app.function(image=server_image, gpu="B200", cpu=8, memory=32768, timeout=600, retries=0)
def server_remote(config: dict):
    import importlib.util

    package = Path(importlib.util.find_spec("vllm").origin).parent
    target = package / "v1/attention/backends/flash_attn.py"
    original_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    if original_hash not in (config["hook_original_sha256"], config["hook_current_sha256"]):
        raise RuntimeError("Installed wheel's attention source differs from our pinned checkout")
    hook = Path("/root/flash_attn_hook.py").read_bytes()
    if hashlib.sha256(hook).hexdigest() != config["hook_current_sha256"]:
        raise RuntimeError("Uploaded attention hook differs from the launch manifest")
    target.write_bytes(hook)
    result = remote.local(config)
    result["hook"] = {"installed_before_sha256": original_hash,
                      "installed_after_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                      "path": str(target)}
    return result


@app.local_entrypoint()
def run(stage: str = "cute", gpus: int = 1, tokens: int = 256, kv_heads: int = 8,
        n: int = 4099, iterations: int = 10, seed: int = 0, selection: str = "stock"):
    validate(stage, gpus, tokens, kv_heads, n, iterations)
    config = dict(stage=stage, gpus=gpus, tokens=tokens, kv_heads=kv_heads,
                  n=n, iterations=iterations, seed=seed, selection=selection)
    if stage == "server":
        from server_experiment import patch
        state = patch("vllm", "status")
        if not state["patched"]:
            raise ValueError("Apply the vLLM hook before running the server stage")
        if seed != 0:
            raise ValueError("The bounded server smoke currently uses fixed seed 0")
        config["server_workload"] = {"seed": 0, "prompt_lengths": [17, 129, 257, 17],
                                     "output_tokens_per_request": 4, "query_heads": 2,
                                     "kv_heads": 1, "head_dim": 128, "layers": 2}
        config.update(hook_original_sha256=state["original_sha256"],
                      hook_current_sha256=state["current_sha256"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    output = ROOT / "results" / run_id
    output.mkdir(parents=True, exist_ok=False)
    files = ("modal_app.py", "experiment.py", "kernels.py", "pyproject.toml", "uv.lock",
             "server_experiment.py", "upstream.json")
    hashes = {}
    for name in files:
        source = (ROOT / name).read_bytes()
        (output / name).write_bytes(source)
        hashes[name] = hashlib.sha256(source).hexdigest()
    if stage == "server":
        (output / "flash_attn_hook.py").write_bytes(
            (ROOT / "upstream/vllm/vllm/v1/attention/backends/flash_attn.py").read_bytes())
    manifest = {"run_id": run_id, "config": config, "source_sha256": hashes,
                "base_image": BASE_IMAGE,
                "image_id": (server_image if stage == "server" else image).object_id,
                "server_wheel": SERVER_WHEEL if stage == "server" else None,
                "status": "started"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Running {stage} on B200:{gpus}; results: {output}")
    try:
        function = server_remote if stage == "server" else remote
        result = function.with_options(gpu=f"B200:{gpus}", cpu=8 if stage == "server" else 4 * gpus,
                                       memory=32768 if stage == "server" else 16384 * gpus).remote(config)
    except Exception:
        manifest.update(status="launch_or_remote_error", error=traceback.format_exc())
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        raise
    (output / "run.log").write_text(result.pop("log"))
    (output / "result.json").write_text(json.dumps(result, indent=2))
    manifest["status"] = "passed" if result["ok"] else "failed"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{manifest['status'].upper()}: {output}")
    if not result["ok"]:
        print((output / "run.log").read_text()[-6000:])
        raise RuntimeError(f"Experiment failed. See {output}")
