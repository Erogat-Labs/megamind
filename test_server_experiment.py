"""CPU contract checks; these do not execute GPU kernels or load models."""

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import server_experiment as se


def native(value, *, scale=1):
    return value * scale


class KernelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_default_and_candidate_have_no_dispatch_wrapper(self):
        self.assertIs(se.select("vllm_attention", native), native)
        with patch.dict(os.environ, DIST_VLLM_ATTENTION="test_server_experiment:native"):
            self.assertIs(se.select("vllm_attention", native), native)

    def test_invalid_candidate_fails_without_stock_fallback(self):
        for spec, error in [
            ("bad", ValueError), ("missing_kernel_module_xyz:run", ModuleNotFoundError),
            ("test_server_experiment:missing", AttributeError),
            ("test_server_experiment:__name__", TypeError),
        ]:
            with self.subTest(spec=spec), patch.dict(os.environ, DIST_VLLM_ATTENTION=spec):
                with self.assertRaises(error):
                    se.select("vllm_attention", native)

    def test_audit_distinguishes_selection_from_successful_call(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, DIST_KERNEL_AUDIT=directory):
                selected = se.select("vllm_attention", native)
                self.assertEqual(selected(4, scale=3), 12)
            records = [json.loads(line) for p in Path(directory).glob("*.jsonl")
                       for line in p.read_text().splitlines()]
            self.assertEqual([r["event"] for r in records], ["selected", "call_returned"])
            self.assertEqual(records[0]["selection"], "stock")
            self.assertEqual(records[1]["kwargs"], {"scale": 3})
            self.assertEqual(records[1]["output"], 12)

    def test_candidate_errors_propagate(self):
        def broken(*args, **kwargs):
            raise RuntimeError("kernel failed")
        with patch.dict(sys.modules, fake_candidate=types.SimpleNamespace(run=broken)):
            with patch.dict(os.environ, DIST_VLLM_ATTENTION="fake_candidate:run"):
                with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                    se.select("vllm_attention", native)(4)

    def test_actual_deepinfer_module_dispatches_stock_and_candidate(self):
        class Tensor:
            shape = (3, 64, 128)
            def unsqueeze(self, dim):
                return self

        def flash(*args, **kwargs):
            return "output", "max_logits", "lse"

        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = Tensor
        flash_module = types.ModuleType("flash_mla")
        flash_module.flash_mla_sparse_fwd = flash
        flash_module.flash_mla_with_kvcache = flash
        flash_module.fused_norm_rope_attn_rope_cast = None
        interface = types.ModuleType("flash_mla.flash_mla_interface")
        interface.FlashMLASchedMeta = type("FlashMLASchedMeta", (), {})
        path = se.ROOT / "upstream/DeepInfer" / se.TARGETS["DeepInfer"][0]
        with patch.dict(sys.modules, {"torch": fake_torch, "flash_mla": flash_module,
                                     "flash_mla.flash_mla_interface": interface}):
            stock = runpy.run_path(str(path))["prefill_attn_varlen_sparse"]
            self.assertEqual(stock(Tensor(), Tensor(), Tensor(), 1.0), ("output", "lse"))
            candidate = lambda *args, **kwargs: ("candidate", "candidate_lse")
            with patch.dict(sys.modules, fake_candidate=types.SimpleNamespace(run=candidate)):
                with patch.dict(os.environ, DIST_DEEPINFER_PREFILL="fake_candidate:run"):
                    selected = runpy.run_path(str(path))["prefill_attn_varlen_sparse"]
                    self.assertIs(selected, candidate)
                    self.assertEqual(selected(None), ("candidate", "candidate_lse"))


class SourcePatchTests(unittest.TestCase):
    def test_both_pinned_sources_apply_restore_and_refuse_local_edits(self):
        for engine, (relative, _, _) in se.TARGETS.items():
            with self.subTest(engine=engine), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                repo = root / "upstream" / engine
                repo.parent.mkdir()
                subprocess.run(
                    ["git", "clone", "-q", "--shared", "--no-checkout",
                     str(se.ROOT / "upstream" / engine), str(repo)], check=True
                )
                (root / "upstream.json").write_bytes((se.ROOT / "upstream.json").read_bytes())
                source = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:{relative}"])
                path = repo / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source)
                self.assertFalse(se.patch(engine, "status", root)["patched"])
                self.assertTrue(se.patch(engine, "apply", root)["patched"])
                compiled = path.read_text()
                compile(compiled, str(path), "exec")
                self.assertTrue(se.patch(engine, "apply", root)["patched"])
                path.write_text(compiled + "\n# unrelated edit\n")
                with self.assertRaisesRegex(RuntimeError, "local edits"):
                    se.patch(engine, "restore", root)
                self.assertTrue(path.read_text().endswith("# unrelated edit\n"))
                path.write_text(compiled)
                self.assertFalse(se.patch(engine, "restore", root)["patched"])
                self.assertEqual(path.read_bytes(), source)
                lock = json.loads((root / "upstream.json").read_text())
                lock["repos"][engine]["commit"] = "wrong_revision"
                (root / "upstream.json").write_text(json.dumps(lock))
                with self.assertRaisesRegex(RuntimeError, "differs"):
                    se.patch(engine, "apply", root)


if __name__ == "__main__":
    unittest.main()
