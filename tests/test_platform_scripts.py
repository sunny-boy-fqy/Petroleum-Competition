#!/usr/bin/env python3
"""平台脚本与 E0 证据的静态契约测试（R3-M3：填补三审指出的覆盖空档）。

覆盖：
  * `run_train.sh::run_env` 的 **base/full profile 顺序**（先装依赖、再做 base 硬校验）；
  * `disk_guard` 调用必须显式 `--data-root`（否则 `E0_disk_budget.json::level` 描述的是 `/` 而非 `/data`）；
  * `run_e0` 必须把**全部** `E0_*.json` 回拷到 repo（R3-H5）；
  * `check_env.py` 的 fallback 折文件名必须真实存在；可选依赖必须可降级（R3-M1/M2）；
  * 已提交的 `reports/E0_data_card.json` 其 `shard_cache.cache_root` 必须是**可复现**路径（R3-C3）。
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))


def _read(rel: str) -> str:
    return (V4 / rel).read_text(encoding="utf-8")


class TestRunTrainOrdering(unittest.TestCase):
    """R2-B1：首次 `--mode env` 曾因「先硬校验、后装依赖」而必然失败。"""

    def setUp(self):
        self.src = _read("run_train.sh")
        body = self.src[self.src.index("run_env()"):]
        self.env_body = body[: body.index("\nrun_data()")]

    def test_setup_deps_runs_before_base_check(self):
        i_deps = self.env_body.index("setup_deps.sh")
        i_base = self.env_body.index("check_env_profile base")
        self.assertLess(i_deps, i_base,
                        "run_env 必须先 setup_deps 再做 profile=base 硬校验")

    def test_base_profile_used_then_full_after_data(self):
        self.assertIn("check_env_profile base", self.env_body)
        self.assertIn("check_env_profile full", self.src)
        # run_data 里才做 full 硬校验
        data_body = self.src[self.src.index("run_data()"):]
        data_body = data_body[: data_body.index("\nrun_e0()")]
        self.assertIn("check_env_profile full", data_body)

    def test_disk_guard_checks_data_root_explicitly(self):
        """R3-H2：不带 --data-root 时 level 描述的是 `/`，与 30 GB 配额无关。"""
        for m in re.finditer(r"disk_guard\.py[^\n]*(?:\n[^\n]*)*?(?=\n\s*(?:log|if|else|fi|$))",
                             self.src):
            block = m.group(0)
            if "--json" in block or "--min-free-gb" in block:
                self.assertIn("--data-root", block, block)

    def test_run_e0_copies_all_e0_evidence(self):
        """R3-H5：此前只回拷 3 个文件，data card 与 score_check/prereg 会互相矛盾。"""
        e0_body = self.src[self.src.index("run_e0()"):]
        e0_body = e0_body[: e0_body.index("\nrun_smoke()")]
        self.assertRegex(e0_body, r'for\s+\w+\s+in\s+"\$REPORTS_DIR"/E0_\*\.json')
        for name in ("E0_score_check.json", "E0_gate_prereg.json",
                     "E0_contract_tests.json", "E0_folds.json"):
            # 不再逐个手写 cp，而是由通配循环覆盖
            self.assertNotIn(f'cp -f "$REPORTS_DIR/{name}"', e0_body)
        self.assertIn("E0_*.json", e0_body)


class TestSetupDepsDiskRoot(unittest.TestCase):
    def test_disk_guard_calls_pass_data_root(self):
        src = _read("E0/code/setup_deps.sh")
        self.assertIn('DATA_ROOT="${V4_DATA_ROOT:-/data}"', src)
        calls = [m.start() for m in re.finditer(r"disk_guard\.py", src)]
        self.assertGreaterEqual(len(calls), 2, "setup_deps 应有安装前/后两次磁盘体检")
        for pos in calls:
            block = src[pos: pos + 400]
            self.assertIn("--data-root", block, src[pos:pos + 200])


class TestCheckEnvFixes(unittest.TestCase):
    def test_folds_fallback_file_exists(self):
        """R3-M1：fallback 曾写成 versions/reference/well_folds.json（不存在）。"""
        src = _read("E0/code/check_env.py")
        self.assertIn("v1_well_folds.json", src)
        self.assertNotRegex(src, r'reference"\s*/\s*"well_folds\.json')
        self.assertTrue((V4 / "versions" / "reference" / "v1_well_folds.json").is_file())

    def test_optional_deps_are_degradable_not_hard(self):
        """R3-M2：onnx/onnxruntime 有降级路径，不能把训练卡死。"""
        src = _read("E0/code/check_env.py")
        m = re.search(r"OPTIONAL_PY_DEPS\s*=\s*\{(.*?)\}", src, re.S)
        self.assertIsNotNone(m)
        self.assertIn("onnx", m.group(1))
        self.assertIn("degraded_paths", src)
        required = re.search(r"REQUIRED_PY_DEPS\s*=\s*\{(.*?)\}", src, re.S).group(1)
        for mod in ("numpy", "pandas", "scipy", "sklearn", "pyarrow", "einops"):
            self.assertIn(mod, required, mod)
        self.assertNotIn("onnx", required)


class TestCommittedE0CacheEvidence(unittest.TestCase):
    """R3-C3：提交里的 cache 证据必须是可复现路径，不能是 /tmp。"""

    def setUp(self):
        p = V4 / "reports" / "E0_data_card.json"
        if not p.is_file():
            self.skipTest("E0_data_card.json missing")
        self.card = json.loads(p.read_text(encoding="utf-8"))

    def test_cache_root_is_portable(self):
        cache = self.card.get("shard_cache") or {}
        if not cache:
            self.skipTest("no shard_cache section")
        root = str(cache.get("cache_root", ""))
        self.assertTrue(root, "cache_root 缺失")
        self.assertFalse(root.startswith("/tmp"),
                         f"cache_root 不能是本机临时路径：{root}")
        self.assertTrue(cache.get("cache_root_portable"),
                        f"cache_root 必须是可复现形式（$V4_CACHE_ROOT / repo 相对）：{root}")
        self.assertIn("cache_root_abs", cache)

    def test_sw_single_label_scale_fact(self):
        """R3-C1：SW 小于 1 的行数必须是 0（单一百分数尺度）。"""
        sw = self.card["target_stats"]["SW"]
        self.assertEqual(sw["valid_rows_only"].get("n_lt_1"), 0)
        self.assertGreater(sw["valid_rows_only"]["min"], 1.0)


class TestCheckEnvCudaSemantics(unittest.TestCase):
    """R4-B1：`torch.version.cuda` 是 **runtime**（12.4），不是驱动能力（12.6）。

    旧实现硬断言 `torch.version.cuda == 12.6`，云端 `--mode env` 会必然 `exit 11`，
    于是 `E0_env.json` / `E0_disk_budget.json` 永远产不出来、`E0_cloud_gate` 永远 blocked。
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "v4_check_env", str(V4 / "E0" / "code" / "check_env.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls.mod = mod

    def test_runtime_expectation_is_124_not_126(self):
        self.assertEqual(tuple(self.mod.EXPECTED_CUDA_RUNTIME), (12, 4))
        self.assertEqual(tuple(self.mod.MIN_CUDA_DRIVER), (12, 6))

    def test_old_wrong_constant_is_gone(self):
        src = _read("E0/code/check_env.py")
        self.assertNotIn("EXPECTED_CUDA_MAJOR_MINOR", src)
        self.assertNotIn('rep.add("cuda_version"', src)

    def test_parse_cuda_driver_from_smi(self):
        f = self.mod.parse_cuda_driver_from_smi
        self.assertEqual(
            f("| NVIDIA-SMI 550.54.15  Driver Version: 550.54.15  CUDA Version: 12.6  |"),
            "12.6")
        self.assertEqual(f("CUDA Version: 12.4"), "12.4")
        self.assertIsNone(f("no cuda version here"))
        self.assertIsNone(f(""))

    def test_expected_json_block_splits_runtime_and_driver(self):
        src = _read("E0/code/check_env.py")
        self.assertIn('"cuda_runtime"', src)
        self.assertIn('"cuda_driver_min"', src)
        # 旧的单一 "cuda" 键会让读者再次把 runtime 当驱动
        self.assertNotIn('"cuda": f"', src)

    def test_both_checks_are_registered(self):
        src = _read("E0/code/check_env.py")
        self.assertIn('"cuda_runtime_version"', src)
        self.assertIn('"cuda_driver_version"', src)
        # 驱动能力必须是 advisory（warn），不能 hard fail
        m = re.search(r'rep\.add\("cuda_driver_version",[^)]*?"(hard|warn)"', src, re.S)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "warn")


class TestRunE0PlanStatsEvidence(unittest.TestCase):
    """R4-H2：`run_e0` 必须同时产出计划行数证据 JSON（否则云端 E0_*.json 集合不自洽）。"""

    def setUp(self):
        src = _read("run_train.sh")
        self.e0_body = src[src.index("run_e0()"):]
        self.e0_body = self.e0_body[: self.e0_body.index("\nrun_smoke()")]

    def test_run_e0_writes_plan_stats_json(self):
        self.assertIn("plan_stats.py", self.e0_body)
        self.assertIn("E0_plan_stats.json", self.e0_body)

    def test_run_e0_still_copies_glob(self):
        self.assertIn("E0_*.json", self.e0_body)


class TestCommittedE0GateRecompute(unittest.TestCase):
    """R4-H1：提交里的 prereg 与 report 必须能被同一个校验器复算通过。"""

    def test_prereg_recompute_passed_is_recorded(self):
        from src.validation import gates as G
        pr = json.loads((V4 / "reports" / "E0_gate_prereg.json").read_text(encoding="utf-8"))
        rep = json.loads((V4 / "reports" / "E0_local_contract_gate.json")
                         .read_text(encoding="utf-8"))
        checks = rep.get("mandatory_checks") or rep.get("checks")
        result = {"checks": checks, "abs_diff": rep.get("abs_diff")}
        out = G.aggregate_gate(pr, result)
        self.assertTrue(out["passed"], json.dumps(out, ensure_ascii=False)[:800])

    def test_report_carries_contract_ok_alias(self):
        rep = json.loads((V4 / "reports" / "E0_local_contract_gate.json")
                         .read_text(encoding="utf-8"))
        checks = rep["mandatory_checks"]
        self.assertIn("contract_ok", checks)
        self.assertEqual(checks["contract_ok"], checks["contract_selftest"])


def _in_git_worktree() -> bool:
    """非 git 工作树（例如 `git archive` 解出的目录）里 `git check-ignore` 无法用。"""
    return (V4 / ".git").exists() and shutil.which("git") is not None


@unittest.skipUnless(_in_git_worktree(), "不在 git 工作树中（git archive 解包目录）")
class TestGitignoreDoesNotShadowSources(unittest.TestCase):
    """.gitignore 的目录模式不得误伤同名源码目录（R3 实际事故）。

    `.gitignore` 里的 `models/` 不带前导 `/` 时会匹配**任意深度**的同名目录，
    于是 `src/models/` 被整体忽略：`src/models/__init__.py` 与 `src/models/row_mlp.py`
    **从未进入 git**。推送后在云端训练会直接缺文件，而本机 `git status` 完全看不到。
    这里用 `git check-ignore` 锁死"源码目录不被忽略"，并确认根级产物目录仍被忽略。
    """

    def _check_ignore(self, path: str) -> bool:
        r = subprocess.run(["git", "check-ignore", "-q", path], cwd=str(V4))
        return r.returncode == 0

    def test_source_dirs_are_not_ignored(self):
        for rel in ("src/models/row_mlp.py", "src/models/__init__.py",
                    "src/losses/score_aligned.py", "src/inference/atomic_gate.py"):
            self.assertFalse(self._check_ignore(rel),
                             f"{rel} 被 .gitignore 忽略了（前导 / 缺失导致的源码遮蔽）")

    def test_root_artifact_dirs_are_still_ignored(self):
        for rel in ("models/x.pt", "cache/x.npz", "experiments/x/result.json",
                    "dist/v4_data.tar.gz", ".v4cache/manifest.json"):
            self.assertTrue(self._check_ignore(rel), f"{rel} 应仍被忽略")

    def test_source_files_are_tracked_by_git(self):
        """已在磁盘上的源码文件必须真的被 git 跟踪（无视规则遮蔽的兜底检查）。"""
        out = subprocess.run(["git", "ls-files", "src/"], cwd=str(V4),
                             capture_output=True, text=True).stdout
        tracked = set(out.split())
        for rel in ("src/models/row_mlp.py", "src/validation/gates.py",
                    "src/losses/score_aligned.py"):
            if (V4 / rel).is_file():
                self.assertIn(rel, tracked,
                              f"{rel} 存在于磁盘但未被 git 跟踪（会被静默排除在仓库之外）")

    def test_no_source_directory_anywhere_is_ignored(self):
        """穷举：仓库里任何**源码**目录/文件都不得被 .gitignore 命中。

        `src/` 下确实存在一个名为 `models` 的目录 —— 它不是冲突，而是必须被跟踪的源码；
        真正的不变量是「源码不被忽略」，因此这里逐个路径做 `git check-ignore` 穷举。
        """
        ignored: list[str] = []
        for root in ("src", "tests", "tools", "docs", "versions"):
            base = V4 / root
            if not base.is_dir():
                continue
            for p in sorted(base.rglob("*")):
                if "__pycache__" in p.parts or p.suffix == ".pyc":
                    continue
                rel = str(p.relative_to(V4))
                if self._check_ignore(rel):
                    ignored.append(rel)
        self.assertEqual(ignored, [],
                         f"仓库内被忽略的源码路径（前导 / 缺失导致）：{ignored}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
