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

import contextlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

DATA = V4.parent / "data"
REAL_TARBALL = V4 / "dist" / "v4_data.tar.gz"


def _read(rel: str) -> str:
    return (V4 / rel).read_text(encoding="utf-8")


def _load_check_env():
    """按文件加载 `E0/code/check_env.py`（`E0` 不是合法包名，不能 `import`）。"""
    spec = importlib.util.spec_from_file_location(
        "v4_check_env", str(V4 / "E0" / "code" / "check_env.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synth_well_text(n_rows: int = 30) -> str:
    """生成一口**列名对齐**的最小合法井（17 列 = DEPTH + 13 曲线 + 3 目标）。"""
    from src import constants as C
    header = ",".join(C.COLUMNS)
    units = ",".join(["m"] + ["u"] * (len(C.COLUMNS) - 1))
    rows = []
    for i in range(n_rows):
        vals = []
        for j, name in enumerate(C.COLUMNS):
            if name == "DEPTH":
                vals.append(f"{100.0 + 0.1 * i:.1f}")
            elif name == "POR":
                vals.append(f"{0.05 + 0.001 * i:.4f}")
            elif name == "PERM":
                vals.append(f"{1.0 + i:.4f}")
            elif name == "SW":
                vals.append(f"{60.0 + 0.1 * i:.4f}")
            else:
                vals.append(f"{10.0 + j + i:.4f}")
        rows.append(",".join(vals))
    return "\n".join([header, units, *rows]) + "\n"


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
        """R3-H2：不带 --data-root 时 level 描述的是 `/`，与 30 GB 配额无关。

        review R7：原实现只对正则**匹配到的**块做断言，正则失效时循环体一次都不执行，
        测试就"零断言通过"。现在先要求至少匹配到一个块。
        """
        blocks = [m.group(0) for m in
                  re.finditer(r"disk_guard\.py[^\n]*(?:\n[^\n]*)*?(?=\n\s*(?:log|if|else|fi|$))",
                              self.src)]
        self.assertTrue(blocks, "静态断言没匹配到任何 disk_guard 调用 —— 正则失效等于空转")
        checked = 0
        for block in blocks:
            if "--json" in block or "--min-free-gb" in block:
                checked += 1
                self.assertIn("--data-root", block, block)
        self.assertTrue(checked, "disk_guard 调用块里没找到 --json/--min-free-gb（正则已过期）")

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
        """R3-M2 + R5-M1：pyarrow/onnx/onnxruntime/tensorboard 有降级路径，不能把训练卡死。"""
        src = _read("E0/code/check_env.py")
        m = re.search(r"OPTIONAL_PY_DEPS\s*:\s*dict\[str,\s*str\s*\|\s*None\]\s*=\s*\{(.*?)\}",
                      src, re.S)
        self.assertIsNotNone(m)
        for mod in ("pyarrow", "onnx", "onnxruntime", "tensorboard"):
            self.assertIn(mod, m.group(1), mod)
        self.assertIn("degraded_paths", src)
        required = re.search(
            r"REQUIRED_PY_DEPS\s*:\s*dict\[str,\s*str\s*\|\s*None\]\s*=\s*\{(.*?)\}",
            src, re.S).group(1)
        for mod in ("numpy", "pandas", "scipy", "sklearn", "einops"):
            self.assertIn(mod, required, mod)
        for mod in ("onnx", "tensorboard"):
            self.assertNotIn(mod, required)


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


class TestCheckEnvTargetProfile(unittest.TestCase):
    """2026-09-20 目标平台：**Ascend 910B + CANN 8.3rc2 + torch 2.8.0 + torch_npu 2.8.0**。

    沿用四审 R4-B1 的教训：hard 只断言"能跑"的底线（major.minor / major），
    声明值（`2.8.0` / `8.3rc2`）一律只做 warn，否则 rc 或补丁漂移会再次把 Gate 卡死。
    硬件画像的单一事实源是 `src/hardware.py::PLATFORM`。
    """

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_check_env()

    def test_declared_stack_is_ascend_910b(self):
        self.assertEqual(self.mod.EXPECTED_TORCH, "2.8.0")
        self.assertEqual(self.mod.EXPECTED_TORCH_NPU, "2.8.0")
        self.assertEqual(self.mod.EXPECTED_CANN, "8.3rc2")
        self.assertEqual(self.mod.EXPECTED_ARCH, "aarch64")
        self.assertEqual(self.mod.TARGET_ACCELERATOR, "npu")
        self.assertEqual(self.mod.DISK_BUDGET_GB, 30.0, "云盘配额仍是 30 GB（64 GiB 是显存）")
        self.assertEqual(self.mod.RAM_BUDGET_GB, 16.0)

    def test_hard_checks_are_major_minor_not_exact_pins(self):
        """torch / torch_npu / CANN 的 hard 判定不得把具体小版本或 rc 钉死。"""
        src = _read("E0/code/check_env.py")
        for name, field in (("torch_version", "torch.__version__"),
                            ("torch_npu_version", "tn_ver")):
            m = re.search(r'rep\.add\("%s",\s*([^,]+),' % name, src)
            self.assertIsNotNone(m, f"找不到 {name} 注册")
            self.assertRegex(m.group(1), r"(version_major_minor|_mm|want_mm)",
                             f"{name} 的 hard 判定必须走 major.minor 比较，实际 {m.group(1)!r}")
        m = re.search(r'rep\.add\("cann_version",\s*([^,]+),', src)
        self.assertIsNotNone(m)
        self.assertRegex(m.group(1), r"(cann_major_minor|cann_mm|want_cann_mm)",
                         f"CANN 的 hard 判定必须走 major.minor 比较，实际 {m.group(1)!r}")
        # 声明值只能出现在 *_declared 那类 warn 检查里
        for decl in ("torch_version_declared", "torch_npu_version_declared",
                     "cann_version_declared"):
            mm = re.search(r'rep\.add\("%s",\s*[^,]+,\s*"(\w+)"' % decl, src)
            self.assertIsNotNone(mm, f"找不到 {decl} 注册")
            self.assertEqual(mm.group(1), "warn", f"{decl} 必须是 warn 级")

    def test_old_cuda_only_constants_are_gone(self):
        src = _read("E0/code/check_env.py")
        self.assertNotIn("EXPECTED_CUDA_MAJOR_MINOR", src)
        self.assertNotIn('rep.add("cuda_version"', src)
        self.assertNotIn("gpu_is_a100", src)
        # CUDA 只作为"兜底分支"存在，不得再是主路径
        self.assertIn('accel == "cuda"', src)

    def test_dev_flag_downgrades_device_not_deps(self):
        """`--allow-non-target-device` 只放宽设备/架构/torch，不得放宽依赖分档。"""
        src = _read("E0/code/check_env.py")
        self.assertIn('"--allow-non-target-device"', src)
        self.assertIn('dest="allow_non_target_device"', src)
        self.assertIn('"--allow-non-a100"', src, "旧名必须保留为别名，避免旧脚本失效")
        # 依赖分档只看 profile
        self.assertIn('required_level = "warn" if profile == "base" else "hard"', src)

    def test_cann_parser_prefers_real_version_over_unrelated_version_cfg(self):
        """Review R8：`version.cfg` 里的 `version=1.0` 不能抢在 CANN 8.3rc2 前面。"""
        import types
        from unittest import mock
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ascend-toolkit" / "latest"
            root.mkdir(parents=True)
            (root / "version.cfg").write_text("version=1.0\n", encoding="utf-8")
            (root / "ascend_toolkit_install.info").write_text(
                "version=8.3.RC2\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"ASCEND_HOME_PATH": str(root)}):
                self.assertEqual(self.mod._cann_from_files(), "8.3rc2")

    def test_cann_from_files_ignores_implausible_version_only(self):
        """只有无关的 version=1.0 时不能把它当 CANN 版本交给 Gate。"""
        from unittest import mock
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "version.cfg"
            bad.write_text("version=1.0\n", encoding="utf-8")
            with mock.patch.object(self.mod, "_cann_candidate_files", return_value=[bad]):
                self.assertIsNone(self.mod._cann_from_files())

    def test_query_cann_version_prefers_torch_npu_official_api(self):
        """torch_npu 2.8 的 CANN API 是 `torch_npu.utils.get_cann_version()`，
        不是不存在的 `torch_npu.version.cann`。"""
        import types
        from unittest import mock
        fake = types.ModuleType("torch_npu")
        fake.utils = types.SimpleNamespace(
            get_cann_version=lambda module="CANN": "8.3.RC2")
        fake.npu = types.SimpleNamespace(
            get_cann_version=lambda module="CANN": "8.3.RC2")
        fake.version = types.SimpleNamespace(__version__="2.8.0")
        with mock.patch.dict(sys.modules, {"torch_npu": fake}):
            version, source = self.mod.query_cann_version()
        self.assertEqual(version, "8.3rc2")
        self.assertIn("torch_npu.utils.get_cann_version", source)

    def test_parse_cann_version_variants(self):
        f = self.mod.parse_cann_version_from_text
        self.assertEqual(f("version=8.3.RC2"), "8.3rc2")
        self.assertEqual(f("CANN Version: 8.3.RC2"), "8.3rc2")
        self.assertEqual(f("Version=8.3.rc2"), "8.3rc2")
        self.assertIsNone(f("no version here"))

    def test_bf16_probe_math_is_correct(self):
        """Review R8-B1：8x8 全 1 matmul 的和是 512（不是 64），不能把真实 bf16 判 False。"""
        try:
            import torch
        except Exception:
            self.skipTest("torch not available")
        if not hasattr(torch, "bfloat16"):
            self.skipTest("torch without bfloat16")
        self.assertTrue(self.mod._bf16_probe(torch, "cpu"))

    def test_parse_cuda_driver_from_smi(self):
        f = self.mod.parse_cuda_driver_from_smi
        self.assertEqual(
            f("| NVIDIA-SMI 570.86.10  Driver Version: 570.86.10  CUDA Version: 12.8  |"),
            "12.8")
        self.assertEqual(f("CUDA Version: 12.6"), "12.6")
        self.assertIsNone(f("no cuda version here"))
        self.assertIsNone(f(""))

    def test_expected_json_comes_from_hardware_single_source(self):
        """`E0_env.json::expected` 必须由 `src/hardware.py::describe()` 派生。

        此前"文档写一份、代码写一份、报告再写一份"导致目标平台变更时漂移；
        现在 expected 块只有一个来源，CUDA 相关的四个旧键也随之消失。
        """
        src = _read("E0/code/check_env.py")
        self.assertIn('"expected": HW.describe()', src)
        for gone in ('"cuda_runtime"', '"cuda_runtime_accepted"',
                     '"cuda_runtime_hard_major"', '"cuda_driver_min"'):
            self.assertNotIn(gone, src, f"{gone} 已被 hardware.describe() 取代")

    def test_npu_checks_are_registered(self):
        src = _read("E0/code/check_env.py")
        for key in ('"torch_npu_version"', '"cann_version"', '"accelerator_available"',
                    '"device_is_910b"', '"machine_arch"', '"torch_npu_version_declared"',
                    '"cann_version_declared"'):
            self.assertIn(key, src, f"缺少 NPU 口径检查 {key}")
        # CUDA 分支（若有）只能是 advisory
        m = re.search(r'rep\.add\("cuda_driver_version",[^)]*?"(hard|warn)"', src, re.S)
        if m:
            self.assertEqual(m.group(1), "warn")

    def test_pyarrow_is_optional_not_required(self):
        """R5-M1：分片缓存是 `.npz`，没有任何代码 import pyarrow；
        把它留在 required 会让 `--mode data` 的 full 校验因缺它而 hard fail。"""
        self.assertIn("pyarrow", self.mod.OPTIONAL_PY_DEPS)
        self.assertNotIn("pyarrow", self.mod.REQUIRED_PY_DEPS)
        src = _read("src/data/dataset.py")
        self.assertIn("npz", src)
        self.assertNotIn("import pyarrow", _read("src/data/dataset.py"))

    def test_required_deps_are_version_agnostic(self):
        """R5-M1：required 依赖不得钉死版本（镜像升级会误报）。"""
        for mod, want in self.mod.REQUIRED_PY_DEPS.items():
            self.assertIsNone(want, f"REQUIRED_PY_DEPS[{mod!r}] 不应钉死版本")

    def test_pip_install_list_matches_declared_deps(self):
        """四审要求：我给出的 pip 安装清单必须与代码里的 REQUIRED 集合一致。"""
        self.assertEqual(set(self.mod.REQUIRED_PY_DEPS),
                         {"numpy", "pandas", "scipy", "sklearn", "einops"})

    def test_requirements_txt_active_lines_match_required_deps(self):
        """`requirements.txt` 里**未被注释**的行必须恰好等于 REQUIRED_PY_DEPS。

        这是用户实际照着敲的清单，所以它是真正的接口；`pandas` 用的是发行名
        (`scikit-learn`)，而 import 名是 `sklearn`，两者都要对得上。
        """
        active = []
        for raw in _read("requirements.txt").splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                active.append(line.split("==")[0].split(">")[0].strip())
        import_name = {"scikit-learn": "sklearn"}
        self.assertEqual(sorted(import_name.get(p, p) for p in active),
                         sorted(self.mod.REQUIRED_PY_DEPS))

    def test_requirements_does_not_declare_torch(self):
        """`pip install -r requirements.txt` 绝不能替换镜像里的 torch。"""
        for raw in _read("requirements.txt").splitlines():
            line = raw.split("#", 1)[0].strip()
            self.assertFalse(line.startswith("torch"),
                             f"requirements.txt 不得声明 torch：{raw!r}")

    def test_setup_deps_fallback_matches_required_deps(self):
        """`setup_deps.sh` 的兜底包列表（lock 缺失时用）不得漂移。"""
        src = _read("E0/code/setup_deps.sh")
        m = re.search(r"PKGS=\(\s*(.*?)\)", src, re.S)
        self.assertIsNotNone(m, "找不到 setup_deps.sh 的兜底 PKGS")
        pkgs = [p.strip().strip('"') for p in m.group(1).split() if p.strip()]
        import_name = {"scikit-learn": "sklearn"}
        self.assertEqual(sorted(import_name.get(p, p) for p in pkgs),
                         sorted(self.mod.REQUIRED_PY_DEPS))
        # 兜底列表里绝不能出现 torch
        self.assertFalse(any(p.startswith("torch") for p in pkgs))


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

    def test_leak_report_requires_full_coverage(self):
        """R7-3：`full_90_wells` 只看 violations==0 会假绿（0 井也 passed）。

        源码必须同时断言覆盖性（checked == 80+10），且已提交的证据必须真的覆盖 90 井。
        """
        src = _read("E0/code/run_all.py")
        self.assertIn('"expected": C.EXPECTED_N_TRAIN_WELLS + C.EXPECTED_N_TEST_WELLS', src)
        m = re.search(r'"passed": \(full_leak\["checked"\].*?\)', src, re.S)
        self.assertIsNotNone(m, "full_90_wells.passed 不再包含覆盖性断言")
        self.assertIn('full_leak["checked"] ==', m.group(0))
        card = json.loads((V4 / "reports" / "E0_data_card.json").read_text(encoding="utf-8"))
        f90 = card["input_leak_regression"]["full_90_wells"]
        self.assertEqual(f90["checked"], f90["expected"], f90)
        self.assertEqual(f90["expected"], 90)
        self.assertTrue(f90["passed"], f90)

    def test_folds_evidence_path_is_portable(self):
        """R7：证据里的折文件路径必须是仓库相对形式（不写死作者机绝对路径）。"""
        folds = json.loads((V4 / "versions" / "folds_sha256.json").read_text(encoding="utf-8"))
        self.assertFalse(folds["source_path"].startswith("/"), folds["source_path"])
        self.assertEqual(folds["source_path"], "versions/reference/v1_well_folds.json")
        self.assertEqual(folds["source_sha256"],
                         "f7c2c58bd035294f0e0d80a9103c366877836249fcd6db42269269c85d94b87e")
        # 读取方必须能解析相对路径
        self.assertIn("V4 / path", _read("tools/verify_reference.py"))


class TestRunTrainAllModeRunsFullPipeline(unittest.TestCase):
    """`--mode all` 必须是 E1→E10 全链路，而不是只跑首个阶段。"""

    def setUp(self):
        self.src = _read("run_train.sh")

    def test_all_mode_has_14_task_runner_and_full_chain(self):
        for token in (
            "ALL_TASK_NAMES=",
            "run_all_task()",
            "run_all_task \"$_n\"",
            "exit 21",
            "all_pipeline_progress.json",
        ):
            self.assertIn(token, self.src)

    def test_all_mode_uses_all_subroutes(self):
        for token in (
            "--channel-independence-ablation",
            "--rel-pos-ablation",
            "--capacity-ablation",
            "--target all",
            "--phase all",
        ):
            self.assertIn(token, self.src)

    def test_e3_pipeline_uses_run_root_and_all_scripts(self):
        block = self.src[self.src.index("    E3)"): self.src.index("    E4)")]
        self.assertIn('--run-root "$RUN_ROOT"', block)
        self.assertNotIn('--out-dir "$RUN_ROOT/E3"', block)
        for script in ("E3/code/train_seq.py", "E3/code/rf_ablation.py",
                       "E3/code/compare_row_vs_seq.py"):
            self.assertIn(script, block)

    def test_progress_file_is_written_to_persistent_data_root(self):
        self.assertIn('ALL_PROGRESS="$STATE_DIR/all_pipeline_progress.json"', self.src)
        self.assertIn('mark_progress "$_n" "$_name" running', self.src)
        self.assertIn('mark_progress "$_n" "$_name" done', self.src)

    def test_training_paths_are_local_and_final_model_publishes_to_network(self):
        for token in (
            'NETWORK_ROOT="${V4_NETWORK_ROOT:-/data}"',
            'LOCAL_ROOT="${V4_LOCAL_ROOT:-}"',
            'export V4_LOCAL_ROOT="$LOCAL_ROOT"',
            'export V4_NETWORK_ROOT="$NETWORK_ROOT"',
            'publish_final_to_network',
            'publish_progress_to_network',
        ):
            self.assertIn(token, self.src)

    def test_bootstrap_searches_network_root_for_tarball(self):
        src = _read("tools/bootstrap_data.sh")
        self.assertIn('V4_NETWORK_ROOT', src)
        self.assertIn('"$NETWORK_ROOT"', src)
        self.assertIn('"$NETWORK_ROOT/dist"', src)


    def test_through_parameter_controls_task_range(self):
        for token in ("--through|--all-to", "1..14", "_n<=ALL_THROUGH"):
            self.assertIn(token, self.src)

    def test_completed_tasks_are_skipped_for_resume(self):
        for token in ("task_status", "already done", "task_local_ready"):
            self.assertIn(token, self.src)


class TestCrossTaskStateMirror(unittest.TestCase):
    """本地 runtime 会随任务结束丢失，必须只把小状态文件镜像到 /data 并恢复。"""

    def test_sync_state_only_copies_selected_patterns(self):
        import subprocess as sp
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src"
            dst = Path(td) / "dst"
            src.mkdir()
            (src / "last.pt").write_bytes(b"checkpoint")
            (src / "E1").mkdir()
            (src / "E1" / "oof.npz").write_bytes(b"oof")
            (src / "big.cache").write_bytes(b"should-not-copy")
            r = sp.run(
                [sys.executable, str(V4 / "tools" / "sync_state.py"),
                 "--src", str(src), "--dst", str(dst), "--once"],
                capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertTrue((dst / "last.pt").is_file())
            self.assertTrue((dst / "E1" / "oof.npz").is_file())
            self.assertFalse((dst / "big.cache").exists())

    def test_run_train_registers_state_mirror_and_readiness(self):
        src = _read("run_train.sh")
        for token in (
            "tools/sync_state.py",
            "REMOTE_RUN_MIRROR",
            "restore_state_from_network",
            "sync_state_to_network",
            "task_local_ready",
        ):
            self.assertIn(token, src)


class TestBootstrapDataTarballResolution(unittest.TestCase):
    """R5-B1：`dist/*.tar.gz` 被 .gitignore 忽略，云端 repo 内**不可能**有分发包。

    五审复现：文档让用户把 tarball 传到云盘 `/data`，而 `bootstrap_data.sh` 只在
    `$V4/dist/` 里找 → `run_train.sh --mode data` 直接 exit 4，`env → data → e0`
    的推荐流程在第二步就断掉。这里用**隔离的假仓库**（无 dist tarball）逐条锁死
    定位与搜索语义，并用小 tarball 证明"从任意路径解压"确实生效。
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        # 假仓库：tools/bootstrap_data.sh + src/（校验块要 import src.constants）+ 空 dist/
        self.fake = self.tmp / "v4"
        (self.fake / "tools").mkdir(parents=True)
        shutil.copy2(V4 / "tools" / "bootstrap_data.sh", self.fake / "tools" / "bootstrap_data.sh")
        shutil.copytree(V4 / "src", self.fake / "src",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (self.fake / "dist").mkdir()
        (self.fake / "versions" / "reference").mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, *args, data_root, extra_env=None):
        env = dict(os.environ)
        env["V4_DATA_ROOT"] = str(data_root)
        env.pop("V4_DATA_TARBALL", None)
        env.pop("V4_DATA_MANIFEST", None)
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["bash", str(self.fake / "tools" / "bootstrap_data.sh"), *args],
            capture_output=True, text=True, env=env)

    def _tiny_tarball(self, path: Path) -> Path:
        """只含 1 口 1 行井的假分发包：足以证明"解压发生了"，又必然触发计数 FAIL。"""
        src = self.tmp / "inner" / "well_a.txt"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("DEPTH,POR\nm,frac\n1.0,0.1\n", encoding="utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "w:gz") as tf:
            tf.add(src, arcname="v4/data/train/well_a.txt")
        return path

    def test_no_tarball_anywhere_exits_4_and_lists_cloud_path(self):
        data_root = self.tmp / "data"
        r = self._run(data_root=data_root)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 4, out)
        self.assertIn("找不到数据分发包", out)
        self.assertIn(f"{data_root}/v4_data.tar.gz", out)      # 云盘根是推荐位置
        self.assertIn(f"{self.fake}/dist/v4_data.tar.gz", out)

    def test_explicit_tarball_is_used_from_arbitrary_path(self):
        tb = self._tiny_tarball(self.tmp / "cloud" / "v4_data.tar.gz")
        r = self._run("--tarball", str(tb), data_root=self.tmp / "data")
        out = r.stdout + r.stderr
        self.assertNotIn("找不到数据分发包", out)
        self.assertIn(f"tarball    : {tb}", out)
        self.assertIn("train wells=1", out)        # 解压真的发生了
        self.assertEqual(r.returncode, 1, out)     # 计数不符 -> 硬校验 FAIL

    def test_env_var_tarball_is_honored(self):
        tb = self._tiny_tarball(self.tmp / "cloud2" / "v4_data.tar.gz")
        r = self._run(data_root=self.tmp / "data",
                      extra_env={"V4_DATA_TARBALL": str(tb)})
        out = r.stdout + r.stderr
        self.assertIn(f"tarball    : {tb}", out)
        self.assertIn("train wells=1", out)

    def test_missing_explicit_tarball_exits_4(self):
        r = self._run("--tarball", str(self.tmp / "nope.tar.gz"), data_root=self.tmp / "data")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 4, out)
        self.assertIn("指向的文件不存在", out)

    def test_cloud_root_is_auto_searched(self):
        data_root = self.tmp / "data"
        tb = self._tiny_tarball(data_root / "v4_data.tar.gz")
        r = self._run(data_root=data_root)
        out = r.stdout + r.stderr
        self.assertIn(f"tarball    : {tb}", out)
        self.assertIn("train wells=1", out)

    def test_cloud_dist_with_manifest_checks_sha(self):
        data_root = self.tmp / "data"
        tb = self._tiny_tarball(data_root / "dist" / "v4_data.tar.gz")
        (data_root / "dist" / "v4_data_manifest.json").write_text(
            json.dumps({"tarball": {"sha256": "0" * 64},
                        "counts": {"n_train_wells": 1, "n_test_wells": 0,
                                   "n_train_rows": 1, "n_test_rows": 0}}),
            encoding="utf-8")
        r = self._run(data_root=data_root)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 3, out)     # sha256 不符必须中止
        self.assertIn("sha256 mismatch", out)

    def test_counts_are_unconditional_and_sourced_from_constants(self):
        """R5-H2 + 单一事实源：井数/行数硬校验不得依赖 manifest，也不得硬编码字面量。"""
        src = _read("tools/bootstrap_data.sh")
        self.assertIn("C.EXPECTED_N_TRAIN_WELLS", src)
        self.assertIn("C.EXPECTED_N_TRAIN_ROWS", src)
        self.assertIn("C.EXPECTED_N_TEST_WELLS", src)
        self.assertIn("C.EXPECTED_N_TEST_ROWS", src)
        for literal in ("730268", "730_268", "95948", "95_948"):
            self.assertNotIn(literal, src, f"bootstrap_data.sh 不得硬编码 {literal}")
        m = re.search(r"ok = \((.*?)\)\n", src, re.S)
        self.assertIsNotNone(m, "找不到 bootstrap_data.sh 的 ok 判定")
        for expr in ("rows_tr == C.EXPECTED_N_TRAIN_ROWS", "rows_te == C.EXPECTED_N_TEST_ROWS",
                     "n_tr == C.EXPECTED_N_TRAIN_WELLS", "n_te == C.EXPECTED_N_TEST_WELLS"):
            self.assertIn(expr, m.group(1), "硬校验必须是**无条件**的")

    def test_explicit_missing_manifest_exits_nonzero(self):
        """R6-L1：显式写下的 manifest 路径写错，不等于"没有 manifest"。"""
        tb = self._tiny_tarball(self.tmp / "cloud3" / "v4_data.tar.gz")
        r = self._run("--tarball", str(tb), "--manifest", str(self.tmp / "nope.json"),
                      data_root=self.tmp / "data")
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 4, out)
        self.assertIn("指向的文件不存在", out)

    def test_explicit_missing_manifest_env_exits_nonzero(self):
        tb = self._tiny_tarball(self.tmp / "cloud4" / "v4_data.tar.gz")
        r = self._run("--tarball", str(tb), data_root=self.tmp / "data",
                      extra_env={"V4_DATA_MANIFEST": str(self.tmp / "nope2.json")})
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 4, out)
        self.assertIn("指向的文件不存在", out)

    def test_manifest_without_sha_exits_3(self):
        """R6-L1：manifest 缺 `tarball.sha256` 不得打印 "tarball sha256 OK"。"""
        data_root = self.tmp / "data"
        tb = self._tiny_tarball(data_root / "v4_data.tar.gz")
        man = data_root / "v4_data_manifest.json"
        man.write_text(json.dumps({"counts": {"n_train_wells": 1, "n_test_wells": 0,
                                              "n_train_rows": 1, "n_test_rows": 0}}),
                       encoding="utf-8")
        r = self._run(data_root=data_root)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 3, out)
        self.assertNotIn("tarball sha256 OK", out)
        self.assertIn("tarball.sha256", out)

    def test_verify_only_writes_nothing(self):
        """R6-L2：`--verify-only` 不得创建 dest（此前无条件 mkdir -p "$DEST"）。"""
        data_root = self.tmp / "data"
        dest = data_root / "v4" / "data"
        r = self._run("--verify-only", data_root=data_root)
        out = r.stdout + r.stderr
        self.assertFalse(dest.exists(), f"--verify-only 竟然创建了 {dest}")
        self.assertFalse((dest / "folds").exists())
        self.assertIn("train wells=0", out)      # 只读校验仍然跑了

    def test_dest_is_honored_in_tarball_mode(self):
        """R6-L2：`--dest` 必须真正生效（此前解压永远写 $DATA_ROOT，校验却读 $DEST）。"""
        data_root = self.tmp / "data"
        tb = self._tiny_tarball(self.tmp / "cloud5" / "v4_data.tar.gz")
        dest = self.tmp / "custom" / "wells"
        r = self._run("--tarball", str(tb), "--dest", str(dest), data_root=data_root)
        out = r.stdout + r.stderr
        self.assertIn("--dest 与默认不同", out)
        self.assertTrue((dest / "train" / "well_a.txt").is_file(),
                        f"数据没有落到自定义 dest：{out}")
        self.assertFalse((data_root / "v4" / "data" / "train").exists(),
                         "数据不应同时写到默认 dest")


class TestRunTrainDataModeFindsCloudTarball(unittest.TestCase):
    """R5-B1 端到端：tarball 只在"云盘"（repo 内无 dist/）时 `--mode data` 必须能部署。

    注：本机开发机通常没有 torch / A100 / 8 GiB 空闲盘，`check_env --profile full`
    可能以 exit 13 结束 —— 那不是本测试的对象。这里断言的是**部署阶段**：
    `RESULT: OK` + 80/10 口井真的落到 `$DATA_ROOT/v4/data`，且失败原因与数据无关。
    """

    @unittest.skipUnless(REAL_TARBALL.is_file(),
                         "需要 dist/v4_data.tar.gz（本机 `tools/pack_dataset.py` 产物）")
    def test_data_mode_deploys_from_cloud_only_layout(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cloud = tmp / "cloud"
            cloud.mkdir()
            shutil.copy2(REAL_TARBALL, cloud / "v4_data.tar.gz")
            man = V4 / "dist" / "v4_data_manifest.json"
            if man.is_file():
                shutil.copy2(man, cloud / "v4_data_manifest.json")
            data_root = tmp / "data"
            env = dict(os.environ)
            env.update({
                "V4_DATA_ROOT": str(data_root),
                "V4_LOG_DIR": str(tmp / "logs"),
                "V4_RUN_ROOT": str(tmp / "runs"),
                "V4_CACHE_ROOT": str(tmp / "cache"),
                # 只把 tarball 放在"云盘"，repo 的 dist/ 完全不参与（文档推荐布局）
                "V4_DATA_TARBALL": str(cloud / "v4_data.tar.gz"),
            })
            env.pop("V4_DATA_MANIFEST", None)
            r = subprocess.run(["bash", str(V4 / "run_train.sh"), "--mode", "data"],
                               capture_output=True, text=True, env=env, cwd=str(tmp))
            out = r.stdout + r.stderr
            self.assertIn("RESULT: OK", out)
            self.assertIn("train wells=80 rows=730268", out)
            self.assertNotIn("找不到数据分发包", out)
            self.assertNotEqual(r.returncode, 4, out[-2000:])
            self.assertEqual(len(list((data_root / "v4" / "data" / "train").glob("*.txt"))), 80)
            self.assertEqual(len(list((data_root / "v4" / "data" / "test").glob("*.txt"))), 10)
            if r.returncode == 0:
                self.assertIn("[data] 数据健康校验通过", out)
            else:
                self.assertEqual(r.returncode, 13, out[-2000:])
                # 失败必须来自本机环境（无 torch / 无 A100 / 磁盘小），而不是数据
                self.assertIn("[OK  ] data_train_80", out)
                self.assertIn("[OK  ] data_test_10", out)

    def test_run_data_forwards_tarball_as_first_class_flag(self):
        """静态兜底：`--tarball`/`V4_DATA_TARBALL` 必须是一等参数。

        不能把它塞进 `EXTRA_ARGS` —— 那会把 `--mode all --epochs 5` 的参数误传给
        `bootstrap_data.sh`（unknown arg -> exit 2）。
        """
        src = _read("run_train.sh")
        self.assertIn('--tarball)  DATA_TARBALL="$2"', src)
        self.assertIn('--manifest) DATA_MANIFEST="$2"', src)
        self.assertIn('bargs+=(--tarball "$DATA_TARBALL")', src)
        self.assertIn('bargs+=(--manifest "$DATA_MANIFEST")', src)
        data_body = src[src.index("run_data()"):]
        data_body = data_body[: data_body.index("\nrun_e0()")]
        self.assertNotIn("EXTRA_ARGS", data_body,
                         "run_data 不得把 EXTRA_ARGS 透传给 bootstrap_data.sh")


class TestCheckDataLeakRequiresData(unittest.TestCase):
    """R5-H1：0 口井时输出 `RESULT: OK` 是**假绿**（数据路径写错/空目录时回归空转）。"""

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, str(V4 / "tools" / "check_data_leak.py"), *args],
            capture_output=True, text=True, cwd=str(V4))

    def test_script_asserts_coverage(self):
        src = _read("tools/check_data_leak.py")
        self.assertIn("EXPECTED_N_TRAIN_WELLS", src)
        self.assertIn("EXPECTED_N_TEST_WELLS", src)
        self.assertIn("--expect-train", src)
        self.assertIn("coverage[", src)

    def test_empty_data_dir_fails(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._run("--data", td)
            out = r.stdout + r.stderr
            self.assertNotEqual(r.returncode, 0, out)
            self.assertIn("RESULT: FAIL", out)
            self.assertIn("coverage[train]", out)

    def test_missing_data_dir_fails(self):
        with tempfile.TemporaryDirectory() as td:
            r = self._run("--data", str(Path(td) / "does_not_exist"))
            out = r.stdout + r.stderr
            self.assertNotEqual(r.returncode, 0, out)
            self.assertIn("RESULT: FAIL", out)

    def test_expect_flags_are_enforced(self):
        """只检查到 1 口井却声明期望 2 口 -> FAIL（覆盖性断言真的在起作用）。"""
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "train"
            src.mkdir()
            (src / "well_synth.txt").write_text(_synth_well_text(), encoding="utf-8")
            r = self._run("--data", td, "--expect-test", "0", "--expect-train", "2")
            out = r.stdout + r.stderr
            self.assertNotEqual(r.returncode, 0, out)
            self.assertIn("coverage[train]", out)
            self.assertIn("train=2", out)

    @unittest.skipUnless((DATA / "train").is_dir() and (DATA / "test").is_dir(),
                         "需要 ../data/{train,test}")
    def test_real_data_passes_with_exactly_90_wells(self):
        r = self._run()
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertIn("RESULT: OK", out)
        self.assertIn("80+10=90", out)


class TestSetupDepsDryRunIsSideEffectFree(unittest.TestCase):
    """`--dry-run` 是预检：不得写任何证据文件，也不得留下本机环境的假结论。"""

    def test_dry_run_writes_no_evidence(self):
        free_gb = shutil.disk_usage(str(V4)).free / 1024 ** 3
        if free_gb < 8.0:
            self.skipTest(f"代码所在磁盘只有 {free_gb:.1f} GiB 空闲（setup_deps 会在 <8 GiB 时 exit 2）")
        with tempfile.TemporaryDirectory() as td:
            reports = Path(td) / "reports"
            env = dict(os.environ)
            env["V4_REPORTS_DIR"] = str(reports)
            # 必须指向**空闲 >= 8 GiB** 的目录，否则脚本按 PLAN §3.4.1 直接 exit 2
            env["V4_DATA_ROOT"] = str(V4)
            env["V4_CACHE_ROOT"] = str(Path(td) / "cache")
            r = subprocess.run(["bash", str(V4 / "E0/code/setup_deps.sh"), "--dry-run"],
                               capture_output=True, text=True, env=env)
            out = r.stdout + r.stderr
            self.assertEqual(r.returncode, 0, out)
            self.assertIn("would run:", out)
            self.assertFalse((reports / "E0_env.json").exists(),
                             "--dry-run 不得写 E0_env.json（会污染 repo 快照）")
            self.assertFalse((reports / "E0_disk_budget.json").exists(),
                             "--dry-run 不得写 E0_disk_budget.json")
            stray = list((V4 / "reports").glob("E0_env.json")) + \
                list((V4 / "reports").glob("E0_disk_budget.json"))
            self.assertEqual(stray, [], f"--dry-run 在 repo 里留下了证据文件：{stray}")


class TestGateFilesUseContractConstants(unittest.TestCase):
    """R6-M2：E0 Gate 与云端环境 Gate 的判据也必须取自 `src/constants.py`。

    `constants.py` 自称唯一事实源，但 R5 之后仍有两处硬编码：
    `E0/code/run_all.py` 的 `730_268`、`E0/code/check_env.py` 的 `80` / `10`。
    这两处正是 Gate 阈值，一旦常量改动就会"常量改了、Gate 口径没改"。
    """

    def test_no_hardcoded_data_counts_in_gate_files(self):
        for rel in ("E0/code/run_all.py", "E0/code/check_env.py"):
            src = _read(rel)
            for literal in ("730268", "730_268", "95948", "95_948"):
                self.assertNotIn(literal, src,
                                 f"{rel} 不得硬编码数据量 {literal}（应用 C.EXPECTED_N_*）")

    def test_gate_files_reference_constants(self):
        ra = _read("E0/code/run_all.py")
        self.assertIn("C.EXPECTED_N_TRAIN_ROWS", ra)
        self.assertIn("C.EXPECTED_N_TEST_ROWS", ra)
        ce = _read("E0/code/check_env.py")
        self.assertIn("EXPECTED_N_TRAIN_WELLS", ce)
        self.assertIn("EXPECTED_N_TEST_WELLS", ce)
        self.assertIn("from src import constants", ce)

    def test_constants_module_is_stdlib_only(self):
        """`check_env.py` 在无 numpy/torch 的开发机上也要能导入 `src.constants`。"""
        src = _read("src/constants.py")
        for bad in ("import numpy", "import pandas", "import torch", "import scipy"):
            self.assertNotIn(bad, src, f"src/constants.py 必须只依赖标准库：{bad}")


class TestDependencyFactsAreSingleSourced(unittest.TestCase):
    """R5-M1/M2：PLAN / image_requirements / requirements.txt / setup_deps / lock 五处口径必须一致。

    五审发现 PLAN.md §3.3.1 仍在装 `pyarrow`/`onnx`/`onnxruntime`、预算表仍写
    "不装 tensorboard"，`image_requirements.md` 的 Dockerfile 又重复钉死版本并装 onnx，
    与 `requirements.txt` / `setup_deps.sh` 直接打架。
    """

    IMPORT_NAME = {"scikit-learn": "sklearn"}
    # 允许出现在 pip 安装命令里的包（required + 可选推荐）
    ALLOWED = {"numpy", "pandas", "scipy", "scikit-learn", "einops", "tensorboard"}
    FORBIDDEN = {"pyarrow", "onnx", "onnxruntime", "torch", "torchvision", "timm"}

    @staticmethod
    def _fenced_blocks(text: str, lang: str | None = None) -> list[list[str]]:
        blocks: list[list[str]] = []
        cur: list[str] = []
        inside = False
        open_lang = ""
        for line in text.splitlines():
            if line.strip().startswith("```"):
                if not inside:
                    inside, open_lang, cur = True, line.strip()[3:].strip(), []
                else:
                    if lang is None or open_lang == lang:
                        blocks.append(cur)
                    inside = False
                continue
            if inside:
                cur.append(line)
        return blocks

    def _pip_lines(self, lines) -> list[str]:
        """从**已取出的代码行**里挑出 pip 安装命令行（忽略注释）。"""
        out = []
        for line in lines:
            code = line.split("#", 1)[0]
            if re.search(r"(^|\s)(python3?\s+-m\s+)?pip\s+install", code):
                out.append(code.strip())
        return out

    @staticmethod
    def _join_continuations(lines) -> list[str]:
        """把 `\\` 续行拼成一行，否则 Dockerfile 式多行 pip 命令只会读到开关。"""
        out: list[str] = []
        buf = ""
        for line in lines:
            s = line.rstrip()
            if s.endswith("\\"):
                buf += s[:-1] + " "
                continue
            out.append(buf + s)
            buf = ""
        if buf:
            out.append(buf)
        return out

    def _all_pip_lines(self, rel: str, lang: str | None = None) -> list[str]:
        blocks = self._fenced_blocks(_read(rel), lang=lang)
        return self._pip_lines(self._join_continuations([ln for b in blocks for ln in b]))

    @staticmethod
    def _pkgs_from_pip_line(line: str) -> list[str]:
        # 去掉 `pip install` 与开关/选项，遇 shell 连接符即停止
        parts = line.split()
        try:
            i = next(k for k, p in enumerate(parts) if p == "install")
        except StopIteration:  # pragma: no cover - 正则已保证有 install
            return []
        pkgs: list[str] = []
        for p in parts[i + 1:]:
            if p in ("&&", "||", ";", "|", ">", ">>", "&", "\\"):
                break
            if p.startswith("-"):
                continue
            pkgs.append(p.strip('"\''))
        return pkgs

    def test_plan_pip_commands_only_use_the_agreed_list(self):
        lines = self._all_pip_lines("PLAN.md")
        self.assertTrue(lines, "PLAN.md 里应保留一份 pip 安装示例")
        for line in lines:
            for pkg in self._pkgs_from_pip_line(line):
                name = pkg.split("==")[0].split(">")[0].split("<")[0].strip()
                self.assertIn(name, self.ALLOWED,
                              f"PLAN.md 的 pip 命令出现不该装的包：{pkg!r}（{line}）")

    def test_plan_does_not_claim_tensorboard_is_not_installed(self):
        src = _read("PLAN.md")
        for bad in ("不装 tensorboard", "不安装：`tensorboard`", "不安装 `tensorboard`"):
            self.assertNotIn(bad, src, f"PLAN.md 仍与「tensorboard 可选推荐」矛盾：{bad}")

    def test_image_dockerfile_only_installs_the_agreed_list(self):
        text = _read("docs/image_requirements.md")
        blocks = self._fenced_blocks(text, lang="dockerfile")
        self.assertTrue(blocks, "找不到 image_requirements.md 的 dockerfile 代码块")
        flat = [ln for b in blocks for ln in b]
        body = "\n".join(flat)
        for pkg in ("pyarrow", "onnx", "onnxruntime"):
            self.assertNotIn(pkg, body,
                             f"镜像 Dockerfile 不得安装 {pkg}（与 requirements.txt 冲突）")
        lines = self._pip_lines(self._join_continuations(flat))
        self.assertTrue(lines, "Dockerfile 应保留 pip 安装行")
        for line in lines:
            for pkg in self._pkgs_from_pip_line(line):
                name = pkg.split("==")[0].strip()
                self.assertIn(name, self.ALLOWED, f"Dockerfile 出现未列入清单的包：{pkg!r}")

    def test_no_claim_that_torch_provides_numpy(self):
        """torch 的 PyPI `Requires-Dist` 里**没有** numpy（2.7.1 与 2.8.0 均如此），不得再写"torch 自带 numpy"。

        这条不是文案洁癖：一旦有人相信"numpy 必然存在"，就会把它从 required 清单里删掉，
        而 numpy 是本项目口径层的唯一硬依赖（`portability.HAS_NUMPY`）。
        """
        for rel in ("requirements.txt", "PLAN.md", "versions/locks/cloud.txt",
                    "docs/dependencies.md", "docs/image_requirements.md",
                    "E0/code/check_env.py", "E0/code/setup_deps.sh"):
            src = _read(rel)
            for bad in ("自带一份 numpy", "随 torch 提供", "torch 自带 numpy", "torch+numpy"):
                self.assertNotIn(bad, src, f"{rel} 仍声称 torch 提供 numpy：{bad}")
        # 反向确认：numpy 必须在 required 清单里
        self.assertIn("numpy", _load_check_env().REQUIRED_PY_DEPS)

    def test_requirements_lock_and_setup_deps_agree(self):
        chk = _load_check_env()
        # 1) requirements.txt 有效行
        active = []
        for raw in _read("requirements.txt").splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                active.append(line.split("==")[0].split(">")[0].strip())
        self.assertEqual(sorted(self.IMPORT_NAME.get(p, p) for p in active),
                         sorted(chk.REQUIRED_PY_DEPS))
        # 2) setup_deps.sh 的 lock 过滤规则产出的包集合必须恰好等于 REQUIRED_PY_DEPS
        src = _read("E0/code/setup_deps.sh")
        m = re.search(r"grep -viE '\^\(([^)]*)\)'", src)
        self.assertIsNotNone(m, "找不到 setup_deps.sh 的加速栈排除正则")
        pattern = "^(?:" + m.group(1) + ")"
        self.assertNotIn("numpy", pattern,
                         "R5-M2：setup_deps.sh 不得把 numpy 排除在安装之外（required 里有它）")
        for must in ("torch", "npu", "ascend", "cann"):
            self.assertIn(must, pattern,
                          f"{must} 必须被排除（禁止 pip 触碰镜像加速栈）")
        rx = re.compile(pattern)
        lock_pkgs = []
        for raw in _read("versions/locks/cloud.txt").splitlines():
            if rx.search(raw):
                continue
            line = re.sub(r"[ \t]*#.*$", "", raw).strip()
            if line:
                lock_pkgs.append(line)
        self.assertEqual(sorted(self.IMPORT_NAME.get(p, p) for p in lock_pkgs),
                         sorted(chk.REQUIRED_PY_DEPS),
                         "versions/locks/cloud.txt 过滤后必须恰好等于 REQUIRED_PY_DEPS")


class TestOnnxWordingMatchesDependencyPolicy(unittest.TestCase):
    """R6-M1：计划不能一边说"不安装 onnx"，一边承诺"优先导出 ONNX 兜底"。

    实测（无 onnx 的本机 venv）：`torch.onnx.export` →
    `OnnxExporterError: Module onnx is not installed!`，torch 2.7 的 dynamo 路径还额外
    需要 `onnxscript`。因此 ONNX 只能是 best-effort，实际兜底是 `.pt` + `.npz` 清单。
    """

    OLD_WORDING = "同时尝试 `torch.onnx.export`（失败不阻塞）"

    def test_plan_no_longer_promises_onnx_export(self):
        src = _read("PLAN.md")
        for bad in ("优先导出 ONNX", "torch 2.7 内置", "ONNX 导出兜底路径"):
            self.assertNotIn(bad, src, f"PLAN.md 仍把 ONNX 写成可依赖的兜底：{bad}")
        self.assertIn("best-effort", src)
        self.assertIn("Module onnx is not installed", src)
        self.assertIn(".npz", src)

    def test_e10_plan_states_onnx_is_best_effort(self):
        src = _read("E10/P0/PLAN.md")
        self.assertIn("best-effort", src)
        self.assertIn("`import onnx`", src)
        self.assertNotIn(self.OLD_WORDING, src,
                         "E10/P0 仍是旧的 ONNX 措辞（生成器改了但没重跑 gen_p_details.py？）")
        self.assertIn(".npz", src, "E10/P0 必须写出真正的兜底：`.npz` 权重清单")

    def test_generator_carries_the_same_wording(self):
        gen = _read("docs/gen_p_details.py")
        self.assertNotIn(self.OLD_WORDING, gen,
                         "生成器仍会产出旧的 ONNX 措辞（改了生成产物但没改生成器）")
        self.assertIn("best-effort", gen)


class TestFrozenPins(unittest.TestCase):
    """版本策略：默认不钉版本 -> 实机 `pip freeze` 回填 -> 需要时用 `--from-frozen` 精确复现。

    这是"你不指定安装的版本吗"的代码答案：钉版本的位置是 `cloud_frozen.txt`（实测事实），
    而不是我在开发机上猜出来的小版本号。
    """

    SYNTHETIC = "\n".join([
        "# synthetic pip freeze",
        "torch==2.8.0",
        "torch_npu==2.8.0",
        "torch-npu==2.8.0",
        "nvidia-cuda-runtime-cu12==12.6.77",
        "triton==3.3.1",
        "cuda-python==12.6.0",
        "numpy==2.1.3",
        "pandas==2.2.3",
        "scipy==1.14.1",
        "scikit_learn==1.5.2",          # 下划线写法必须归一到 scikit-learn
        "einops==0.8.0",
        "tensorboard==2.18.0",
        "setuptools==75.1.0",
        "pip==24.2",
        "protobuf==5.28.2",
        "-e git+https://example.com/x.git#egg=y",
        "",
    ])

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "v4_frozen_pins", str(V4 / "E0" / "code" / "frozen_pins.py"))
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_picks_exact_versions_for_direct_deps_only(self):
        pins = self.mod.parse_freezes(self.SYNTHETIC)
        self.assertEqual(pins["numpy"], "numpy==2.1.3")
        self.assertEqual(pins["pandas"], "pandas==2.2.3")
        self.assertEqual(pins["scipy"], "scipy==1.14.1")
        self.assertEqual(pins["scikit-learn"], "scikit-learn==1.5.2")
        self.assertEqual(pins["einops"], "einops==0.8.0")
        self.assertEqual(pins["tensorboard"], "tensorboard==2.18.0")
        # 传递依赖 / 打包工具不得出现
        for junk in ("setuptools", "pip", "protobuf"):
            self.assertNotIn(junk, pins)

    def test_never_contains_forbidden_series(self):
        """白名单式构造：结果集与 torch/nvidia/cuda/triton **无交集**（含 `-e` 行也不受影响）。"""
        pins = self.mod.parse_freezes(self.SYNTHETIC)
        for name in pins:
            for bad in self.mod.FORBIDDEN:
                self.assertFalse(name.startswith(bad),
                                 f"frozen 结果不得包含 {bad}* （实际 {name}）")
        # 再来一个只有禁装系列的 freeze：必须得到空结果
        only_bad = ("torch==2.8.0\ntorch_npu==2.8.0\nascend-toolkit==1.0\ncann==8.3\n"
                    "nvidia-cudnn-cu12==9.5\ntriton==3.3.1\ncuda-python==1.0\n")
        self.assertEqual(self.mod.parse_freezes(only_bad), {})

    def test_name_normalization_follows_pep503(self):
        text = "Scikit.Learn==1.5.2\nEIN0OPS==1.0\n"
        self.assertIn("scikit-learn", self.mod.parse_freezes(text))
        self.assertEqual(self.mod.parse_freezes("Scikit_Learn==1.5.2")["scikit-learn"],
                         "scikit-learn==1.5.2")

    def test_missing_and_empty_files_return_nonzero(self):
        with tempfile.TemporaryDirectory() as td, \
                contextlib.redirect_stderr(io.StringIO()):
            missing = Path(td) / "nope.txt"
            self.assertEqual(self.mod.main(["--frozen", str(missing)]), 2)
            empty = Path(td) / "empty.txt"
            empty.write_text("# nothing here\n", encoding="utf-8")
            self.assertEqual(self.mod.main(["--frozen", str(empty)]), 1)

    def test_setup_deps_wires_from_frozen(self):
        """静态兜底：`--from-frozen` 必须真的走 frozen_pins.py，而不是被忽略。"""
        src = _read("E0/code/setup_deps.sh")
        self.assertIn("--from-frozen) FROM_FROZEN=1", src)
        self.assertIn("frozen_pins.py", src)
        self.assertIn("cloud_frozen.txt", src)
        # frozen 路径下不得把 white-list 之外的包塞进 pip
        self.assertIn("V4_FROZEN_LOCK", src)

    def test_setup_deps_from_frozen_dry_run(self):
        """端到端（需 >= 8 GiB 空闲，否则脚本按 PLAN §3.4.1 直接 exit 2）。"""
        free_gb = shutil.disk_usage(str(V4)).free / 1024 ** 3
        if free_gb < 8.0:
            self.skipTest(f"代码所在磁盘只有 {free_gb:.1f} GiB 空闲")
        with tempfile.TemporaryDirectory() as td:
            frozen = Path(td) / "frozen.txt"
            frozen.write_text(self.SYNTHETIC, encoding="utf-8")
            env = dict(os.environ)
            env.update({"V4_REPORTS_DIR": str(Path(td) / "reports"),
                        "V4_DATA_ROOT": str(V4),
                        "V4_CACHE_ROOT": str(Path(td) / "cache"),
                        "V4_FROZEN_LOCK": str(frozen)})
            r = subprocess.run(["bash", str(V4 / "E0/code/setup_deps.sh"),
                                "--from-frozen", "--dry-run"],
                               capture_output=True, text=True, env=env)
            out = r.stdout + r.stderr
            self.assertEqual(r.returncode, 0, out)
            self.assertIn("numpy==2.1.3", out)
            self.assertIn("scikit-learn==1.5.2", out)
            self.assertNotIn("torch", out.split("would run:")[-1])
            self.assertNotIn("nvidia", out.split("would run:")[-1])


class TestPushProtocolIsDocumented(unittest.TestCase):
    """用户已完成 remote + SSH 配置（2026-09-19）：文档里必须常驻"开跑前先 push"的协议。

    起因：平台只克隆**已 push** 的代码，而 agent 曾在给出云端开跑清单时漏掉 push ——
    任务会静默跑在旧代码上，日志里完全看不出来。这条回归防止文档被后续重构悄悄改掉。
    """

    REMOTE = "git@github.com:sunny-boy-fqy/Petroleum-Competition.git"
    HTTPS = "https://github.com/sunny-boy-fqy/Petroleum-Competition.git"

    def test_every_entry_doc_carries_the_push_step(self):
        for rel in ("README.md", "docs/platform_setup.md", "docs/training_tasks.md", "PLAN.md"):
            src = _read(rel)
            self.assertIn("git push", src, f"{rel} 必须保留开跑前的 push 步骤")
            self.assertIn(self.REMOTE, src, f"{rel} 必须写明已配置的 remote")
            self.assertIn("master", src, f"{rel} 必须写明远端有 master 分支")

    def test_push_updates_both_branches(self):
        """2026-09-20：远端同时维护 main 与 master（平台分支字段默认 main）。

        漏推一个分支会让平台静默跑到旧代码上，所以 push 命令必须一次推两个 ref。
        """
        for rel in ("README.md", "docs/platform_setup.md", "docs/training_tasks.md", "PLAN.md"):
            src = _read(rel)
            self.assertIn("HEAD:master HEAD:main", src,
                          f"{rel} 的 push 命令必须同时更新 master 与 main")

    def test_platform_repo_field_is_https_never_ssh(self):
        """平台【仓库地址】字段值必须是 HTTPS（平台侧没有本机 SSH key）。

        注意（2026-09-20 教训）：这条**不是**"任务已创建但失败"的解释——平台表单正则
        `/^(https?:\\/\\/|git@)[\\w\\-.~/]+(\\.git)?$/i` 会直接拒绝 scp 形式
        `git@github.com:owner/repo.git`（`:` 不匹配），所以那种填法根本提交不了。
        文档里不得再把它写成失败根因。
        """
        for rel in ("README.md", "docs/platform_setup.md", "docs/training_tasks.md"):
            src = _read(rel)
            self.assertIn(self.HTTPS, src, f"{rel} 必须给出平台用的 HTTPS 仓库地址")
        for rel in ("docs/platform_setup.md", "docs/training_tasks.md"):
            # 只看"字段名恰好是 仓库地址"的表格行（故障排查表里的正文提到它不算字段）
            rows = [ln for ln in _read(rel).splitlines()
                    if re.match(r"^\|\s*仓库地址\s*\|", ln)]
            self.assertTrue(rows, f"{rel} 必须保留平台「仓库地址」字段行")
            for line in rows:
                # 字段**值**（行内第一个 URL）必须是 HTTPS；行内其余位置允许出现
                # SSH 地址作为"不要这样填"的反面警告。
                first = re.search(r"(https?://\S+|git@\S+)", line)
                self.assertIsNotNone(first, f"{rel} 的仓库地址行没有 URL：{line.strip()}")
                self.assertTrue(first.group(1).startswith("https://"),
                                f"{rel} 的平台仓库地址字段值必须是 HTTPS，实际是 {first.group(1)}")

    def test_no_doc_claims_ssh_url_was_the_failure_cause(self):
        """反面纪律：文档不得再把"填了 SSH 地址"写成那次失败的（已证伪的）根因。"""
        for rel in ("README.md", "docs/platform_setup.md", "docs/training_tasks.md", "PLAN.md"):
            src = _read(rel)
            self.assertNotIn("表现为「错误 + 无日志」", src, f"{rel} 仍把 SSH 地址写成失败根因")
            for line in src.splitlines():
                if "Permission denied" in line:
                    self.assertIn("不", line,
                                  f"{rel} 提到 Permission denied 时必须同时说明该填法提交不了：{line.strip()}")

    def test_docs_record_platform_frontend_facts(self):
        """把从前端 bundle 读到的硬事实固化：状态枚举没有「错误」、Git 源工作目录、URL 正则。"""
        src = _read("docs/platform_setup.md")
        for needle in ("排队中", "运行中", "失败", "暂无日志",
                       "/code/workspace", r"git@)[\w\-.~/]+", "日志连接失败"):
            self.assertIn(needle, src, f"docs/platform_setup.md 必须记录平台事实：{needle}")
        self.assertIn("没有「错误」这个状态", src)
        # 必须给出 2x2 判定法与"根因未定"的纪律
        self.assertIn("2×2", src)
        self.assertIn("根因未定", src)

    def test_docs_explain_the_prep_stage_silent_failure(self):
        """「失败 + 无日志」必须被文档解释，并给出云盘日志这条判定路径。"""
        src = _read("docs/platform_setup.md")
        self.assertIn("容器从未启动", src)
        self.assertIn("无日志", src)
        self.assertIn("/data/v4/logs", src)
        # 零歧义探针不能含命令替换（要能排除"平台不解析 $( )"这一因素）
        probe = [ln for ln in src.splitlines() if "find /code/workspace -maxdepth 3" in ln]
        self.assertTrue(probe, "docs/platform_setup.md 必须给出零歧义探针命令")
        self.assertNotIn("$(", probe[0])

    def test_launcher_header_warns_about_ssh_vs_https(self):
        src = _read("run_train.sh")
        self.assertIn(self.HTTPS, src)
        # 说明为什么 scp 形式不可用（表单正则），而不是断言一个未经验证的失败故事
        self.assertIn(r"git@)[\w\-.~/]+", src)
        self.assertIn("不要预设原因", src)

    def test_plan_records_the_sync_protocol(self):
        src = _read("PLAN.md")
        self.assertIn("代码同步协议", src)
        self.assertIn("`git push`", src)
        self.assertIn("ssh", src.lower())

    def test_no_placeholder_remote_or_main_branch_left(self):
        for rel in ("docs/platform_setup.md", "docs/training_tasks.md", "README.md"):
            src = _read(rel)
            for bad in ("<你的仓库地址>", "<你的 git 仓库地址>", "git push -u origin main",
                        "git remote add origin <"):
                self.assertNotIn(bad, src, f"{rel} 仍有过期占位符：{bad}")

    def test_repo_remote_matches_documented_remote(self):
        """文档写的 remote 必须与实际 `git remote` 一致（非工作树时跳过）。"""
        if not _in_git_worktree():
            self.skipTest("不在 git 工作树中")
        out = subprocess.run(["git", "remote", "get-url", "origin"], cwd=str(V4),
                             capture_output=True, text=True).stdout.strip()
        self.assertEqual(out, self.REMOTE)


class TestStartCommandsAreLocationIndependent(unittest.TestCase):
    """仓库根 = `v4/` 的内容，所以平台克隆目录**不叫 `v4`**。

    文档若硬编码 `/code/workspace/v4/run_train.sh`，平台任务会直接 `No such file`。
    所有入口文档与 P 级计划必须用 `find` 自定位写法。
    """

    RESOLVER = 'bash "$(find /code/workspace -name run_train.sh | head -1)"'

    def test_no_doc_hardcodes_the_clone_path(self):
        docs = ["README.md", "PLAN.md", "dist/README.md", "docs/platform_setup.md",
                "docs/training_tasks.md", "docs/image_requirements.md", "E0/docs/data_card.md"]
        docs += [str(p.relative_to(V4)) for p in sorted(V4.glob("E*/P*/PLAN.md"))]
        docs += [str(p.relative_to(V4)) for p in sorted(V4.glob("E*/PLAN.md"))]
        bad: list[str] = []
        for rel in docs:
            if not (V4 / rel).is_file():
                continue
            src = _read(rel)
            if "/code/workspace/v4" in src:
                bad.append(rel)
        self.assertEqual(bad, [], f"这些文件仍硬编码 /code/workspace/v4：{bad}")

    def test_entry_docs_use_the_resolver(self):
        for rel in ("README.md", "docs/platform_setup.md", "docs/training_tasks.md",
                    "dist/README.md"):
            self.assertIn(self.RESOLVER, _read(rel),
                          f"{rel} 必须给出位置无关的启动命令写法")

    def test_generator_emits_the_resolver(self):
        """生成器里的内层引号必须转义（否则会写出 Python 字符串语法错误）。"""
        gen = _read("docs/gen_p_details.py")
        self.assertIn('find /code/workspace -name run_train.sh', gen)
        self.assertIn('bash \\"$(find /code/workspace -name run_train.sh | head -1)\\"', gen)
        self.assertNotIn("/code/workspace/v4", gen)
        # 生成的 P 级计划里必须是**未转义**的真实命令
        self.assertIn(self.RESOLVER, _read("E0/P0/PLAN.md"))

    def test_resolver_command_is_under_the_platform_limit(self):
        """启动命令上限 500 字符（平台硬约束）。"""
        for suffix in (" --mode env", " --mode data", " --mode e0", " --mode all"):
            self.assertLess(len(self.RESOLVER + suffix), 500)

    def test_cloud_commands_never_use_a_bare_v4_prefix(self):
        """review R7 第 5 条的分类版：`v4/...` 只在**本机**上下文里合法。

        本机开发机上目录**确实叫** `v4/`（项目父目录下），所以 `python3 v4/tools/...`
        是正确的；但云端克隆目录不叫 `v4`，**以「云端」为上下文**的命令绝不能出现
        `v4/` 前缀。这里按代码块内的注释跟踪上下文，逐行断言。
        """
        files = ["README.md", "docs/platform_setup.md", "docs/training_tasks.md",
                 "dist/README.md", "E0/P0/PLAN.md", "E1/P0/PLAN.md"]
        checked = 0
        for rel in files:
            if not (V4 / rel).is_file():
                continue
            for block in TestDependencyFactsAreSingleSourced._fenced_blocks(_read(rel)):
                ctx = None
                for line in block:
                    s = line.strip()
                    if not s:
                        continue
                    if s.startswith("#"):
                        if "云端" in s:
                            ctx = "cloud"
                        elif "本机" in s:
                            ctx = "local"
                        continue
                    if ctx != "cloud":
                        continue
                    checked += 1
                    # 只抓**相对**的 `v4/`（命令开头的仓库目录前缀）；
                    # `/data/v4/...` 是数据路径、`$V4/...` 是环境变量，都合法。
                    self.assertIsNone(
                        re.search(r"(?<![A-Za-z0-9_/$])v4/", s),
                        f"{rel} 的云端命令出现本机才有效的 `v4/`：{s}")
        self.assertGreater(checked, 0,
                           "没解析到任何「云端」上下文命令 —— 上下文跟踪失效，断言空转")


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
        for root in ("src", "tests", "tools", "docs", "versions", "E0", "configs"):
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
