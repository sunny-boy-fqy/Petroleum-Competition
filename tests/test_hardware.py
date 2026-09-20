"""目标平台画像测试（2026-09-20：A100/CUDA → **Ascend 910B / CANN 8.3rc2 / torch_npu**）。

锁定 `src/hardware.py` 作为**唯一事实源**：`check_env.py`、`src/constants.py` 的磁盘预算、
生成的计划与文档都必须从它派生。任何"文档/代码各写一份"的漂移都在这里被挡住。
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src import constants as C  # noqa: E402
from src import hardware as HW  # noqa: E402


class TestPlatformProfile(unittest.TestCase):
    def test_declared_profile_is_ascend_910b(self):
        self.assertEqual(HW.PLATFORM["resource_spec"], "Ascend910B-1-64G")
        self.assertEqual(HW.PLATFORM["accelerator"], "npu")
        self.assertEqual(HW.PLATFORM["accelerator_model"], "910B")
        # 三量必须分开命名（2026-09-20 曾把云盘误写成 64 GiB）：
        self.assertEqual(HW.PLATFORM["device_memory_gb"], 64, "64 GB 是**显存**（HBM）")
        self.assertEqual(HW.PLATFORM["host_ram_gb"], 16, "16 GB 是**内存**")
        self.assertEqual(HW.PLATFORM["cloud_disk_gb"], 30, "30 GB 是**云盘 /data 配额**")
        self.assertEqual(HW.PLATFORM["arch"], "aarch64")
        self.assertEqual(HW.PLATFORM["python"], (3, 11))
        self.assertEqual(HW.PLATFORM["torch"], "2.8.0")
        self.assertEqual(HW.PLATFORM["torch_npu"], "2.8.0")
        self.assertEqual(HW.PLATFORM["cann"], "8.3rc2")
        self.assertEqual(tuple(HW.PLATFORM["cann_accepted_major_minor"]), (8, 3))

    def test_torch_and_torch_npu_share_minor_version(self):
        """torch_npu 必须与 torch 同小版本（否则 torch.npu 不可用）。"""
        self.assertEqual(HW.version_major_minor(HW.PLATFORM["torch"]),
                         HW.version_major_minor(HW.PLATFORM["torch_npu"]))

    def test_constants_disk_budget_matches_profile(self):
        """预算常量必须来自硬件画像的正确字段（磁盘=云盘配额，不是显存）。"""
        self.assertEqual(C.DISK_BUDGET_GB, float(HW.PLATFORM["cloud_disk_gb"]))
        self.assertEqual(C.RAM_BUDGET_GB, float(HW.PLATFORM["host_ram_gb"]))
        self.assertEqual(C.DISK_BUDGET_GB, 30.0)

    def test_three_sizes_are_never_confused(self):
        """反混淆回归：云盘 ≠ 显存 ≠ 内存，且文档不得把它们写混。

        真实事故（2026-09-20）：平台规格改成 `Ascend910B-1-64G | … | 16G | 64Gi` 后，
        agent 把**云盘**从 30 GB 改成了 64 GiB —— 全部磁盘纪律会因此建立在错误容量上。
        这条测试同时锁数值与文案。
        """
        self.assertNotEqual(HW.PLATFORM["cloud_disk_gb"], HW.PLATFORM["device_memory_gb"])
        self.assertNotEqual(HW.PLATFORM["cloud_disk_gb"], HW.PLATFORM["host_ram_gb"])
        profile = HW.describe()
        self.assertEqual(profile["cloud_disk_gb"], 30)
        self.assertEqual(profile["device_memory_gb"], 64)
        self.assertEqual(profile["host_ram_gb"], 16)
        # 文案：不得再出现"64 GiB 磁盘 / 云盘 64 / 磁盘配额 64"这类说法
        bad = ("64 GiB 磁盘", "64GiB 磁盘", "磁盘 64 GiB", "云盘 64", "磁盘配额 64",
               "64 GiB 磁盘纪律")
        for rel in ("PLAN.md", "README.md", "docs/platform_setup.md", "docs/training_tasks.md",
                    "docs/dependencies.md", "docs/image_requirements.md", "docs/PROJECT_FILES.md",
                    "E0/code/check_env.py", "E0/code/setup_deps.sh", "src/data/disk_guard.py",
                    "src/constants.py", "run_train.sh"):
            src = (V4 / rel).read_text(encoding="utf-8")
            for b in bad:
                self.assertNotIn(b, src, f"{rel} 把云盘容量写成了显存容量：{b!r}")


class TestCannVersionNormalization(unittest.TestCase):
    def test_variants_normalize_to_one_form(self):
        for text in ("8.3rc2", "8.3.RC2", "8.3.rc2", "8.3-RC2", "8.3rc2 "):
            self.assertEqual(HW.normalize_cann(text), "8.3rc2", text)

    def test_major_minor(self):
        self.assertEqual(HW.cann_major_minor("8.3.RC2"), (8, 3))
        self.assertEqual(HW.cann_major_minor("8.2.RC1"), (8, 2))
        self.assertEqual(HW.cann_major_minor("8.3"), (8, 3))
        self.assertEqual(HW.cann_major_minor("garbage"), (-1, -1))
        self.assertEqual(HW.cann_major_minor(None), (-1, -1))

    def test_rc_drift_is_same_major_minor(self):
        """rc 后缀漂移不能变成 hard failure（R4-B1 的同类风险）。"""
        self.assertEqual(HW.cann_major_minor("8.3.RC3"), HW.cann_major_minor("8.3rc2"))
        self.assertNotEqual(HW.normalize_cann("8.3.RC3"), HW.normalize_cann("8.3rc2"))


class TestArch(unittest.TestCase):
    def test_arm64_and_aarch64_are_the_same_arch(self):
        self.assertEqual(HW.norm_arch("arm64"), "aarch64")
        self.assertEqual(HW.norm_arch("aarch64"), "aarch64")
        self.assertTrue(HW.arch_matches("arm64"))
        self.assertTrue(HW.arch_matches("aarch64"))

    def test_x86_is_not_the_target_arch(self):
        self.assertEqual(HW.norm_arch("x86_64"), "x86_64")
        self.assertFalse(HW.arch_matches("x86_64"))


class TestAcceleratorDetection(unittest.TestCase):
    """用假 torch 对象注入桩（不需要 torch/torch_npu 真的安装）。"""

    @staticmethod
    def _fake(npu_avail: bool = False, cuda_avail: bool = False):
        def _mod(avail):
            return types.SimpleNamespace(is_available=lambda: avail)
        t = types.SimpleNamespace()
        t.npu = _mod(npu_avail)
        t.cuda = _mod(cuda_avail)
        return t

    def test_npu_wins_over_cuda(self):
        self.assertEqual(HW.detect_accelerator(self._fake(True, True), try_import_accel=False),
                         "npu")

    def test_cuda_used_when_no_npu(self):
        self.assertEqual(HW.detect_accelerator(self._fake(False, True), try_import_accel=False),
                         "cuda")

    def test_cpu_fallback(self):
        self.assertEqual(HW.detect_accelerator(self._fake(False, False), try_import_accel=False),
                         "cpu")

    def test_missing_backends_do_not_raise(self):
        self.assertEqual(HW.detect_accelerator(types.SimpleNamespace(),
                                               try_import_accel=False), "cpu")

    def test_backend_raising_is_treated_as_unavailable(self):
        def _boom():
            raise RuntimeError("device init failed")
        t = types.SimpleNamespace(npu=types.SimpleNamespace(is_available=_boom),
                                  cuda=types.SimpleNamespace(is_available=lambda: False))
        self.assertEqual(HW.detect_accelerator(t, try_import_accel=False), "cpu")

    def test_none_falls_back_to_real_torch_or_none(self):
        """`torch_mod=None` 时会去 import 真 torch：装了就是 cpu/npu/cuda，没装才是 none。"""
        self.assertIn(HW.detect_accelerator(None, try_import_accel=False), ("none", "cpu"))

    def test_device_string(self):
        self.assertEqual(HW.device_string("npu", 0), "npu:0")
        self.assertEqual(HW.device_string("cuda", 1), "cuda:1")
        self.assertEqual(HW.device_string("cpu"), "cpu")


class TestForbiddenInstallPrefixes(unittest.TestCase):
    def test_accel_stack_is_forbidden(self):
        for name in ("torch", "torch_npu", "torch-npu", "torch_npu==", "npu-tools",
                     "ascend-toolkit", "cann", "cann-toolkit", "nvidia-cudnn-cu12",
                     "cuda-python", "triton", "apex", "deepspeed", "flash-attn", "xformers"):
            self.assertTrue(HW.is_forbidden_install(name), name)

    def test_lightweight_deps_are_allowed(self):
        for name in ("numpy", "pandas", "scipy", "scikit-learn", "einops", "tensorboard"):
            self.assertFalse(HW.is_forbidden_install(name), name)

    def test_version_specifiers_are_stripped(self):
        self.assertTrue(HW.is_forbidden_install("torch_npu==2.8.0"))
        self.assertTrue(HW.is_forbidden_install("torch>=2.0"))
        self.assertFalse(HW.is_forbidden_install("numpy==1.26.4"))


class TestDocsFollowTheProfile(unittest.TestCase):
    """文档不得再宣称 A100 / CUDA / torch 2.7.1 / 30 GB 是目标平台。"""

    def test_no_stale_target_platform_claims(self):
        files = ["PLAN.md", "README.md", "docs/platform_setup.md", "docs/training_tasks.md",
                 "docs/dependencies.md", "docs/image_requirements.md", "requirements.txt",
                 "run_train.sh", "E0/code/check_env.py", "E0/code/setup_deps.sh"]
        bad = ("A100", "CUDA 12.8", "torch 2.7.1", "2.7.1+cu128", "sm_80")
        for rel in files:
            src = (V4 / rel).read_text(encoding="utf-8")
            for b in bad:
                if b == "A100":
                    # 允许出现在"变更说明"里（A100 → Ascend 的历史记录）
                    hits = [ln for ln in src.splitlines()
                            if "A100" in ln and "Ascend" not in ln]
                    self.assertEqual(hits, [], f"{rel} 仍把 A100 当目标平台：{hits[:2]}")
                    continue
                self.assertNotIn(b, src, f"{rel} 仍有过期硬件口径：{b}")

    def test_docs_declare_the_new_versions(self):
        joined = "\n".join((V4 / f).read_text(encoding="utf-8")
                           for f in ("PLAN.md", "docs/dependencies.md",
                                     "docs/image_requirements.md", "EC/" + "x" if False else "README.md"))
        for needle in ("Ascend 910B", "torch_npu", "CANN 8.3rc2", "2.8.0", "arm64",
                       "30 GB", "64 GB"):
            self.assertIn(needle, joined, f"文档必须声明 {needle}")

    def test_generators_use_the_new_profile(self):
        for rel in ("docs/gen_plans.py", "docs/gen_p_details.py"):
            src = (V4 / rel).read_text(encoding="utf-8")
            self.assertNotIn("A100", src, f"{rel} 仍生成 A100 文案")
            self.assertIn("Ascend 910B", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
