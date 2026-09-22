#!/usr/bin/env python3
"""目标达成度审计（**纯标准库**）：对照 E1–E10 的交付物，逐项给出"本地已完成 / 待云端"。

审计口径
-------
对每个阶段核验四件事（缺一即 `local_complete=false`）：

1. **可运行代码**：入口脚本存在（清单复用 `tools/check_pipeline.py` 的 `STAGE_SCRIPTS`）；
2. **单测**：该阶段有对应测试文件（`tests/test_*.py`）且包含 `def test_` 用例；
3. **Gate/报告命名**：阶段关键报告名在代码或计划里出现（复用 `REPORT_TOKENS`）；
4. **接线**：`run_train.sh` 有该阶段分支；文档（`docs/PROJECT_FILES.md`）提到该阶段代码。

输出还把**只能在云端完成**的部分显式列出（正式 Gate 数值、A 榜、B0 复核、官方数据复现），
避免把"本地契约层已通过"误读成"比赛成绩已拿到"。

用法::

    python3 tools/audit_goal.py --json reports/E0_goal_audit.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]

STAGE_TESTS: dict[str, tuple[str, ...]] = {
    "E0": ("tests/test_contract.py", "tests/test_gates.py", "tests/test_score.py",
           "tests/test_plan_stats.py", "tests/test_platform_scripts.py",
           "tests/test_disk_guard.py", "tests/test_hardware.py", "tests/test_parse.py",
           "tests/test_atomic_gate.py", "tests/test_pipeline_check.py"),
    "E1": ("tests/test_e1_pipeline.py", "tests/test_row_dataset.py",
           "tests/test_training_core.py"),
    "E2": ("tests/test_features_e2.py",),
    "E3": ("tests/test_seq_pipeline.py", "tests/test_models_seq.py"),
    "E4": ("tests/test_patchtf_e4.py", "tests/test_e4_pipeline.py"),
    "E5": ("tests/test_target_heads_e5.py", "tests/test_frozen_e5.py",
           "tests/test_e5_pipeline.py", "tests/test_e5_perm_pipeline.py",
           "tests/test_e5_sw_pipeline.py", "tests/test_e5_aggregate.py"),
    "E6": ("tests/test_state_train_e6.py", "tests/test_e6_pipeline.py", "tests/test_e6_tau.py",
           "tests/test_e6_gate.py", "tests/test_e6_p2.py", "tests/test_metrics_auc.py",
           "tests/test_registry_writes.py"),
    "E7": ("tests/test_loss_ablation_lib.py", "tests/test_e7_ablate_loss.py",
           "tests/test_e7_decode.py"),
    "E8": ("tests/test_models_e8.py", "tests/test_e8_mmoe.py", "tests/test_e8_well_branch.py",
           "tests/test_e8_transductive.py", "tests/test_e8_ensemble.py"),
    "E9": ("tests/test_e9_aggregate.py", "tests/test_e9_choose.py", "tests/test_e9_leakage.py",
           "tests/test_e9_confirm.py", "tests/test_e9_submit.py"),
    "E10": ("tests/test_e10_final_train.py", "tests/test_e10_package.py",
            "tests/test_e10_submit.py", "tests/test_e10_b0.py", "tests/test_predict_pd1.py"),
}
CROSS_CUTTING: dict[str, tuple[str, ...]] = {
    "推理入口": ("predict.py", "tests/test_predict_pd1.py"),
    "训练入口": ("train.py", "tests/test_e10_final_train.py"),
    "PD1 注册": ("src/versioning/registry.py", "tests/test_registry_writes.py"),
    "集成融合": ("src/ensemble/blend.py", "tests/test_ensemble_blend.py"),
    "序列主干": ("src/models/unet1d.py", "src/models/tcn.py", "src/models/patchtf.py"),
    "冻结骨干": ("src/training/frozen.py", "tests/test_frozen_e5.py"),
    "两阶段件": ("src/training/state_train.py", "tests/test_state_train_e6.py"),
    "解码层": ("src/inference/decode.py", "tests/test_e7_decode.py"),
    "EMA/SWA": ("src/training/ema.py", "tests/test_models_e8.py"),
}
CLOUD_PENDING: tuple[str, ...] = (
    "E1 5 折 OOF ≥ 78.0（正式 Gate，需云端跑 80 井）",
    "E2 全折单组消融 + 内存画像（正式 Gate）",
    "E3 5 折 OOF ≥ 81.0 + 感受野消融 + 边界体检（正式 Gate）",
    "E4 网格搜索/CI·位置消融的正式数值（本地仅链路预检）",
    "E5 三目标连续切片 Acc 的配对 CI（需冻结骨干权重）",
    "E6 原子 Acc ≥ 0.99 / 召回 ≥ 0.98 / 联合 AUC ≥ 0.97（正式 Gate）",
    "E7 λ1/λ2/边界聚焦/PERM 截断/exp7 L_phys 的正式消融结论（代码已实现，需云端真实数据验证）",
    "E8 MMoE/井级/transductive/集成的正式 CI 结论（需真实 OOF 成员）",
    "E9 护栏达标与 A 榜首轮成绩（需真实候选 + 平台配额）",
    "E10 全量重训、干净目录复现（官方 10 井 95,948 行）、B0 兜底当场验证",
)


def _load_pipeline_mod():
    spec = importlib.util.spec_from_file_location(
        "v4_check_pipeline_for_audit", str(V4 / "tools" / "check_pipeline.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def count_tests(path: Path) -> int:
    if not path.is_file():
        return 0
    return len(re.findall(r"^\s*def test_", path.read_text(encoding="utf-8", errors="ignore"),
                          flags=re.M))


def audit(repo: Path = V4) -> dict:
    cp = _load_pipeline_mod()
    pipeline = cp.run(repo)
    docs = (repo / "docs" / "PROJECT_FILES.md")
    docs_text = docs.read_text(encoding="utf-8") if docs.is_file() else ""
    run_train = (repo / "run_train.sh")
    rt_text = run_train.read_text(encoding="utf-8") if run_train.is_file() else ""

    stages: dict[str, dict] = {}
    local_ok = True
    for stage, scripts in cp.STAGE_SCRIPTS.items():
        code_missing = [s for s in scripts if not (repo / s).is_file()]
        tests = STAGE_TESTS.get(stage, ())
        test_files = [t for t in tests if (repo / t).is_file()]
        n_tests = sum(count_tests(repo / t) for t in tests)
        gate_tokens = list(cp.REPORT_TOKENS.get(stage, ()))
        blob = ""
        for p in [*(repo / stage).glob("code/*.py"), (repo / stage / "PLAN.md")]:
            if p.is_file():
                blob += p.read_text(encoding="utf-8", errors="ignore")
        gates_ok = all(tok in blob for tok in gate_tokens) if gate_tokens else True
        wiring = (f"{stage})" in rt_text) if stage != "E0" else True
        documented = stage in docs_text
        ok = bool(not code_missing and test_files and n_tests > 0 and gates_ok and wiring
                  and documented)
        local_ok = local_ok and ok
        stages[stage] = {"ok": bool(ok), "code_missing": code_missing,
                         "tests": list(test_files), "n_tests": int(n_tests),
                         "gate_tokens": gate_tokens, "gates_ok": bool(gates_ok),
                         "wiring": bool(wiring), "documented": bool(documented),
                         "cloud_pending": [c for c in CLOUD_PENDING if c.startswith(stage)]}

    cross: dict[str, dict] = {}
    for name, paths in CROSS_CUTTING.items():
        missing = [p for p in paths if not (repo / p).is_file()]
        cross[name] = {"ok": bool(not missing), "missing": missing}
        local_ok = local_ok and not missing

    total_tests = sum(count_tests(p) for p in sorted((repo / "tests").glob("test_*.py")))
    n_test_files = len(list((repo / "tests").glob("test_*.py")))
    return {"ok": bool(local_ok and pipeline["ok"]),
            "local_complete": bool(local_ok and pipeline["ok"]),
            "repo": str(repo),
            "pipeline_check": {"ok": pipeline["ok"], "problems": pipeline["problems"]},
            "stages": stages, "cross_cutting": cross,
            "test_suite": {"n_test_files": n_test_files, "n_test_methods": total_tests},
            "cloud_pending": list(CLOUD_PENDING),
            "note": ("本审计只证明**本地契约层**交付完整；正式 Gate 数值、A 榜成绩与"
                     "官方数据复现属于云端任务，见 cloud_pending")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="目标达成度审计")
    ap.add_argument("--repo", default=str(V4))
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    res = audit(Path(args.repo))
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
    print(json.dumps({k: v for k, v in res.items() if k != "stages"},
                     ensure_ascii=False, indent=2))
    print("阶段状态：" + "  ".join(
        f"{k}={'OK' if v['ok'] else 'GAP'}" for k, v in res["stages"].items()))
    return 0 if res["local_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
