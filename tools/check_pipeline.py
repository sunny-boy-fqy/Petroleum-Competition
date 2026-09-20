#!/usr/bin/env python3
"""流水线完整性检查（**纯标准库**）：阶段脚本 / 入口 / 接线 / 报告命名一次性核对。

为什么需要它
-----------
PLAN.md 里的"输出契约"很容易只写在文档里而**代码没有对应物**。本检查把 E1–E10 的
关键产物固化成清单，任何一项缺失即失败（退出码 1），从而避免"文档说做过、仓库里没有"。

检查四类
-------
1. **PLAN 文件**：每个阶段目录都有 `PLAN.md`（含 P 子目录的 PLAN）；
2. **阶段脚本**：E1–E10 的入口脚本逐个存在（清单见 `STAGE_SCRIPTS`）；
3. **接线**：`run_train.sh` 有对应 case 分支；`train.py` / `predict.py` / `configs/v4.yaml`
   / `requirements.txt` 存在；`predict.py` 能列出 PD1；
4. **报告命名**：每个阶段的关键报告文件名在计划/脚本里出现（`REPORT_TOKENS`），
   防止"脚本写出的报告名与计划不一致"。

用法::

    python3 tools/check_pipeline.py            # 人读摘要 + JSON 落 reports/
    python3 tools/check_pipeline.py --json /tmp/x.json
    python3 tools/check_pipeline.py --repo /path/to/v4   # 指定仓库根（测试用）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]

STAGE_SCRIPTS: dict[str, tuple[str, ...]] = {
    "E0": ("E0/code/run_all.py", "E0/code/check_env.py"),
    "E1": ("E1/code/train_row.py",),
    "E2": ("E2/code/build_features.py", "E2/code/ablate_groups.py"),
    "E3": ("E3/code/train_seq.py", "E3/code/rf_ablation.py", "E3/code/compare_row_vs_seq.py"),
    "E4": ("E4/code/train_patchtf.py",),
    "E5": ("E5/code/head_por.py", "E5/code/head_perm.py", "E5/code/head_sw.py",
           "E5/code/evaluate_targets.py", "E5/code/e5_common.py"),
    "E6": ("E6/code/train_state.py", "E6/code/search_tau.py", "E6/code/build_pd1.py",
           "E6/code/gate.py"),
    "E7": ("E7/code/ablate_loss.py", "E7/code/decode_search.py"),
    "E8": ("E8/code/train_mmoe.py", "E8/code/well_branch.py", "E8/code/pseudo_label.py",
           "E8/code/ensemble.py"),
    "E9": ("E9/code/aggregate_oof.py", "E9/code/choose_submission.py",
           "E9/code/leakage_audit.py", "E9/code/confirm_check.py", "E9/code/submit_batch.py"),
    "E10": ("E10/code/final_train.py", "E10/code/export_cpu.py", "E10/code/verify_inference.py",
            "E10/code/build_submission.py", "E10/code/build_b0_fallback.py", "E10/code/submit.py"),
}
ENTRY_FILES: tuple[str, ...] = ("train.py", "predict.py", "run_train.sh", "requirements.txt",
                               "configs/v4.yaml")
RUN_TRAIN_STAGES: tuple[str, ...] = tuple(f"E{i}" for i in range(1, 11))
REPORT_TOKENS: dict[str, tuple[str, ...]] = {
    "E1": ("E1_gate.json", "E1_metrics.json"),
    "E2": ("E2_ablation.json", "E2_throughput.json", "E2_mem_profile.json"),
    "E3": ("E3_gate.json", "E3_boundary_report.json"),
    "E4": ("E4_gate.json", "E4_patchtf.json"),
    "E5": ("E5_gate.json", "E5_per_target.json"),
    "E6": ("E6_gate.json", "E6_atomic_report.json", "E6_tau_search.json"),
    "E7": ("E7_loss_ablation.json", "E7_decode_search.json"),
    "E8": ("E8_gate.json", "E8_ensemble_report.json", "E8_mmoe.json",
           "E8_well_branch.json", "E8_transductive.json"),
    "E9": ("E9_validation_report.json", "E9_submission_decision.json",
           "E9_leakage_audit.json", "E9_a_board_log.json"),
    "E10": ("E10_final_train.json", "E10_reproduce_report.json", "E10_submission_log.json",
            "E10_B0_fallback.json"),
}


def check_plans(repo: Path) -> list[str]:
    problems = []
    for stage in STAGE_SCRIPTS:
        d = repo / stage
        if not (d / "PLAN.md").is_file():
            problems.append(f"缺少 {stage}/PLAN.md")
        for p in sorted(d.glob("P*/PLAN.md")):
            if p.stat().st_size < 200:
                problems.append(f"{p.relative_to(repo)} 内容过短（<200B）")
    return problems


def check_scripts(repo: Path) -> list[str]:
    problems = []
    for stage, scripts in STAGE_SCRIPTS.items():
        for rel in scripts:
            p = repo / rel
            if not p.is_file():
                problems.append(f"缺少阶段脚本 {rel}")
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            if "argparse" not in text and not rel.endswith(("e5_common.py", "run_all.py")):
                problems.append(f"{rel} 没有 argparse（无法作为可运行入口？）")
    return problems


def check_wiring(repo: Path) -> list[str]:
    problems = []
    for rel in ENTRY_FILES:
        if not (repo / rel).is_file():
            problems.append(f"缺少入口/配置文件 {rel}")
    rt = repo / "run_train.sh"
    if rt.is_file():
        src = rt.read_text(encoding="utf-8")
        for stage in RUN_TRAIN_STAGES:
            if f"{stage})" not in src:
                problems.append(f"run_train.sh 缺少 {stage} 分支")
    train = repo / "train.py"
    if train.is_file():
        src = train.read_text(encoding="utf-8")
        for stage in ("E1", "E3", "E4", "E5", "E6", "E8", "E10"):
            if f'"{stage}"' not in src:
                problems.append(f"train.py 未接线阶段 {stage}")
    pred = repo / "predict.py"
    if pred.is_file() and "PD1" not in pred.read_text(encoding="utf-8"):
        problems.append("predict.py 未接线 PD1")
    return problems


def check_reports(repo: Path) -> list[str]:
    """报告命名一致性：`REPORT_TOKENS` 至少要在该阶段的计划或脚本里出现。"""
    problems = []
    for stage, tokens in REPORT_TOKENS.items():
        blob = ""
        d = repo / stage
        for p in [d / "PLAN.md", *sorted(d.glob("P*/PLAN.md")), *sorted(d.glob("code/*.py"))]:
            if p.is_file():
                blob += p.read_text(encoding="utf-8", errors="ignore")
        for tok in tokens:
            if tok not in blob:
                problems.append(f"{stage} 的报告名 {tok} 未在计划/脚本中出现（命名不一致？）")
    return problems


def run(repo: Path) -> dict:
    problems = {
        "plans": check_plans(repo),
        "scripts": check_scripts(repo),
        "wiring": check_wiring(repo),
        "reports": check_reports(repo),
    }
    flat = [f"[{k}] {v}" for k, vs in problems.items() for v in vs]
    return {"ok": not flat, "n_problems": len(flat), "problems": flat, "detail": problems,
            "n_stages": len(STAGE_SCRIPTS),
            "n_scripts": sum(len(v) for v in STAGE_SCRIPTS.values()),
            "repo": str(repo)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="流水线完整性检查")
    ap.add_argument("--repo", default=str(V4))
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    res = run(Path(args.repo))
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
