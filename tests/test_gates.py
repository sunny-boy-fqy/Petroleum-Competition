"""Gate 校验器测试（R2-B2/H4：metric 类型、绝对门槛、boolean Gate、模板 vs 生效）。"""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src.validation import gates as G  # noqa: E402

CORE = list(G.CORE_MANDATORY)


def _prereg(**over) -> dict:
    base = {
        "gate_id": "E9_P9_gate", "stage": "E9", "p_stage": "P9",
        "created_at": "2026-01-01T00:00:00+00:00",
        "gate_type": "delta", "primary_metric": "oof_total",
        "primary_threshold_key": "min_delta",
        "baseline_version": "E1_PD0", "baseline_artifact": "experiments/E1/oof.npz",
        "baseline_manifest_sha256": "a" * 64,
        "thresholds": {"min_delta": 0.0, "min_effect_floor": 0.0},
        "alpha": 0.05, "multiplicity": "none", "candidate_budget": 1,
        "bootstrap_iters": 1000, "bootstrap_unit": "well_row_weighted_cluster",
        "pilot_std": None, "mde_units": 80, "min_detectable_effect": None,
        "planned_task_training_h": 1.0, "mandatory_checks": list(CORE),
        "decisions_locked": [],
    }
    base.update(over)
    return base


class TestValidatePrereg(unittest.TestCase):
    def test_valid_passes(self):
        self.assertEqual(G.validate_prereg(_prereg()), [])

    def test_missing_field(self):
        d = _prereg(); d.pop("thresholds")
        self.assertTrue(G.validate_prereg(d))

    def test_placeholder_rejected_when_effective(self):
        # 生效预注册（非模板）里出现空/占位基线 -> 必须被拒
        self.assertTrue(any("placeholder" in e for e in
                            G.validate_prereg(_prereg(baseline_artifact=""))))
        self.assertTrue(any("placeholder" in e for e in
                            G.validate_prereg(_prereg(baseline_version="TBD"))))
        # baseline_manifest_sha256 对 delta 类 Gate 必填
        self.assertTrue(any("baseline_manifest_sha256" in e for e in
                            G.validate_prereg(_prereg(baseline_manifest_sha256=""))))
        # 对 boolean 类 Gate 可留空（无上游 manifest）
        d = _prereg(gate_type="boolean", primary_metric="env_hard_checks_passed",
                    baseline_manifest_sha256="")
        self.assertFalse(any("baseline_manifest_sha256" in e
                             for e in G.validate_prereg(d)))

    def test_template_allows_placeholders(self):
        d = _prereg(created_at="<ISO8601>", baseline_artifact="<path>",
                    baseline_manifest_sha256="<sha256>")
        self.assertTrue(G.is_template(d))
        self.assertEqual(G.validate_prereg(d), [])
        # 显式 template=True 时即使没有占位符也放宽内容检查（仓库模板场景）
        d2 = _prereg(created_at="", baseline_version="", baseline_artifact="",
                     baseline_manifest_sha256="")
        self.assertEqual(G.validate_prereg(d2, template=True), [])

    def test_multiplicity_none_requires_single_candidate(self):
        self.assertTrue(G.validate_prereg(_prereg(candidate_budget=3)))
        self.assertEqual(G.validate_prereg(_prereg(candidate_budget=3,
                                                   multiplicity="holm")), [])

    def test_gate_id_consistency(self):
        self.assertTrue(any("gate_id" in e for e in
                            G.validate_prereg(_prereg(gate_id="X_Y_gate"))))

    def test_core_mandatory_or_exempt(self):
        d = _prereg(mandatory_checks=["contract_ok"],
                    mandatory_exempt=[c for c in CORE if c != "contract_ok"])
        self.assertEqual(G.validate_prereg(d), [])
        d2 = _prereg(mandatory_checks=["contract_ok"])
        self.assertTrue(any("missing core" in e for e in G.validate_prereg(d2)))

    def test_mde_consistency(self):
        mde = G.mde_two_sided(1.0, 80)
        self.assertEqual(G.validate_prereg(_prereg(pilot_std=1.0,
                                                   min_detectable_effect=mde)), [])
        self.assertTrue(G.validate_prereg(_prereg(pilot_std=1.0,
                                                  min_detectable_effect=mde * 2)))


class TestGateTypes(unittest.TestCase):
    def _inferred(self, **over):
        d = _prereg(**over)
        d.pop("gate_type", None)          # 测"按 primary_metric 推断"
        return d

    def test_inference(self):
        self.assertEqual(G.gate_type(self._inferred(primary_metric="oof_total")), "delta")
        self.assertEqual(
            G.gate_type(self._inferred(primary_metric="env_hard_checks_passed")),
            "boolean")
        self.assertEqual(
            G.gate_type(self._inferred(primary_metric="seq_pipeline_ok")), "boolean")
        self.assertEqual(
            G.gate_type(self._inferred(primary_metric="confirm_non_inferiority")),
            "non_inferior")
        # 显式声明优先于推断
        self.assertEqual(
            G.gate_type(_prereg(gate_type="absolute",
                                primary_metric="env_hard_checks_passed")),
            "absolute")

    def test_boolean_gate_ignores_ci(self):
        d = _prereg(gate_type="boolean", primary_metric="env_hard_checks_passed",
                    thresholds={"min_delta": 0.0, "min_effect_floor": 0.0})
        r = G.aggregate_gate(d, {"checks": {c: True for c in CORE}})
        self.assertTrue(r["passed"], r)

    def test_boolean_gate_fails_on_mandatory(self):
        d = _prereg(gate_type="boolean", primary_metric="env_hard_checks_passed")
        chk = {c: True for c in CORE}; chk["contract_ok"] = False
        self.assertFalse(G.aggregate_gate(d, {"checks": chk})["passed"])


class TestDeltaGateAbsoluteThreshold(unittest.TestCase):
    """R2-B2 核心：oof_total_min 等绝对门槛必须参与判定。"""

    def _d(self):
        return _prereg(thresholds={"min_delta": 0.0, "min_effect_floor": 0.0,
                                   "oof_total_min": 81.0})

    def test_pass_requires_both_delta_and_absolute(self):
        chk = {c: True for c in CORE}
        # delta/CI 达标但绝对分不够 -> 必须 False（修复前会 True）
        r = G.aggregate_gate(self._d(), {"delta": 0.5, "paired_ci_low": 0.1,
                                         "oof_total": 80.0, "checks": chk})
        self.assertFalse(r["passed"], r)
        self.assertIn("oof_total_min", r["details"]["absolute_checks"])
        # 两者都达标 -> True
        r2 = G.aggregate_gate(self._d(), {"delta": 0.5, "paired_ci_low": 0.1,
                                          "oof_total": 81.5, "checks": chk})
        self.assertTrue(r2["passed"], r2)

    def test_placeholder_absolute_threshold_is_enforced(self):
        """审查 WP0：min_placeholder_acc 必须真正参与 Gate，0.9723 不得判过。"""
        d = _prereg(thresholds={"min_delta": 0.0, "min_effect_floor": 0.0,
                                "min_placeholder_acc": 0.98})
        chk = {c: True for c in CORE}
        r = G.aggregate_gate(d, {"delta": 0.5, "paired_ci_low": 0.1,
                                 "placeholder_min_acc": 0.9723, "checks": chk})
        self.assertFalse(r["passed"], r)
        self.assertFalse(r["details"]["absolute_checks"]["min_placeholder_acc"]["ok"])
        r2 = G.aggregate_gate(d, {"delta": 0.5, "paired_ci_low": 0.1,
                                  "placeholder_min_acc": 0.99, "checks": chk})
        self.assertTrue(r2["passed"], r2)

    def test_delta_floor(self):
        d = _prereg(thresholds={"min_delta": 0.0, "min_effect_floor": 0.2})
        self.assertAlmostEqual(G.effective_threshold(d), 0.2)
        chk = {c: True for c in CORE}
        self.assertFalse(G.aggregate_gate(d, {"delta": 0.1, "paired_ci_low": 0.05,
                                              "checks": chk})["passed"])


class TestNonInferiority(unittest.TestCase):
    def test_margin_semantics(self):
        d = _prereg(gate_type="non_inferior", primary_metric="confirm_non_inferiority",
                    primary_threshold_key="non_inferiority_margin",
                    thresholds={"non_inferiority_margin": 1.0, "min_effect_floor": 0.0})
        chk = {c: True for c in CORE}
        self.assertTrue(G.aggregate_gate(d, {"delta": -0.5, "paired_ci_low": -0.9,
                                             "checks": chk})["passed"])
        self.assertFalse(G.aggregate_gate(d, {"delta": -1.5, "paired_ci_low": -0.9,
                                              "checks": chk})["passed"])


class TestAbsoluteThresholdDirection(unittest.TestCase):
    """R3-H1：`min_*` 用 `>=`、`max_*` 用 `<=`，并映射到正确的 result 字段。

    修复前 `aggregate_gate` 把所有绝对键一律按 `have >= want` 判定，于是
    `max_degradation=0.1` 被当成下界（要求 degradation>=0.1），E9/P2 的
    `degradation=0.01` 被误判为 FAIL；`max_minutes`/`max_memory_gb`/`max_point_diff`
    在 boolean Gate 下则完全不参与判定（静默失效）。
    """

    def test_direction_registry(self):
        for k in ("oof_total_min", "min_auc", "min_atom_acc", "min_atom_recall"):
            self.assertEqual(G.absolute_direction(k), "min", k)
        for k in ("max_hard_failures", "max_point_diff", "max_minutes",
                  "max_memory_gb", "max_degradation", "abs_tolerance"):
            self.assertEqual(G.absolute_direction(k), "max", k)
        with self.assertRaises(KeyError):
            G.absolute_direction("max_not_a_real_key")

    def test_every_absolute_key_has_result_field_mapping(self):
        """每个登记的方向键都必须有 result 字段映射，否则会取不到值而静默失败。"""
        for k in G.ABSOLUTE_KEYS:
            self.assertIn(k, G.METRIC_RESULT_FIELDS, k)
            self.assertTrue(G.METRIC_RESULT_FIELDS[k], k)

    def _tpl(self, stage: str, pstage: str) -> dict:
        """读取仓库里真实的 P 级预注册模板（P 级 PLAN.md 里的 ```json 块）。"""
        f = V4 / stage / pstage / "PLAN.md"
        m = re.search(r"```json\n(.*?)\n```", f.read_text(encoding="utf-8"), re.S)
        self.assertIsNotNone(m, f)
        return json.loads(m.group(1))

    def _checks(self, prereg: dict) -> dict:
        return {c: True for c in prereg["mandatory_checks"]}

    def test_e9_p2_max_degradation_is_upper_bound(self):
        """E9/P2：degradation=0.01 必须 PASS；超界必须 FAIL；缺指标必须 FAIL。"""
        pr = self._tpl("E9", "P2")
        self.assertIn("max_degradation", pr["thresholds"])
        r = G.aggregate_gate(pr, {"checks": self._checks(pr), "degradation": 0.01})
        self.assertTrue(r["passed"], r)
        self.assertTrue(r["details"]["absolute_checks"]["max_degradation"]["ok"])
        self.assertEqual(r["details"]["absolute_checks"]["max_degradation"]["direction"], "max")
        # 修复前这里会被判 FAIL：max_degradation 被当成下界 0.1
        r2 = G.aggregate_gate(pr, {"checks": self._checks(pr), "degradation": 0.35})
        self.assertFalse(r2["passed"], r2)
        # 拿不到指标 -> 未通过（不可复算的 Gate 不能算过）
        r3 = G.aggregate_gate(pr, {"checks": self._checks(pr)})
        self.assertFalse(r3["passed"], r3)

    def test_e10_p0_resource_limits_are_enforced(self):
        """E10/P0：boolean Gate 的 max_minutes / max_memory_gb 必须真正生效。"""
        pr = self._tpl("E10", "P0")
        self.assertIn("max_minutes", pr["thresholds"])
        self.assertIn("max_memory_gb", pr["thresholds"])
        chk = self._checks(pr)
        self.assertTrue(G.aggregate_gate(
            pr, {"checks": chk, "minutes": 28.0, "memory_gb": 6.0})["passed"])
        self.assertFalse(G.aggregate_gate(
            pr, {"checks": chk, "minutes": 31.0, "memory_gb": 6.0})["passed"])
        self.assertFalse(G.aggregate_gate(
            pr, {"checks": chk, "minutes": 28.0, "memory_gb": 9.0})["passed"])
        # 不提供指标 -> 必须在结果里失败，而不是当作没声明
        res = G.aggregate_gate(pr, {"checks": chk})["details"]["absolute_checks"]
        self.assertEqual(set(res), {"max_minutes", "max_memory_gb"})
        self.assertTrue(all(not v["ok"] for v in res.values()))

    def test_e10_p1_point_diff_and_abs_tolerance_are_upper_bounds(self):
        """E10/P1：max_point_diff 必须真正生效；abs_tolerance 比较的是偏差绝对值。"""
        pr = self._tpl("E10", "P1")
        chk = self._checks(pr)
        self.assertTrue(G.aggregate_gate(pr, {"checks": chk, "point_diff": 1e-9})["passed"])
        self.assertFalse(G.aggregate_gate(pr, {"checks": chk, "point_diff": 1e-3})["passed"])
        # abs_tolerance -> result["abs_diff"]，方向为 max
        d = _prereg(gate_type="boolean", primary_metric="data_card_recomputable",
                    primary_threshold_key="abs_tolerance",
                    thresholds={"abs_tolerance": 1e-4})
        self.assertTrue(G.aggregate_gate(
            d, {"checks": {c: True for c in CORE}, "abs_diff": 5e-5})["passed"])
        self.assertFalse(G.aggregate_gate(
            d, {"checks": {c: True for c in CORE}, "abs_diff": 5e-2})["passed"])

    def test_metric_specific_field_mapping(self):
        """state_auc -> auc；atomic_f1 -> atomic_acc/atomic_f1；逐目标原子指标同名映射。"""
        cases = [
            ({"min_auc": 0.97}, "auc", 0.98, 0.95),
            ({"min_atomic_acc": 0.99}, "atomic_acc", 0.995, 0.98),
            ({"min_atom_recall": 0.98}, "atom_recall", 0.99, 0.90),
            ({"min_por_acc": 0.99}, "por_acc", 0.995, 0.97),
            ({"max_hard_failures": 0}, "hard_failures", 0, 3),
        ]
        for th, field, good, bad in cases:
            d = _prereg(thresholds={"min_delta": 0.0, **th})
            base = {"delta": 1.0, "paired_ci_low": 0.2, "checks": {c: True for c in CORE}}
            self.assertTrue(G.aggregate_gate(d, {**base, field: good})["passed"],
                            (th, field, good))
            self.assertFalse(G.aggregate_gate(d, {**base, field: bad})["passed"],
                             (th, field, bad))

    def test_unknown_absolute_key_is_rejected(self):
        """方向未知的 min_*/max_* 键必须在 validate_prereg 阶段就报错，而不是静默放过。"""
        d = _prereg(thresholds={"min_delta": 0.0, "max_made_up_thing": 1.0})
        self.assertTrue(any("方向未知" in e for e in G.validate_prereg(d)))
        # 模板也要查（否则坏模板会一直躺在仓库里）
        d2 = _prereg(created_at="<ISO8601>", thresholds={"min_delta": 0.0,
                                                         "max_made_up_thing": 1.0})
        self.assertTrue(any("方向未知" in e for e in G.validate_prereg(d2, template=True)))

    def test_boolean_gate_still_requires_mandatory(self):
        d = _prereg(gate_type="boolean", primary_metric="env_hard_checks_passed",
                    primary_threshold_key="max_hard_failures",
                    thresholds={"max_hard_failures": 0})
        chk = {c: True for c in CORE}
        self.assertTrue(G.aggregate_gate(d, {"checks": chk, "hard_failures": 0})["passed"])
        chk2 = dict(chk); chk2["contract_ok"] = False
        self.assertFalse(G.aggregate_gate(d, {"checks": chk2, "hard_failures": 0})["passed"])


class TestGateTypeInferenceSingleSource(unittest.TestCase):
    """R3-H1：`infer_gate_type` 是唯一事实源；生成器与校验器不得各抄一份清单。"""

    def test_infer_gate_type_matches_gate_type(self):
        for pm in sorted(G.VALID_PRIMARY) + ["a_board_no_breakdown", "cpu_inference_ok"]:
            self.assertEqual(G.infer_gate_type(pm), G.gate_type({"primary_metric": pm}),
                             pm)

    def test_explicit_field_wins(self):
        self.assertEqual(
            G.infer_gate_type("env_hard_checks_passed", "absolute"), "absolute")
        self.assertEqual(
            G.infer_gate_type("oof_total", "boolean"), "boolean")



    """仓库内 33 份模板必须全部通过浅层校验；E0 实际预注册必须通过严格校验。"""

    def test_all_templates_valid(self):
        files = sorted(V4.glob("E*/P*/PLAN.md"))
        self.assertEqual(len(files), 33, f"expect 33 P plans, got {len(files)}")
        for f in files:
            m = re.search(r"```json\n(.*?)\n```", f.read_text(encoding="utf-8"), re.S)
            self.assertIsNotNone(m, f)
            d = json.loads(m.group(1))
            self.assertEqual(G.validate_prereg(d, template=True), [], f)

    def test_e0_actual_prereg_is_strict_valid(self):
        p = V4 / "reports" / "E0_gate_prereg.json"
        if not p.is_file():
            self.skipTest("E0_gate_prereg.json missing")
        d = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(G.validate_prereg(d), [])


class TestAbsoluteGateMetricMapping(unittest.TestCase):
    """R4-M3：`gate_type=absolute` 的判定必须走指标字段映射，而不是永远读 `score`。

    旧实现在 absolute 分支里硬编码 `result["score"]`，因此
    `primary_threshold_key=min_auc` / `max_minutes` 这类键会取错值（或取不到值而
    静默判失败）。33 份模板目前没有 absolute Gate，所以这是个 latent bug。
    """

    def _abs(self, key: str, want: float):
        return _prereg(gate_type="absolute", primary_metric="state_auc",
                       primary_threshold_key=key, thresholds={key: want})

    def test_min_auc_uses_auc_field_not_score(self):
        d = self._abs("min_auc", 0.97)
        checks = {c: True for c in d["mandatory_checks"]}
        # auc 达标但 score 很低：旧实现会因为读 score 而误判
        ok = G.aggregate_gate(d, {"checks": checks, "auc": 0.98, "score": 1.0})
        self.assertTrue(ok["passed"], ok)
        self.assertEqual(ok["details"]["field"], "auc")
        bad = G.aggregate_gate(d, {"checks": checks, "auc": 0.95, "score": 100.0})
        self.assertFalse(bad["passed"])
        self.assertEqual(bad["details"]["field"], "auc")

    def test_max_minutes_uses_minutes_field_and_max_direction(self):
        d = self._abs("max_minutes", 40.0)
        checks = {c: True for c in d["mandatory_checks"]}
        ok = G.aggregate_gate(d, {"checks": checks, "minutes": 31.0, "score": 0.0})
        self.assertTrue(ok["passed"], ok)
        self.assertEqual(ok["details"]["direction"], "max")
        self.assertEqual(ok["details"]["field"], "minutes")
        bad = G.aggregate_gate(d, {"checks": checks, "minutes": 55.0})
        self.assertFalse(bad["passed"])

    def test_state_auc_alias_is_accepted(self):
        d = self._abs("min_auc", 0.9)
        checks = {c: True for c in d["mandatory_checks"]}
        ok = G.aggregate_gate(d, {"checks": checks, "state_auc": 0.95})
        self.assertTrue(ok["passed"])
        self.assertEqual(ok["details"]["field"], "state_auc")

    def test_missing_metric_fails_not_passes(self):
        d = self._abs("min_auc", 0.9)
        checks = {c: True for c in d["mandatory_checks"]}
        res = G.aggregate_gate(d, {"checks": checks, "score": 100.0})
        self.assertFalse(res["passed"], "拿不到 auc 时必须判失败（宁严勿松）")
        self.assertIn("缺少", res["details"]["error"])

    def test_legacy_score_key_still_works(self):
        """未登记映射的键（如 'score'）保留旧行为：回落到 result['score']。"""
        d = self._abs("score", 60.0)
        checks = {c: True for c in d["mandatory_checks"]}
        self.assertTrue(G.aggregate_gate(d, {"checks": checks, "score": 61.0})["passed"])
        self.assertFalse(G.aggregate_gate(d, {"checks": checks, "score": 59.0})["passed"])


class TestE0PreregRecompute(unittest.TestCase):
    """R4-H1：E0 的 prereg 与实物报告必须能用**同一个**校验器复算通过。

    四审实测：prereg 有 13 项（含 `contract_ok`），report 只有 12 项（含
    `contract_selftest`），且 report 没有 `abs_diff` -> `aggregate_gate` 直接
    `passed=False`。这个回归把"预注册能否被实物复算"变成硬断言。
    """

    def _load(self, name: str) -> dict | None:
        p = V4 / "reports" / name
        if not p.is_file():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def test_reports_exist(self):
        for name in ("E0_gate_prereg.json", "E0_local_contract_gate.json"):
            self.assertIsNotNone(self._load(name), f"缺少 {name}")

    def test_aggregate_gate_recompute_passes(self):
        prereg = self._load("E0_gate_prereg.json")
        report = self._load("E0_local_contract_gate.json")
        checks = report.get("mandatory_checks") or report.get("checks")
        result = {"checks": checks}
        for k in ("abs_diff", "score_total_abs_diff"):
            if k in report:
                result[k] = report[k]
        out = G.aggregate_gate(prereg, result)
        self.assertEqual(out["prereg_errors"], [], out["prereg_errors"])
        self.assertEqual(out["mandatory_failures"], [], out["mandatory_failures"])
        self.assertTrue(out["passed"], json.dumps(out, ensure_ascii=False)[:1200])

    def test_prereg_and_report_declare_the_same_checks(self):
        prereg = self._load("E0_gate_prereg.json")
        report = self._load("E0_local_contract_gate.json")
        declared = set(prereg["mandatory_checks"])
        actual = set(report.get("mandatory_checks") or report.get("checks") or {})
        self.assertEqual(declared - actual, set(),
                         "prereg 声明了 report 没有的 mandatory check（无法复算）")
        self.assertEqual(actual - declared, set(),
                         "report 有 prereg 未声明的 mandatory check（预注册漏记）")

    def test_report_exposes_abs_tolerance_metric(self):
        report = self._load("E0_local_contract_gate.json")
        self.assertIn("abs_diff", report)
        self.assertLessEqual(abs(float(report["abs_diff"])), 1e-4,
                             "常数基线锚点偏差必须 <= abs_tolerance=1e-4")


if __name__ == "__main__":
    unittest.main(verbosity=2)
