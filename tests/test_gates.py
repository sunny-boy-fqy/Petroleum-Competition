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


class TestRepoPreregTemplates(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
