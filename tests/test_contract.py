"""提交契约测试（R2-B6：井数/每井行数/深度对齐/SW 尺度必须硬校验）。"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V4))

from src import constants as C            # noqa: E402
from src.inference import contract as CT  # noqa: E402

DATA = V4.parent / "data"
TEST_DIR = DATA / "test"


def _good(n_rows: int = 2, well: str = "w1") -> dict:
    return {
        "modelId": "",
        "modelName": "t",
        "version": "1.0",
        "resultData": [{
            "logId": well,
            "predictions": [
                {"depth": 1.0 + 0.1 * i, "POR": 0.1, "PERM": 0.01, "SW": 99.9}
                for i in range(n_rows)
            ],
        }],
    }


class TestPayloadBasics(unittest.TestCase):
    def test_good(self):
        r = CT.validate_payload(_good(), expected_rows=2, expected_wells=1)
        self.assertTrue(r.ok, r.errors)

    def test_missing_toplevel(self):
        d = _good(); d.pop("version")
        self.assertFalse(CT.validate_payload(d, expected_rows=2, expected_wells=1).ok)

    def test_perm_nonpositive(self):
        d = _good(); d["resultData"][0]["predictions"][1]["PERM"] = 0.0
        self.assertFalse(CT.validate_payload(d, expected_rows=2, expected_wells=1).ok)

    def test_uppercase_depth_rejected(self):
        d = _good()
        p0 = d["resultData"][0]["predictions"][0]
        p0["DEPTH"] = p0.pop("depth")
        self.assertFalse(CT.validate_payload(d, expected_rows=2, expected_wells=1).ok)

    def test_nan_rejected(self):
        d = _good(); d["resultData"][0]["predictions"][0]["SW"] = float("nan")
        self.assertFalse(CT.validate_payload(d, expected_rows=2, expected_wells=1).ok)

    def test_sw_guard_not_bypassed_by_partial_row_scales(self):
        d = _good()
        d["resultData"][0]["predictions"][0]["SW"] = 12345.0
        r = CT.validate_payload(d, expected_rows=2, expected_wells=1,
                                row_scales={"POR": (0.0, 60.0)})
        self.assertFalse(r.ok, r.errors)

    def test_rowcount_mismatch(self):
        self.assertFalse(CT.validate_payload(_good(2), expected_rows=3,
                                             expected_wells=1).ok)

    def test_wellcount_mismatch(self):
        self.assertFalse(CT.validate_payload(_good(2, "w1"), expected_rows=2,
                                             expected_wells=2).ok)


class TestScaleGuard(unittest.TestCase):
    def test_sw_normalized_to_0_1_rejected(self):
        """R2-B6：把 SW 归一化到 [0,1] 后直接输出必须被拒。"""
        d = _good(n_rows=3)
        for p in d["resultData"][0]["predictions"]:
            p["SW"] = 0.4
        r = CT.validate_payload(d, expected_rows=3, expected_wells=1)
        self.assertFalse(r.ok)
        self.assertTrue(any("SW median" in e for e in r.errors), r.errors)

    def test_real_sw_scale_accepted(self):
        d = _good(n_rows=3)
        for p, sw in zip(d["resultData"][0]["predictions"], (45.0, 82.0, 99.9)):
            p["SW"] = sw
        r = CT.validate_payload(d, expected_rows=3, expected_wells=1)
        self.assertTrue(r.ok, r.errors)


@unittest.skipUnless(TEST_DIR.is_dir(), "test split not found")
class TestAgainstRealTestSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wells = sorted(f.stem for f in TEST_DIR.glob("*.txt"))
        cls.payload = {
            "modelId": "", "modelName": "t", "version": "1.0",
            "resultData": [
                {"logId": w, "predictions": [
                    {"depth": d, "POR": 0.1, "PERM": 0.01, "SW": 99.9}
                    for d in CT._input_depths(TEST_DIR, w)]}
                for w in cls.wells],
        }

    def test_rowcount_per_well_is_checked(self):
        """删掉某口井最后一行 -> 必须被"每井行数"检查拒绝。"""
        d = json.loads(json.dumps(self.payload))
        d["resultData"][0]["predictions"] = d["resultData"][0]["predictions"][:-1]
        r = CT.validate_payload(d, test_dir=TEST_DIR, expected_rows=None,
                                expected_wells=len(self.wells))
        self.assertFalse(r.ok)
        self.assertTrue(any("rows !=" in e for e in r.errors), r.errors[:3])

    def test_full_payload_passes(self):
        total = sum(len(it["predictions"]) for it in self.payload["resultData"])
        r = CT.validate_payload(self.payload, test_dir=TEST_DIR, expected_rows=total,
                                expected_wells=len(self.wells))
        self.assertTrue(r.ok, r.errors[:5])
        self.assertEqual(total, C.EXPECTED_N_TEST_ROWS)

    def test_depth_alignment_report_flags_misalignment(self):
        depths = {it["logId"]: [p["depth"] for p in it["predictions"]]
                  for it in self.payload["resultData"]}
        rep = CT.depth_alignment_report(depths, TEST_DIR)
        self.assertTrue(all(v["ok"] for v in rep.values()), rep)
        # 人为错位
        bad = dict(depths)
        w0 = self.wells[0]
        bad[w0] = [d + 1.0 for d in bad[w0]]
        rep2 = CT.depth_alignment_report(bad, TEST_DIR)
        self.assertFalse(rep2[w0]["ok"])


class TestE0ArtifactsConsistency(unittest.TestCase):
    """文档/注册表与 JSON 的一致性（R2-B4/M7）。"""

    def test_data_card_test_schema_is_clean(self):
        card = V4 / "reports" / "E0_data_card.json"
        if not card.is_file():
            self.skipTest("E0_data_card.json missing")
        d = json.loads(card.read_text(encoding="utf-8"))
        # R2-B5：测试集不应被列为非规范 schema
        self.assertEqual(len(d["test"]["noncanonical_schema_wells"]), 0)
        # 训练集恰好 3 口
        self.assertEqual(len(d["train"]["noncanonical_schema_wells"]), 3)
        # 泄漏回归（含全量 90 井）
        self.assertTrue(d["input_leak_regression"]["passed"])
        self.assertTrue(d["input_leak_regression"]["full_90_wells"]["passed"])
        self.assertEqual(d["input_leak_regression"]["full_90_wells"]["violations"], 0)
        # 缓存两项（R2-B3）
        self.assertTrue(d["shard_cache"]["built"])
        self.assertTrue(d["shard_cache"]["input_cols_ok"])
        # SW<1 为 0（R2-B4）
        self.assertEqual(d["target_stats"]["SW"]["valid_rows_only"]["n_lt_1"], 0)

    def test_candidates_total_identity(self):
        p = V4 / "versions" / "candidates.json"
        if not p.is_file():
            self.skipTest("candidates.json missing")
        d = json.loads(p.read_text(encoding="utf-8"))
        for c in d["candidates"]:
            cv = c.get("cv") or {}
            if "total" not in cv:
                continue
            want = 100.0 * (C.SCORE_WEIGHTS["POR"] * cv["por"]
                            + C.SCORE_WEIGHTS["PERM"] * cv["perm"]
                            + C.SCORE_WEIGHTS["SW"] * cv["sw"])
            self.assertAlmostEqual(cv["total"], want, delta=1e-5, msg=c["candidate_id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
