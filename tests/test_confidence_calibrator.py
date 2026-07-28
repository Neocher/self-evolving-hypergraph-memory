"""测试过度巩固防护 — ConfidenceCalibrator"""
import time
import pytest
from core.confidence_calibrator import ConfidenceCalibrator, CalibratorConfig


class TestConfidenceCalibrator:
    def test_direct_high_confidence(self):
        cal = ConfidenceCalibrator()
        cal_conf, flagged = cal.calibrate(
            "地球是圆的", 0.95, "direct")
        assert cal_conf > 0.8  # 直接观察几乎不衰减
        assert not flagged

    def test_hearsay_decay(self):
        cal = ConfidenceCalibrator()
        cal_conf, _ = cal.calibrate(
            "听说火星上有水", 0.9, "hearsay")
        assert cal_conf < 0.5  # hearsay 权重 0.4

    def test_repeated_consolidation_flags(self):
        cal = ConfidenceCalibrator(
            CalibratorConfig(decay_rate=0.2, min_confidence=0.15))
        text = "多次整合的内容"
        for i in range(10):
            cal.record_consolidation(text, "inferred")
        cal_conf, flagged = cal.calibrate(text, 0.9, "inferred")
        assert cal_conf < 0.2  # 严重衰减
        assert flagged  # 标记审查

    def test_no_calibration_without_consolidation(self):
        cal = ConfidenceCalibrator()
        cal_conf, _ = cal.calibrate("新内容", 0.8, "direct")
        assert cal_conf > 0.7  # 近乎不变

    def test_state_summary(self):
        cal = ConfidenceCalibrator()
        cal.calibrate("A", 0.5, "inferred")
        cal.calibrate("B", 0.5, "hearsay")
        for i in range(15):
            cal.record_consolidation("C", "inferred")
        cal.calibrate("C", 0.5, "inferred")
        s = cal.state()
        assert s["total_tracked"] >= 3
        assert s["flagged"] >= 1
        assert s["high_consolidation"] >= 1

    def test_flagged_items(self):
        cal = ConfidenceCalibrator(
            CalibratorConfig(decay_rate=0.3, min_confidence=0.1))
        for i in range(8):
            cal.record_consolidation("可疑信息", "hearsay")
        cal.calibrate("可疑信息", 0.9, "hearsay")
        items = cal.flagged_items()
        assert len(items) >= 1
        assert items[0]["calibrated_confidence"] < 0.5

    def test_set_source_type(self):
        cal = ConfidenceCalibrator()
        cal.calibrate("X", 0.9, "hearsay")
        cal.set_source_type("X", "direct")  # LLM判定后升级
        cal_conf, _ = cal.calibrate("X", 0.9, "direct")
        assert cal_conf > 0.8  # 源类型升级后信心恢复
