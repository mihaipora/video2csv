import numpy as np
import pytest

from video2csv.extract import clean_ocr_text, extract_value, extract_frame
from video2csv.models import ROI, ValueType, FrameResult


class MockOCREngine:
    def __init__(self, response: str = "123"):
        self.response = response
        self.call_count = 0

    def image_to_string(self, image: np.ndarray) -> str:
        self.call_count += 1
        return self.response


class TestCleanOcrText:
    def test_digits_only(self):
        assert clean_ocr_text("2040", signed=False) == "2040"

    def test_strips_unit_text(self):
        assert clean_ocr_text("2040 mbar", signed=False) == "2040"

    def test_strips_percent(self):
        assert clean_ocr_text("109 %", signed=False) == "109"

    def test_preserves_negative(self):
        assert clean_ocr_text("-20 mbar", signed=True) == "-20"

    def test_strips_negative_when_unsigned(self):
        assert clean_ocr_text("-20 mbar", signed=False) == "20"

    def test_preserves_decimal(self):
        assert clean_ocr_text("14.7", signed=False) == "14.7"

    def test_negative_decimal(self):
        assert clean_ocr_text("-3.5", signed=True) == "-3.5"

    def test_whitespace_handling(self):
        assert clean_ocr_text("  2040  ", signed=False) == "2040"

    def test_empty_string(self):
        assert clean_ocr_text("", signed=False) == ""

    def test_garbage_text(self):
        assert clean_ocr_text("abc", signed=False) == ""


class TestExtractValue:
    def _make_roi(self, value_type=ValueType.INT, signed=False):
        return ROI(
            name="test", x=0, y=0, width=50, height=30,
            value_type=value_type, signed=signed,
        )

    def _make_frame(self):
        return np.full((100, 200, 3), 200, dtype=np.uint8)

    def test_int_cast(self):
        engine = MockOCREngine("2040 mbar")
        roi = self._make_roi(ValueType.INT)
        result = extract_value(self._make_frame(), roi, engine)
        assert result == 2040
        assert isinstance(result, int)

    def test_float_cast(self):
        engine = MockOCREngine("14.7")
        roi = self._make_roi(ValueType.FLOAT)
        result = extract_value(self._make_frame(), roi, engine)
        assert result == 14.7
        assert isinstance(result, float)

    def test_negative_signed(self):
        engine = MockOCREngine("-20 mbar")
        roi = self._make_roi(ValueType.INT, signed=True)
        result = extract_value(self._make_frame(), roi, engine)
        assert result == -20

    def test_negative_unsigned_strips_minus(self):
        engine = MockOCREngine("-20 mbar")
        roi = self._make_roi(ValueType.INT, signed=False)
        result = extract_value(self._make_frame(), roi, engine)
        assert result == 20

    def test_garbage_returns_none(self):
        engine = MockOCREngine("garbage")
        roi = self._make_roi()
        result = extract_value(self._make_frame(), roi, engine)
        assert result is None

    def test_empty_returns_none(self):
        engine = MockOCREngine("")
        roi = self._make_roi()
        result = extract_value(self._make_frame(), roi, engine)
        assert result is None

    def test_int_from_float_string(self):
        engine = MockOCREngine("2040.0")
        roi = self._make_roi(ValueType.INT)
        result = extract_value(self._make_frame(), roi, engine)
        assert result == 2040
        assert isinstance(result, int)


class TestExtractFrame:
    def test_extracts_all_rois(self):
        engine = MockOCREngine("100")
        rois = [
            ROI(name="a", x=0, y=0, width=10, height=10, value_type=ValueType.INT),
            ROI(name="b", x=0, y=0, width=10, height=10, value_type=ValueType.INT),
        ]
        frame = np.full((100, 200, 3), 200, dtype=np.uint8)
        result = extract_frame(frame, 0, 0.0, rois, engine)
        assert isinstance(result, FrameResult)
        assert result.values == {"a": 100, "b": 100}
        assert engine.call_count == 2

    def test_partial_failure(self):
        """One ROI succeeds, one fails — both are present in result."""
        class AlternatingEngine:
            def __init__(self):
                self._calls = 0
            def image_to_string(self, image):
                self._calls += 1
                return "100" if self._calls == 1 else "garbage"

        rois = [
            ROI(name="good", x=0, y=0, width=10, height=10, value_type=ValueType.INT),
            ROI(name="bad", x=0, y=0, width=10, height=10, value_type=ValueType.INT),
        ]
        frame = np.full((100, 200, 3), 200, dtype=np.uint8)
        result = extract_frame(frame, 5, 166.7, rois, AlternatingEngine())
        assert result.values["good"] == 100
        assert result.values["bad"] is None
        assert result.frame_index == 5
        assert result.timestamp_ms == 166.7
