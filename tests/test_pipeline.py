import csv
from pathlib import Path

import numpy as np

from video2csv.config import AppConfig
from video2csv.models import ROI, ValueType
from video2csv.ocr.factory import register_engine
from video2csv.pipeline import run


class MockOCREngine:
    def __init__(self):
        self.call_count = 0

    def image_to_string(self, image: np.ndarray) -> str:
        self.call_count += 1
        return "42"


class TestPipeline:
    def test_end_to_end(self, tmp_path: Path, tiny_video: Path):
        register_engine("mock", MockOCREngine)

        config = AppConfig(
            rois=[
                ROI(name="test_val", x=0, y=0, width=32, height=32, value_type=ValueType.INT),
            ],
            ocr_engine="mock",
        )
        output = tmp_path / "output.csv"
        run(config, tiny_video, output)

        assert output.exists()
        with open(output) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 5
        assert rows[0]["frame"] == "0"
        assert rows[0]["test_val"] == "42"
        assert "timestamp_ms" in rows[0]

    def test_skips_failed_ocr(self, tmp_path: Path, tiny_video: Path):
        class FailEveryOtherEngine:
            def __init__(self):
                self._calls = 0

            def image_to_string(self, image: np.ndarray) -> str:
                self._calls += 1
                return "100" if self._calls % 2 == 1 else "garbage"

        register_engine("fail_alternate", FailEveryOtherEngine)

        config = AppConfig(
            rois=[
                ROI(name="a", x=0, y=0, width=32, height=32, value_type=ValueType.INT),
            ],
            ocr_engine="fail_alternate",
        )
        output = tmp_path / "output.csv"
        run(config, tiny_video, output)

        with open(output) as f:
            rows = list(csv.DictReader(f))

        # 5 frames, every other OCR fails -> frames 0,2,4 succeed (odd calls: 1,3,5)
        # Frame 0: call 1 (odd) -> "100" OK
        # Frame 1: call 2 (even) -> "garbage" SKIP
        # Frame 2: call 3 (odd) -> "100" OK
        # Frame 3: call 4 (even) -> "garbage" SKIP
        # Frame 4: call 5 (odd) -> "100" OK
        assert len(rows) == 3
