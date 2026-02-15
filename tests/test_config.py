import json
from pathlib import Path

import pytest

from video2csv.config import load_config
from video2csv.models import ValueType


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return path


class TestLoadConfig:
    def test_loads_valid_config(self, tmp_path: Path):
        path = _write_config(tmp_path, {
            "ocr_engine": "paddleocr",
            "rois": [
                {
                    "name": "boost_pressure (mbar)",
                    "x": 100,
                    "y": 200,
                    "width": 120,
                    "height": 40,
                    "value_type": "int",
                }
            ],
        })
        config = load_config(path)
        assert config.ocr_engine == "paddleocr"
        assert len(config.rois) == 1
        assert config.rois[0].name == "boost_pressure (mbar)"
        assert config.rois[0].value_type is ValueType.INT
        assert config.rois[0].signed is False

    def test_signed_field(self, tmp_path: Path):
        path = _write_config(tmp_path, {
            "rois": [
                {
                    "name": "diff (mbar)",
                    "x": 0, "y": 0, "width": 10, "height": 10,
                    "value_type": "int",
                    "signed": True,
                }
            ],
        })
        config = load_config(path)
        assert config.rois[0].signed is True

    def test_default_ocr_engine(self, tmp_path: Path):
        path = _write_config(tmp_path, {
            "rois": [
                {"name": "a", "x": 0, "y": 0, "width": 10, "height": 10, "value_type": "int"}
            ],
        })
        config = load_config(path)
        assert config.ocr_engine == "paddleocr"

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.json")

    def test_invalid_json_raises(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("{invalid json")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_config(path)

    def test_missing_rois_raises(self, tmp_path: Path):
        path = _write_config(tmp_path, {"ocr_engine": "paddleocr"})
        with pytest.raises(ValueError, match="missing required field 'rois'"):
            load_config(path)

    def test_missing_roi_field_raises(self, tmp_path: Path):
        path = _write_config(tmp_path, {
            "rois": [{"name": "test"}],
        })
        with pytest.raises(ValueError, match="missing required field"):
            load_config(path)

    def test_invalid_value_type_raises(self, tmp_path: Path):
        path = _write_config(tmp_path, {
            "rois": [
                {"name": "a", "x": 0, "y": 0, "width": 10, "height": 10, "value_type": "string"}
            ],
        })
        with pytest.raises(ValueError, match="must be 'int' or 'float'"):
            load_config(path)

    def test_float_value_type(self, tmp_path: Path):
        path = _write_config(tmp_path, {
            "rois": [
                {"name": "afr", "x": 0, "y": 0, "width": 10, "height": 10, "value_type": "float"}
            ],
        })
        config = load_config(path)
        assert config.rois[0].value_type is ValueType.FLOAT
