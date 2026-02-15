import json
import logging
from dataclasses import dataclass
from pathlib import Path

from video2csv.models import ROI, ValueType

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AppConfig:
    rois: list[ROI]
    ocr_engine: str


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}") from e

    if "rois" not in raw:
        raise ValueError("Config missing required field 'rois'")

    rois = [_parse_roi(r) for r in raw["rois"]]
    ocr_engine = raw.get("ocr_engine", "paddleocr")

    return AppConfig(rois=rois, ocr_engine=ocr_engine)


def _parse_roi(raw: dict) -> ROI:
    required = ["name", "x", "y", "width", "height", "value_type"]
    for field in required:
        if field not in raw:
            raise ValueError(f"ROI missing required field '{field}'")

    try:
        value_type = ValueType(raw["value_type"])
    except ValueError:
        raise ValueError(
            f"value_type must be 'int' or 'float', got '{raw['value_type']}'"
        )

    return ROI(
        name=raw["name"],
        x=raw["x"],
        y=raw["y"],
        width=raw["width"],
        height=raw["height"],
        value_type=value_type,
        signed=raw.get("signed", False),
    )
