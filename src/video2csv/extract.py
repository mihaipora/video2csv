import logging
import re

from video2csv.models import ROI, ValueType, FrameResult
from video2csv.ocr.base import OCREngine
from video2csv.preprocess import crop_roi
from video2csv.video import Frame

logger = logging.getLogger(__name__)


def clean_ocr_text(raw: str, signed: bool) -> str:
    """Strip OCR artifacts, keep only digits, dots, and optionally a leading minus."""
    text = raw.strip()
    # Fix common OCR confusions before stripping
    text = text.replace("l", "1").replace("L", "1")
    text = text.replace("O", "0").replace("o", "0")
    if signed and text.startswith("-"):
        cleaned = "-" + re.sub(r"[^0-9.]", "", text[1:])
    else:
        cleaned = re.sub(r"[^0-9.]", "", text)
    return cleaned


def extract_value(
    frame: Frame,
    roi: ROI,
    engine: OCREngine,
) -> int | float | None:
    cropped = crop_roi(frame, roi.x, roi.y, roi.width, roi.height)
    raw_text = engine.image_to_string(cropped)
    cleaned = clean_ocr_text(raw_text, roi.signed)

    logger.debug(
        "ROI '%s': raw=%r cleaned=%r", roi.name, raw_text, cleaned
    )

    if not cleaned or cleaned == "-":
        logger.warning("ROI '%s': empty OCR result (raw=%r)", roi.name, raw_text)
        return None

    try:
        if roi.value_type is ValueType.INT:
            return int(float(cleaned))
        else:
            return float(cleaned)
    except (ValueError, TypeError):
        logger.warning(
            "ROI '%s': could not cast %r to %s",
            roi.name,
            cleaned,
            roi.value_type.value,
        )
        return None


def extract_frame(
    frame: Frame,
    frame_index: int,
    timestamp_ms: float,
    rois: list[ROI],
    engine: OCREngine,
) -> FrameResult:
    values: dict[str, int | float | None] = {}
    for roi in rois:
        values[roi.name] = extract_value(frame, roi, engine)
    return FrameResult(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        values=values,
    )
