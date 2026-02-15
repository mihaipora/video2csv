import csv
import logging
from pathlib import Path

from video2csv.models import ROI, FrameResult

logger = logging.getLogger(__name__)


class CSVWriter:
    def __init__(self, output_path: Path, rois: list[ROI]) -> None:
        self._output_path = output_path
        self._fieldnames = ["frame", "timestamp_ms"] + [r.name for r in rois]
        self._file = None
        self._writer = None

    def __enter__(self) -> "CSVWriter":
        self._file = open(self._output_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        self._writer.writeheader()
        return self

    def __exit__(self, *exc) -> None:
        if self._file:
            self._file.close()

    def write_row(self, result: FrameResult, *, skip_on_any_none: bool = True) -> bool:
        if skip_on_any_none and any(v is None for v in result.values.values()):
            logger.warning(
                "Frame %d: skipping row due to failed OCR (values=%s)",
                result.frame_index,
                result.values,
            )
            return False

        row = {
            "frame": result.frame_index,
            "timestamp_ms": f"{result.timestamp_ms:.1f}",
        }
        for name, value in result.values.items():
            row[name] = value if value is not None else ""
        self._writer.writerow(row)
        return True
