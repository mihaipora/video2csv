import logging
from pathlib import Path

import numpy as np

from video2csv.config import AppConfig
from video2csv.extract import extract_frame
from video2csv.models import ROI, FrameResult
from video2csv.ocr import create_engine
from video2csv.preprocess import crop_roi
from video2csv.video import Frame, open_video, iter_frames, frame_timestamp_ms
from video2csv.writer import CSVWriter

logger = logging.getLogger(__name__)


def _crop_rois(frame: Frame, rois: list[ROI]) -> list[np.ndarray]:
    """Crop all ROI regions from a frame."""
    return [crop_roi(frame, r.x, r.y, r.width, r.height) for r in rois]


def _rois_changed(
    current_crops: list[np.ndarray],
    previous_crops: list[np.ndarray],
    threshold: float = 5.0,
) -> bool:
    """Check if any ROI region changed between frames.

    Compares mean absolute pixel difference per ROI.
    A threshold of ~5.0 accounts for video compression artifacts
    without missing real value changes.
    """
    for curr, prev in zip(current_crops, previous_crops):
        diff = np.mean(np.abs(curr.astype(np.int16) - prev.astype(np.int16)))
        if diff > threshold:
            return True
    return False


def analyze_change_rate(
    config: AppConfig,
    video_path: Path,
    max_frames: int | None = None,
) -> None:
    """Fast pass over video to measure how often ROI values change.

    Decodes frames and compares ROI crops — no OCR.
    Reports statistics to help choose an optimal --frame-step value.
    """
    cap, meta = open_video(video_path)
    frames_to_process = min(max_frames, meta.total_frames) if max_frames else meta.total_frames

    logger.info(
        "Analyzing: %s | %.2f fps | %d frames",
        meta.path, meta.fps, frames_to_process,
    )

    prev_crops: list[np.ndarray] | None = None
    change_frames: list[int] = []
    gaps: list[int] = []

    for frame_idx, frame in iter_frames(cap, frames_to_process):
        current_crops = _crop_rois(frame, config.rois)

        if prev_crops is None:
            change_frames.append(frame_idx)
        elif _rois_changed(current_crops, prev_crops):
            change_frames.append(frame_idx)
            gaps.append(frame_idx - change_frames[-2])
            prev_crops = [c.copy() for c in current_crops]
            continue

        prev_crops = [c.copy() for c in current_crops]

    if not gaps:
        logger.info("No ROI changes detected in %d frames.", frames_to_process)
        return

    gaps_arr = np.array(gaps)
    min_gap = int(gaps_arr.min())
    max_gap = int(gaps_arr.max())
    median_gap = float(np.median(gaps_arr))
    mean_gap = float(gaps_arr.mean())
    p5 = float(np.percentile(gaps_arr, 5))

    logger.info("--- ROI Change Rate Analysis ---")
    logger.info("Total frames analyzed: %d", frames_to_process)
    logger.info("Total changes detected: %d", len(change_frames))
    logger.info("Gap between changes (frames): min=%d, max=%d, median=%.0f, mean=%.1f", min_gap, max_gap, median_gap, mean_gap)
    logger.info("Gap between changes (ms):     min=%.0f, max=%.0f, median=%.0f, mean=%.0f",
                min_gap / meta.fps * 1000, max_gap / meta.fps * 1000,
                median_gap / meta.fps * 1000, mean_gap / meta.fps * 1000)
    logger.info("5th percentile gap: %.0f frames (%.0f ms)", p5, p5 / meta.fps * 1000)
    safe_step = max(1, int(p5 // 2))
    logger.info("Recommended --frame-step: %d (half of 5th percentile, safe margin)", safe_step)


def _small_path(output_path: Path) -> Path:
    """Derive the _small.csv path from the main output path."""
    return output_path.with_stem(output_path.stem + "_small")


def run(
    config: AppConfig,
    video_path: Path,
    output_path: Path,
    max_frames: int | None = None,
) -> None:
    engine = create_engine(config.ocr_engine)
    cap, meta = open_video(video_path)
    small_output = _small_path(output_path)

    frames_to_process = min(max_frames, meta.total_frames) if max_frames else meta.total_frames

    logger.info(
        "Video: %s | %dx%d | %.2f fps | %d frames | %.1f seconds",
        meta.path,
        meta.width,
        meta.height,
        meta.fps,
        meta.total_frames,
        meta.duration_ms / 1000.0,
    )
    if max_frames:
        logger.info("Processing first %d frames (of %d)", frames_to_process, meta.total_frames)
    logger.info("ROIs: %s", [r.name for r in config.rois])
    logger.info("Output: %s", output_path)
    logger.info("Output (changes only): %s", small_output)

    written = 0
    skipped_ocr_fail = 0
    reused = 0
    ocr_calls = 0
    prev_crops: list[np.ndarray] | None = None
    prev_values: dict[str, int | float | None] | None = None

    with CSVWriter(output_path, config.rois) as writer, \
         CSVWriter(small_output, config.rois) as small_writer:
        for frame_idx, frame in iter_frames(cap, frames_to_process):
            ts = frame_timestamp_ms(frame_idx, meta.fps)
            current_crops = _crop_rois(frame, config.rois)

            if prev_crops is not None and not _rois_changed(current_crops, prev_crops):
                # ROIs unchanged — reuse previous OCR values with new timestamp
                reused += 1
                result = FrameResult(
                    frame_index=frame_idx,
                    timestamp_ms=ts,
                    values=dict(prev_values),
                )
            else:
                # ROIs changed — run OCR
                prev_crops = [c.copy() for c in current_crops]
                ocr_calls += 1
                result = extract_frame(frame, frame_idx, ts, config.rois, engine)
                prev_values = result.values
                vals = " | ".join(f"{k}={v}" for k, v in result.values.items())
                logger.info("Frame %d @ %.1f ms: %s", frame_idx, ts, vals)
                # Write to changes-only CSV
                small_writer.write_row(result)

            if writer.write_row(result):
                written += 1
            else:
                skipped_ocr_fail += 1

    logger.info(
        "Done. Wrote %d rows (%d OCR, %d reused), skipped %d OCR failures.",
        written,
        ocr_calls,
        reused,
        skipped_ocr_fail,
    )
