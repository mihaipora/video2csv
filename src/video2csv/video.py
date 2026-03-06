import logging
import time
from collections.abc import Generator
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from video2csv.models import VideoMeta

logger = logging.getLogger(__name__)

Frame = NDArray[np.uint8]


def open_video(path: Path) -> tuple[cv2.VideoCapture, VideoMeta]:
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_ms = total_frames / fps * 1000.0 if fps > 0 else 0.0

    meta = VideoMeta(
        path=str(path),
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        duration_ms=duration_ms,
    )
    return cap, meta


def iter_frames(
    cap: cv2.VideoCapture, total_frames: int, step: int = 1,
) -> Generator[tuple[int, Frame], None, None]:
    try:
        last_log_time = 0.0
        frame_idx = 0
        while frame_idx < total_frames:
            if step > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning(
                    "Failed to read frame %d / %d, stopping", frame_idx, total_frames
                )
                break
            now = time.monotonic()
            if frame_idx == 0 or now - last_log_time >= 5.0:
                pct = frame_idx / total_frames * 100
                logger.info("Frame %d / %d (%.1f%%)", frame_idx, total_frames, pct)
                last_log_time = now
            yield frame_idx, frame
            frame_idx += step
    finally:
        cap.release()


def read_frame_at(cap: cv2.VideoCapture, frame_index: int) -> Frame | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def frame_timestamp_ms(frame_index: int, fps: float) -> float:
    return frame_index / fps * 1000.0
