"""Dump 5 frames from a video at 20%, 40%, 60%, 80%, 100% progress.

Usage: python -m video2csv.dump_frames <video_path>

Outputs frame_0.png through frame_4.png in the current directory.
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

from video2csv.video import open_video, read_frame_at, frame_timestamp_ms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PERCENTAGES = [0.2, 0.4, 0.6, 0.8, 1.0]


def _find_last_readable_frame(cap: cv2.VideoCapture, total_frames: int) -> int:
    """Binary search for the last frame OpenCV can actually read."""
    lo, hi = max(total_frames - 50, 0), total_frames - 1
    last_good = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        frame = read_frame_at(cap, mid)
        if frame is not None:
            last_good = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return last_good


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump 5 frames from a video at 20%/40%/60%/80%/100% progress."
    )
    parser.add_argument("video", type=Path, help="Path to the input video file.")
    args = parser.parse_args()

    if not args.video.exists():
        logger.error("Video file not found: %s", args.video)
        sys.exit(1)

    cap, meta = open_video(args.video)

    logger.info(
        "Video: %s | %dx%d | %.2f fps | %d frames | %.1f seconds",
        meta.path,
        meta.width,
        meta.height,
        meta.fps,
        meta.total_frames,
        meta.duration_ms / 1000.0,
    )

    last_readable = _find_last_readable_frame(cap, meta.total_frames)
    logger.info("Last readable frame: %d", last_readable)

    for i, pct in enumerate(PERCENTAGES):
        frame_idx = min(int(pct * last_readable), last_readable)
        frame_idx = max(frame_idx, 0)

        frame = read_frame_at(cap, frame_idx)
        if frame is None:
            logger.warning("Could not read frame %d (%.0f%%)", frame_idx, pct * 100)
            continue

        ts = frame_timestamp_ms(frame_idx, meta.fps)
        output_path = f"frame_{i}.png"
        cv2.imwrite(output_path, frame)
        logger.info(
            "Saved %s — frame %d / %d (%.0f%%) @ %.1f ms",
            output_path,
            frame_idx,
            meta.total_frames,
            pct * 100,
            ts,
        )

    cap.release()
    logger.info("Done.")


if __name__ == "__main__":
    main()
