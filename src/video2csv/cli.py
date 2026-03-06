import argparse
import logging
import sys
from pathlib import Path

from video2csv.config import load_config
from video2csv.pipeline import analyze_change_rate, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video2csv",
        description="Extract numeric data from screen-captured datalog videos to CSV.",
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Path to the input MP4 video file.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Path to the JSON ROI configuration file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path for the output CSV file. Defaults to <video_stem>.csv.",
    )
    parser.add_argument(
        "-n",
        "--max-frames",
        type=int,
        default=None,
        help="Stop after processing N frames (for testing).",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every Nth frame (skip N-1 frames between reads). Default: 1 (every frame).",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze ROI change rate and recommend --frame-step. No OCR, no output.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.video.exists():
        logging.error("Video file not found: %s", args.video)
        sys.exit(1)

    config = load_config(args.config)

    if args.analyze:
        analyze_change_rate(config, args.video, max_frames=args.max_frames)
        return

    output = args.output or args.video.with_suffix(".csv")
    run(config, args.video, output, max_frames=args.max_frames, frame_step=args.frame_step)
