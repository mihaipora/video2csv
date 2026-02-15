# CLAUDE.md

## Project overview

`video2csv` extracts numeric values from screen-captured automotive diagnostic software videos using OCR. It reads every frame, crops configured ROIs, detects changes via pixel comparison, runs PaddleOCR only on changed frames, and outputs CSV with millisecond timestamps.

## Tech stack

- Python 3.13+, managed with `uv`
- OpenCV (`opencv-python-headless`) for video decoding
- PaddleOCR 3.4 + PaddlePaddle for OCR
- NumPy for frame/image operations
- Hatchling for packaging

## Project structure

```
src/video2csv/
  cli.py          — CLI entry point (argparse)
  pipeline.py     — Main orchestration: frame loop, change detection, OCR dispatch
  video.py        — Video decoding, frame iteration, timestamp calculation
  extract.py      — Per-frame extraction: crop → OCR → clean → cast
  preprocess.py   — Image preprocessing (crop_roi, threshold, upscale)
  writer.py       — CSV output with context manager
  config.py       — JSON config loading and validation
  models.py       — Data classes: ROI, VideoMeta, FrameResult, ValueType
  dump_frames.py  — Standalone tool to dump sample frames from a video
  ocr/
    base.py                  — OCREngine protocol (interface)
    factory.py               — Engine registry and create_engine()
    paddleocr_engine.py      — Full pipeline (server det + rec) — default
    paddleocr_mobile_engine.py — Mobile det + rec (lighter, same accuracy)
    paddleocr_rec_engine.py  — Recognition only, no detection (fastest, loses minus signs)
```

## Commands

```bash
# Run tests (64 tests, all unit tests, no video/OCR required)
python -m pytest tests/

# Run extraction
python -m video2csv data/recording.mp4 -c config.json

# Analyze change rate (fast, no OCR)
python -m video2csv data/recording.mp4 -c config.json --analyze

# Dump sample frames from a video
python -m video2csv.dump_frames data/recording.mp4
```

## Key design decisions

- **One row per frame**: The full CSV always has one row per video frame. Values are reused from the previous OCR result when the frame hasn't changed.
- **Change detection**: ROI crops are compared frame-to-frame using mean absolute pixel difference (threshold=5.0). This skips ~97% of OCR calls.
- **Two CSV outputs**: `<name>.csv` (full, every frame) and `<name>_small.csv` (changes only).
- **No preprocessing for PaddleOCR**: PaddleOCR handles its own image preprocessing. Passing raw BGR crops gives better results than grayscale/threshold.
- **Tall ROIs for Y-drift**: The diagnostic software UI drifts vertically (~65px over 28 min). ROIs are configured taller than the text, and PaddleOCR's detection finds the text within.
- **Signed values**: ROIs with `"signed": true` preserve leading minus signs during OCR text cleaning.
- **OCR text selection**: When multiple text regions are detected in an ROI, prefer text containing digits, then pick the largest bounding box area. This avoids picking unit labels like "mbar" over the actual number.

## OCR engines

Three engines are registered in `ocr/factory.py`, selectable via `"ocr_engine"` in config.json:

| Engine | Config value | Speed | Notes |
|--------|-------------|-------|-------|
| Server (default) | `paddleocr` | ~0.43s/call | Best accuracy, handles minus signs |
| Mobile | `paddleocr-mobile` | ~0.28s/call | 1.6x faster, same accuracy |
| Rec-only | `paddleocr-rec` | ~0.06s/call | 7x faster, but drops some minus signs |

## Testing

Tests are in `tests/` and use pytest. They are all unit tests that mock OpenCV and OCR — no video files or PaddleOCR needed to run them.

```bash
python -m pytest tests/ -v
```

## Data files

- `data/*.mp4` and `data/*.csv` are gitignored (large files)
- `data/frame_*.png` are tracked (sample frames for ROI coordinate reference)
- `config.json` contains ROI definitions for the current diagnostic video setup
