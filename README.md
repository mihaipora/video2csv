# video2csv

Extract numeric data from screen-captured diagnostic software videos into CSV files.

## What it does

`video2csv` takes an MP4 video recording of automotive diagnostic software (e.g., Autocom/Delphi DS) and extracts numeric parameter values frame-by-frame using OCR. The output is a CSV file with millisecond-accurate timestamps — one row per video frame — turning a screen recording into structured, plottable data.

## How it works

1. **Video decoding** — OpenCV reads the video frame by frame at the native framerate.

2. **ROI extraction** — For each frame, user-defined rectangular regions of interest (ROIs) are cropped. Each ROI corresponds to a parameter displayed on screen (e.g., boost pressure, RPM, temperature).

3. **Change detection** — Before running OCR, the tool compares each ROI crop to the previous frame using pixel-level difference. If nothing changed, it reuses the previous values. This skips OCR on ~97% of frames, reducing processing time from days to under an hour.

4. **OCR** — Changed regions are passed to PaddleOCR, which recognizes the numeric text. The raw OCR output is cleaned (non-numeric characters stripped) and cast to the configured type (int or float). Negative values are supported for signed parameters.

5. **CSV output** — Two files are produced:
   - `<video_name>.csv` — Full output with one row per frame, including frames where values were reused from previous OCR.
   - `<video_name>_small.csv` — Compact output containing only the rows where values actually changed.

## Installation

Requires Python 3.13+.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Usage

### Basic extraction

```bash
video2csv recording.mp4 -c configs/config.json
```

This produces `recording.csv` and `recording_small.csv` in the same directory as the video.

### Options

```
video2csv <video> -c <config> [options]

positional arguments:
  video                   Path to the input MP4 video file

required arguments:
  -c, --config CONFIG     Path to the JSON ROI configuration file

optional arguments:
  -o, --output OUTPUT     Path for the output CSV file (default: <video_stem>.csv)
  -n, --max-frames N      Stop after processing N frames (useful for testing)
  --frame-step N          Process every Nth frame (default: 1, every frame).
                          Use --analyze to find a safe value.
  --analyze               Analyze ROI change rate without running OCR.
                          Reports how often values change and recommends
                          a --frame-step value.
  -v, --verbose           Enable DEBUG-level logging
```

### Analyze change rate

Before running a full extraction, you can analyze how often values change in the video:

```bash
video2csv recording.mp4 -c configs/config.json --analyze
```

This runs a fast pass (no OCR) and reports change frequency statistics, including the total number of changes detected and a recommended `--frame-step` value. Since OCR only runs on changed frames, this tells you how long the full extraction will take.

### Frame stepping

For long videos, you can skip frames to reduce processing time:

```bash
video2csv recording.mp4 -c configs/config.json --frame-step 12
```

This processes every 12th frame instead of every frame. The `--analyze` command recommends a safe step value (half the 5th percentile gap between changes) that won't miss any value updates. Change detection still runs on every processed frame, so OCR is only called when values actually change.

### Test with a few frames

```bash
video2csv recording.mp4 -c configs/config.json -n 100
```

### Dump sample frames

To visually inspect the video and determine ROI coordinates:

```bash
python -m video2csv.dump_frames recording.mp4
```

This saves 5 PNG frames at 20%, 40%, 60%, 80%, and 100% through the video.

## Configuration

The ROI configuration is a JSON file that defines which screen regions to read and how to interpret the values.

```json
{
  "ocr_engine": "paddleocr",
  "rois": [
    {
      "name": "boost_pressure (mbar)",
      "x": 1580,
      "y": 295,
      "width": 270,
      "height": 130,
      "value_type": "int",
      "signed": false
    },
    {
      "name": "boost_pressure_loop_diff (mbar)",
      "x": 1580,
      "y": 475,
      "width": 270,
      "height": 130,
      "value_type": "int",
      "signed": true
    }
  ]
}
```

### ROI fields

| Field | Description |
|-------|-------------|
| `name` | Column header in the CSV. Include the unit for clarity (e.g., `"rpm"`, `"boost (mbar)"`). |
| `x`, `y` | Top-left corner of the region in pixels. |
| `width`, `height` | Size of the region in pixels. Use a taller region than the text to account for slight vertical drift in the recording. |
| `value_type` | `"int"` or `"float"`. |
| `signed` | Set to `true` if the value can be negative (e.g., pressure differences). |

### How to find ROI coordinates

1. Run `python -m video2csv.dump_frames recording.mp4` to extract sample frames.
2. Open the frames in an image editor.
3. Note the pixel coordinates (x, y) and size (width, height) of each numeric value on screen.
4. Make the height ~2-3x the actual text height to handle any vertical drift in the recording.

## CSV output format

```csv
frame,timestamp_ms,boost_pressure (mbar),boost_pressure_loop_diff (mbar),boost_pressure_ref (mbar)
0,0.0,1017,-4,1010
1,21.1,1017,-4,1010
2,42.1,1017,-4,1010
...
47,989.7,1020,-5,1013
```

- `frame` — Video frame index (0-based).
- `timestamp_ms` — Frame timestamp in milliseconds, calculated from the frame index and video framerate.
- Remaining columns — One per configured ROI, using the `name` from the config as the header.

Rows where OCR failed for any parameter are skipped (logged as warnings).

## Performance

Processing time depends on how often the on-screen values change. The tool only runs OCR when a change is detected — unchanged frames reuse previous values at negligible cost.

For a typical 28-minute diagnostic recording at 47.5 fps (~81,000 frames):
- **~2 minutes** for the frame decode and change detection pass
- **~0.4 seconds** per OCR call (3 ROIs per changed frame)
- With ~2,200 value changes: **~50 minutes** total

Use `--analyze` to estimate processing time for your specific video before starting.

## Quality checking

After extraction, you can verify the results by visually comparing frame images against CSV values using Claude CLI (Haiku model). The quality check script is fully generic — it reads parameter names, crop coordinates, and count from the config file.

### Running a quality check

```bash
# Full check (100 frames)
python -m video2csv.quality_check \
  -c configs/config_20260304.json \
  --csv data/20260304_191424_small.csv \
  --video data/20260304_191424.mp4

# Quick test
python -m video2csv.quality_check \
  -c configs/config_20260304.json \
  --csv data/20260304_191424_small.csv \
  --video data/20260304_191424.mp4 \
  --stop-after 5

# Sample fewer frames
python -m video2csv.quality_check \
  -c configs/config_20260304.json \
  --csv data/20260304_191424_small.csv \
  --video data/20260304_191424.mp4 \
  -n 50
```

The script:
1. Reads ROI definitions from the config file (parameter names, coordinates, types).
2. Samples N evenly spaced frames from the small CSV (changes-only file).
3. Computes a bounding crop box from ROI coordinates and dumps value-column crops as PNGs.
4. Calls `claude -p` (Haiku) for each frame to read the values from the image.
5. Compares Claude's reading against the CSV and reports mismatches.
6. Saves a report to `data/quality_check/report.md` and `report.json`.
