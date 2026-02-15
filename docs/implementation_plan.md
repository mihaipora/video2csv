# video2csv - Implementation Plan

## Overview

Extract numeric data from screen-captured automotive datalog videos, frame by frame, into CSV files with millisecond timestamps.

**Pipeline:** Video (OpenCV) -> Frame extraction -> Crop ROIs -> Preprocess -> OCR (PaddleOCR) -> Cast to numeric -> CSV

---

## 1. Project Structure

```
video2csv/
├── pyproject.toml
├── config.example.json
├── docs/
│   └── implementation_plan.md
├── src/
│   └── video2csv/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── config.py
│       ├── models.py
│       ├── video.py
│       ├── preprocess.py
│       ├── ocr/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── paddleocr_engine.py
│       │   └── factory.py
│       ├── extract.py
│       ├── writer.py
│       └── pipeline.py
└── tests/
    ├── conftest.py
    ├── test_video.py
    ├── test_preprocess.py
    ├── test_ocr.py
    ├── test_extract.py
    ├── test_writer.py
    ├── test_config.py
    └── test_pipeline.py
```

---

## 2. Dependencies

### Runtime (`[project.dependencies]`)

| Package | Purpose |
|---------|---------|
| `opencv-python-headless` | Video decoding, image manipulation (no GUI) |
| `paddleocr` | OCR engine — pure pip install, no system deps, lightweight (<10MB models) |
| `paddlepaddle` | PaddleOCR's deep learning backend |
| `numpy` | Image array types (implicit via opencv) |

### Dev (`[dependency-groups]` -> `dev`)

| Package | Purpose |
|---------|---------|
| `pytest` | Test runner |

No system-level dependencies required. Everything installs via `uv pip install` / `uv sync`.

---

## 3. Module Design

### 3.1 `models.py` - Shared Data Structures

```python
class ValueType(Enum):
    INT = "int"
    FLOAT = "float"

@dataclass(frozen=True, slots=True)
class ROI:
    name: str               # CSV column header, e.g. "boost_pressure (mbar)"
    x: int                  # Top-left x (pixels)
    y: int                  # Top-left y (pixels)
    width: int
    height: int
    value_type: ValueType
    signed: bool = False    # If True, OCR text may start with "-"

@dataclass(frozen=True, slots=True)
class VideoMeta:
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_ms: float

@dataclass(slots=True)
class FrameResult:
    frame_index: int
    timestamp_ms: float
    values: dict[str, int | float | None]   # ROI name -> parsed value or None
```

Notes:
- `name` includes the unit of measure and becomes the CSV column header directly (e.g. `"boost_pressure (mbar)"`).
- `signed` flag indicates the value can be negative. Used in `extract.py` to clean OCR text: strip everything except digits, `.`, and `-` (leading only). If `signed=False`, the minus sign is stripped.

### 3.2 `config.py` - Configuration Loading

```python
@dataclass(frozen=True, slots=True)
class AppConfig:
    rois: list[ROI]
    ocr_engine: str          # e.g. "paddleocr"

def load_config(path: Path) -> AppConfig: ...
```

Raises `FileNotFoundError` or `ValueError` on invalid input.

### 3.3 `video.py` - Video Decoding (first standalone piece)

```python
Frame = NDArray[np.uint8]   # HxWxC BGR image

def open_video(path: Path) -> tuple[cv2.VideoCapture, VideoMeta]: ...
def iter_frames(cap: cv2.VideoCapture, total_frames: int) -> Generator[tuple[int, Frame], None, None]: ...
def frame_timestamp_ms(frame_index: int, fps: float) -> float: ...
```

Key design:
- `open_video` and `iter_frames` are separate so tests can mock the capture object.
- `iter_frames` is a generator: only one frame in memory at a time.
- Progress logged every 100 frames: `"Frame 300 / 15000"`.
- Capture object released when generator completes.

### 3.4 `preprocess.py` - Image Preprocessing

```python
def crop_roi(frame: Frame, x: int, y: int, width: int, height: int) -> Frame: ...
def preprocess_roi(image: Frame, *, upscale_factor: int = 3, threshold_value: int = 127, invert: bool = True) -> Frame: ...
```

Preprocessing pipeline per ROI crop:
1. **Grayscale** - `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`
2. **Binary threshold** - `cv2.threshold(gray, 127, 255, THRESH_BINARY_INV)` (dark text on light bg)
3. **Upscale 3x** - `cv2.resize(..., INTER_CUBIC)` (OCR works better on larger images)

Ordering rationale: grayscale before threshold (single channel needed), threshold before upscale (avoids interpolation artifacts on gray pixels).

### 3.5 `ocr/base.py` - OCR Abstraction (Protocol)

```python
@runtime_checkable
class OCREngine(Protocol):
    def image_to_string(self, image: NDArray[np.uint8]) -> str: ...
```

Using `Protocol` (structural subtyping) over ABC:
- No forced inheritance
- `runtime_checkable` for isinstance verification
- Any class with matching `image_to_string` satisfies it

### 3.6 `ocr/paddleocr_engine.py` - PaddleOCR Implementation

```python
class PaddleOCREngine:
    def __init__(self) -> None:
        """Initialize PaddleOCR with recognition-only mode.

        Uses rec-only (no detection) since we already crop exact ROIs.
        Digit-optimized: use_angle_cls=False, lang="en".
        Models are downloaded once on first run, cached locally.
        """
        self._ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            show_log=False,
            # rec-only: we feed pre-cropped images, no detection needed
        )

    def image_to_string(self, image: NDArray[np.uint8]) -> str:
        """Run OCR on a preprocessed image.

        Returns the recognized text string, or empty string on failure.
        PaddleOCR returns list of (bbox, (text, confidence)) tuples.
        We extract just the text from the highest-confidence result.
        """
        ...
```

Key PaddleOCR notes:
- **Recognition-only mode**: since we already crop exact ROIs, we skip text detection and only run recognition. This is faster.
- **No angle classification**: datalog text is always horizontal.
- **Models cached locally**: first run downloads ~10MB of models, subsequent runs are instant.
- **Suppress logging**: `show_log=False` to avoid PaddleOCR's verbose startup logs.

### 3.7 `ocr/factory.py` - Engine Registry

```python
_REGISTRY: dict[str, type] = {"paddleocr": PaddleOCREngine}

def create_engine(name: str, **kwargs) -> OCREngine: ...
def register_engine(name: str, cls: type) -> None: ...
```

To add a new engine: create a class with `image_to_string`, register it in `_REGISTRY`. No existing code changes needed.

### 3.8 `extract.py` - Per-Frame Extraction

```python
def extract_value(frame: Frame, roi: ROI, engine: OCREngine) -> int | float | None: ...
def extract_frame(frame: Frame, frame_index: int, timestamp_ms: float, rois: list[ROI], engine: OCREngine) -> FrameResult: ...
```

`extract_value`: crop -> preprocess -> OCR -> clean -> cast. Returns `None` + logs warning on failure.

Text cleaning step (between OCR and cast):
1. Strip whitespace
2. If `roi.signed` is True, preserve a leading `-` character
3. Remove all characters except digits, `.`, and leading `-`
4. Cast to `int()` or `float()` based on `roi.value_type`

This handles OCR artifacts like stray letters while preserving negative signs for parameters like boost pressure loop difference.

### 3.9 `writer.py` - CSV Writing

```python
class CSVWriter:
    def __init__(self, output_path: Path, rois: list[ROI]) -> None: ...
    def __enter__(self) -> "CSVWriter": ...
    def __exit__(self, *exc) -> None: ...
    def write_row(self, result: FrameResult, *, skip_on_any_none: bool = True) -> bool: ...
```

Context manager. Header: `frame, timestamp_ms, <roi_names...>`. ROI names include the unit of measure (e.g. `boost_pressure (mbar)`), taken directly from the `name` field in config. Skips rows where any value is None.

### 3.10 `pipeline.py` - Main Orchestration

```python
def run(config: AppConfig, video_path: Path, output_path: Path) -> None: ...
```

Flow: create engine -> open video -> log metadata -> iterate frames -> extract -> write CSV -> log summary.

### 3.11 `cli.py` - CLI Interface

```
usage: video2csv [-h] -c CONFIG [-o OUTPUT] [-v] video
```

- `video` (positional): path to MP4
- `-c/--config` (required): path to ROI JSON config
- `-o/--output` (optional): CSV output path, defaults to `<video_stem>.csv`
- `-v/--verbose`: DEBUG logging

---

## 4. JSON Config Format

```json
{
  "ocr_engine": "paddleocr",
  "rois": [
    {
      "name": "accelerator_pedal (%)",
      "x": 100,
      "y": 80,
      "width": 120,
      "height": 40,
      "value_type": "int"
    },
    {
      "name": "boost_pressure (mbar)",
      "x": 100,
      "y": 200,
      "width": 120,
      "height": 40,
      "value_type": "int"
    },
    {
      "name": "boost_pressure_loop_diff (mbar)",
      "x": 100,
      "y": 320,
      "width": 120,
      "height": 40,
      "value_type": "int",
      "signed": true
    },
    {
      "name": "boost_pressure_ref (mbar)",
      "x": 100,
      "y": 440,
      "width": 120,
      "height": 40,
      "value_type": "int"
    }
  ]
}
```

- `name`: unique, becomes CSV column header — include unit of measure (e.g. `"boost_pressure (mbar)"`)
- `x, y`: top-left corner in pixel coordinates of full frame
- `width, height`: bounding box dimensions in pixels
- `value_type`: `"int"` or `"float"` for casting OCR output
- `signed` (optional, default `false`): if `true`, negative values are preserved during OCR text cleaning
- `ocr_engine` (optional): defaults to `"paddleocr"`

---

## 5. Data Flow

```
CLI args
  -> load_config(config_path) -> AppConfig
  -> open_video(video_path) -> (cap, VideoMeta)
  -> create_engine(config.ocr_engine) -> OCREngine  (PaddleOCR model loaded once here)
  -> CSVWriter(output_path, config.rois)
  -> for frame_idx, frame in iter_frames(cap, total_frames):
       ts = frame_timestamp_ms(frame_idx, fps)
       for each ROI:
         crop_roi(frame, x, y, w, h)
         preprocess_roi(cropped)
         engine.image_to_string(processed)
         int(text) or float(text)  -> value or None + warning
       -> FrameResult(frame_index, timestamp_ms, {name: value|None})
       -> writer.write_row(result) -> written or skipped
  -> log summary (written / skipped counts)
```

Memory: only one frame in memory at any time (generator-based iteration).

---

## 6. Error Handling

| Layer | Error | Handling |
|-------|-------|----------|
| CLI | Video/config file not found | Log error, `sys.exit(1)` |
| Config | Invalid JSON / missing fields | `ValueError` with context |
| Video | Cannot open file | `RuntimeError` |
| Video | `cap.read()` returns False | Generator stops cleanly |
| OCR | PaddleOCR model download fails | Exception propagates with clear message |
| OCR | OCR returns empty/garbage text | `int()`/`float()` ValueError caught, returns `None`, logs warning |
| Writer | Any ROI value is None | Row skipped, warning logged |
| Writer | Disk failure | `OSError` propagates (crash with traceback) |

### Logging Levels

- **INFO**: Start/end summary, video metadata, row counts
- **WARNING**: OCR cast failures, skipped rows
- **DEBUG** (`-v`): Frame-by-frame progress, raw OCR text
- **ERROR**: Fatal conditions (file not found, video won't open)

---

## 7. Implementation Order

### Phase 1: Project Scaffolding
1. Create `pyproject.toml` (metadata, dependencies, `[project.scripts]` entry point)
2. Create `src/video2csv/__init__.py`
3. Create `tests/conftest.py`
4. Run `uv sync` to install dependencies

### Phase 2: Shared Models
1. Implement `models.py` (pure dataclasses, no external deps)

### Phase 3: Video Decoding (standalone, testable first)
1. Implement `video.py` (`open_video`, `iter_frames`, `frame_timestamp_ms`)
2. Implement `dump_frames.py` — standalone CLI tool for verification:
   - Takes a video path as argument
   - Dumps 5 frames as PNG at 20%, 40%, 60%, 80%, 100% progress
   - Output: `frame_0.png` through `frame_4.png` in current directory
   - Usage: `uv run python -m video2csv.dump_frames <video_path>`
   - Serves as a manual verification tool for video decoding
3. Write `tests/test_video.py`:
   - `frame_timestamp_ms` with known values (pure arithmetic)
   - `open_video` raises on missing path
   - `iter_frames` with mock `cv2.VideoCapture`
4. Verify: `uv run pytest tests/test_video.py -v`

### Phase 4: Image Preprocessing
1. Implement `preprocess.py` (`crop_roi`, `preprocess_roi`)
2. Write `tests/test_preprocess.py`:
   - `crop_roi` on synthetic array with known pixels
   - `preprocess_roi` output shape, dtype, channel count

### Phase 5: OCR Abstraction + PaddleOCR
1. Implement `ocr/base.py`, `ocr/paddleocr_engine.py`, `ocr/factory.py`, `ocr/__init__.py`
2. Write `tests/test_ocr.py`:
   - Protocol conformance (`isinstance` check)
   - Factory lookup + `KeyError` on unknown
   - OCR on synthetic digit image (`cv2.putText`)

### Phase 6: Per-Frame Extraction
1. Implement `extract.py`
2. Write `tests/test_extract.py` with mock OCR engine

### Phase 7: CSV Writer
1. Implement `writer.py`
2. Write `tests/test_writer.py` (write + read back, skip logic)

### Phase 8: Config Loading
1. Implement `config.py`
2. Write `tests/test_config.py` (valid load, validation errors, defaults)

### Phase 9: Pipeline + CLI
1. Implement `pipeline.py`, `cli.py`, `__main__.py`
2. Write `tests/test_pipeline.py` (integration with synthetic video + mock OCR)
3. End-to-end: `uv run video2csv 20260207_133847.mp4 -c config.json -v`

---

## 8. Testing Strategy

Each component is independently testable:

| Component | Mocking | External Deps |
|-----------|---------|---------------|
| `models.py` | None | None |
| `video.py` | Mock `cv2.VideoCapture` | None |
| `preprocess.py` | None | `numpy`, `cv2` (real) |
| `ocr/` | Stub class for protocol | PaddleOCR (for integration) |
| `extract.py` | Mock `OCREngine` | None |
| `writer.py` | None | None (temp files) |
| `config.py` | None | None (temp JSON) |
| `pipeline.py` | Synthetic video + mock OCR | `cv2` (real) |

### Shared Test Fixtures (`conftest.py`)

- `sample_frame`: 640x480 BGR numpy array
- `sample_rois`: two ROIs within 640x480 bounds
- `tiny_video`: 5-frame 64x64 MP4 created with `cv2.VideoWriter`
- `MockOCREngine`: returns predetermined strings, tracks call count
